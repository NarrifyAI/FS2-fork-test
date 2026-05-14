import argparse
import copy
import os
import re
from pathlib import Path

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from forge_utils import progress

_progress_bar = progress.tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

try:
    from forge.events import EventType, emit
except ImportError:  # pragma: no cover - optional Forge integration
    EventType = None
    emit = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_LOSS_METRICS = (
    "total_loss",
    "mel_loss",
    "mel_postnet_loss",
    "pitch_loss",
    "energy_loss",
    "duration_loss",
)


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def _split_combined_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("Forge FastSpeech2 config must be a YAML mapping")
    if {"preprocess", "model", "train"}.issubset(raw):
        return (
            copy.deepcopy(raw["preprocess"]),
            copy.deepcopy(raw["model"]),
            copy.deepcopy(raw["train"]),
        )
    if {"preprocess_config", "model_config", "train_config"}.issubset(raw):
        return (
            copy.deepcopy(raw["preprocess_config"]),
            copy.deepcopy(raw["model_config"]),
            copy.deepcopy(raw["train_config"]),
        )
    raise ValueError(
        "Forge FastSpeech2 config must contain preprocess/model/train sections"
    )


def _apply_runtime_overrides(configs, args):
    preprocess_config, model_config, train_config = configs
    if args.data_dir:
        preprocess_config.setdefault("path", {})["preprocessed_path"] = args.data_dir
    if args.checkpoint_dir:
        train_config.setdefault("path", {})["ckpt_path"] = args.checkpoint_dir
    if args.log_dir:
        paths = train_config.setdefault("path", {})
        paths["log_path"] = args.log_dir
        paths["result_path"] = os.path.join(args.log_dir, "result")
    return preprocess_config, model_config, train_config


def load_configs(args):
    if args.config:
        configs = _split_combined_config(_load_yaml(args.config))
    else:
        missing = [
            name
            for name, value in (
                ("--preprocess_config", args.preprocess_config),
                ("--model_config", args.model_config),
                ("--train_config", args.train_config),
            )
            if not value
        ]
        if missing:
            raise SystemExit(
                "Either --config or all legacy config files are required: "
                + ", ".join(missing)
            )
        configs = (
            _load_yaml(args.preprocess_config),
            _load_yaml(args.model_config),
            _load_yaml(args.train_config),
        )
    return _apply_runtime_overrides(configs, args)


def _emit_metric(metric_id, value, *, step, chart=None):
    if emit is None or EventType is None:
        return
    payload = {"id": metric_id, "value": round(float(value), 6), "epoch": int(step)}
    if chart is not None:
        payload["chart"] = chart
    emit(EventType.METRIC, **payload)


def _emit_checkpoint(path, *, step):
    if emit is None or EventType is None:
        return
    emit(EventType.CHECKPOINT, path=str(Path(path).name), epoch=int(step), is_best=False)


def _torch_load(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _model_state_dict(model):
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def _load_model_state(model, checkpoint):
    state = checkpoint.get("model") or checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError("checkpoint does not contain a model state")
    try:
        model.load_state_dict(state)
    except RuntimeError:
        if all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
            model.load_state_dict(state)
        else:
            raise


def _parse_step_from_name(path):
    match = re.search(r"(\d+)\.pth\.tar$", Path(path).name)
    return int(match.group(1)) if match else 0


def _find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(
        checkpoint_dir.glob("*.pth.tar"),
        key=lambda item: (_parse_step_from_name(item), item.stat().st_mtime),
    )
    return candidates[-1] if candidates else None


def _load_training_checkpoint(path, model, optimizer):
    checkpoint = _torch_load(path)
    _load_model_state(model, checkpoint)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = int(checkpoint.get("step") or _parse_step_from_name(path) or 0)
    epoch = int(checkpoint.get("epoch") or 1)
    optimizer.current_step = step
    print(f"Resumed FastSpeech2 checkpoint {path} at step {step}")
    return step + 1, epoch


def _load_finetune_checkpoint(path, model):
    checkpoint = _torch_load(path)
    _load_model_state(model, checkpoint)
    print(f"Loaded FastSpeech2 fine-tune checkpoint {path}")


def _save_checkpoint(path, *, model, optimizer, configs, step, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": _model_state_dict(model),
            "optimizer": optimizer._optimizer.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "configs": configs,
            "checkpoint_format": "forge_fastspeech2_v1",
        },
        path,
    )
    _emit_checkpoint(path, step=step)


def _log_losses(logger, train_log_path, step, total_step, losses):
    values = [loss.item() for loss in losses]
    message1 = "Step {}/{}, ".format(step, total_step)
    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *values
    )
    with open(os.path.join(train_log_path, "log.txt"), "a", encoding="utf-8") as handle:
        handle.write(message1 + message2 + "\n")
    log(logger, step, losses=values)
    _emit_metric("step", step, step=step)
    for name, value in zip(_LOSS_METRICS, values):
        _emit_metric(name, value, step=step, chart="Loss")
    return message1 + message2


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=False
    )
    if len(dataset) == 0:
        raise ValueError("FastSpeech2 train.txt contains no samples")
    batch_size = int(train_config["optimizer"]["batch_size"])
    group_size = 4
    loader_batch_size = min(len(dataset), max(batch_size, batch_size * group_size))
    loader = DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    model, optimizer = get_model(args, configs, device, train=True)
    start_step = args.restore_step + 1
    epoch = 1
    if args.restore_step == 0:
        resume_path = os.environ.get("RESUME_CHECKPOINT") or _find_latest_checkpoint(
            train_config["path"]["ckpt_path"]
        )
        if resume_path:
            start_step, epoch = _load_training_checkpoint(resume_path, model, optimizer)
        elif os.environ.get("FINETUNE_CHECKPOINT"):
            _load_finetune_checkpoint(os.environ["FINETUNE_CHECKPOINT"], model)

    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    vocoder = get_vocoder(model_config, device)

    for path in train_config["path"].values():
        os.makedirs(path, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    step = start_step
    grad_acc_step = int(train_config["optimizer"]["grad_acc_step"])
    grad_clip_thresh = float(train_config["optimizer"]["grad_clip_thresh"])
    total_step = int(train_config["step"]["total_step"])
    log_step = int(train_config["step"].get("log_step", 0) or 0)
    save_step = int(train_config["step"].get("save_step", 0) or 0)
    synth_step = int(train_config["step"].get("synth_step", 0) or 0)
    val_step = int(train_config["step"].get("val_step", 0) or 0)

    outer_bar = _progress_bar(total=total_step, desc="Training", position=0)
    outer_bar.n = max(0, step - 1)
    outer_bar.refresh()

    try:
        while step <= total_step:
            inner_bar = _progress_bar(total=len(loader), desc="Epoch {}".format(epoch), position=1)
            for batchs in loader:
                for batch in batchs:
                    batch = to_device(batch, device)

                    output = model(*(batch[2:]))
                    losses = loss_fn(batch, output)
                    total_loss = losses[0] / grad_acc_step

                    total_loss.backward()
                    if step % grad_acc_step == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                        optimizer.step_and_update_lr()
                        optimizer.zero_grad()

                    if log_step > 0 and step % log_step == 0:
                        outer_bar.write(
                            _log_losses(train_logger, train_log_path, step, total_step, losses)
                        )

                    if synth_step > 0 and step % synth_step == 0:
                        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                        )

                    if val_step > 0 and step % val_step == 0:
                        model.eval()
                        message, val_losses = evaluate(
                            model,
                            step,
                            configs,
                            val_logger,
                            vocoder,
                            return_losses=True,
                        )
                        with open(
                            os.path.join(val_log_path, "log.txt"),
                            "a",
                            encoding="utf-8",
                        ) as handle:
                            handle.write(message + "\n")
                        outer_bar.write(message)
                        if val_losses is not None:
                            for name, value in zip(_LOSS_METRICS, val_losses):
                                _emit_metric("val_" + name, value, step=step, chart="Validation")
                        model.train()

                    should_save = save_step > 0 and step % save_step == 0
                    final_step = step >= total_step
                    if should_save or final_step:
                        latest_path = os.path.join(
                            train_config["path"]["ckpt_path"], "latest.pt"
                        )
                        _save_checkpoint(
                            latest_path,
                            model=model,
                            optimizer=optimizer,
                            configs=configs,
                            step=step,
                            epoch=epoch,
                        )
                        if should_save:
                            candidate_path = os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            )
                            _save_checkpoint(
                                candidate_path,
                                model=model,
                                optimizer=optimizer,
                                configs=configs,
                                step=step,
                                epoch=epoch,
                            )

                    if final_step:
                        return
                    step += 1
                    outer_bar.update(1)

                inner_bar.update(1)
            epoch += 1
    finally:
        train_logger.close()
        val_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--config", type=str, default=None, help="Forge combined config")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default=None,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default=None, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, default=None, help="path to train.yaml"
    )
    args = parser.parse_args()

    main(args, load_configs(args))
