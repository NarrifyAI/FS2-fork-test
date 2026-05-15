import argparse
import copy
import os
import re
import time
from pathlib import Path

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_progress_bar = tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import (
    amp_autocast,
    iter_device_batches,
    log,
    make_grad_scaler,
    resolve_amp_config,
    resolve_gpu_prefetch,
    synth_one_sample,
)
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

try:
    from forge.events import EventType, emit
except ImportError:  # pragma: no cover - optional Forge integration
    EventType = None
    emit = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def _split_combined_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("Forge FastSpeech2 config must be a YAML mapping")
    if {"preprocess", "model", "train"}.issubset(raw):
        train_config = copy.deepcopy(raw["train"])
        if "data" in raw and "data" not in train_config:
            train_config["data"] = copy.deepcopy(raw["data"])
        return (
            copy.deepcopy(raw["preprocess"]),
            copy.deepcopy(raw["model"]),
            train_config,
        )
    if {"preprocess_config", "model_config", "train_config"}.issubset(raw):
        train_config = copy.deepcopy(raw["train_config"])
        if "data" in raw and "data" not in train_config:
            train_config["data"] = copy.deepcopy(raw["data"])
        return (
            copy.deepcopy(raw["preprocess_config"]),
            copy.deepcopy(raw["model_config"]),
            train_config,
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
    message = (
        "Step {}/{}, Total Loss: {:.4f}, Mel Loss: {:.4f}, "
        "Mel PostNet Loss: {:.4f}"
    ).format(
        step,
        total_step,
        values[0],
        values[1],
        values[2],
    )
    with open(os.path.join(train_log_path, "log.txt"), "a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    log(logger, step, losses=values)
    return message


def _profile_state(train_config, device, train_log_path):
    config = train_config.get("profile", {}) or {}
    enabled = bool(config.get("enabled", False))
    return {
        "enabled": enabled,
        "interval": max(1, int(config.get("interval_step", 100) or 100)),
        "sync_cuda": enabled
        and device.type == "cuda"
        and bool(config.get("cuda_sync", True)),
        "log_path": os.path.join(train_log_path, "profile.txt"),
        "stats": [],
    }


def _profile_sync(profile):
    if profile["sync_cuda"]:
        torch.cuda.synchronize()


def _profile_now(profile):
    _profile_sync(profile)
    return time.perf_counter()


def _profile_elapsed_ms(profile, started_at):
    _profile_sync(profile)
    return (time.perf_counter() - started_at) * 1000.0


def _as_int(value):
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _profile_batch_stats(batch):
    samples = len(batch[0])
    mel_lens = batch[7]
    max_mel_len = _as_int(batch[8])
    if torch.is_tensor(mel_lens):
        mel_frames = int(mel_lens.sum().item())
    else:
        mel_frames = int(sum(mel_lens))
    padded_mel_frames = max(1, samples * max_mel_len)
    padding_ratio = max(
        0.0,
        1.0 - (float(mel_frames) / float(padded_mel_frames)),
    )
    return {
        "samples": samples,
        "mel_frames": mel_frames,
        "max_mel_len": max_mel_len,
        "padding_ratio": padding_ratio,
    }


def _profile_log(profile, outer_bar, logger, step, message):
    with open(profile["log_path"], "a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    outer_bar.write(message)
    if logger is not None:
        logger.add_text("Profile/log", message, step)


def _profile_record(profile, outer_bar, logger, step, sample):
    if not profile["enabled"]:
        return

    profile["stats"].append(sample)
    if step % profile["interval"] != 0:
        return

    stats = profile["stats"]
    profile["stats"] = []
    count = len(stats)
    total_ms = sum(item["step_ms"] for item in stats)
    total_seconds = max(total_ms / 1000.0, 1e-9)
    samples = sum(item["samples"] for item in stats)
    mel_frames = sum(item["mel_frames"] for item in stats)
    message = (
        "Perf Step {}, avg {} steps: step {:.1f}ms, data {:.1f}ms, "
        "forward {:.1f}ms, backward {:.1f}ms, optimizer {:.1f}ms, "
        "samples/s {:.1f}, mel_frames/s {:.0f}, padding {:.1f}%, max_mel {:.0f}"
    ).format(
        step,
        count,
        total_ms / count,
        sum(item["data_ms"] for item in stats) / count,
        sum(item["forward_ms"] for item in stats) / count,
        sum(item["backward_ms"] for item in stats) / count,
        sum(item["optimizer_ms"] for item in stats) / count,
        samples / total_seconds,
        mel_frames / total_seconds,
        100.0 * sum(item["padding_ratio"] for item in stats) / count,
        sum(item["max_mel_len"] for item in stats) / count,
    )
    _profile_log(profile, outer_bar, logger, step, message)


def _profile_event(profile, outer_bar, logger, step, label, elapsed_ms):
    if not profile["enabled"]:
        return
    message = "Perf {} Step {}, elapsed {:.1f}ms".format(label, step, elapsed_ms)
    _profile_log(profile, outer_bar, logger, step, message)


_EARLY_STOP_METRICS = {
    "val_total_loss": 0,
    "val_mel_loss": 1,
    "val_mel_postnet_loss": 2,
    "val_pitch_loss": 3,
    "val_energy_loss": 4,
}


def _early_stop_state(train_config):
    config = train_config.get("early_stop", {}) or {}
    enabled = bool(config.get("enabled", False))
    metric = str(config.get("metric", "val_total_loss"))
    if metric not in _EARLY_STOP_METRICS:
        supported = ", ".join(sorted(_EARLY_STOP_METRICS))
        raise ValueError(
            f"Unsupported FastSpeech2 early_stop.metric={metric!r}; "
            f"supported values: {supported}"
        )

    patience = int(config.get("patience", 20))
    if patience < 1:
        raise ValueError("FastSpeech2 early_stop.patience must be >= 1")

    min_delta = float(config.get("min_delta", 0.0))
    if min_delta < 0:
        raise ValueError("FastSpeech2 early_stop.min_delta must be >= 0")

    return {
        "enabled": enabled,
        "metric": metric,
        "min_step": max(0, int(config.get("min_step", 0))),
        "patience": patience,
        "min_delta": min_delta,
        "best": None,
        "best_step": None,
        "bad_validations": 0,
    }


def _update_early_stop(state, step, val_losses):
    if not state["enabled"] or val_losses is None or step < state["min_step"]:
        return False, None

    metric = state["metric"]
    value = float(val_losses[_EARLY_STOP_METRICS[metric]])
    best = state["best"]
    if best is None or value < best - state["min_delta"]:
        state["best"] = value
        state["best_step"] = step
        state["bad_validations"] = 0
        return (
            False,
            f"Early Stop: {metric} improved to {value:.6f} at step {step}",
        )

    state["bad_validations"] += 1
    message = (
        f"Early Stop: {metric}={value:.6f} did not improve beyond "
        f"{best:.6f} from step {state['best_step']} "
        f"({state['bad_validations']}/{state['patience']})"
    )
    return state["bad_validations"] >= state["patience"], message


def _write_training_message(train_log_path, message):
    with open(os.path.join(train_log_path, "log.txt"), "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _dataloader_kwargs(train_config):
    data_config = train_config.get("data", {}) or {}
    num_workers = max(0, int(data_config.get("num_workers", 0) or 0))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(data_config.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            data_config.get("persistent_workers", True)
        )
        kwargs["prefetch_factor"] = max(
            1,
            int(data_config.get("prefetch_factor", 2) or 2),
        )
    return kwargs


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=False
    )
    if len(dataset) == 0:
        raise ValueError("FastSpeech2 train.txt contains no samples")
    batch_size = int(train_config["optimizer"]["batch_size"])
    data_config = train_config.get("data", {}) or {}
    group_size = max(1, int(data_config.get("batch_group_size", 4) or 4))
    loader_batch_size = min(len(dataset), max(batch_size, batch_size * group_size))
    loader = DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        **_dataloader_kwargs(train_config),
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

    if torch.cuda.device_count() > 1:
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
    amp = resolve_amp_config(train_config, device)
    scaler = make_grad_scaler(amp)
    gpu_prefetch = resolve_gpu_prefetch(train_config, device)
    profile = _profile_state(train_config, device, train_log_path)
    if amp["enabled"]:
        print(
            "FastSpeech2 AMP enabled: "
            f"dtype={amp['dtype_name']}, grad_scaler={scaler is not None}"
        )
    elif amp["requested"]:
        print("FastSpeech2 AMP requested but CUDA is unavailable; using float32.")
    if gpu_prefetch:
        print("FastSpeech2 GPU prefetch enabled.")
    if profile["enabled"]:
        sync_label = "with CUDA sync" if profile["sync_cuda"] else "without CUDA sync"
        print(
            "FastSpeech2 profiling enabled: "
            f"interval={profile['interval']} steps, {sync_label}."
        )
    early_stop = _early_stop_state(train_config)
    if early_stop["enabled"] and val_step <= 0:
        raise ValueError(
            "FastSpeech2 early_stop.enabled requires train.step.val_step > 0"
        )

    outer_bar = _progress_bar(total=total_step, desc="Training", position=0)
    outer_bar.n = max(0, step - 1)
    outer_bar.refresh()

    try:
        while step <= total_step:
            batch_iter = iter_device_batches(loader, device, prefetch=gpu_prefetch)
            while step <= total_step:
                data_started_at = _profile_now(profile)
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                data_ms = _profile_elapsed_ms(profile, data_started_at)
                early_stop_requested = False
                batch_stats = _profile_batch_stats(batch) if profile["enabled"] else {}

                forward_started_at = _profile_now(profile)
                with amp_autocast(amp):
                    output = model(*(batch[2:]))
                    losses = loss_fn(batch, output)
                    total_loss = losses[0] / grad_acc_step
                forward_ms = _profile_elapsed_ms(profile, forward_started_at)

                backward_started_at = _profile_now(profile)
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                backward_ms = _profile_elapsed_ms(profile, backward_started_at)

                optimizer_ms = 0.0
                if step % grad_acc_step == 0:
                    optimizer_started_at = _profile_now(profile)
                    if scaler is not None:
                        scaler.unscale_(optimizer._optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    if scaler is not None:
                        optimizer.update_learning_rate()
                        scaler.step(optimizer._optimizer)
                        scaler.update()
                    else:
                        optimizer.step_and_update_lr()
                    optimizer.zero_grad()
                    optimizer_ms = _profile_elapsed_ms(profile, optimizer_started_at)

                step_ms = _profile_elapsed_ms(profile, data_started_at)
                if profile["enabled"]:
                    batch_stats.update(
                        {
                            "data_ms": data_ms,
                            "forward_ms": forward_ms,
                            "backward_ms": backward_ms,
                            "optimizer_ms": optimizer_ms,
                            "step_ms": step_ms,
                        }
                    )
                    _profile_record(profile, outer_bar, train_logger, step, batch_stats)

                if log_step > 0 and step % log_step == 0:
                    outer_bar.write(
                        _log_losses(
                            train_logger,
                            train_log_path,
                            step,
                            total_step,
                            losses,
                        )
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
                    validation_started_at = _profile_now(profile)
                    message, val_losses = evaluate(
                        model,
                        step,
                        configs,
                        val_logger,
                        vocoder,
                        return_losses=True,
                    )
                    validation_ms = _profile_elapsed_ms(
                        profile,
                        validation_started_at,
                    )
                    with open(
                        os.path.join(val_log_path, "log.txt"),
                        "a",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(message + "\n")
                    outer_bar.write(message)
                    _profile_event(
                        profile,
                        outer_bar,
                        train_logger,
                        step,
                        "Validation",
                        validation_ms,
                    )
                    early_stop_requested, early_stop_message = _update_early_stop(
                        early_stop,
                        step,
                        val_losses,
                    )
                    if early_stop_message:
                        _write_training_message(train_log_path, early_stop_message)
                        outer_bar.write(early_stop_message)
                    model.train()

                should_save = save_step > 0 and step % save_step == 0
                final_step = step >= total_step or early_stop_requested
                if should_save or final_step:
                    checkpoint_started_at = _profile_now(profile)
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
                    checkpoint_ms = _profile_elapsed_ms(profile, checkpoint_started_at)
                    _profile_event(
                        profile,
                        outer_bar,
                        train_logger,
                        step,
                        "Checkpoint",
                        checkpoint_ms,
                    )

                if final_step:
                    return
                step += 1
                outer_bar.update(1)

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
