import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import torch
import yaml
from scipy.io import wavfile
from torch.utils.data import DataLoader

from utils.model import get_model
from utils.tools import (
    amp_autocast,
    iter_device_batches,
    log,
    resolve_amp_config,
    resolve_gpu_prefetch,
    synth_one_sample,
)
from model import FastSpeech2Loss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_load_eval(path, device):
    path = Path(path)
    if not path.is_file() and path.with_name(path.name + ".zip").is_file():
        zip_path = path.with_name(path.name + ".zip")
        with zipfile.ZipFile(zip_path) as archive:
            member = path.name
            if member not in archive.namelist():
                candidates = [
                    name for name in archive.namelist() if Path(name).name == path.name
                ]
                if not candidates:
                    raise FileNotFoundError(f"{path} not found in {zip_path}")
                member = candidates[0]
            with archive.open(member) as handle:
                payload = io.BytesIO(handle.read())
        try:
            return torch.load(payload, map_location=device, weights_only=False)
        except TypeError:
            payload.seek(0)
            return torch.load(payload, map_location=device)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_eval_hifigan_vocoder(model_config, speaker, device):
    import hifigan

    hifigan_dir = Path(__file__).resolve().parent / "hifigan"
    with (hifigan_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = hifigan.AttrDict(json.load(handle))

    vocoder = hifigan.Generator(config)
    if speaker == "LJSpeech":
        checkpoint = hifigan_dir / "generator_LJSpeech.pth.tar"
    elif speaker == "universal":
        checkpoint = hifigan_dir / "generator_universal.pth.tar"
    else:
        raise ValueError(f"Unsupported HiFi-GAN speaker: {speaker}")

    state = _torch_load_eval(checkpoint, device)
    vocoder.load_state_dict(state["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    model_config["vocoder"] = {"model": "HiFi-GAN", "speaker": speaker}
    return vocoder


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


def evaluate(
    model,
    step,
    configs,
    logger=None,
    vocoder=None,
    return_losses=False,
    synth_audio_dir=None,
):
    preprocess_config, model_config, train_config = configs

    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    if len(dataset) == 0:
        message = "Validation Step {}, skipped: val.txt contains no samples".format(step)
        if return_losses:
            return message, None
        return message

    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        **_dataloader_kwargs(train_config),
    )

    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    amp = resolve_amp_config(train_config, device)
    gpu_prefetch = resolve_gpu_prefetch(train_config, device)

    loss_sums = [0 for _ in range(6)]
    last_batch = None
    last_output = None
    for batch in iter_device_batches(loader, device, prefetch=gpu_prefetch):
        with torch.no_grad():
            with amp_autocast(amp):
                output = model(*(batch[2:]))
                losses = loss_fn(batch, output)

            weight = len(batch[0])
            for i in range(len(losses)):
                loss_sums[i] += losses[i].item() * weight
        last_batch = batch
        last_output = output

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = (
        "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, "
        "Mel PostNet Loss: {:.4f}"
    ).format(
        step,
        loss_means[0],
        loss_means[1],
        loss_means[2],
    )

    if logger is not None:
        log(logger, step, losses=loss_means)
    should_synthesize = logger is not None or synth_audio_dir is not None
    if should_synthesize and last_batch is not None and last_output is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            last_batch,
            last_output,
            vocoder,
            model_config,
            preprocess_config,
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        if logger is not None:
            log(
                logger,
                fig=fig,
                tag="Validation/step_{}_{}".format(step, tag),
            )
            log(
                logger,
                audio=wav_reconstruction,
                sampling_rate=sampling_rate,
                tag="Validation/step_{}_{}_reconstructed".format(step, tag),
            )
            log(
                logger,
                audio=wav_prediction,
                sampling_rate=sampling_rate,
                tag="Validation/step_{}_{}_synthesized".format(step, tag),
            )
        if synth_audio_dir is not None and wav_prediction is not None:
            output_dir = Path(synth_audio_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"step_{step}_{tag}"
            wavfile.write(
                output_dir / f"{prefix}_reconstructed.wav",
                sampling_rate,
                wav_reconstruction,
            )
            wavfile.write(
                output_dir / f"{prefix}_synthesized.wav",
                sampling_rate,
                wav_prediction,
            )

    if return_losses:
        return message, loss_means
    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--vocoder-model",
        type=str,
        default="HiFi-GAN",
        choices=["HiFi-GAN", "none"],
        help="vocoder to use for standalone evaluation audio",
    )
    parser.add_argument(
        "--vocoder-speaker",
        type=str,
        default="universal",
        choices=["universal", "LJSpeech"],
        help="HiFi-GAN checkpoint variant for standalone evaluation audio",
    )
    parser.add_argument(
        "--synth-audio-dir",
        type=str,
        default=None,
        help=(
            "directory for reconstructed/synthesized eval wavs; "
            "defaults to train result_path/eval when a vocoder is enabled"
        ),
    )
    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    model = get_model(args, configs, device, train=False).to(device)
    vocoder = None
    if args.vocoder_model != "none":
        vocoder = get_eval_hifigan_vocoder(model_config, args.vocoder_speaker, device)

    synth_audio_dir = args.synth_audio_dir
    if synth_audio_dir is None and vocoder is not None:
        synth_audio_dir = os.path.join(train_config["path"]["result_path"], "eval")

    message = evaluate(
        model,
        args.restore_step,
        configs,
        vocoder=vocoder,
        synth_audio_dir=synth_audio_dir,
    )
    print(message)
