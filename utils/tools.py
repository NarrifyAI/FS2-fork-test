import contextlib
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_config(train_config, device):
    config = train_config.get("amp", {}) or {}
    requested = bool(config.get("enabled", False))
    dtype_name = str(config.get("dtype", "float16") or "float16").lower()
    if dtype_name in {"float16", "fp16", "half"}:
        dtype = torch.float16
        normalized_dtype = "float16"
    elif dtype_name in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
        normalized_dtype = "bfloat16"
    else:
        raise ValueError(
            f"Unsupported FastSpeech2 AMP dtype: {dtype_name}. "
            "Use float16 or bfloat16."
        )

    enabled = requested and device.type == "cuda"
    return {
        "requested": requested,
        "enabled": enabled,
        "dtype": dtype,
        "dtype_name": normalized_dtype,
        "use_grad_scaler": (
            enabled
            and dtype == torch.float16
            and bool(config.get("grad_scaler", True))
        ),
    }


def amp_autocast(amp):
    if not amp["enabled"]:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type="cuda", dtype=amp["dtype"])
        except TypeError:
            return torch.amp.autocast("cuda", dtype=amp["dtype"])
    return torch.cuda.amp.autocast(dtype=amp["dtype"])


def make_grad_scaler(amp):
    if not amp["use_grad_scaler"]:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=True)


def resolve_gpu_prefetch(train_config, device):
    data_config = train_config.get("data", {}) or {}
    return bool(data_config.get("gpu_prefetch", False)) and device.type == "cuda"


def _is_fastspeech2_batch(value):
    if not isinstance(value, (list, tuple)) or len(value) not in (6, 12, 13):
        return False
    if len(value) >= 3 and isinstance(value[2], (list, tuple)):
        return False
    return isinstance(value[0], (list, tuple)) and isinstance(value[1], (list, tuple))


def iter_cpu_batches(loader):
    for item in loader:
        if _is_fastspeech2_batch(item):
            yield item
            continue
        for batch in item:
            if not _is_fastspeech2_batch(batch):
                raise ValueError("Unexpected FastSpeech2 DataLoader batch shape")
            yield batch


def _record_stream(value, stream):
    if torch.is_tensor(value):
        if value.device.type == "cuda":
            value.record_stream(stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream(item, stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)


def iter_device_batches(loader, device, *, prefetch=False):
    batches = iter_cpu_batches(loader)
    if not prefetch or device.type != "cuda":
        for batch in batches:
            yield to_device(batch, device)
        return

    stream = torch.cuda.Stream()

    def preload():
        try:
            batch = next(batches)
        except StopIteration:
            return None
        with torch.cuda.stream(stream):
            return to_device(batch, device)

    next_batch = preload()
    while next_batch is not None:
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(stream)
        batch = next_batch
        _record_stream(batch, current_stream)
        next_batch = preload()
        yield batch


def _to_tensor(value, device, dtype=None):
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.from_numpy(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device, non_blocking=True)


def to_device(data, device):
    if len(data) in (12, 13):
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data[:12]

        speakers = _to_tensor(speakers, device, torch.long)
        texts = _to_tensor(texts, device, torch.long)
        src_lens = _to_tensor(src_lens, device)
        mels = _to_tensor(mels, device, torch.float32)
        mel_lens = _to_tensor(mel_lens, device)
        pitches = _to_tensor(pitches, device, torch.float32)
        energies = _to_tensor(energies, device)
        durations = _to_tensor(durations, device, torch.long)

        batch = (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )
        if len(data) == 13:
            prosodies = _to_tensor(data[12], device, torch.float32)
            batch = batch + (prosodies,)
        return batch

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = _to_tensor(speakers, device, torch.long)
        texts = _to_tensor(texts, device, torch.long)
        src_lens = _to_tensor(src_lens, device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def _to_numpy(value):
    if torch.is_tensor(value):
        if value.is_floating_point():
            value = value.float()
        return value.cpu().numpy()
    return value


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = _to_numpy(targets[11][0, :src_len].detach())
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = _to_numpy(targets[9][0, :src_len].detach())
        pitch = expand(pitch, duration)
    else:
        pitch = _to_numpy(targets[9][0, :mel_len].detach())
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = _to_numpy(targets[10][0, :src_len].detach())
        energy = expand(energy, duration)
    else:
        energy = _to_numpy(targets[10][0, :mel_len].detach())

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (_to_numpy(mel_prediction), pitch, energy),
            (_to_numpy(mel_target), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = _to_numpy(predictions[5][i, :src_len].detach())
        if predictions[2] is None:
            if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
                pitch = _to_numpy(targets[9][i, :src_len].detach())
                pitch = expand(pitch, _to_numpy(targets[11][i, :src_len].detach()))
            else:
                pitch = _to_numpy(targets[9][i, :mel_len].detach())
        elif preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = _to_numpy(predictions[2][i, :src_len].detach())
            pitch = expand(pitch, duration)
        else:
            pitch = _to_numpy(predictions[2][i, :mel_len].detach())
        if predictions[3] is None:
            if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
                energy = _to_numpy(targets[10][i, :src_len].detach())
                energy = expand(
                    energy, _to_numpy(targets[11][i, :src_len].detach())
                )
            else:
                energy = _to_numpy(targets[10][i, :mel_len].detach())
        elif preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = _to_numpy(predictions[3][i, :src_len].detach())
            energy = expand(energy, duration)
        else:
            energy = _to_numpy(predictions[3][i, :mel_len].detach())

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (_to_numpy(mel_prediction), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
