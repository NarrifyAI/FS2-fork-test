import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from text import load_phoneme_inventory, text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        data_config = train_config.get("data", {}) or {}
        self.return_tensors = bool(data_config.get("pin_memory", False))
        self.prosody_config = preprocess_config["preprocessing"].get("prosody", {})
        self.use_frame_prosody = self.prosody_config.get("enabled", False)
        self.prosody_features = self.prosody_config.get(
            "features", ["log_pitch", "voiced", "energy"]
        )
        self.derive_missing_prosody = self.prosody_config.get(
            "derive_from_pitch_energy", False
        )

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        self.phoneme_inventory = load_phoneme_inventory(
            os.path.join(self.preprocessed_path, "phoneme_inventory.json")
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(
            text_to_sequence(self.text[idx], self.cleaners, self.phoneme_inventory)
        )
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        prosody = None
        if self.use_frame_prosody:
            prosody_path = os.path.join(
                self.preprocessed_path,
                "prosody",
                "{}-prosody-{}.npy".format(speaker, basename),
            )
            if os.path.exists(prosody_path):
                prosody = np.load(prosody_path)
            elif self.derive_missing_prosody:
                prosody = self.build_frame_prosody(pitch, energy)
            else:
                raise FileNotFoundError(
                    f"Missing frame prosody file for {speaker}/{basename}: "
                    f"{prosody_path}"
                )
            if prosody.shape[0] != mel.shape[0]:
                raise ValueError(
                    f"Prosody frame count mismatch for {speaker}/{basename}: "
                    f"prosody={prosody.shape[0]}, mel={mel.shape[0]}"
                )

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }
        if prosody is not None:
            sample["prosody"] = prosody

        return sample

    def build_frame_prosody(self, pitch, energy):
        values = []
        voiced = pitch > 0
        for feature in self.prosody_features:
            if feature == "pitch":
                values.append(pitch)
            elif feature == "log_pitch":
                log_pitch = np.zeros_like(pitch, dtype=np.float32)
                log_pitch[voiced] = np.log(np.maximum(pitch[voiced], 1e-5))
                values.append(log_pitch)
            elif feature == "voiced":
                values.append(voiced.astype(np.float32))
            elif feature == "energy":
                values.append(energy)
            else:
                raise ValueError(f"Unsupported frame prosody feature: {feature}")
        return np.stack(values, axis=-1).astype(np.float32)

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        prosodies = (
            [data[idx]["prosody"] for idx in idxs] if self.use_frame_prosody else None
        )

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        if prosodies is not None:
            prosodies = pad_2D(prosodies)

        batch = (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )
        if prosodies is not None:
            batch = batch + (prosodies,)
        if self.return_tensors:
            batch = tuple(
                torch.from_numpy(item) if isinstance(item, np.ndarray) else item
                for item in batch
            )
        return batch

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        self.phoneme_inventory = load_phoneme_inventory(
            os.path.join(self.preprocessed_path, "phoneme_inventory.json")
        )
        with open(
            os.path.join(
                self.preprocessed_path, "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(
            text_to_sequence(self.text[idx], self.cleaners, self.phoneme_inventory)
        )

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
