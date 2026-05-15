import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        inventory_path = os.path.join(
            preprocess_config["path"]["preprocessed_path"], "phoneme_inventory.json"
        )
        if os.path.exists(inventory_path):
            with open(inventory_path, "r", encoding="utf-8") as f:
                inventory = json.load(f)
            if isinstance(inventory, dict) and inventory:
                ids = []
                for value in inventory.values():
                    try:
                        ids.append(int(value))
                    except (TypeError, ValueError):
                        pass
                if ids:
                    model_config["n_src_vocab"] = max(ids) + 1
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        if "multi_speaker" in model_config:
            raise ValueError(
                "model.multi_speaker speaker-ID conditioning is no longer supported; "
                "use model.speaker_conditioning.mode=external_embedding"
            )
        speaker_config = model_config.get("speaker_conditioning")
        if not isinstance(speaker_config, dict):
            raise ValueError("model.speaker_conditioning is required")
        if speaker_config.get("mode") != "external_embedding":
            raise ValueError(
                "model.speaker_conditioning.mode must be external_embedding"
            )
        if speaker_config.get("projection") != "linear":
            raise ValueError("model.speaker_conditioning.projection must be linear")
        speaker_input_dim = int(speaker_config.get("input_dim", 0) or 0)
        if speaker_input_dim <= 0:
            raise ValueError("model.speaker_conditioning.input_dim must be positive")
        self.speaker_input_dim = speaker_input_dim
        self.speaker_emb = nn.Linear(
            speaker_input_dim,
            model_config["transformer"]["encoder_hidden"],
        )

    def forward(
        self,
        speaker_embeddings,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        prosody_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if speaker_embeddings is None:
            raise ValueError("speaker_embeddings are required")
        if speaker_embeddings.dim() != 2:
            raise ValueError("speaker_embeddings must have shape [batch, dim]")
        if speaker_embeddings.size(1) != self.speaker_input_dim:
            raise ValueError(
                "speaker_embeddings dimension "
                f"{speaker_embeddings.size(1)} does not match {self.speaker_input_dim}"
            )
        speaker_conditioning = self.speaker_emb(speaker_embeddings)
        output = output + speaker_conditioning.unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            prosody_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
