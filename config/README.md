# Config
Here are the config files used to train the single/multi-speaker TTS models.
4 different configurations are given:
- LJSpeech: suggested configuration for LJSpeech dataset.
- LibriTTS: suggested configuration for LibriTTS dataset.
- AISHELL3: suggested configuration for AISHELL-3 dataset.
- LJSpeech_paper: closed to the setting proposed in the original FastSpeech 2 paper.

Some important hyper-parameters are explained here.

## preprocess.yaml
- **path.lexicon_path**: the lexicon (which maps words to phonemes) used by Montreal Forced Aligner. 
  We provide an English lexicon and a Mandarin lexicon. 
  Erhua (ㄦ化音) is handled in the Mandarin lexicon.
- **mel.stft.mel_fmax**: set it to 8000 if HiFi-GAN vocoder is used, and set it to null if MelGAN is used.
- **pitch.feature & energy.feature**: the original paper proposed to predict and apply frame-level pitch and energy features to the inputs of the TTS decoder to control the pitch and energy of the synthesized utterances. 
  However, in our experiments, we find that using phoneme-level features makes the prosody of the synthesized utterances more natural.
- **pitch.normalization & energy.normalization**: to normalize the pitch and energy values or not. 
  The original paper did not normalize these values.
- **prosody.enabled & prosody.features**: when enabled, preprocessing writes a frame-level ``prosody`` array aligned to mel frames. Supported default features are ``log_pitch``, ``voiced``, and ``energy``. This is intended for external frame-level prosody conditioning instead of the built-in pitch/energy bucket embeddings. Set ``derive_from_pitch_energy`` only for compatibility with old frame-level pitch/energy exports; fresh oracle runs should use explicit ``prosody`` files.

## train.yaml
- **optimizer.grad_acc_step**: the number of batches of gradient accumulation before updating the model parameters and call optimizer.zero_grad(), which is useful if you wish to train the model with a large batch size but you do not have sufficient GPU memory.
- **optimizer.anneal_steps & optimizer.anneal_rate**: the learning rate is reduced at the **anneal_steps** by the ratio specified with **anneal_rate**.

## model.yaml
- **transformer.decoder_layer**: the original paper used a 4-layer decoder, but we find it better to use a 6-layer decoder, especially for multi-speaker TTS.
- **variance_embedding.pitch_quantization**: when the pitch values are normalized as specified in ``preprocess.yaml``, it is not valid to use log-scale quantization bins as proposed in the original paper, so we use linear-scaled bins instead. 
- **prosody_conditioning.mode**: set to ``external_frame`` to inject continuous frame-level prosody features after the length regulator and before the decoder. In this mode, ``train_variance_predictors`` can be disabled so the renderer learns from external frame prosody rather than optimizing internal pitch/energy predictors.
- **duration_conditioning.mode**: set to ``external`` when phoneme durations come from another model. FastSpeech2 then uses the supplied duration targets for length regulation and does not optimize its internal duration predictor.
- **speaker_conditioning**: Forge uses ``mode: external_embedding`` with ``input_dim: 192`` and ``projection: linear``. The batch item formerly used for speaker IDs now carries a float32 speaker embedding loaded from ``speaker_emb/*.npy``. The legacy ``multi_speaker`` speaker-ID table is not supported by this fork.
- **vocoder.speaker**: should be set to 'universal' if any dataset other than LJSpeech is used.
