import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None, return_losses=False):
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
    )

    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)

    loss_sums = [0 for _ in range(6)]
    last_batch = None
    last_output = None
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                output = model(*(batch[2:]))
                losses = loss_fn(batch, output)

                weight = len(batch[0])
                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * weight
            last_batch = batch
            last_output = output

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [loss for loss in loss_means])
    )

    if logger is not None:
        log(logger, step, losses=loss_means)
        if last_batch is not None and last_output is not None:
            fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                last_batch,
                last_output,
                vocoder,
                model_config,
                preprocess_config,
            )
            log(
                logger,
                fig=fig,
                tag="Validation/step_{}_{}".format(step, tag),
            )
            sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
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
    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    model = get_model(args, configs, device, train=False).to(device)
    vocoder = get_vocoder(model_config, device)

    message = evaluate(model, args.restore_step, configs, vocoder=vocoder)
    print(message)
