# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import numpy as np
import pytorch_lightning as pl
from datetime import datetime
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import TensorDataset, DataLoader
from train import FactVerificationTransformer


def get_dataloader(model, args):
    filepath = Path(args.in_file)
    assert filepath.exists(), f"Cannot find [{filepath}]"
    dataset_type = filepath.stem
    feature_list = model.create_features(dataset_type, filepath)
    return DataLoader(
        TensorDataset(*feature_list),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    return args


def main():
    args = build_args()

    model = FactVerificationTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_file,
        strict=True if args.strict else False,
    )
    model.freeze()

    params = {}
    params["precision"] = model.hparams.precision

    trainer = pl.Trainer.from_argparse_args(
        args, logger=False, checkpoint_callback=False, **params
    )

    t_start = datetime.now()
    predictions = trainer.predict(model, get_dataloader(model, args))
    t_delta = datetime.now() - t_start
    rank_zero_info(f"Prediction took '{t_delta}'")

    probs = np.vstack([p for p in predictions])
    rank_zero_info(f"Save output probabilities to {args.out_file}")
    np.savetxt(args.out_file, probs, delimiter=" ", fmt="%.5f")


if __name__ == "__main__":
    main()
