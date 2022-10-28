# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from argparse import Namespace
from filelock import FileLock
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from lightning_base import BaseTransformer, generic_train
from modeling_base import BaseModel
from processors import (
    FactVerificationProcessor,
    compute_metrics,
    convert_examples_to_features,
)

MODEL_NAMES_MAPPING = {"base": BaseModel}


class FactVerificationTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        num_labels = len(FactVerificationProcessor().get_labels())

        rank_zero_info(f"model_name: {hparams.model_name}")
        model = MODEL_NAMES_MAPPING[hparams.model_name](hparams, num_labels)

        super().__init__(
            hparams,
            num_labels=num_labels,
            model=model,
            config=None if model is None else model.config,
        )

    def create_features(self, set_type, filepath):
        rank_zero_info(f"Create features from [{filepath}]")
        hparams = self.hparams
        processor = FactVerificationProcessor()

        examples = processor.get_examples(
            filepath, set_type, self.training, hparams.use_title
        )

        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=hparams.max_seq_length,
            label_list=processor.get_labels(),
            threads=hparams.num_workers,
        )

        num_examples = processor.get_length(filepath)

        def empty_tensor_1():
            return torch.empty(num_examples, dtype=torch.long)

        def empty_tensor_2():
            return torch.empty((num_examples, hparams.max_seq_length), dtype=torch.long)

        input_ids = empty_tensor_2()
        attention_mask = empty_tensor_2()
        token_type_ids = empty_tensor_2()
        labels = empty_tensor_1()

        for i, feature in enumerate(features):
            input_ids[i] = torch.tensor(feature.input_ids)
            attention_mask[i] = torch.tensor(feature.attention_mask)
            if feature.token_type_ids is not None:
                token_type_ids[i] = torch.tensor(feature.token_type_ids)
            labels[i] = torch.tensor(feature.label)

        return [input_ids, attention_mask, token_type_ids, labels]

    def cached_feature_file(self, mode):
        dirname = "ewc_" + Path(self.hparams.data_dir).parts[-1]
        feat_dirpath = Path(self.hparams.cache_dir) / dirname
        feat_dirpath.mkdir(parents=True, exist_ok=True)
        pt = self.hparams.pretrained_model_name.replace("/", "__")
        return (
            feat_dirpath
            / f"cached_{mode}_{pt}_{self.hparams.max_seq_length}_{self.hparams.seed}"
        )

    def prepare_data(self):
        if self.training:
            if self.hparams.multiple_training_datasets:
                dataset_types = [
                    "train1",
                    "train2",
                    "dev",
                ]  # support up to two training datasets
            else:
                dataset_types = ["train", "dev"]

            for dataset_type in dataset_types:
                if dataset_type == "dev" and self.hparams.skip_validation:
                    continue
                cached_feature_file = self.cached_feature_file(dataset_type)
                lock_path = cached_feature_file.with_suffix(".lock")
                with FileLock(lock_path):
                    if (
                        cached_feature_file.exists()
                        and not self.hparams.overwrite_cache
                    ):
                        rank_zero_info(f"Feature file [{cached_feature_file}] exists!")
                        continue

                    filepath = Path(self.hparams.data_dir) / f"{dataset_type}.jsonl"
                    assert filepath.exists(), f"Cannot find [{filepath}]"
                    feature_list = self.create_features(dataset_type, filepath)
                    rank_zero_info(f"\u2728 Saving features to [{cached_feature_file}]")
                    torch.save(feature_list, cached_feature_file)

    def init_parameters(self):
        base_name = self.config.model_type  # e.g., bert, roberta, ...
        no_init = [base_name]
        rank_zero_info(f"\U0001F4A5 Force no_init to [{base_name}]")
        if self.hparams.load_weights:
            no_init += ["classifier"]
            rank_zero_info("\U0001F4A5 Force no_init to [classifier]")
        if self.hparams.no_init:
            no_init += self.hparams.no_init
            rank_zero_info(f"\U0001F4A5 Force no_init to {self.hparams.no_init}")
        for n, p in self.model.named_parameters():
            if any(ni in n for ni in no_init):
                continue
            rank_zero_info(f"Initialize [{n}]")
            if "bias" not in n:
                p.data.normal_(mean=0.0, std=self.config.initializer_range)
            else:
                p.data.zero_()

    def _get_dataloader(self, mode, batch_size, num_workers):
        if self.training and mode == "dev" and self.hparams.skip_validation:
            return None
        cached_feature_file = self.cached_feature_file(mode)
        assert cached_feature_file.exists(), f"Cannot find [{cached_feature_file}]"
        feature_list = torch.load(cached_feature_file)
        shuffle = True if "train" in mode and self.training else False
        rank_zero_info(
            f"Load features from [{cached_feature_file}] with shuffle={shuffle}"
        )
        return DataLoader(
            TensorDataset(*feature_list),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_dataloader(self, mode, batch_size, num_workers):
        if mode == "train" and self.hparams.multiple_training_datasets:
            if self.hparams.use_concat_dataset:
                feature_lists = [
                    torch.load(self.cached_feature_file(f"{mode}{i}"))
                    for i in ["1", "2"]
                ]
                rank_zero_info(f"\u2728 Concatenate {len(feature_lists)} datasets")
                return DataLoader(
                    ConcatDataset([TensorDataset(*fl) for fl in feature_lists]),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                )
            else:
                return {
                    f"{mode}{i}": self._get_dataloader(
                        f"{mode}{i}", batch_size, num_workers
                    )
                    for i in ["1", "2"]
                }
        else:
            return self._get_dataloader(mode, batch_size, num_workers)

    def init_diag_fisher(self):
        diag_fisher = {}
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            diag_fisher[n] = torch.zeros_like(p.data)

        assert isinstance(self.train_loader, dict)
        rank_zero_info("\u2728 Use [train2] to compute diagonal matrix")
        train_loader = self.train_loader["train2"]
        num_batches = len(train_loader)
        dataset_size = len(train_loader.dataset)
        self.eval().cuda()
        for batch in tqdm(train_loader, total=num_batches, desc="Compute diag matrix"):
            batch = [b.cuda() for b in batch]
            inputs = self.build_inputs(batch)
            outputs = self(**inputs)
            self.zero_grad()
            outputs.loss.backward()
            for n, p in self.model.named_parameters():
                if n in diag_fisher and p.grad is not None:
                    grad = p.grad.detach().cpu()

                    if self.hparams.regularizer == "aewc":
                        v = abs(grad)
                    else:
                        v = grad**2

                    diag_fisher[n] += v / dataset_size

        rank_zero_info("\U0001F4A5 Register diagonal Fisher to buffer")
        for n, p in self.model.named_parameters():
            if n not in diag_fisher:
                continue
            _n = n.replace(".", "__") + "_F"
            self.model.register_buffer(
                _n, diag_fisher[n].detach().clone(), persistent=False
            )
        self.train()

    def forward(self, **inputs):
        return self.model(**inputs)

    def build_inputs(self, batch):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.config.model_type not in {"distilbert", "bart"}:
            inputs["token_type_ids"] = (
                batch[2]
                if self.config.model_type in ["bert", "xlnet", "albert"]
                else None
            )
        return inputs

    def base_training_step(self, inputs, batch_idx):
        outputs = self(**inputs)
        self.log_dict(
            {
                "train_loss": outputs.loss.detach().cpu(),
                "lr": self.lr_scheduler.get_last_lr()[-1],
            }
        )
        return outputs.loss

    def _ewc_loss(self):
        loss = 0
        for n, p in self.model.named_parameters():
            n = n.replace(".", "__")
            n_F = n + "_F"
            if not (hasattr(self.model, n) and hasattr(self.model, n_F)):
                continue
            p0 = getattr(self.model, n)
            diag_fisher = getattr(self.model, n_F)

            if self.hparams.regularizer == "aewc":
                v = abs(p - p0)
            else:
                v = (p - p0) ** 2

            loss += (diag_fisher * v).sum()
        return loss

    def ewc_training_step(self, inputs, batch_idx, eps=1e-8):
        outputs = self(**inputs)
        ewc_loss = self._ewc_loss()
        if self.hparams.regularizer == "ewc":
            ewc_loss = 0.5 * ewc_loss

        if self.hparams.regularizer == "rewc":
            ewc_loss = (ewc_loss + eps) ** 0.5

        final_loss = outputs.loss + (self.hparams.lambda_reg * ewc_loss)
        self.log_dict(
            {
                "train_loss_base": outputs.loss.detach().cpu(),
                "train_loss_ewc": ewc_loss.detach().cpu(),
                "train_loss_final": final_loss.detach().cpu(),
                "lr": self.lr_scheduler.get_last_lr()[-1],
            }
        )
        return final_loss

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            inputs = self.build_inputs(batch["train1"])
        else:
            inputs = self.build_inputs(batch)

        if "ewc" in self.hparams.regularizer:
            return self.ewc_training_step(inputs, batch_idx)
        else:
            return self.base_training_step(inputs, batch_idx)

    def validation_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return {
            "loss": outputs.loss.detach().cpu(),
            "probs": probs.detach().cpu().numpy(),
            "labels": inputs["labels"].detach().cpu().numpy(),
        }

    def predict_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs.detach().cpu().numpy()

    def validation_epoch_end(self, outputs):
        avg_loss = (
            torch.stack([x["loss"] for x in outputs]).mean().detach().cpu().item()
        )
        labels = np.concatenate([x["labels"] for x in outputs], axis=0)
        probs = np.concatenate([x["probs"] for x in outputs], axis=0)
        results = {
            **{"loss": avg_loss},
            **compute_metrics(probs, labels),
        }
        self.log_dict({f"val_{k}": torch.tensor(v) for k, v in results.items()})

    def register_weights(self):
        rank_zero_info("\U0001F4A5 Register pre-trained weights to buffer")
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            n = n.replace(".", "__")
            self.model.register_buffer(n, p.detach().clone(), persistent=False)

    def load_weights(self, checkpoint):
        rank_zero_info(f"Loading model weights from [{checkpoint}]")
        checkpoint = torch.load(
            checkpoint,
            map_location=lambda storage, loc: storage,
        )

        ckpt_lr = checkpoint["hyper_parameters"]["learning_rate"]
        rank_zero_info(f"\U0001F4AB Checkpoint's learning_rate = {ckpt_lr}")

        if "lr0" in checkpoint["hyper_parameters"]:
            ckpt_lr0 = checkpoint["hyper_parameters"]["lr0"]
            assert ckpt_lr0 == self.hparams.lr0
            rank_zero_info(f"\U0001F4AB Checkpoint's lr0 = {ckpt_lr0}")

        if "num_prior_training" in checkpoint["hyper_parameters"]:
            ckpt_num_prior_training = checkpoint["hyper_parameters"][
                "num_prior_training"
            ]
            assert self.hparams.num_prior_training > ckpt_num_prior_training
            rank_zero_info(
                f"\U0001F4AB Checkpoint's num_prior_training = {ckpt_num_prior_training}"
            )

        if self.hparams.lr_decay > 0.0:
            assert self.hparams.lr0 >= ckpt_lr
            assert self.hparams.num_prior_training > 0
            new_lr = (
                1.0 / (1.0 + self.hparams.lr_decay * self.hparams.num_prior_training)
            ) * self.hparams.lr0
            assert new_lr <= ckpt_lr

            self.hparams.learning_rate = new_lr
            rank_zero_info(
                f"\U0001F4A5 New learning_rate = {new_lr:.8f} computed from"
                f" lr0 = {self.hparams.lr0}, lr_decay = {self.hparams.lr_decay},"
                f" and num_prior_training = {self.hparams.num_prior_training}"
            )

        ckpt_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
        ckpt_dict = {k: v for k, v in ckpt_dict.items() if k in model_dict}
        assert len(ckpt_dict), "Cannot find shareable weights"
        model_dict.update(ckpt_dict)
        self.load_state_dict(model_dict)

    @staticmethod
    def add_model_specific_args(parser):
        BaseTransformer.add_model_specific_args(parser)
        parser.add_argument("--cache_dir", type=str, default="/tmp")
        parser.add_argument("--overwrite_cache", action="store_true")
        parser.add_argument("--save_all_checkpoints", action="store_true")
        parser.add_argument("--max_seq_length", type=int, default=128)
        parser.add_argument("--use_title", action="store_true")
        parser.add_argument("--no_init", nargs="+", default=[])
        parser.add_argument("--classifier_dropout_prob", type=float, default=0.1)
        parser.add_argument(
            "--regularizer",
            type=str,
            choices=["ewc", "rewc", "aewc", "none"],
            default="none",
        )
        parser.add_argument("--lambda_reg", type=float, default=1.0)
        parser.add_argument("--load_weights", type=str, default=None)
        parser.add_argument("--multiple_training_datasets", action="store_true")
        parser.add_argument("--use_concat_dataset", action="store_true")
        parser.add_argument("--num_prior_training", type=int, default=0)
        parser.add_argument("--lr0", type=float, default=None)
        parser.add_argument("--lr_decay", type=float, default=0.01)
        return parser


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FactVerificationTransformer.add_model_specific_args(parser)
    return parser.parse_args()


def main():
    args = build_args()

    if args.regularizer != "none":
        assert args.load_weights, "Must specify --load_weights [checkpoint_file]"

    if "ewc" in args.regularizer:
        assert (
            args.multiple_training_datasets
        ), "Must specify --multiple_training_datasets"
        rank_zero_info(f"Use regularizer={args.regularizer}")

    if args.multiple_training_datasets:
        for dataset in ["train1", "train2"]:
            filepath = Path(args.data_dir) / f"{dataset}.jsonl"
            assert filepath.exists(), f"Cannot find file {filepath}"

    if args.multiple_training_datasets and not args.use_concat_dataset:
        if "ewc" not in args.regularizer:
            rank_zero_info("\u2728 Concatenate multiple training datasets")
            args.use_concat_dataset = True

    if args.lr0 is None:
        rank_zero_info(f"Global lr0 set to {args.learning_rate}")
        args.lr0 = args.learning_rate

    if args.seed > 0:
        pl.seed_everything(args.seed)

    model = FactVerificationTransformer(args)

    if args.load_weights is not None:
        model.load_weights(args.load_weights)
        if args.regularizer != "none":
            model.register_weights()

    ckpt_dirpath = Path(args.default_root_dir) / "checkpoints"
    ckpt_dirpath.mkdir(parents=True, exist_ok=True)

    monitor, mode, ckpt_filename = None, "min", "{epoch}-{step}"
    dev_filepath = Path(args.data_dir) / "dev.jsonl"
    if dev_filepath.exists() and not args.skip_validation:
        monitor, mode = "val_acc", "max"
        ckpt_filename = "{epoch}-{step}-{" + monitor + ":.4f}"

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dirpath,
            filename=ckpt_filename,
            monitor=monitor,
            mode=mode,
            save_top_k=-1 if args.save_all_checkpoints else 1,
        )
    )

    if monitor is not None:
        callbacks.append(
            EarlyStopping(monitor=monitor, mode=mode, patience=args.patience)
        )

    generic_train(model, args, callbacks)


if __name__ == "__main__":
    main()
