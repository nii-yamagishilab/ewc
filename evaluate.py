# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import io
import jsonlines
import pandas as pd
import numpy as np
from processors import FactVerificationProcessor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def load_pred_labels(filename):
    probs = np.loadtxt(filename, dtype=np.float64)
    pred_labels = np.argmax(probs, axis=1)
    i2label = {
        i: label for i, label in enumerate(FactVerificationProcessor().get_labels())
    }
    return [i2label[pred][0] for pred in pred_labels]


def load_gold_labels(filename):
    return [line["label"][0] for line in jsonlines.open(filename)]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = build_args()
    gold_labels = load_gold_labels(args.gold_file)
    pred_labels = load_pred_labels(args.prob_file)
    labels = FactVerificationProcessor().get_labels()
    prec = (
        precision_score(
            gold_labels, pred_labels, labels=labels, average=None, zero_division=0
        )
        * 100.0
    )
    rec = (
        recall_score(
            gold_labels, pred_labels, labels=labels, average=None, zero_division=0
        )
        * 100.0
    )
    f1 = (
        f1_score(gold_labels, pred_labels, labels=labels, average=None, zero_division=0)
        * 100.0
    )
    acc = accuracy_score(gold_labels, pred_labels) * 100.0
    mat = confusion_matrix(gold_labels, pred_labels, labels=labels)
    df = pd.DataFrame(mat, columns=labels, index=labels)
    df2 = pd.DataFrame([prec, rec, f1], columns=labels, index=["Prec:", "Rec:", "F1:"])
    results = "\n".join(
        [
            "Confusion Matrix:",
            f"{df}",
            "",
            f"{df2.round(2)}",
            "",
            f"Acc: {acc.round(2)}",
        ]
    )

    print(results)

    with io.open(args.out_file, "w", encoding="utf-8") as f:
        f.write(results + "\n")


if __name__ == "__main__":
    main()
