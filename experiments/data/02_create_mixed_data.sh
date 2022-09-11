#!/bin/bash
#
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

set -ex

# Create mixed full training sets
mixed_datasets='mnli+fever fever+vitc mnli+vitc'
for dir in $mixed_datasets; do
  if [[ ! -d "${dir}" ]]; then
    mkdir -p "${dir}"
    data1="${dir%+*}"
    data2="${dir#*+}"
    # Just use symbolic links
    ln -nfs "../${data1}/train.jsonl" "${dir}/train1.jsonl"
    ln -nfs "../${data2}/train.jsonl" "${dir}/train2.jsonl"
    wc -l "${dir}"/*.jsonl
  fi
done

dir='mnli+fever+vitc'
if [[ ! -d "${dir}" ]]; then
  mkdir -p "${dir}"
  # The current implementation supports two training files only
  # So, merge mnli/fever and use a symbolic link for vitc
  cat "mnli/train.jsonl" "fever/train.jsonl" > "${dir}/train1.jsonl"
  ln -nfs "../vitc/train.jsonl" "${dir}/train2.jsonl"
  wc -l "${dir}"/*.jsonl
fi


# Randomly select 1%
pct=1
dirs='mnli_1 fever_1'
for dir in $dirs; do
  orig="${dir%_*}"
  mkdir -p "${dir}"
  if [[ ! -f "${dir}/train.jsonl" ]]; then
    python 'permute.py' \
      --in_file "${orig}/train.jsonl" \
      --out_file "${dir}/train.jsonl" \
      --percentage "${pct}"
    wc -l "${dir}/train.jsonl"
  fi
done


# Create mixed partial/full training sets
dirs='mnli_1+fever fever_1+vitc mnli_1+vitc'
for dir in $dirs; do
  mkdir -p "${dir}"
  data1="${dir#*+}"
  sub2="${dir%+*}"
  data2="${sub2%_*}"
  # train1 acts as the current (full) data
  ln -nfs "../${data1}/train.jsonl" "${dir}/train1.jsonl"
  # train2 acts as the prior (partial) data used for estimating empirical Fisher
  ln -nfs "../${sub2}/train.jsonl" "${dir}/train2.jsonl"
  # Merge dev sets
  cat "${data1}/dev.jsonl" "${data2}/dev.jsonl" > "${dir}/dev.jsonl"
  wc -l "${dir}"/*.jsonl
done

dir='mnli_1+fever_1+vitc'
mkdir -p "${dir}"
ln -nfs "../vitc/train.jsonl" "${dir}/train1.jsonl"
cat "mnli_1/train.jsonl" "fever_1/train.jsonl" > "${dir}/train2.jsonl"
cat "vitc/dev.jsonl" "mnli/dev.jsonl" "fever/dev.jsonl" > "${dir}/dev.jsonl"
wc -l "${dir}"/*.jsonl
