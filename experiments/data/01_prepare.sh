#!/bin/bash
#
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

set -ex

# Download datasets and uncompress
url='https://github.com/TalSchuster/talschuster.github.io/raw/master/static'
for name in \
    'vitaminc' \
    'vitaminc_real'; do
  if [[ ! -f "${name}.zip" ]]; then
    wget "${url}/${name}.zip"
  fi
  if [[ ! -d "${name}" ]]; then
    unzip "${name}.zip"
  fi
done

url="${url}/vitaminc_baselines"
for name in \
    'fever' \
    'fever_adversarial' \
    'fever_symmetric' \
    'fever_triggers' \
    'mnli'; do
  if [[ ! -f "${name}.zip" ]]; then
    wget "${url}/${name}.zip"
  fi
  if [[ ! -d "${name}" ]]; then
    unzip "${name}.zip"
  fi
done


# Extract real train/dev sets from vitaminc
for split in 'train' 'dev'; do
  real="vitaminc_real/${split}.jsonl"
  if [[ ! -f "${real}" ]]; then
    grep ': "real"' "vitaminc/${split}.jsonl" > "${real}"
  fi
done


# Create balanced dev sets for fever/vitaminc_real
for dir in 'fever' 'vitaminc_real'; do
  if [[ ! -f "${dir}/dev.orig.jsonl" ]]; then
    mv "${dir}/dev.jsonl" "${dir}/dev.orig.jsonl"
    python 'permute.py' \
      --in_file "${dir}/dev.orig.jsonl" \
      --out_file "${dir}/dev.perm.jsonl"
    head -n 9000 "${dir}/dev.perm.jsonl" > "${dir}/dev.jsonl"
  fi
done


# mnli needs a special treatment because the original dev/test sets are identical
# Select 9k examples from the train set to be the dev set
dir='mnli'
if [[ ! -f "${dir}/train.orig.jsonl" ]]; then
  mv "${dir}/train.jsonl" "${dir}/train.orig.jsonl"
  python 'permute.py' \
    --in_file "${dir}/train.orig.jsonl" \
    --out_file "${dir}/train.perm.jsonl"
  tail -n 9000 "${dir}/train.perm.jsonl" > "${dir}/dev.jsonl"
  head -n 383702 "${dir}/train.perm.jsonl" > "${dir}/train.jsonl"
fi


# Simplify dirnames
ln -nfs vitaminc_real vitc
ln -nfs fever_adversarial adversarial
ln -nfs fever_symmetric symmetric
ln -nfs fever_triggers triggers

wc -l {mnli,fever,vitc}/{train,dev,test}.jsonl
