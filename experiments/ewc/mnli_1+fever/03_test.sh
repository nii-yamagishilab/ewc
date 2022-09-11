#!/bin/bash
#
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=test
#SBATCH --out=test.%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --array=1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate ewc
fi

set -ex

split='test'
max_len=128
pretrained='bert-base-uncased'

# Specify model/out dirs with the best lambda observed from dev set
lambda=$(cat 'lambda_best.txt')
model_dir="${pretrained}-${max_len}-${lambda}-mod"
out_dir="${pretrained}-${max_len}-${lambda}-out"
mkdir -p "${out_dir}"

unset -v latest

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ "${file}" -nt "${latest}" ]] && latest="${file}"
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

datasets='mnli fever vitc adversarial symmetric triggers'

for dataset in $datasets; do
  out_file="${out_dir}/${dataset}.${split}.prob"
  if [[ ! -f "${out_file}" ]]; then
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python '../../../predict.py' \
      --checkpoint_file "${latest}" \
      --in_file "../../data/${dataset}/${split}.jsonl" \
      --out_file "${out_file}" \
      --batch_size 128 \
      --gpus 1
  fi

  eval_file="${out_dir}/eval.${dataset}.${split}.txt"
  if [[ ! -f "${eval_file}" ]]; then
     python '../../../evaluate.py' \
      --gold_file "../../data/${dataset}/${split}.jsonl" \
      --prob_file "${out_dir}/${dataset}.${split}.prob" \
      --out_file "${eval_file}"
  fi
done
