#!/bin/bash
#
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=train
#SBATCH --out=train.%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --array=1-5

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate ewc
fi

set -ex

seed=3435
lambda=$(sed -n "$SLURM_ARRAY_TASK_ID"p lambdas.txt)
dir="$(basename "$PWD")"

# Get parent dirname
reg="$(basename "$(dirname "$PWD")")"

# Remove suffix until "+"
prior="${dir%+*}"

# Remove suffix until "_"
prior="${prior%_*}"

data_dir="../../data/${dir}"
max_len=128
pretrained='bert-base-uncased'
prior_model_dir="${pretrained}-${max_len}-mod"

# Specify model with lambda
model_dir="${pretrained}-${max_len}-${lambda}-mod"

unset -v ckpt
for file in "../../base/${prior}/${prior_model_dir}/checkpoints"/*.ckpt; do
  [[ "${file}" -nt "${ckpt}" ]] && ckpt="${file}"
done

if [[ -d "${model_dir}" ]]; then
  echo "${model_dir} exists! Skip training."
  exit
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python '../../../train.py' \
  --data_dir "${data_dir}" \
  --default_root_dir "${model_dir}" \
  --pretrained_model_name "${pretrained}" \
  --max_seq_length "${max_len}" \
  --seed "${seed}" \
  --cache_dir "/local/$(whoami)" \
  --overwrite_cache \
  --max_epochs 3 \
  --skip_validation \
  --multiple_training_datasets \
  --regularizer "${reg}" \
  --lambda_reg "${lambda}" \
  --load_weights "${ckpt}" \
  --lr_decay 0.01 \
  --lr0 2e-5 \
  --num_prior_training 1 \
  --train_batch_size 32 \
  --accumulate_grad_batches 8 \
  --adafactor \
  --warmup_ratio 0.02 \
  --gradient_clip_val 1.0 \
  --precision 16 \
  --deterministic true \
  --gpus 1
