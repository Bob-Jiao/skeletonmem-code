#!/usr/bin/env bash
set -euo pipefail

root=/data/jiaoguanbo/skeletonmem
base_model=/data/jiaoguanbo/models/lingbot-va-base
train_run=${root}/results/stage2/gate_a/full_train/full4k_gacc10_20260818T134819Z
eval_root=${root}/results/stage2/gate_a

steps=(1000 2000 3000 4000)

if [ ! -d "${base_model}" ]; then
  echo "missing base model: ${base_model}" >&2
  exit 2
fi
if [ ! -d "${train_run}/checkpoints" ]; then
  echo "missing train checkpoints: ${train_run}/checkpoints" >&2
  exit 3
fi

for step in "${steps[@]}"; do
  ckpt=${train_run}/checkpoints/checkpoint_step_${step}
  model=${eval_root}/full4k_eval_model_step${step}

  if [ ! -f "${ckpt}/transformer/diffusion_pytorch_model.safetensors" ]; then
    echo "missing transformer safetensors: ${ckpt}" >&2
    exit 4
  fi

  mkdir -p "${model}"
  for component in assets text_encoder tokenizer vae; do
    if [ ! -e "${base_model}/${component}" ]; then
      echo "missing base component: ${base_model}/${component}" >&2
      exit 5
    fi
    if [ -e "${model}/${component}" ] || [ -L "${model}/${component}" ]; then
      current=$(readlink -f "${model}/${component}" || true)
      expected=$(readlink -f "${base_model}/${component}")
      if [ "${current}" != "${expected}" ]; then
        echo "refusing to overwrite non-matching component: ${model}/${component}" >&2
        exit 6
      fi
    else
      ln -s "${base_model}/${component}" "${model}/${component}"
    fi
  done

  if [ -e "${model}/transformer" ] || [ -L "${model}/transformer" ]; then
    current=$(readlink -f "${model}/transformer" || true)
    expected=$(readlink -f "${ckpt}/transformer")
    if [ "${current}" != "${expected}" ]; then
      echo "refusing to overwrite non-matching transformer: ${model}/transformer" >&2
      exit 7
    fi
  else
    ln -s "${ckpt}/transformer" "${model}/transformer"
  fi

  sha=$(sha256sum "${model}/transformer/diffusion_pytorch_model.safetensors" | awk '{print $1}')
  printf 'step=%s model=%s transformer_sha256=%s\n' "${step}" "${model}" "${sha}"
done
