#!/usr/bin/env bash
set -euo pipefail

dataset_root=${1:-/data/jiaoguanbo/LIBERO/libero_all}
target_base=${2:-/data/yaoyifei/dataset/fastwam/libero_mujoco3.3.2}

declare -A suite_map=(
  [libero_10_no_noops_lingbot]=libero_10_no_noops_lerobot
  [libero_goal_no_noops_lingbot]=libero_goal_no_noops_lerobot
  [libero_object_no_noops_lingbot]=libero_object_no_noops_lerobot
  [libero_spatial_no_noops_lingbot]=libero_spatial_no_noops_lerobot
)

for lingbot_suite in "${!suite_map[@]}"; do
  lerobot_suite=${suite_map[$lingbot_suite]}
  suite_dir="${dataset_root}/${lingbot_suite}"
  if [ ! -d "${suite_dir}" ]; then
    echo "missing suite dir: ${suite_dir}" >&2
    exit 2
  fi

  for kind in data videos; do
    link_path="${suite_dir}/${kind}"
    target_path="${target_base}/${lerobot_suite}/${kind}"
    if [ ! -d "${target_path}" ]; then
      echo "missing target: ${target_path}" >&2
      exit 3
    fi
    if [ -e "${link_path}" ] && [ ! -L "${link_path}" ]; then
      echo "refusing to replace non-symlink: ${link_path}" >&2
      exit 4
    fi
    rm -f "${link_path}"
    ln -s "${target_path}" "${link_path}"
    printf '%s -> %s\n' "${link_path}" "$(readlink "${link_path}")"
  done
done

echo "RELINK_DONE"
