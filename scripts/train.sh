#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
usage() {
  cat <<'EOF'
Usage: bash scripts/train.sh --dataset PATH --resume_from CHECKPOINT [options]

  --dataset, --dataset_folder PATH  Paired dataset directory (DATASET_DIR)
  --resume_from PATH               Checkpoint file (RESUME_FROM)
  --gpu_ids IDS                    Two GPU IDs; default: 2,3 (GPU_IDS)
  --output_dir PATH                Default: outputs/experiment_2 (OUTPUT_DIR)
  --main_process_port PORT         Default: 29501 (MASTER_PORT)
  -h, --help                       Show this launcher help

Options accept --name VALUE or --name=VALUE and override environment variables.
Other arguments are forwarded to the Python training script.
Relative paths are resolved from the release directory.
EOF
}
training_args=()
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dataset|--dataset_folder|--resume_from|--gpu_ids|--output_dir|--main_process_port|--dataset=*|--dataset_folder=*|--resume_from=*|--gpu_ids=*|--output_dir=*|--main_process_port=*)
      option="${1%%=*}"
      if [[ "$1" == *=* ]]; then
        value="${1#*=}"
        shift
      else
        if (($# < 2)) || [[ "$2" == --* ]]; then
          echo "Error: $option requires a value" >&2; exit 2
        fi
        value="$2"
        shift 2
      fi
      if [[ -z "$value" ]]; then
        echo "Error: $option requires a nonempty value" >&2; exit 2
      fi
      case "$option" in
        --dataset|--dataset_folder) DATASET_DIR="$value" ;;
        --resume_from) RESUME_FROM="$value" ;;
        --gpu_ids) GPU_IDS="$value" ;;
        --output_dir) OUTPUT_DIR="$value" ;;
        --main_process_port) MASTER_PORT="$value" ;;
      esac
      ;;
    *) training_args+=("$1"); shift ;;
  esac
done
if [[ -z "${DATASET_DIR:-}" || -z "${RESUME_FROM:-}" ]]; then
  echo "Error: supply --dataset PATH and --resume_from CHECKPOINT, or set DATASET_DIR and RESUME_FROM." >&2
  usage >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/preflight.py --dataset_folder "$DATASET_DIR" --resume_from "$RESUME_FROM" --gpu_ids "${GPU_IDS:-2,3}"
exec "$PYTHON_BIN" -m accelerate.commands.launch \
  --config_file configs/accelerate_2gpu.yaml --multi_gpu --num_processes=2 \
  --gpu_ids "${GPU_IDS:-2,3}" --main_process_port "${MASTER_PORT:-29501}" \
  src/train_pretrained_pix2pix_turbo.py \
  --output_dir "${OUTPUT_DIR:-outputs/experiment_2}" \
  --dataset_folder "$DATASET_DIR" --resolution=512 --train_batch_size=1 \
  --enable_xformers_memory_efficient_attention --resume_from "$RESUME_FROM" \
  --viz_freq 25 --report_to None --lambda_clipsim 0 \
  --eval_freq 4170 --num_samples_eval 4170 --gradient_checkpointing "${training_args[@]}"
