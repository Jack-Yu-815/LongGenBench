#!/bin/bash
set -e

# ----------------------------
# Required argument parsing
# ----------------------------
if [ $# -lt 1 ]; then
  echo "Usage: $0 --gpu_idx <GPU_INDEX>"
  exit 1
fi

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --gpu_idx)
      GPU_IDX="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$GPU_IDX" ]; then
  echo "Error: --gpu_idx is required"
  exit 1
fi

# ----------------------------
# Fixed parameters
# ----------------------------
MAX_LENGTH=16384
NUM_GPUS=1
INPUT_DIR="../Dataset/Dataset_short.json"
OUTPUT_DIR="./results"
CSV_PATH="./results/accuracy_results.csv"

mkdir -p "$OUTPUT_DIR"

# ----------------------------
# Conditional mapping
# ----------------------------
# Define mapping based on GPU index
case $GPU_IDX in
  0)
    PRESSES=("knorm")
    MODELS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-3-12b-it")
    ;;
  1)
    PRESSES=("adakv_expected_attention_e2")
    MODELS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-3-12b-it")
    ;;
  2)
    PRESSES=("streaming_llm")
    MODELS=( "meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-3-12b-it")
    ;;
  3)
    PRESSES=("keydiff")
    MODELS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-3-12b-it")
    ;;
  4)
    PRESSES=("knorm_8")
    MODELS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-3-12b-it")
    ;;
  5)
    PRESSES=("adakv_expected_attention_e2_8")
    MODELS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-3-12b-it")
    ;;
  6)
    PRESSES=("streaming_llm_8")
    MODELS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-3-12b-it")
    ;;
  7)
    PRESSES=("keydiff_8")
    MODELS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-3-12b-it")
    ;;
  *)
    echo "Error: No mapping defined for gpu_idx=$GPU_IDX"
    exit 1
    ;;
esac

# ----------------------------
# Export GPU visibility
# ----------------------------
export CUDA_VISIBLE_DEVICES=$GPU_IDX

# ----------------------------
# Run inference
# ----------------------------
echo "Running on GPU $GPU_IDX with presses: ${PRESSES[*]} and models: ${MODELS[*]}"
python inference_press.py \
  --max_length $MAX_LENGTH \
  --gpu $NUM_GPUS \
  --input_file $INPUT_DIR \
  --model_names "${MODELS[@]}" \
  --presses "${PRESSES[@]}"

# ----------------------------
# Run eval (uncomment if needed)
# ----------------------------
# python eval_api.py --data "$OUTPUT_FILE" --csv "$CSV_PATH" --gpu $NUM_GPUS