#!/bin/bash

# 定义 inference.py 需要的参数
# MODEL_TYPE="mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_TYPE="THUDM/LongWriter-llama3.1-8b"
MODEL_TYPE="meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL_TYPE="google/gemma-2-9b"
# MODEL_TYPE="In2Training/FILM-7B"
# export VLLM_ATTENTION_BACKEND=FLASHINFER



MODEL_NAME=$(basename $MODEL_TYPE)
MAX_LENGTH=16384
NUM_GPUS=1
INPUT_DIR="../Dataset/Dataset_short.json"
OUTPUT_DIR="./results"
CSV_PATH="./results/accuracy_results.csv"
export CUDA_VISIBLE_DEVICES=5
# 确保输出目录存在




# 运行 inference.py
python inference_press.py --max_length $MAX_LENGTH --gpu $NUM_GPUS  --input_file $INPUT_DIR --model_names "meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-3-12b-it"

# 运行 eval.py
# python eval_api.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS
