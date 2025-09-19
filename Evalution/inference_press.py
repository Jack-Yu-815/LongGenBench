import argparse
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline
import torch
import os
from tqdm import tqdm
import kvpress
from kvpress import (
    AdaKVPress,
    BlockPress,
    ChunkKVPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVzipPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)
from kvpress.presses.generation.decoding_press import DecodingPress


def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM with command line arguments.')
    parser.add_argument('--max_length', type=int, default=8000, help='Maximum length of generation.')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--input_file', type=str, required=True, help='input file path.')
    parser.add_argument('--model_names', type=str, nargs='+', 
                       default=["meta-llama/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B", "google/gemma-3-12b-it"],
                       help='List of model names to evaluate.')
  
    args = parser.parse_args()
    return args

# Process output to split blocks and count words
def process_output(output: str) -> dict:
    blocks = output.split('#*#')
    word_count = len(output.split())
    return {"blocks": blocks, "word_count": word_count}

def read_json(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        return json.load(file)
    
def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_to_json(data: list, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_inputs(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Combine inputs, results and word counts and save them
def process_and_save_results(inputs: list, results: list, filename: str) -> None:
    combined = []
    for input_data, result_data in zip(inputs, results):
        combined.append({
            "input": input_data["prompt"],
            "checks_once": input_data["checks_once"],
            "checks_range": input_data["checks_range"],
            "checks_periodic": input_data["checks_periodic"],
            "type": input_data["type"],
            "number": input_data['number'],
            "output_blocks": result_data["blocks"],
            "word_count": result_data["word_count"]  # Adding word count here
        })
    save_to_json(combined, filename)

args = parse_args()

#input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/Dataset_short.json'
inputs = load_inputs(args.input_file)



press_dict = {
    "knorm": DecodingPress(base_press=KnormPress(), compression_interval=256, target_size=4096),
    "adakv_expected_attention_e2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)), compression_interval=256, target_size=4096, hidden_states_buffer_size=256),
    "streaming_llm": DecodingPress(base_press=StreamingLLMPress(), compression_interval=256, target_size=4096),
    # "tova": DecodingPress(base_press=TOVAPress(), compression_interval=256, target_size=4096, hidden_states_buffer_size=256),
    "keydiff": DecodingPress(base_press=KeyDiffPress(), compression_interval=256, target_size=4096, hidden_states_buffer_size=256),
    # "qfilter": DecodingPress(base_press=QFilterPress(), compression_interval=256, target_size=4096),
    
    "knorm_8": DecodingPress(base_press=KnormPress(), compression_interval=256, target_size=8192),
    "adakv_expected_attention_e2_8": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)), compression_interval=256, target_size=8192, hidden_states_buffer_size=256),
    "streaming_llm_8": DecodingPress(base_press=StreamingLLMPress(), compression_interval=256, target_size=8192),
    "keydiff_8": DecodingPress(base_press=KeyDiffPress(), compression_interval=256, target_size=8192, hidden_states_buffer_size=256),
    # "tova_8": DecodingPress(base_press=TOVAPress(), compression_interval=256, target_size=8192, hidden_states_buffer_size=256),
    # "qfilter_8": DecodingPress(base_press=QFilterPress(), compression_interval=256, target_size=8192),
    
    "knorm_2": DecodingPress(base_press=KnormPress(), compression_interval=256, target_size=2048),
    "streaming_llm_2": DecodingPress(base_press=StreamingLLMPress(), compression_interval=256, target_size=2048),
    "keydiff_2": DecodingPress(base_press=KeyDiffPress(), compression_interval=256, target_size=2048, hidden_states_buffer_size=256),
    # "tova_2": DecodingPress(base_press=TOVAPress(), compression_interval=256, target_size=2048, hidden_states_buffer_size=256),
    # "qfilter_2": DecodingPress(base_press=QFilterPress(), compression_interval=256, target_size=2048),
    "adakv_expected_attention_e2_2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)), compression_interval=256, target_size=2048, hidden_states_buffer_size=256),
    
}

for model_name in args.model_names:
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set generation parameters
    generation_config = {
        "max_new_tokens": args.max_length,
        "temperature": 0.95,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.005
    }

    # Set random seed for reproducibility
    torch.manual_seed(6211027)

    prompts = [input_data['prompt'] for input_data in inputs]

    start_time = time.time()
    results = []

    attn_implementation = "sdpa"  # use "eager" for ObservedAttentionPress and "sdpa" if you can't use "flash_attention_2"
    pipe = pipeline("kv-press-text-generation", model=model_name, device_map="auto", model_kwargs={"attn_implementation":attn_implementation, "dtype": torch.bfloat16})
    
    for press_name, press in press_dict.items():
        # Process each prompt individually (batch size = 1)
        for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts", unit="prompt")):
            try:
                cache = DynamicCache()
                with torch.no_grad():
                    generated_text = pipe(" ", question=prompt, press=press, cache=cache, max_new_tokens=args.max_length)["answer"]

                # Stop at '*** finished' if present
                if '*** finished' in generated_text:
                    generated_text = generated_text.split('*** finished')[0]
                
                # Process the output
                result = process_output(inputs[i]['prefix'] + generated_text)
                results.append(result)
                
            except Exception as e:
                tqdm.write(f"Error processing prompt {i+1}: {e}")
                # Add empty result to maintain alignment
                results.append({"blocks": [""], "word_count": 0})

        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

        output_file = f"./results/{model_name}_maxlen{args.max_length}_{press_name}.json"
        process_and_save_results(inputs, results, output_file)
        print(f"\nSaved result to {output_file}")
