import json
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from functools import partial
import sys, os
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import argparse
import hashlib
import os
sys.path.append(f"{base_dir}/train/LLaMA-Factory/src/llamafactory/model/hdlm_utils")
from hdlm_depth2 import Depth2_HdLMModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def inference_worker(rank, data_path, model_path,
                      intermediate_layer_index=25, total_workers=4):
    model_name = os.path.basename(model_path)
    output_dir = Path(f'./aqua_sub_results_{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'result_part_{rank}.json'
    
    start_gpu = rank * 2
    gpu_list = list(range(start_gpu, start_gpu + 1))
    gpu_list_str = ','.join(map(str, gpu_list))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list_str
    device = torch.device('cuda:0') 
    
    full_data = []
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            full_data = json.load(f)
    elif data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    full_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSONL line: {e}")
    else:
        raise ValueError(f"Unsupported file format: {data_path}. Only .json and .jsonl are supported.")

    data_slice = np.array_split(full_data, total_workers)[rank]
    
    existing_results = {}
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    input_hash = hashlib.md5((item["question"]).encode()).hexdigest()
                    existing_results[input_hash] = item
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            existing_results = {}
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Depth2_HdLMModel.from_pretrained(
        model_path,
        tokenizer_path=model_path,
        think_layer_index=intermediate_layer_index,
        device_map="auto",
    )

    results = []
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
        except:
            results = []
    
    for idx, item in enumerate(tqdm(data_slice, desc=f'GPU-{gpu_list_str}')):
        input_hash = hashlib.md5((item["question"]).encode()).hexdigest()
        if input_hash in existing_results:
            continue
            
        try:
            l1, l2 = model.chat(
                tokenizer=tokenizer,
                query=item["question"],
                subtask=item['options'],
                think=True,
                dataset="commonsenseqa"
            )
            result_item = {
                "input": item["question"],
                "l1": str(l1),
                "l2": str(l2),
                "true_l1": item["rationale"],
                "true_l2": item["correct"]
            }
            
            results.append(result_item)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                f.flush() 
            existing_results[input_hash] = result_item
        except Exception as e:
            print(f"Error processing item: {e}")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                f.flush()

def evaluate_acc(all_results):
    correct = 0
    total = len(all_results)

    for res in all_results:
        if res["l2"] == res["true_l2"]:
            correct += 1

    accuracy = correct / total
    return {'accuracy': round(accuracy, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model")
    parser.add_argument("--intermediate_layer_index", type=int, default=25,
                      help="First classification layer index (default: 60)")
    args = parser.parse_args()
    multiprocessing.set_start_method('spawn')
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist")
    data_path = "{base_dir}/data/AQuA/raw/test.jsonl"
    num_gpus = 4  # according your gpu number
    with Pool(num_gpus) as pool:
        pool.map(partial(inference_worker,
                        data_path=data_path,
                        model_path=args.model_path,
                        intermediate_layer_index=args.intermediate_layer_index),
                 range(num_gpus))
    
    model_name = os.path.basename(args.model_path)
    output_dir = Path(f'./aqua_sub_results_{model_name}')
    all_results = []
    for rank in range(num_gpus):
        part_file = output_dir / f'result_part_{rank}.json'
        if part_file.exists():
            try:
                with open(part_file, 'r') as f:
                    all_results.extend(json.load(f))
            except Exception as e:
                print(f"Error loading part {rank}: {e}")
    
    full_result_path = output_dir / 'full_results.json'
    with open(full_result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    metrics = {}
    metrics.update(evaluate_acc(all_results))

    print(f"Evaluation Metrics: {metrics}")
    
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

