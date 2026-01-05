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
# add system path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
base_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(f"{base_dir}/train/LLaMA-Factory/src/llamafactory/model/hdlm_utils")

from hdlm_depth2 import Depth2_HdLMModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def inference_worker(rank, data_path, model_path,
                      intermediate_layer_index=16, total_workers=8):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    with open(data_path, 'r') as f:
        full_data = json.load(f)

    data_slice = np.array_split(full_data, total_workers)[rank]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Depth2_HdLMModel.from_pretrained(
        model_path,
        intermediate_layer_index = intermediate_layer_index
    ).to(device)
    
    results = []
    
    for item in tqdm(data_slice, desc=f'GPU-{rank}'):
        l1, l2 = model.chat(
            tokenizer=tokenizer,
            query=item["input"],
            system_prompt=item["system"],
            think=True,
            dataset = "WOS"
        )
        results.append({
            "l1": str(l1),
            "l2": str(l2),
            "true_l1": item["think"],
            "true_l2": item["assitant"]
        })
    
    with open(f'./wos_sub_results/result_part_{rank}.json', 'w') as f:
        json.dump(results, f)
def evaluate_hierarchical(all_results):

    true_joint = [f"{res['true_l1']}-{res['true_l2']}" for res in all_results]
    pred_joint = [f"{res['l1']}-{res['l2']}" for res in all_results]

    correct_joint = sum(1 for t, p in zip(true_joint, pred_joint) if t == p)
    acc_joint = correct_joint / len(all_results)

    unique_labels = list(set(true_joint + pred_joint))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    true_ids = [label_to_id[label] for label in true_joint]
    pred_ids = [label_to_id[label] for label in pred_joint]

    micro_f1 = f1_score(true_ids, pred_ids, average='micro')
    macro_f1 = f1_score(true_ids, pred_ids, average='macro')

    return {
        'acc_joint': round(acc_joint, 4),
        'micro_f1': round(micro_f1, 4),
        'macro_f1': round(macro_f1, 4)
    }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model")
    parser.add_argument("--intermediate_layer_index", type=int, default=16,
                      help="First classification layer index (default: 16)")
    args = parser.parse_args()
    multiprocessing.set_start_method('spawn')
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist")
    data_path = "HdLM/data/Depth2/WoS/alpaca_datasets/wos_test_data.json"
    system_prompt = '''You are a research assistant, your task is to categorize a research article by determining its domain and area. Your judgment will be based on the article's abstract and keywords, and you will use provided lists of domains and areas. You must strictly output the ID corresponding to your classification at each step.'''  # 保持原有prompt
    
    num_gpus = 8  # according your gpu number
    with Pool(num_gpus) as pool:
        pool.map(partial(inference_worker,
                        data_path=data_path,
                        model_path=args.model_path,
                        intermediate_layer_index=args.intermediate_layer_index),
                 range(num_gpus))
    
    all_results = []
    for rank in range(num_gpus):
        with open(f'./wos_sub_results/result_part_{rank}.json') as f:
            all_results.extend(json.load(f))
    metrics = evaluate_hierarchical(all_results)

    print(f"Joint ACC (L1+L2): {metrics['acc_joint']}")
    print(f"Micro-F1: {metrics['micro_f1']}")
    print(f"Macro-F1: {metrics['macro_f1']}")