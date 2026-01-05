import json
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from functools import partial
import sys, os
from pathlib import Path
import argparse
import hashlib

# 添加模型工具路径（确保路径正确）
sys.path.append("/vepfs/group04/user/xiningyuan/HdLM_git/HdLM/utils/LLaMA-Factory/src/llamafactory/model/hdlm_utils")
from hdlm_depth2 import Depth2_HdLMModel
from transformers import AutoTokenizer


def inference_worker(rank, data_path, model_path,
                     intermediate_layer_index=25, total_workers=2):
    model_name = os.path.basename(model_path)
    output_dir = Path(f'./cmqa_sub_results_acc_{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'result_part_{rank}.json'

    start_gpu = rank * 2
    gpu_list = list(range(start_gpu, start_gpu + 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_list))
    device = torch.device('cuda:0')

    full_data = []
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            full_data = json.load(f)
    elif data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            for line in f:
                full_data.append(json.loads(line.strip()))
    else:
        raise ValueError("Unsupported file format")

    data_slice = np.array_split(full_data, total_workers)[rank]

    existing_results = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            for item in json.load(f):
                key = hashlib.md5((item["input"] + item["system"]).encode()).hexdigest()
                existing_results[key] = item

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Depth2_HdLMModel.from_pretrained(
        model_path,
        tokenizer_path=model_path,
        think_layer_index=intermediate_layer_index,
        device_map="auto",
    )

    results = list(existing_results.values())

    for item in tqdm(data_slice, desc=f'GPU-{rank}'):
        key = hashlib.md5((item["input"] + item["system"]).encode()).hexdigest()
        if key in existing_results:
            continue

        l1, l2 = model.chat(
            tokenizer=tokenizer,
            query=item["input"],
            system_prompt=item["system"],
            subtask=item['subtask'],
            think=True,
            dataset="commonsenseqa"
        )

        result_item = {
            "input": item["input"],
            "system": item["system"],
            "l1": str(l1),
            "l2": str(l2),
            "true_l1": item["think"],
            "true_l2": item["assistant"]
        }

        results.append(result_item)
        existing_results[key] = result_item

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)


def evaluate_l2_only(all_results):
    correct = 0
    total = len(all_results)

    for res in all_results:
        if res["true_l2"] in res["l2"]:
            correct += 1

    return {"accuracy_l2_loose": round(correct / total, 4)}


def evaluate_l2_strict_acc(all_results):
    correct = 0
    total = 0

    for res in all_results:
        correct += int(res["l2"] == res["true_l2"])
        total += 1

    return {"accuracy_l2_strict": round(correct / total, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--intermediate_layer_index", type=int, default=25)
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn')

    data_path = "/vepfs/group04/user/xiningyuan/HdLM_git/HdLM/data/Depth2/CommonSenseQA/commonsense_qa_alpaca_train.jsonl"

    num_gpus = 4
    with Pool(num_gpus) as pool:
        pool.map(
            partial(
                inference_worker,
                data_path=data_path,
                model_path=args.model_path,
                intermediate_layer_index=args.intermediate_layer_index,
                total_workers=num_gpus
            ),
            range(num_gpus)
        )

    model_name = os.path.basename(args.model_path)
    output_dir = Path(f'./cmqa_sub_results_acc_{model_name}')

    all_results = []
    for rank in range(num_gpus):
        part_file = output_dir / f'result_part_{rank}.json'
        if part_file.exists():
            with open(part_file, 'r') as f:
                all_results.extend(json.load(f))

    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ===== 同时计算两种 acc =====
    metrics = {}
    metrics.update(evaluate_l2_only(all_results))
    metrics.update(evaluate_l2_strict_acc(all_results))

    print("Evaluation Metrics:", metrics)

    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

