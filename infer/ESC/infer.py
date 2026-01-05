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
            system_prompt=item['system'],
            think=True,
            dataset="ESC_Emo"
        )
        results.append({
            "l1": l1,
            "true_l1": str(item.get('think', "-9")),
            "l2": l2,
            "true_l2": item["assistant"]
        })
    
    with open(f'./esc_sub_results/result_part_{rank}.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model")
    parser.add_argument("--intermediate_layer_index", type=int, default=16,
                      help="First classification layer index (default: 16)")
    parser.add_argument("--merged_model", type=bool, default=False,
                      help="Use DailyDialog and ESC merged model")
    args = parser.parse_args()
    multiprocessing.set_start_method('spawn')
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist")
    data_path = "HdLM/data/Depth2/ESC/demo_cases.json"
    
    num_gpus = 8  # according to your gpus
    with Pool(num_gpus) as pool:
        pool.map(partial(inference_worker,
                        data_path=data_path,
                        model_path=args.model_path,
                        intermediate_layer_index=args.intermediate_layer_index),
                 range(num_gpus))
    
    all_results = []
    for rank in range(num_gpus):
        with open(f'./esc_sub_results/result_part_{rank}.json') as f:
            all_results.extend(json.load(f))

    correct_think = 0
    true_l1_all = []
    pred_l1_all = []
    true_ans_all = []
    pred_ans_all = []
    for result in all_results:
        pred_l1 = result['l1']
        true_l1 = result['true_l1']
        pred_ans = result['l2']
        true_ans = result['true_l2']
        true_l1_all.append(true_l1)
        pred_l1_all.append(pred_l1)
        true_ans_all.append(true_ans)
        pred_ans_all.append(pred_ans)
        if pred_l1 == true_l1:
            correct_think += 1
    combined_answer=[
        {"True Stratey": true_l, "Pred Stratey": pred_l,"True Answer": true, "Pred Answer":pred}
        for true_l, pred_l, true, pred in zip(true_l1_all, pred_l1_all, true_ans_all, pred_ans_all)
    ]
    total = len(true_l1_all)
    acc_l1 = round(correct_think / total * 100, 2)
    macro_f1 = f1_score(true_l1_all, pred_l1_all, average="macro")
    micro_f1 = f1_score(true_l1_all, pred_l1_all, average="micro")

    print(f"Strategy Result:")
    print(f"L1 Acc: {acc_l1}% ({correct_think}/{total})")
    print(f"L1 Macro F1 score: {macro_f1}")
    print(f"L1 Micro F1 score: {micro_f1}")
    
    output_file = "./infer_result.json"
    with open(output_file,"w",encoding='utf-8') as json_file:
        json.dump(combined_answer,json_file,ensure_ascii=False,indent=2)
    print("Combined answers data saved to filepath successfully!")
    

