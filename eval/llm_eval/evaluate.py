import argparse
from tqdm import tqdm
from utils.read_file import FileReader
from utils.logging import get_logger
from utils.data_process import *
import time
from collections import Counter
from pycocoevalcap.cider.cider import Cider
logger = get_logger()
file_reader = FileReader()


def compute_cider_score(candidate, references):
    cider = Cider()

    print("Start compute CIDEr scores...")
    start_time = time.time()
    score, scores = cider.compute_score(candidate, references)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Computing CIDEr time cost: {elapsed_time:.4f} sec")
    return score, scores

def calculate_dist_2(texts):
    print("Start compute Distinct-2 scores")
    start_time = time.time()
    total_bigrams = 0
    unique_bigrams = set()
    for text in tqdm(texts, desc="Processing texts"):
        tokens = text.split()
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            unique_bigrams.add(bigram)
            total_bigrams += 1

    end_time = time.time() 
    elapsed_time = end_time - start_time

    print(f"Total bigram num: {total_bigrams}")
    print(f"Special bigram num: {len(unique_bigrams)}")

    if total_bigrams == 0:
        print("There is no bignarm, return score with 0.0")
        return 0.0

    distinct_2_score = len(unique_bigrams) / total_bigrams
    print(f"Computing Distinct-2 time cost: {elapsed_time:.4f} sec")
    return distinct_2_score




def main(args):
    data = file_reader.read_file(args.file_type, args.file_path)
    if args.eval_method.lower() == 'dist2':
        if args.data_type.lower() == 'sharegpt':
            content_list = []
            for i, x in enumerate(tqdm(data)):
                content_list.append(extract_content(x))
            original_list = sum(content_list, [])
            response_list = list(filter(lambda x: x is not None, original_list))
        elif args.data_type.lower() == 'alpaca':
            response_list = [line[args.key_new_content] for line in data if line[args.key_new_content] != 'None' and line[args.key_new_content] is not None]

        distinct_2 = calculate_dist_2(response_list)
        logger.info(f"Dist-2 score: {distinct_2 * 100:.2f}%")
    elif args.eval_method.lower() == 'cider':
        if args.data_type.lower() == 'sharegpt':
            content_old_content_list = []
            for i, x in enumerate(tqdm(data)):
                content_old_content_list.append(extract_content_and_old_content(x))
            all_content_old_content_list = sum(content_old_content_list, [])
            candidate, references = process_sharegpt_data(all_content_old_content_list)
        elif args.data_type.lower() == 'alpaca':
            if not args.key_old_content:
                raise ValueError("key_old_content must be provided for CIDEr evaluation.")
            candidate, references = process_data(data, args.key_new_content, args.key_old_content)
        score, _ = compute_cider_score(candidate, references)
        logger.info(f"CIDEr score: {score * 100:.2f}%")
    else:
        logger.error("Unknown evaluation method: %s", args.eval_method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate text data.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--file_type', type=str, required=True, help='Type of the input file (e.g., jsonl, txt).')
    parser.add_argument('--eval_method', type=str, required=True, choices=['dist2', 'cider'], help='Evaluation method to use.')
    parser.add_argument('--data_type', type=str, required=True, help='the data type.')
    parser.add_argument('--key_new_content', type=str, help='Key name for new content in the data.')
    parser.add_argument('--key_old_content', type=str, help='Key name for old content in the data (required for CIDEr evaluation).')

    args = parser.parse_args()
    main(args)

# Only for cider and dist-2, B-2 and R-L scores' code need to be complished