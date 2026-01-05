#!/bin/bash

cd llm_eval

file_path="/HdLM/infra/ESC/result.json"


echo "Processing $path with dist2..."
python evaluate.py \
    --file_path "$file_path" \
    --file_type jsonArray \
    --eval_method dist2 \
    --data_type alpaca \
    --key_new_content 'Pred Answer'

echo "Processing $path with cider..."
python evaluate.py \
    --file_path "$file_path" \
    --file_type jsonArray \
    --eval_method cider \
    --data_type alpaca \
    --key_new_content 'Pred Answer' \
    --key_old_content 'True Answer'
done

