import json
from wos_prompts import sys_prompt,usr_prompt,Domain_dict,Area_dict,area_task_prompt
def format_json(input_path, output_path, domain_list):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    alpaca_data = []

    for item in data:
        keywords = item.get("Keywords", "")
        abstract = item.get("Abstract", "")
        label1 = item.get("Label1", "")
        label2 = item.get("Label2", "")

        data = {
             "system": sys_prompt,
             "input": usr_prompt.format(abstract=abstract, keywords=keywords, domain_list=domain_list),
             "think": str(label1),
             "subtask": area_task_prompt.format(Area_subdict = Area_dict[Domain_dict[label1]]),
             "assistant": str(label2)
        }
        alpaca_data.append(data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=4)


train_input_path = "splited_jsondata/train_data.json"
train_output_path = "alpaca_datasets/wos_train_data.json" 
test_input_path = "splited_jsondata/test_data.json"
test_output_path = "alpaca_datasets/wos_test_data.json" 
domain_list = Domain_dict

format_json(train_input_path, train_output_path, domain_list)
format_json(test_input_path, test_output_path, domain_list)

with open(train_output_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
print(len(data1))

with open(test_output_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
print(len(data1))