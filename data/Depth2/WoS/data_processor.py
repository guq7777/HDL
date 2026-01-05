import pandas as pd
import json
from sklearn.model_selection import train_test_split

excel_file_path = "Data.xlsx"

output_json_file_path = "splited_jsondata/wos_data.json"

df = pd.read_excel(excel_file_path)

required_columns = ['keywords', 'Abstract', 'Y1', 'Y2', 'Y']
df = df[required_columns]

df['keywords'] = df['keywords'].str.strip()
df['Abstract'] = df['Abstract'].str.strip()

df.rename(columns={'keywords':'Keywords', 'Y1': 'Label1', 'Y2': 'Label2', 'Y': 'Label'}, inplace=True)

data = df.to_dict(orient='records')

with open(output_json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)
### split train val test
train_output_path = "splited_jsondata/train_data.json"
val_output_path = "splited_jsondata/val_data.json"
test_output_path = "splited_jsondata/test_data.json"
df = pd.DataFrame(data)
print("Total Len:", len(df))

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Label'])


train_data = train_df.to_dict(orient='records')
print("Len Train: ",len(train_data))
val_data = val_df.to_dict(orient='records')
print("Len Val: ",len(val_data))
test_data = test_df.to_dict(orient='records')
print("Len Test: ",len(test_data))

with open(train_output_path, 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, indent=4, ensure_ascii=False)
with open(val_output_path, 'w', encoding='utf-8') as val_file:
    json.dump(val_data, val_file, indent=4, ensure_ascii=False)
with open(test_output_path, 'w', encoding='utf-8') as test_file:
    json.dump(test_data, test_file, indent=4, ensure_ascii=False)