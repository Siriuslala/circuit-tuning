
import json
import jsonlines
import pandas as pd
from collections import defaultdict

train_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad/squad_v2/train-00000-of-00001.parquet"
dev_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad/squad_v2/validation-00000-of-00001.parquet"

df_train = pd.read_parquet(train_path, engine='pyarrow')
data_train = df_train.to_dict(orient='records')

df_dev = pd.read_parquet(dev_path, engine='pyarrow')
data_dev = df_dev.to_dict(orient='records')


tgt_train_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad/data/train.jsonl"
tgt_dev_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad/data/dev.jsonl"

# with jsonlines.open(tgt_train_path, "w") as f:
#     for line in data_train:
#         line["answer"] = line["answers"]["text"].tolist()
#         del line["answers"]
#         f.write(line)

# with jsonlines.open(tgt_dev_path, "w") as f:
#     for line in data_dev:
#         line["answer"] = line["answers"]["text"].tolist()
#         del line["answers"]
#         f.write(line)

# titles = defaultdict(int)
# with jsonlines.open(tgt_train_path, "r") as f:
#     for line in f:
#         titles[line["title"]] += 1

# titles_dev = defaultdict(int)   
# with jsonlines.open(tgt_dev_path, "r") as f:
#     for line in f:
#         titles_dev[line["title"]] += 1

# print(len(titles), sum(titles.values()))
# print(len(titles_dev), sum(titles_dev.values()))


with open("/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/squad/data_raw/dev-v2.0.json", "r") as f:
    data = json.load(f)

print(len(data["data"]))
print(data["data"][0].keys())

titles = defaultdict(int)
samples = []
for article in data["data"]:
    items = [] 
    # titles[article["title"]] += len(article["paragraphs"])
    for paragraph in article["paragraphs"]:
        para_data = {"context": paragraph["context"], "qas": []}
        for qa in paragraph["qas"]:
            item = {
                "id": qa["id"],
                "question": qa["question"],
                "answer": [ans["text"] for ans in qa["answers"]] if qa["answers"] else ["[No Answer]"]
            }
            para_data["qas"].append(item)
        samples.append({"title": article["title"], "paragraph": para_data})

with jsonlines.open(tgt_dev_path, "w") as f:
    for line in samples:
        f.write(line)

# with jsonlines.open(tgt_dev_path, "w") as f:
#     for line in data_dev:
#         line["answer"] = line["answers"]["text"].tolist()
#         del line["answers"]
#         f.write(line)