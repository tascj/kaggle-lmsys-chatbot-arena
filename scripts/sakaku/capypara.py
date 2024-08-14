import datasets
import pandas as pd
import json
from datasets import load_dataset, concatenate_datasets
import random
from tqdm import tqdm
random.seed(42)
dataset1 = load_dataset("argilla/Capybara-Preferences", split="train")
df_lmsys21k = pd.read_csv('../artifacts/lmsys-33k-deduplicated.csv')
columns = df_lmsys21k.columns
new_df = []
for dd in tqdm(dataset1):
    response_chosen = dd['chosen']
    response_rejected = dd['rejected']
    prompt = []
    r_chosen = []
    r_rejected = []
    for i in range(len(response_chosen)):
        if response_chosen[i]['role'] == 'user':
            prompt.append(response_chosen[i]['content'])
        if response_chosen[i]['role'] == 'assistant':
            r_chosen.append(response_chosen[i]['content'])
        if response_rejected[i]['role'] == 'assistant':
            r_rejected.append(response_rejected[i]['content'])
    if random.random() > 0.5:
        winner_model_a = 1
        winner_model_b = 0
        response_a = r_chosen
        response_b = r_rejected
    else:
        winner_model_a = 0
        winner_model_b = 1
        response_a = r_rejected
        response_b = r_chosen
    new_df.append(['a', 'a', 'a', json.dumps(prompt), json.dumps(response_a), json.dumps(response_b), winner_model_a, winner_model_b, 0])
    # print('a')
new_df = pd.DataFrame(new_df, columns=columns)
new_df.to_csv('../artifacts/Capybara.csv', index=False)
print('a')
