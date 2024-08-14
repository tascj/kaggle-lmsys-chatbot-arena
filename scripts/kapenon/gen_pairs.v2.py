# %%
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# from datasets import load_dataset
# ds = load_dataset("lmsys/lmsys-chat-1m")
# df = ds["train"].to_pandas() # lmsys 1M has no test set
# df.to_parquet("../artifacts/lmsys-1m.parquet")

df = pd.read_parquet("../artifacts/lmsys-1m.parquet")


def separate_user_assistant(conversation):
    res = defaultdict(list)
    for c in conversation:
        assert c["role"] in {"user", "assistant"}
        assert type(c["content"]) == str
        res[c["role"]].append(c["content"])
    assert len(res["user"]) > 0
    assert len(res["user"]) == len(res["assistant"])
    return res


sep = df["conversation"].map(separate_user_assistant)
df["prompt"] = sep.map(lambda x: x["user"])
df["response"] = sep.map(lambda x: x["assistant"])

# Generate a unique id for each prompt
df["pid"] = df["prompt"].map(str).map(hash)
pid_to_count = dict(df["pid"].value_counts())
df["n_pid"] = df["pid"].map(pid_to_count)

# remove
print("BEFORE removal", len(df))
df["prompt_response_hash"] = (df["prompt"].map(str) + df["response"].map(str)).map(hash)
df = df.drop_duplicates(["prompt_response_hash"])
print("AFTER removal", len(df))

df = df.drop(columns=["prompt_response_hash"])

# %%
# shuffle
df = df.sample(len(df), random_state=42).reset_index(drop=True)

# The result dictionary
res = dict(id=[], model_a=[], model_b=[], prompt=[], response_a=[], response_b=[])
cand_df = df[df["n_pid"] > 1]
cand_pids = cand_df["pid"].unique()
for i, pid in tqdm(enumerate(cand_pids)):

    tmp = cand_df[cand_df["pid"] == pid]
    tmp_models = tmp["model"].unique()
    if len(tmp_models) == 1:
        continue

    # Create pairs AMAP. Each model appears only once.
    for j in range(0, len(tmp_models) - 1, 2):
        model_a, model_b = list(tmp_models)[j : j + 2]
        a = tmp[tmp["model"] == model_a].iloc[0]
        b = tmp[tmp["model"] == model_b].iloc[0]

        assert a.prompt == b.prompt
        assert model_a != model_b
        assert len(a.prompt) == len(a.response) == len(b.response)

        if a.response == b.response:
            print("Responses are identical. Skipping.")
            print(model_a, a.response)
            print(model_b, b.response)
            continue

        res["id"].append(f"lmsys-1m_{i}")
        res["model_a"].append(model_a)
        res["model_b"].append(model_b)
        res["prompt"].append(a.prompt)
        res["response_a"].append(a.response)
        res["response_b"].append(b.response)


res = pd.DataFrame(res)

# %%
# remove deduplicates
df = pd.read_csv("../data/train.csv")
df_lmsys33k = pd.read_csv("../artifacts/lmsys-33k-deduplicated.csv")
for col in ["prompt", "response_a", "response_b"]:
    df[col] = df[col].map(json.loads)
    df_lmsys33k[col] = df_lmsys33k[col].map(json.loads)
df = pd.concat([df, df_lmsys33k], ignore_index=True)

# check intersection
train_prompts = df["prompt"].map(str)
res_prompts = res["prompt"].map(str)
inter = set(train_prompts).intersection(set(res_prompts))

# rm
res = res[~res_prompts.isin(inter)]

res["prompt"] = res["prompt"].map(json.dumps)
res["response_a"] = res["response_a"].map(json.dumps)
res["response_b"] = res["response_b"].map(json.dumps)
res["winner_model_a"] = 1
res["winner_model_b"] = 0
res["winner_tie"] = 0
res.to_parquet("../artifacts/lmsys_pairs_110k.parquet")
