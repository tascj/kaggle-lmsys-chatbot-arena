import torch
import pandas as pd


def load_pred(path):
    pred = torch.from_numpy(torch.load(path))
    return pred


# 130k
preds = [
    load_pred(f"../artifacts/stage1/m{idx}/pseudo_labels_130k.pth") for idx in [0, 1, 3]
]
pred_130k = torch.stack(preds, dim=0).mean(0).numpy()

dfs = [
    pd.read_csv(f"../artifacts/{name}")
    for name in [
        "Capybara.csv",
        "RLHFLow_PKU-SafeRLHF-30K-standard.csv",
        "RLHFLow_Argilla-Math-DPO-standard.csv",
        "RLHFLow_CodeUltraFeedback-standard.csv",
        "HelpSteer.csv",
    ]
]
df = pd.concat(dfs).reset_index(drop=True)
df["winner_model_a"] = pred_130k[:, 0]
df["winner_model_b"] = pred_130k[:, 1]
df["winner_tie"] = pred_130k[:, 2]
df.to_parquet("../artifacts/dpseudo_m013_130k.parquet", index=False)


# 110k from lmsys
preds = [
    load_pred(f"../artifacts/stage1/m{idx}/pseudo_labels_110k.pth") for idx in [0, 1, 3]
]
pred_110k = torch.stack(preds, dim=0).mean(0).numpy()
df = pd.read_parquet("../artifacts/lmsys_pairs_110k.parquet")
df["winner_model_a"] = pred_110k[:, 0]
df["winner_model_b"] = pred_110k[:, 1]
df["winner_tie"] = pred_110k[:, 2]
df.to_parquet("../artifacts/dpseudo_m013_110k.parquet", index=False)
