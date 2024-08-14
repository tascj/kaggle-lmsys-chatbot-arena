import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("../data/train.csv")

sgkf = StratifiedGroupKFold(n_splits=5, random_state=0, shuffle=True)
group_id = df["prompt"]
label_id = df["winner_model_a winner_model_b winner_tie".split()].values.argmax(1)
splits = list(sgkf.split(df, label_id, group_id))

df["fold"] = -1
for fold, (_, valid_idx) in enumerate(splits):
    df.loc[valid_idx, "fold"] = fold

print(df["fold"].value_counts())
df.to_csv("../artifacts/dtrainval.csv", index=False)
