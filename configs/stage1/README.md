# Stage 1

## Models

0. google/gemma-2-9b-it
1. google/gemma-2-127b-it
3. RLHFlow/ArmoRM-Llama3-8B-v0.1

## Data

1. 55k Kaggle competition data
2. 21k from [abdullahmeda](https://www.kaggle.com/datasets/abdullahmeda/lmsys-additional-33k-labelled-conversations)

Validataion is 20% from 55k, StratifiedGroupKFold, group by prompt. Use `scripts/prepare_dataset.py` to make train/val split.

## Run

```
torchrun --nproc_per_node=4 main.py configs/stage1/m0.py
torchrun --nproc_per_node=4 main.py configs/stage1/m1.py
torchrun --nproc_per_node=4 main.py configs/stage1/m3.py
```
