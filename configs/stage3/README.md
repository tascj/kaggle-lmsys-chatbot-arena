# Stage 3

## Models

0. google/gemma-2-9b-it
3. RLHFlow/ArmoRM-Llama3-8B-v0.1

## Data

1. 55k Kaggle competition data
2. 21k from [abdullahmeda](https://www.kaggle.com/datasets/abdullahmeda/lmsys-additional-33k-labelled-conversations)

## Run

```
torchrun --nproc_per_node=4 main.py configs/stage3/m0.py
torchrun --nproc_per_node=4 main.py configs/stage3/m3.py
```
