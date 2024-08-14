# Stage 2

## Models

0. google/gemma-2-9b-it
3. RLHFlow/ArmoRM-Llama3-8B-v0.1

## Data

1. 130k preference data from
    * argilla/Capybara-Preferences
    * RLHFlow/PKU-SafeRLHF-30K-standard
    * RLHFlow/Argilla-Math-DPO-standard
    * RLHFlow/CodeUltraFeedback-standard
    * RLHFlow/Helpsteer-preference-standard
2. 110k pairs from lmsys-1m

## Run

```
torchrun --nproc_per_node=4 main.py configs/stage2/m0.py
torchrun --nproc_per_node=4 main.py configs/stage2/m3.py
```
