# LMSYS - Chatbot Arena Human Preference Predictions

[Competition](https://www.kaggle.com/competitions/lmsys-chatbot-arena)

## Requirements

### Hardware

A100 SXM 80G x4

### Software

Base Image
```
nvcr.io/nvidia/pytorch:24.04-py3
```

Packages
```
detectron2==0.6
transformers==4.43.3
datasets==2.19.0
flash-attn==2.6.2
optimi==0.2.1
```


## Training

Directory structure should be as follows.

```
├── data
│   ├── train.csv
│   └── test.csv
├── artifacts
│   ├── dtrainval.csv
│   ├── lmsys-33k-deduplicated.csv
│   ├── ...
│   ├── stage1
│   ├── ...
│   └── stage3
└── src  # this repo
    ├── configs
    ├── human_pref
    └── main.py
```

1.  `python scripts/prepare_dataset.py` and download 21k external data from  [abdullahmeda](https://www.kaggle.com/datasets/abdullahmeda/lmsys-additional-33k-labelled-conversations)
2. [stage1](configs/stage1/README.md)
3. [make pseudo labels](configs/stage1_generate_pseudo_labels/README.md)
4. [stage2](configs/stage2/README.md)
4. [stage3](configs/stage3/README.md)

## Inference

Reference scripts to convert checkpoints for inference.
```
python scripts/prepare_gemma2_for_submission.py
python scripts/prepare_llama3_for_submission.py
```

[Kaggle Notebook](https://www.kaggle.com/code/tascj0/lmsys-0805)
