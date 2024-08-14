# Make pseudo labels

Same as stage1, replaced val dataset.

## Data

1. 130k preference data from
    * argilla/Capybara-Preferences
    * RLHFlow/PKU-SafeRLHF-30K-standard
    * RLHFlow/Argilla-Math-DPO-standard
    * RLHFlow/CodeUltraFeedback-standard
    * RLHFlow/Helpsteer-preference-standard
2. 110k pairs from lmsys-1m

## Run

### Prepare csv/parquet files

You may need to modify file paths in the scripts.
```
python scripts/kapenon/gen_pairs.v2.py

python scripts/sakaku/capypara.py
python scripts/sakaku/rlhflow1.py
python scripts/sakaku/rlhflow2.py
```

### Generate pseudo labels

```
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m0_110k.py --load-from ../artifacts/stage1/m0/update_last.pth --eval-only --out ../artifacts/stage1/m0/pseudo_labels_110k.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m0_130k.py --load-from ../artifacts/stage1/m0/update_last.pth --eval-only --out ../artifacts/stage1/m0/pseudo_labels_130k.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m1_110k.py --load-from ../artifacts/stage1/m1/update_last.pth --eval-only --out ../artifacts/stage1/m1/pseudo_labels_110k.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m1_130k.py --load-from ../artifacts/stage1/m1/update_last.pth --eval-only --out ../artifacts/stage1/m1/pseudo_labels_130k.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m3_110k.py --load-from ../artifacts/stage1/m3/update_last.pth --eval-only --out ../artifacts/stage1/m3/pseudo_labels_110k.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m3_130k.py --load-from ../artifacts/stage1/m3/update_last.pth --eval-only --out ../artifacts/stage1/m3/pseudo_labels_130k.pth
```

```
python scripts/prepare_pseudo_label.py
```
