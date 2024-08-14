import functools

import torch
from detectron2.config import LazyCall as L

# from detectron2.config import instantiate
from fvcore.common.param_scheduler import CompositeParamScheduler, LinearParamScheduler
from transformers import AutoTokenizer
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from human_pref.data.dataset import LMSYSDataset
from human_pref.data.processors import ProcessorPAB
from human_pref.models.modeling_gemma2_fast import (
    Gemma2DecoderLayer,
    Gemma2ForSequenceClassification,
)
from human_pref.anyprecision_optimizer import AnyPrecisionAdamW


model_name_or_path = "google/gemma-2-27b-it"


# model config
def build_model():
    model = Gemma2ForSequenceClassification.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        num_labels=3,
        _attn_implementation="flash_attention_2",
    )
    # re-initilize the head here
    # Gemma2ForSequenceClassification init is weird
    hdim = model.config.hidden_size
    model.score = torch.nn.Sequential(
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hdim, hdim // 2),
        torch.nn.Dropout(0.1),
        torch.nn.GELU(),
        torch.nn.Linear(hdim // 2, 3),
    ).bfloat16()

    return model


model = L(build_model)()
optimizer = L(AnyPrecisionAdamW)(
    lr=0.25e-5,
    use_kahan_summation=True,
    betas=(0.9, 0.99),
    eps=1e-6,
    weight_decay=0.01,
)


# data config
def build_dataset(fold, training, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True
    processor = ProcessorPAB(
        tokenizer=tokenizer,
        max_length=max_length,
        support_system_role=False,
    )
    if training:
        dataset0 = LMSYSDataset(
            csv_file="../artifacts/dtrainval.csv",
            query=f"fold != {fold}",
            processor=processor,
            include_swap=True,
        )
        dataset1 = LMSYSDataset(
            csv_file="../artifacts/lmsys-33k-deduplicated.csv",
            query=None,
            processor=processor,
            include_swap=True,
        )
        dataset = torch.utils.data.ConcatDataset([dataset0, dataset1])
    else:
        dataset = torch.utils.data.ConcatDataset(
            [
                LMSYSDataset(
                    csv_file=f"../artifacts/{name}",
                    query=None,
                    processor=processor,
                    is_parquet=False,
                )
                for name in [
                    "Capybara.csv",
                    "RLHFLow_PKU-SafeRLHF-30K-standard.csv",
                    "RLHFLow_Argilla-Math-DPO-standard.csv",
                    "RLHFLow_CodeUltraFeedback-standard.csv",
                    "HelpSteer.csv",
                ]
            ]
        )
    return dataset


def build_data_loader(dataset, batch_size, num_workers, training=True):
    from human_pref.data.collators import VarlenCollator, ShardedMaxTokensCollator

    max_tokens = 1024 * 2
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=training,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=training,
        collate_fn=ShardedMaxTokensCollator(
            max_tokens=max_tokens,
            base_collator=VarlenCollator(),
            sort_samples=training,
        ),
    )


VAL_FOLD = 0
dataloader = dict(
    train=L(build_data_loader)(
        dataset=L(build_dataset)(fold=VAL_FOLD, training=True, max_length=4096),
        batch_size=80,
        num_workers=4,
        training=True,
    ),
    val=L(build_data_loader)(
        dataset=L(build_dataset)(fold=VAL_FOLD, training=False, max_length=4096),
        batch_size=80,
        num_workers=4,
        training=False,
    ),
)

max_epochs = 1
lr_multiplier = L(CompositeParamScheduler)(
    schedulers=[
        L(LinearParamScheduler)(start_value=0.001, end_value=1),
        L(LinearParamScheduler)(start_value=1, end_value=0.001),
    ],
    lengths=[0.1, 0.9],
    interval_scaling=["rescaled", "rescaled"],
)

train = dict(
    device="cuda",
    max_epochs=max_epochs,
    log_interval=1,
    checkpoint_interval=200,
    eval_interval=1,
    cast_to_bf16=False,
    log_buffer_size=20,
    clip_grad=False,
    seed=3,
)

fsdp = dict(
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Gemma2DecoderLayer},
    ),
    sharding_strategy=ShardingStrategy.NO_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision. For rotary_emb in this case.
        buffer_dtype=torch.float32,
    ),
)
