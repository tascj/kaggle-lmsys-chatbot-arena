# ruff: noqa: E402
import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HOME"] = "/path/to/fast/storage"

import argparse
import time
import shutil
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from pathlib import Path


from detectron2.config import LazyConfig, instantiate
from detectron2.solver import LRMultiplier
from detectron2.engine.hooks import LRScheduler
from detectron2.utils.env import seed_all_rng

from human_pref.logging import get_logger
from human_pref.utils import to_gpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--load-from", default=None, type=str)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output-root", default="../artifacts")
    parser.add_argument(
        "--opts",
        help="""
Modify config options at the end of the command, use "path.key=value".
        """.strip(),
        default=[],
        nargs=argparse.ZERO_OR_MORE,
    )
    parser.add_argument("--out", default=None, type=str)
    return parser.parse_args()


class LogLossBuffer:
    """Circular buffer for storing log loss values"""

    def __init__(self, size, device="cuda"):
        self.buffer = torch.zeros(size, device=device)
        self.size = size
        self.idx = 0
        self.num = 0

    def append(self, value):
        self.buffer[self.idx] = value
        self.idx = (self.idx + 1) % self.size
        self.num = min(self.num + 1, self.size)

    def mean(self):
        return self.buffer.sum().item() / self.num


@torch.no_grad()
def do_test(cfg, model):
    logger = get_logger("lmsys")
    logger.info("Evaluation start")

    val_loader = instantiate(cfg.dataloader.val)

    model.eval()
    from tqdm import tqdm

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        prog_bar = tqdm(val_loader)
    else:
        prog_bar = val_loader

    probs = []
    for batch in prog_bar:
        for micro_batch in batch:
            micro_batch = to_gpu(micro_batch)
            prob = model(micro_batch["input_ids"], micro_batch["cu_seqlens"]).softmax(
                dim=-1
            )
            gather_probs = [torch.zeros_like(prob) for _ in range(world_size)]
            dist.all_gather(gather_probs, prob)
            prob = torch.stack(gather_probs, dim=1).flatten(0, 1)
            probs.append(prob.data.cpu())

    result = torch.cat(probs, dim=0).numpy()
    # the last batch maybe padded to be divisible by world_size
    result = result[: len(val_loader.dataset)]

    logger.info("Evaluation prediction done")
    if not hasattr(val_loader.dataset, "evaluate"):
        eval_result = {"info": f"Not implemented for {type(val_loader.dataset)}"}
    else:
        eval_result = val_loader.dataset.evaluate(result)
    logger.info("Evaluation end")
    return result, eval_result


def save_checkpoint(model, optimizer, work_dir, checkpoint_path):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if dist.get_rank() == 0:
        checkpoint = {
            "model": cpu_state,
            # "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)


def do_train(cfg, model):
    cfg.optimizer.params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    max_epochs = cfg.train.max_epochs
    lr_scheduler = LRMultiplier(
        optimizer,
        multiplier=instantiate(cfg.lr_multiplier),
        max_iter=max_epochs * len(train_loader),
    )
    best_param_group_id = LRScheduler.get_best_param_group_id(optimizer)

    logger = get_logger("lmsys")
    loss_history = LogLossBuffer(cfg.train.get("log_buffer_size", 100))
    total_updates = 0

    rank = dist.get_rank()
    fsdp_loss = torch.zeros(2).to(rank)

    clip_grad = cfg.train.get("clip_grad", True)
    for curr_epoch in range(max_epochs):
        model.train()
        for curr_iter, batch in enumerate(train_loader):
            total_batch_size = sum(micro_batch["batch_size"] for micro_batch in batch)
            fsdp_loss.zero_()
            for micro_batch in batch:
                micro_batch = to_gpu(micro_batch)
                logits = model(micro_batch["input_ids"], micro_batch["cu_seqlens"])
                loss = F.cross_entropy(logits, micro_batch["label"])
                fsdp_loss[0] += loss.detach() * micro_batch["batch_size"]
                fsdp_loss[1] += micro_batch["batch_size"]
                loss = loss * (micro_batch["batch_size"] / total_batch_size)
                loss.backward()

                dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)

            if clip_grad:
                grad_norm = model.clip_grad_norm_(1.0)
                grad_norm = grad_norm.item()
            else:
                grad_norm = 0
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_history.append(fsdp_loss[0] / fsdp_loss[1])
            total_updates += 1
            lr_scheduler.step()
            if total_updates % cfg.train.log_interval == 0:
                lr = optimizer.param_groups[best_param_group_id]["lr"]
                loss_val = loss_history.mean()
                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                logger.info(
                    f"Epoch [{curr_epoch+1}/{max_epochs}] Iter [{curr_iter+1}/{len(train_loader)}]"
                    f" lr: {lr:.4e}, loss: {loss_val:.4f}, grad_norm: {grad_norm:.4f}, max_mem: {max_mem_mb:.0f}M"
                )

            # save every N updates
            if total_updates % cfg.train.checkpoint_interval == 0:
                checkpoint_path = (
                    Path(cfg.train.work_dir) / f"update_{total_updates}.pth"
                )
                logger.info(f"Save checkpoint: {checkpoint_path}")
                save_checkpoint(model, optimizer, cfg.train.work_dir, checkpoint_path)
                logger.info("Save checkpoint done.")
                dist.barrier()

        # end of epoch checkpoint
        checkpoint_path = Path(cfg.train.work_dir) / "update_last.pth"
        logger.info(f"Save checkpoint: {checkpoint_path}")
        save_checkpoint(model, optimizer, cfg.train.work_dir, checkpoint_path)
        logger.info("Save checkpoint done.")

        dist.barrier()

        # evaluate
        if (curr_epoch + 1) % cfg.train.get("eval_interval", 1) == 0:
            result, eval_result = do_test(cfg, model)
            if rank == 0:
                logger.info(f"Epoch {curr_epoch+1} evaluation result: {eval_result}")
                torch.save(
                    result,
                    Path(cfg.train.work_dir) / f"result_epoch_{curr_epoch+1}.pth",
                )


def setup(args):
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

    cfg = LazyConfig.load(args.config)
    # default work_dir
    cfg_path = Path(args.config)
    work_dir_root = Path(args.output_root)
    work_dir = str(work_dir_root / cfg_path.relative_to("configs/").with_suffix(""))
    cfg.train.work_dir = work_dir
    # override config
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    Path(cfg.train.work_dir).mkdir(parents=True, exist_ok=True)

    # dump config
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not args.eval_only and dist.get_rank() == 0:
        # LazyConfig.save(cfg, str(Path(work_dir) / f"{timestamp}.yaml"))
        shutil.copy(args.config, Path(work_dir) / f"{timestamp}.py")

    # logger
    if args.eval_only or args.no_log_file:
        log_file = None
    else:
        log_file = Path(work_dir) / f"{timestamp}.log"
    logger = get_logger("lmsys", log_file=log_file)
    logger.info("Start")

    # seed
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = cfg.train.get("seed", 0)
    seed_all_rng(seed)
    logger.info(f"Set random seed: {seed}")

    return cfg


def clean_up():
    dist.destroy_process_group()


def main():
    args = parse_args()
    cfg = setup(args)
    model = instantiate(cfg.model)
    logger = get_logger("lmsys")
    if args.init_only:
        init_path = Path(cfg.train.work_dir) / "initialized.pth"
        torch.save(model.state_dict(), init_path)
        logger.info(f"Saved initialized model: {init_path}")

    if cfg.train.get("cast_to_bf16", False):
        logger.info("Casting model to BF16")
        # for name, m in model.named_modules():
        #     m.to(torch.bfloat16)
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16)

    load_from = cfg.train.get("load_from", None)
    if args.load_from is not None:
        load_from = args.load_from

    if load_from is not None:
        checkpoint = torch.load(load_from, map_location="cpu")
        if "model" not in checkpoint:
            checkpoint = {"model": checkpoint}
        load_result = model.load_state_dict(checkpoint["model"], strict=False)
        logger.info(f"Load checkpoint: {load_from}")
        logger.info(f"Load checkpoint: {load_result}")

    logger.info(f"Use sharding strategy: {cfg.fsdp.sharding_strategy}")

    model = FSDP(
        model,
        auto_wrap_policy=cfg.fsdp.auto_wrap_policy,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        device_id=torch.cuda.current_device(),
        mixed_precision=cfg.fsdp.mixed_precision,
    )
    apply_activation_checkpointing(model, auto_wrap_policy=cfg.fsdp.auto_wrap_policy)

    if args.eval_only:
        result, eval_result = do_test(cfg, model)
        logger.info(f"Evaluation result: {eval_result}")
        if args.out is not None:
            torch.save(result, args.out)
    else:
        do_train(cfg, model)

    clean_up()


if __name__ == "__main__":
    main()
