import torch


class VarlenCollator:
    def __call__(self, samples):
        seq_lens = []
        cu_seqlens = [0]
        end = 0
        input_idss = []
        position_idss = []
        for sample in samples:
            seq_len = sample["input_ids"].size(0)
            seq_lens.append(seq_len)
            end += seq_len
            cu_seqlens.append(end)
            input_idss.append(sample["input_ids"])
            position_idss.append(torch.arange(seq_len))
        input_ids = torch.cat(input_idss, dim=0)
        position_ids = torch.cat(position_idss, dim=0)
        data = dict(
            batch_size=len(samples),
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            seq_lens=seq_lens,
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
            max_seq_len=max(seq_lens),
        )

        data["label"] = torch.stack([sample["label"] for sample in samples], dim=0)
        for key in samples[0]:
            if key not in data.keys():
                data[key] = [sample[key] for sample in samples]
        return [data]


class ShardedMaxTokensCollator:
    def __init__(self, max_tokens, base_collator, sort_samples=False):
        self.max_tokens = max_tokens
        self.base_collator = base_collator
        self.sort_samples = sort_samples
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def __call__(self, samples):
        # flatten if a 'sample' is a list of samples
        if isinstance(samples[0], list):
            samples = [sample for s in samples for sample in s]
        if self.sort_samples:
            samples = sorted(samples, key=lambda x: x["input_ids"].size(0))
        while len(samples) % self.world_size != 0:
            samples.append(samples[-1])
            print(
                f"Padding samples to make them divisible by num_shards={self.world_size}"
            )

        sample_index_matrix = torch.arange(len(samples)).reshape(-1, self.world_size)
        size_matrix = torch.tensor(
            [sample["input_ids"].size(0) for sample in samples]
        ).reshape(-1, self.world_size)
        micro_batch_segments = []
        # (start, end) that (size_matrix[start:end].sum(0) <= self.max_tokens).all()
        start = 0
        for end in range(size_matrix.size(0)):
            # look ahead
            if (
                size_matrix[start : end + 1].sum(0) > self.max_tokens
            ).any() and end > start:
                micro_batch_segments.append((start, end))
                start = end
        if start < size_matrix.size(0):
            micro_batch_segments.append((start, size_matrix.size(0)))

        micro_batches = []
        for start, end in micro_batch_segments:
            micro_batch_samples = [
                samples[i] for i in sample_index_matrix[start:end, self.rank]
            ]
            micro_batches.extend(self.base_collator(micro_batch_samples))
        return micro_batches
