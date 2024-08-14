import torch


def to_gpu(data):
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)
    elif isinstance(data, dict):
        return {key: to_gpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_gpu(value) for value in data]
    elif isinstance(data, str):
        return data
    elif isinstance(data, int):
        return data
    elif data is None:
        return data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
