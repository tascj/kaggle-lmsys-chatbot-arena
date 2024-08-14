import copy
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import log_loss


class LMSYSDataset(Dataset):
    LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

    def __init__(
        self,
        csv_file,
        processor,
        query=None,
        include_swap=False,
        is_parquet=False,
    ):
        if is_parquet:
            df = pd.read_parquet(csv_file)
        else:
            df = pd.read_csv(csv_file)
        if query is not None:
            df = df.query(query).reset_index(drop=True)
        self.df = df
        if not isinstance(processor, list):
            processor = [processor]
        self.processor = processor
        self.include_swap = include_swap

    def __len__(self):
        return len(self.df)  # 100

    def __getitem__(self, idx):
        data = self.df.iloc[idx].to_dict()
        ret = [processor.build_input(data) for processor in self.processor]
        if self.include_swap:
            data_swap = copy.deepcopy(data)
            data_swap["model_a"] = data["model_b"]
            data_swap["model_b"] = data["model_a"]
            data_swap["response_a"] = data["response_b"]
            data_swap["response_b"] = data["response_a"]
            data_swap["winner_model_a"] = data["winner_model_b"]
            data_swap["winner_model_b"] = data["winner_model_a"]
            ret.extend(
                [processor.build_input(data_swap) for processor in self.processor]
            )
        # a list of samples
        # maybe including swapped samples
        # maybe including samples from multiple processors
        return ret

    def evaluate(self, results):
        labels = self.df[self.LABEL_COLS].values
        loss = log_loss(labels, results)
        return {"log_loss": loss}
