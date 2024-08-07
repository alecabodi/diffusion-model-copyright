# Created by Chen Henry Wu
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
import random

class Preprocessor(object):

    def __init__(self, args):
        self.args = args

    def preprocess(self, raw_datasets: DatasetDict, cache_root: str):
        assert len(raw_datasets) == 3  # Not always.
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return {
            'train': train_dataset,
            'dev': dev_dataset,
            'test': test_dataset,
        }


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        n_samples = 10
        n_data = 100
        filepath = 'raw_data/pool.txt' # adjust path as needed
        
        # Open the file and read the prompts
        if filepath != '':
            with open(filepath, 'r') as file:
                prompts = file.readlines()

        prompts = [prompt.strip() for prompt in prompts]

        assert len(prompts) == 100, "The prompt list must contain exactly 100 prompts."

        # Sample 10 prompts uniformly at random
        sampled_indices = random.sample(range(n_data), n_samples)

        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "prompt": prompts[idx % len(prompts)],
                "model_kwargs": ["sample_id", "prompt"]
            }
            for idx in sampled_indices
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        n_samples = 10
        n_data = 100
        filepath = 'raw_data/pool.txt' # adjust path as needed
        
        # Open the file and read the prompts
        if filepath != '':
            with open(filepath, 'r') as file:
                prompts = file.readlines()

        prompts = [prompt.strip() for prompt in prompts]

        assert len(prompts) == 100, "The prompt list must contain exactly 100 prompts."

        # Sample 10 prompts uniformly at random
        sampled_indices = random.sample(range(n_data), n_samples)

        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "prompt": prompts[idx % len(prompts)],
                "model_kwargs": ["sample_id", "prompt"]
            }
            for idx in sampled_indices
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        n_samples = 10
        n_data = 100
        filepath = 'raw_data/pool.txt' # adjust path as needed
        
        # Open the file and read the prompts
        if filepath != '':
            with open(filepath, 'r') as file:
                prompts = file.readlines()

        prompts = [prompt.strip() for prompt in prompts]

        assert len(prompts) == 100, "The prompt list must contain exactly 100 prompts."

        # Sample 10 prompts uniformly at random
        sampled_indices = random.sample(range(n_data), n_samples)

        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "prompt": prompts[idx % len(prompts)],
                "model_kwargs": ["sample_id", "prompt"]
            }
            for idx in sampled_indices
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
