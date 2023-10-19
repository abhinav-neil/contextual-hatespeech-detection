import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import datasets
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score
import wandb


class HSDataset(Dataset):
    """
    Dataset class for hate speech classification.
    """

    def __init__(self, data_path, tokenizer, split=None, max_len=128):
        """
        Initialize the HateSpeechDataset.

        Args:
            data_path (str): Path to the data.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
            split (str): Split of the data. Must be one of 'train', 'val', or 'test'. Defaults to None.
            max_len (int): Maximum sequence length.
        """
        df = pd.read_csv(data_path)
        df = df[df.split == split] if split else df
        df = df.loc[:, ['text', 'label']]
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text, label = self.df.iloc[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        

def create_dataloaders(data_path, tokenizer, max_len=128, bsz=64, num_workers=12, resample=False):
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_path (str): Path to the data.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        max_len (int): Maximum sequence length.
         (int): Batch size.
        num_workers (int): Number of workers for the data loaders
        resample (bool): Whether to resample the training data to have equal number of samples per class.
    Returns:
        train_data_loader (torch.utils.data.DataLoader): Data loader for training.
        val_data_loader (torch.utils.data.DataLoader): Data loader for validation.
        test_data_loader (torch.utils.data.DataLoader): Data loader for testing.
    """
    
    # create train, val, test datasets
    train_dataset = HSDataset(data_path, tokenizer, split='train', max_len=max_len)
    val_dataset = HSDataset(data_path, tokenizer, split='val', max_len=max_len)
    test_dataset = HSDataset(data_path, tokenizer, split='test', max_len=max_len)
    
    # resampling
    if resample:
        # Compute sample weights
        train_labels = train_dataset.df['label'].values
        class_sample_count = np.array([len(np.where(train_labels==t)[0]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=False, num_workers=12, sampler=sampler)
    else:        
        train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=12)

    train_loader = DataLoader(train_dataset, batch_size=bsz, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bsz, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=bsz, num_workers=num_workers)

    return train_loader, val_loader, test_loader

