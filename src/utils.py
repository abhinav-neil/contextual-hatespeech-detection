import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score
import wandb


class HateSpeechDataset(Dataset):
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
        df = df.loc[:, ['tweet', 'class']]
        # df = df[['tweet', 'label']]  # Only keep necessary columns
        df['class'] = df['class'].apply(lambda x: 1 if x == 0 else 0)  # binary indicator of hate_speech or not
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        tweet, label = self.df.iloc[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(data_path, tokenizer, max_len=128, batch_size=64, num_workers=12):
    """
    Create data loaders for training, validation, and testing.

    Args:
        train_df (pandas.DataFrame): Training data.
        val_df (pandas.DataFrame): Validation data.
        test_df (pandas.DataFrame): Test data.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        max_len (int): Maximum sequence length.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the data loaders
    Returns:
        train_data_loader (torch.utils.data.DataLoader): Data loader for training.
        val_data_loader (torch.utils.data.DataLoader): Data loader for validation.
        test_data_loader (torch.utils.data.DataLoader): Data loader for testing.
    """
    train_dataset = HateSpeechDataset(data_path, tokenizer, 'train', max_len)
    val_dataset = HateSpeechDataset(data_path, tokenizer, 'val', max_len)
    test_dataset = HateSpeechDataset(data_path, tokenizer, 'test', max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader

