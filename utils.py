import os
import numpy as np
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

    def __init__(self, tweets, labels, tokenizer, max_len):
        """
        Initialize the HateSpeechDataset.

        Args:
            tweets (ndarray): Array of tweets.
            labels (ndarray): Array of labels.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
            max_len (int): Maximum sequence length.
        """
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]

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

def create_data_loaders(train_df, val_df, test_df, tokenizer, max_len, batch_size):
    """
    Create data loaders for training, validation, and testing.

    Args:
        train_df (pandas.DataFrame): Training data.
        val_df (pandas.DataFrame): Validation data.
        test_df (pandas.DataFrame): Test data.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        max_len (int): Maximum sequence length.
        batch_size (int): Batch size.

    Returns:
        train_data_loader (torch.utils.data.DataLoader): Data loader for training.
        val_data_loader (torch.utils.data.DataLoader): Data loader for validation.
        test_data_loader (torch.utils.data.DataLoader): Data loader for testing.
    """
    train_dataset = HateSpeechDataset(train_df.tweet.to_numpy(), train_df.label.to_numpy(), tokenizer, max_len)
    val_dataset = HateSpeechDataset(val_df.tweet.to_numpy(), val_df.label.to_numpy(), tokenizer, max_len)
    test_dataset = HateSpeechDataset(test_df.tweet.to_numpy(), test_df.label.to_numpy(), tokenizer, max_len)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    return train_data_loader, val_data_loader, test_data_loader

class HateSpeechClassifier(nn.Module):
    """
    Hate speech classifier using a transformer-based model.
    """

    def __init__(self, pretrained_model_name, num_labels):
        """
        Initialize the HateSpeechClassifier.

        Args:
            pretrained_model_name (str): Pretrained model name or path.
            num_labels (int): Number of output labels.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

def train_epoch(model, data_loader, loss_fn, optimizer, device, accumulation_steps=1):
    """
    Train the model for one epoch with gradient accumulation.

    Args:
        model (torch.nn.Module): Model to train.
        data_loader (torch.utils.data.DataLoader): Data loader for training.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to use ('cuda' or 'cpu').
        accumulation_steps (int): Number of steps to accumulate gradients before performing optimization.

    Returns:
        float: Average training loss.
    """
    model = model.train()
    losses = []
    total_loss = 0.0

    for step, batch in enumerate(tqdm(data_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)

        loss.backward()

        total_loss += loss.item()
        losses.append(loss.item())

        if (step + 1) % accumulation_steps == 0:
            # perform optimization and gradient update
            optimizer.step()
            optimizer.zero_grad()

    # perform final optimization step if there are remaining accumulated gradients
    if accumulation_steps > 1 and (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # release GPU memory
    del loss, outputs, input_ids, attention_mask, labels
    torch.cuda.empty_cache()
    
    return np.mean(losses)


def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, device, num_epochs, accumulation_steps=10, patience=2, resume_ckpt=None, wandb_project='aiahs', wandb_run=None):
    """
    Train the model.

    Args:
        model (torch.nn.Module): Model to train.
        train_data_loader (torch.utils.data.DataLoader): Data loader for training.
        val_data_loader (torch.utils.data.DataLoader): Data loader for validation.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to use ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
        resume_ckpt (str): Path to a checkpoint file to resume training from. Default is None.
        wandb_project (str): Name of the W&B project to log to. Default is 'aiahs'.
        wandb_run (wandb.sdk.wandb_run.Run): W&B run object. Default is None.

    Returns:
        float: Best validation accuracy achieved during training.
    """
    # create checkpoint directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    # Initialize W&B project and run
    wandb.init(project=wandb_project, name=wandb_run)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    counter = 0

    # Load checkpoint if provided
    if resume_ckpt is not None:
        checkpoint = torch.load(resume_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        counter = checkpoint['counter']
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']

    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, accumulation_steps)

        # Evaluation
        val_loss, val_acc = evaluate(model, val_data_loader, loss_fn, device)

        # Logging with wandb
        wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

        # Early stopping and checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'counter': counter,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc
            }, 'checkpoints/ckpt_best.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
        # empty GPU cache
        if device == 'cuda':
            torch.cuda.empty_cache()

    wandb.finish()
    return best_val_acc

def evaluate(model, data_loader, loss_fn, device):
    """
    Evaluate the model on the given data.

    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation.
        loss_fn (torch.nn.Module): Loss function.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        float: Average evaluation loss.
        float: Accuracy.
    """
    model = model.eval()
    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            predicted_labels = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    accuracy = accuracy_score(targets, predictions)

    return np.mean(losses), accuracy
