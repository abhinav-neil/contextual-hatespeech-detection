import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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


def train(model, train_loader, val_loader, args):
    """
    Train the model.

    Args:
        model (torch.nn.Module): Model to train.
        train_data_loader (torch.utils.data.DataLoader): Data loader for training.
        val_data_loader (torch.utils.data.DataLoader): Data loader for validation.
        args: training args containing the following attributes:
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
    wandb.init(project=args.wandb_project, name=args.wandb_run)

    # Load checkpoint if provided
    if args.resume_ckpt is not None:
        checkpoint = torch.load(args.resume_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        counter = checkpoint['counter']
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']

    # set loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # train
    model.to(args.device)
    model.train()
    best_val_loss = float('inf')
    best_val_acc = 0.0
    counter = 0

    for epoch in range(args.num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, args.device, args.accumulation_steps)

        # Evaluation
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, args.device)

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
            if counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
        # empty GPU cache
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            
        print(f'epoch {epoch+1}: train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, val acc: {val_acc:.3f}')

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
    model.to(device)
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

def train_pl(model, train_loader, val_loader, args):
    """
    Trains the given model with the provided arguments.

    Args:
        model (pl.LightningModule): The PyTorch Lightning model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation data.
        args (dict): A dictionary containing the following keys:
            - num_epochs (int, optional): The number of epochs for training. Defaults to 10.
            - patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 5.
            - ckpt_dir (str, optional): The directory where the model checkpoints will be saved. Defaults to 'model_ckpts/'.
            - ckpt_name (str, optional): The name of the model checkpoint file. Defaults to 'ckpt_best'.
            - resume_ckpt (str, optional): The path to a checkpoint from which training will resume. Defaults to None.
            - precision (int, optional): The precision to use for training. Defaults to 32.
    """

    # Define early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=args.get('patience', 5),
        verbose=True,
        mode='min'
    )

    # Define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath= args.get('ckpt_dir', 'model_ckpts'),
        filename=args.get('ckpt_name', 'ckpt_best'),
        save_top_k=1,
        mode='min'
    )
    
    # Define logger
    logger = TensorBoardLogger("lightning_logs", name=model.__class__.__name__)


    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=args.get('num_epochs', 10),
        precision = args.get('precision', 32),
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=5,
        logger=logger,
    )
    
    # print trainer args
    # print("\ntrainer args:")
    # for arg, value in vars(trainer).items():
    #     print(f"{arg}: {value}")
    # print("\n")

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.get('resume_ckpt', None))
    print(f'training on device: {model.device}')
    
    return model, trainer