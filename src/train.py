import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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