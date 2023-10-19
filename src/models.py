import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModel
import pytorch_lightning as pl
import torchmetrics

    
class HateSpeechClassifier(pl.LightningModule):
    """
    Hate speech classifier using a transformer-based model.
    """

    def __init__(
        self,                      lm_name: str='roberta-large',
        # num_classes: int=2,
        pos_weight: float=10.0,
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        **kwargs,
    ):
        """
        Initialize the HateSpeechClassifier.

        Args:
            lm_name (str): Pretrained model name or path.
            num_classes (int): Number of output labels.
            pos_weight (float): Weight for positive class in CrossEntropyLoss.
            lr (float): Learning rate for optimizer (Adam).
            weight_decay (float): Weight decay for optimizer (Adam).
        """
        super().__init__()
        self.save_hyperparameters()
        
        # instantiate BERT model
        self.model = AutoModel.from_pretrained(lm_name)
        # self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.model.config.hidden_size, 1)
        
        # instantiate loss function
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        # instantiate accuracy metric
        self.accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.pooler_output # for BERT models
        pooled_output = outputs.last_hidden_state[:, 0]
        # pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output).squeeze(-1)
        return logits
    
    def step(self, batch):
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        logits = self.forward(input_ids, attention_mask)
        loss = self.loss(logits, labels.float())
        return loss, logits, labels


    def training_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        acc = self.accuracy(torch.sigmoid(logits), y)  # Compute accuracy
        self.log("train_loss", loss, batch_size=len(y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        acc = self.accuracy(torch.sigmoid(logits), y)  # Compute accuracy
        self.log("val_loss", loss, batch_size=len(y), on_epoch=True)
        self.log("val_acc", acc, batch_size=len(y), on_epoch=True)  # Log accuracy
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        acc = self.accuracy(torch.sigmoid(logits), y)  # Compute accuracy
        self.log("test_acc", acc, batch_size=len(y), on_epoch=True)  # Log accuracy
        return loss
    
    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        logits = self.forward(input_ids, attention_mask)
        preds = (torch.sigmoid(logits) > 0.5).long()
        return labels, preds
    
    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer