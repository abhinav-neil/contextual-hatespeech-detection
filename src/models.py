import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModel
import pytorch_lightning as pl
import torchmetrics

# class HateSpeechClassifier(nn.Module):
#     """
#     Hate speech classifier using a transformer-based model.
#     """

#     def __init__(self, pretrained_model_name, num_labels):
#         """
#         Initialize the HateSpeechClassifier.

#         Args:
#             pretrained_model_name (str): Pretrained model name or path.
#             num_labels (int): Number of output labels.
#         """
#         super().__init__()
#         self.model = AutoModel.from_pretrained(pretrained_model_name)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(self.model.config.hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.fc(pooled_output)
#         return logits
    
class HateSpeechClassifier(pl.LightningModule):
    """
    Hate speech classifier using a transformer-based model.
    """

    def __init__(
        self,                      pretrained_model_name: str='roberta-large',
        # num_classes: int=2,
        pos_weight: float=5.0,
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        **kwargs,
    ):
        """
        Initialize the HateSpeechClassifier.

        Args:
            pretrained_model_name (str): Pretrained model name or path.
            num_classes (int): Number of output labels.
            pos_weight (float): Weight for positive class in CrossEntropyLoss.
            lr (float): Learning rate for optimizer (Adam).
            weight_decay (float): Weight decay for optimizer (Adam).
        """
        super().__init__()
        self.save_hyperparameters()
        
        # instantiate BERT model
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        # self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.model.config.hidden_size, 1)
        
        # instantiate loss function
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        # instantiate accuracy metric
        self.accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
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
        self.log("train_loss", loss, batch_size=len(y))
        self.log("train_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        acc = self.accuracy(torch.sigmoid(logits), y)  # Compute accuracy
        self.log("val_loss", loss, batch_size=len(y))
        self.log("val_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        acc = self.accuracy(torch.sigmoid(logits), y)  # Compute accuracy
        self.log("test_loss", loss, batch_size=len(y))
        self.log("test_acc", acc, batch_size=len(y))  # Log accuracy
        return loss
    
    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        logits = self.forward(input_ids, attention_mask)
        preds = (torch.sigmoid(logits) > 0.5).long()
        return labels, preds
    
    def on_train_epoch_end(self): 
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy.compute())

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer