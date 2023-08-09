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
        num_classes: int=2,
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        **kwargs,
    ):
        """
        Initialize the HateSpeechClassifier.

        Args:
            pretrained_model_name (str): Pretrained model name or path.
            num_classes (int): Number of output labels.
            lr (float): Learning rate for optimizer (Adam).
            weight_decay (float): Weight decay for optimizer (Adam).
        """
        super().__init__()
        self.save_hyperparameters()
        
        # instantiate BERT model
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        # self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.model.config.hidden_size, num_classes)
        
        # instantiate loss function
        self.loss = nn.CrossEntropyLoss()

        # instantiate accuracy metric
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
    
    def step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        y_hat = self.forward(input_ids, attention_mask)
        loss = self.loss(y_hat, labels)
        return loss, y_hat, labels


    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("train_loss", loss, batch_size=len(y))
        self.log("train_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("val_loss", loss, batch_size=len(y))
        self.log("val_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("test_loss", loss, batch_size=len(y))
        self.log("test_acc", acc, batch_size=len(y))  # Log accuracy
        return loss
    
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