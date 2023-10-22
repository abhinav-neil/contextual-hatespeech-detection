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
        self,                      
        lm_name: str,
        pos_weight: float=1.0,
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        metrics: list = ['acc', 'precision', 'recall', 'f1'],
        **kwargs,
    ):
        """
        Initialize the HateSpeechClassifier.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # instantiate BERT model
        self.model = AutoModel.from_pretrained(lm_name)
        self.fc = nn.Linear(self.model.config.hidden_size, 1)
        
        # instantiate loss function
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        # instantiate metrics
        self.metrics_dict = {}
        if 'acc' in metrics:
            self.metrics_dict['acc'] = torchmetrics.Accuracy(task='binary', average='macro')
        if 'precision' in metrics:
            self.metrics_dict['precision'] = torchmetrics.Precision(task='binary', average='macro')
        if 'recall' in metrics:
            self.metrics_dict['recall'] = torchmetrics.Recall(task='binary', average='macro')
        if 'f1' in metrics:
            self.metrics_dict['f1'] = torchmetrics.F1Score(task='binary', average='macro')

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.fc(pooled_output).squeeze(-1)
        return logits
    
    def step(self, batch):
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        logits = self.forward(input_ids, attention_mask)
        loss = self.loss(logits, labels.float())
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        self.log("train_loss", loss, batch_size=len(batch["labels"]), on_epoch=True)
        return loss

    def compute_metrics(self, logits, labels, prefix):
        preds = (torch.sigmoid(logits) > 0.5).long()
        # move metrics to the device of the logits
        for metric in self.metrics_dict.values():
            metric.to(logits.device)
        for metric_name, metric in self.metrics_dict.items():
            metric_val = metric(preds, labels)
            self.log(f"{prefix}_{metric_name}", metric_val, batch_size=len(labels), on_epoch=True)

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.compute_metrics(logits, labels, prefix="val")
        self.log("val_loss", loss, batch_size=len(labels), on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.compute_metrics(logits, labels, prefix="test")
        return loss
    
    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        logits = self.forward(input_ids, attention_mask)
        preds = (torch.sigmoid(logits) > 0.5).long()
        return labels, preds
    
    def on_train_start(self):
        device = next(self.parameters()).device  # get the device of the model
        for metric in self.metrics_dict.values():
            metric.to(device)  # move each metric to the model's device

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
