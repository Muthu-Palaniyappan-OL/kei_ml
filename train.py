import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from model import Model
from torchmetrics import Accuracy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from data import *


class CustomModel(L.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = Model()
        self.accuracy = Accuracy(task="multilabel", num_labels=(11 * 11))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        outputs = outputs.reshape(outputs.shape[0], 11, 11)
        loss = self.custom_loss(outputs, targets)
        metric = self.custom_metric(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        outputs = outputs.reshape(outputs.shape[0], 11, 11)
        loss = self.custom_loss(outputs, targets)
        metric = self.custom_metric(outputs, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def custom_loss(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        return F.binary_cross_entropy_with_logits(outputs, targets)

    def custom_metric(self, outputs, targets):
        outputs = outputs.view(-1, 121)
        targets = targets.view(-1, 121)
        return self.accuracy(outputs, targets)

    def train_dataloader(self):
        return DataLoader(
            HighEncodedVocDataset(
                root="/tmp/tmp", image_set="train", transform=ToTensor()
            ),
            batch_size=32,
            shuffle=True,
            num_workers=7,
        )

    def val_dataloader(self):
        return DataLoader(
            HighEncodedVocDataset(
                root="/tmp/tmp", image_set="valid", transform=ToTensor()
            ),
            batch_size=32,
            num_workers=7,
        )


model = CustomModel()

trainer = L.Trainer(
    max_epochs=128,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")],
)
trainer.fit(model)

torch.save(model.model.state_dict(), "model.pth")
