import pytorch_lightning as L
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim

#class TimmClassifier(L.LightningModule):
    #def __init__(
        #self,
        #model_name,
        #num_classes,
        #pretrained=True,
        #lr=1e-3,
        #weight_decay=1e-5,
        #scheduler_patience=3,
        #scheduler_factor=0.1,
        #min_lr=1e-6
    # ):
        #super().__init__()
        #self.save_hyperparameters()
        #self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        #self.criterion = nn.CrossEntropyLoss()

    #def forward(self, x):
        #return self.model(x)

class TimmClassifier(LightningModule):
    def __init__(self,lr: float = 1e-3):
        self,
        self.lr = lr
        super(TimmClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)    

    #def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        #return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
        
    
        
