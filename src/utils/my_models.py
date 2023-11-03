import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import TQDMProgressBar
import torchmetrics.functional.classification as tfcl

# import torchvision as tv
# import tensorflow as tf
# import tensorflow_addons as tfa

class TissueClassifier(pl.LightningModule):
    def __init__(self, in_channels, img_size=None, modelArch=None, num_target_classes=2):
        super().__init__()
        self.classes = num_target_classes
        modelArch = modelArch.lower()
        if modelArch == 'resnet':
            # init a pretrained resnet
            backbone = tv.models.resnet50(weights="DEFAULT")
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            layers[0] = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), 
                                                    padding=(3, 3), bias=False)
            layers.append(nn.Flatten())
            layers.append(nn.Linear(num_filters, num_target_classes))
            layers.append(nn.Softmax())
            self.predictor = nn.Sequential(*layers)
        elif modelArch == 'cnn':
            self.predictor = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Flatten(),
                nn.Linear(128*7*7, 60),
                nn.ReLU(),
                nn.Dropout(p=0.8),
                nn.Linear(60, num_target_classes),
                nn.Softmax())
        elif modelArch == 'unet':
            backbone = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                in_channels=in_channels, 
                                out_channels=1, 
                                init_features=in_channels)
            classifier = torch.nn.Sequential()
            classifier.add_module('flatten', nn.Flatten())
            classifier.add_module('fc', nn.Linear(img_size*img_size, num_target_classes))
            classifier.add_module('act', nn.Softmax())
            self.predictor = nn.Sequential(*[backbone, classifier])
        elif modelArch == 'mlp':
            self.predictor = nn.Sequential(
                nn.Linear(in_channels, 30),
                nn.ReLU(),
                nn.Linear(30, 10),
                nn.ReLU(),
                nn.Linear(10, num_target_classes),
                nn.Softmax())
        elif modelArch == 'lr':
            self.predictor = nn.Sequential(
                nn.Linear(in_channels, 1),
                nn.Sigmoid())

    def forward(self, x):
        self.predictor.eval()
        pred = self.predictor(x)
        return pred

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def execute_and_get_metric(self,batch,mode):
        x, target = batch
        target = F.one_hot(target, num_classes=self.classes).float()
        pred = self.predictor(x)
        metric_dict = log_metrics(mode, pred, target)
        return metric_dict

    def training_step(self, train_batch, batch_idx):
        metric_dict=self.execute_and_get_metric(train_batch, 'train')
        self.log_dict(metric_dict, on_step=False, on_epoch=True, prog_bar=False)
        return metric_dict['train_bce']

    def validation_step(self, val_batch, batch_idx):
        metric_dict=self.execute_and_get_metric(val_batch, 'val')
        self.log_dict(metric_dict, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, test_batch, batch_idx):
        metric_dict=self.execute_and_get_metric(test_batch, 'test')
        self.log_dict(metric_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

def log_metrics(mode, preds, target):
    # classification metrics
    if preds.shape[1] == 1:
        # Create a new column that is 1 minus the first column
        new_col = 1 - preds[:, 0]
        # Append the new column to the original matrix
        preds = torch.column_stack((preds, new_col))
    bce = F.binary_cross_entropy_with_logits(preds, target)
    
    preds = torch.argmax(preds, dim=1).float()
    target = torch.argmax(target, dim=1).float()
    test_acc = tfcl.binary_accuracy(preds, target)
    bmc = tfcl.binary_matthews_corrcoef(preds, target).float()
    auroc = tfcl.binary_auroc(preds,target)
    f1 = tfcl.binary_f1_score(preds,target)
    precision = tfcl.binary_precision(preds,target)
    recall = tfcl.binary_recall(preds,target)
    metric_dict = {mode+'_bce':bce, 
                   mode+'_precisio1n':precision, 
                   mode+'_recall':recall,
                   mode+'_bmc':bmc,
                   mode+'_auroc':auroc,
                   mode+'_f1':f1,
                   mode+'_acc':test_acc}
    return metric_dict
    
def get_prediction(model, data_loader):
    m = nn.Softmax(dim=1)
    preds = []
    labels = []
    for x, y in iter(data_loader):
        pred = model(x)
        if pred.shape[1] == 1:
            # Create a new column that is 1 minus the first column
            new_col = 1 - pred[:, 0]
            # Append the new column to the original matrix
            pred = torch.column_stack((pred, new_col))
        preds.append(m(pred)[:,1])
        labels.append(y)
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    return preds, labels
