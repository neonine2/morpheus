import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from torchmetrics.functional.classification import binary_accuracy,binary_matthews_corrcoef,\
binary_auroc,binary_f1_score, binary_recall, binary_precision

def log_metrics(mode, preds, target):
    # classification metrics
    bce = F.binary_cross_entropy_with_logits(preds, target)
    
    preds = torch.argmax(preds, dim=1).float()
    target = torch.argmax(target, dim=1).float()
    test_acc = binary_accuracy(preds, target)
    bmc = binary_matthews_corrcoef(preds, target).float()
    auroc = binary_auroc(preds,target)
    f1 = binary_f1_score(preds,target)
    precision = binary_precision(preds,target)
    recall = binary_recall(preds,target)
    metric_dict = {mode+'_bce':bce,mode+'_precision':precision,mode+'_recall':recall,mode+'_bmc':bmc,
                   mode+'_auroc':auroc,mode+'_f1':f1,mode+'_acc':test_acc}
    return metric_dict
    
class TissueClassifier(pl.LightningModule):
    def __init__(self, num_channels, num_target_classes=2):
        super().__init__()
        self.classes = num_target_classes
        
        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        layers[0] = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_extractor = nn.Sequential(*layers)
        # self.feature_extractor.requires_grad_(False)
        
        # use the pretrained model to classify patches
        self.classifier = nn.Linear(num_filters, num_target_classes)
        # self.predictor = nn.Sequential(
        #     nn.Conv2d(num_channels, 64, 3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(64, 128, 3),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Flatten(),
        #     nn.Linear(128*7*7, 60),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.8),
        #     nn.Linear(60, num_target_classes))

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        preds = self.classifier(representations)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
        return optimizer
    
    def execute_and_get_metric(self,batch,mode):
        x, target = batch
        target = F.one_hot(target, num_classes=self.classes).float()
        representations = self.feature_extractor(x).flatten(1)
        preds = self.classifier(representations)
        metric_dict = log_metrics(mode, preds, target)
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
 