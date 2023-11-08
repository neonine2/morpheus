import os

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint,TQDMProgressBar
import pytorch_lightning as pl

from utils.data_prep import generate_split_from_data
from utils.dataset import make_torch_dataloader, set_seed
from utils.models import TissueClassifier
# from utils.plotting_fun import *

# Function for setting the seed
set_seed(42)

#create folder/path for model and splitdata
modelArch = 'unet'
DATA_NAME = 'cedarsLiver_sz48_pxl3_nc44'
metadata_path = '/home/jerrywang/Thomson Lab Dropbox/IMC'
    
# Parameters
in_channels = int(DATA_NAME.split('nc')[1])
img_size = 16
params = {'batch_size': 64*2,'num_workers': 4,'pin_memory':True}
split_param = {'eps':0.01, "train_lb":0.63, "split_ratio":[0.63,0.16,0.21]}

output_path = generate_split_from_data(DATA_NAME, metadata_path, param=split_param)
train_loader, val_loader, test_loader = make_torch_dataloader(output_path, img_size, model=modelArch, params=params)
model_path = os.path.join(output_path, f'model/{modelArch}')
print(model_path)

# model
model = TissueClassifier(in_channels, img_size, modelArch=modelArch)

# Change the learning rate of the optimizer
# model.configure_optimizers().param_groups[0]['lr'] = 0.001

# training
trainer = pl.Trainer(accelerator='gpu', 
                     devices=1, 
                     precision=16,
                     max_epochs=100, 
                     callbacks=[
                         ModelCheckpoint(monitor='val_bmc', mode="max", save_top_k=1, save_weights_only=True, verbose=False),
                         EarlyStopping(monitor="val_bmc", min_delta=0, patience=7, verbose=False, mode="max"),
                         TQDMProgressBar(refresh_rate=10)
                         ], 
                     default_root_dir=model_path)

trainer.fit(model, train_loader, val_loader)

# testing
trainer.test(dataloaders=test_loader)
