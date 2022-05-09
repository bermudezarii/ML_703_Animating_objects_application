"""
    Author: Ariana Bermudez
    Date: 18/04/2022.

    This file run the training and testing of the model.
"""

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from hyperparameters import parameters as params
from dataset import CustomDataModule
from network import Net

pl.seed_everything(params['seed'])

name = 'Efficientnet'

wandb_logger = WandbLogger(project="CNN_Binary_Template", name=name)

# setup data
dataset = CustomDataModule(aug=True)
dataset.setup() 

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val/acc',
    dirpath='../weights/',
    filename=name+'_weights-{epoch:02d}',
    mode='max',
)
early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.000001, patience=10, verbose=False, mode="min")

trainer = pl.Trainer(
    logger=wandb_logger,    # W&B integration
    log_every_n_steps=2,   # set the logging frequency
    min_epochs=params['min_epochs'],
    max_epochs=params['epochs'],
    precision=params['precision'],
    accumulate_grad_batches=params['accumulate_grad_batches'],
    gpus=-1,                # use all GPUs
    deterministic=True,      # keep it deterministic
    callbacks=[checkpoint_callback]#, early_stop_callback]
)

# setup model
device = 'cuda' if cuda.is_available() else 'cpu'
# model = Net.load_from_checkpoint('../weights/CHANGE TO TRAINED WEIGHTS').to(device)
# model.freeze_weights()

# fit the model
# trainer.fit(model)

# evaluate the model on a test set
trainer.test(datamodule=dataset, ckpt_path="/home/ariana.venegas/Documents/II_Semester/ML703_project/CNN_Binary_Template_2.0/weights/Efficientnet_weights-epoch=00-v1.ckpt") # uses best model
