from config import params
from base_loader import BaseLoader
from torch.utils import data
from train_noatt import trainer
import torch

state = params.mode

if state == "Train":
    torch.backends.cudnn.enabled = False
    train_dataset = BaseLoader("train")
    val_dataset = BaseLoader("val")
    train_loader = data.DataLoader(train_dataset, batch_size=params.batch_size,collate_fn=train_dataset.collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size=params.batch_size, collate_fn=val_dataset.collate_fn)

    agent = trainer()
    agent.train(train_loader, val_loader)
    # =trainer()

