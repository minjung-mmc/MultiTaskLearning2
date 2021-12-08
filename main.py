from config import params
from base_loader import BaseLoader
from torch.utils import data
from train import trainer
import torch
from effdet.MaP import Map

state = params.mode

if state == "Train":
    torch.backends.cudnn.enabled = False
    train_dataset = BaseLoader("train")
    val_dataset = BaseLoader("val")
    test_dataset = BaseLoader("test")
    train_loader = data.DataLoader(
        train_dataset, batch_size=params.batch_size, collate_fn=train_dataset.collate_fn
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=params.batch_size, collate_fn=val_dataset.collate_fn
    )
    test_loader = data.DataLoader(
        val_dataset, batch_size=params.batch_size, collate_fn=val_dataset.collate_fn
    )

    agent = trainer()
    agent.train(train_loader, val_loader)
    # predict, target = agent.test(test_loader)
    # print("predict, target", target)

    # =trainer()

