#%%
import torch.optim as optim
import torch.nn as nn
import torch
import random
import cv2
import pickle
from config import params
from doubleUnet.model import doubleUNet

# from utils import visualize, get_path
import utils
import time
from tensorboardX import SummaryWriter
from torchvision.transforms import ToPILImage
import numpy as np
from effdet.efficientdet_o import EfficientDet
from effdet.config_bifpn import get_efficientdet_config
from effdet import AnchorLabeler, Anchors
from effdet.loss import DetectionLoss
import torch.nn.functional as F
import os
import natsort

name = "efficientdet_d3"
config = params.d3
#%%


#%%

import torch
import torch.nn as nn


writer = SummaryWriter()


class inverse_huber_loss(nn.Module):
    def __init__(self):
        super(inverse_huber_loss, self).__init__()

    def forward(self, depth_est, depth_gt):
        absdiff = torch.abs(depth_est - depth_gt)
        C = 0.2 * torch.max(absdiff).item()
        return torch.mean(
            torch.where(absdiff < C, absdiff, (absdiff * absdiff + C * C) / (2 * C))
        )


# class siloss(nn.Module):
#     def __init__(self, variance_focus):
#         super(siloss, self).__init__()
#         self.variance_focus = variance_focus

#     def forward(self, depth_est, depth_gt, mask):
#         depth_est[depth_est <= 0] = 0.00001
#         depth_gt[depth_gt == 0] = 0.00001
#         d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])

#         return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class trainer:
    def train(self, data_loader_train, data_loader_val):

        model = self.init_model(name)
        if config["load_network"] == True:
            net = os.listdir(config["load_network_path"])
            net = natsort.natsorted(net)[-2]
            model.load_state_dict(torch.load(config["load_network_path"] + "/" + net))

        # config = get_efficientdet_config("efficientdet_d0")
        # model = EfficientDet(config, 34, 1, pretrained_backbone=False).to(params.device)

        optimizer = self.init_optimizer(model)

        # optimizer_seg = self.init_optimizer(model)
        # optimizer_depth = self.init_optimizer(model)

        criterion = nn.CrossEntropyLoss()
        # criterion1 = nn.MSELoss()
        # criterion1 = nn.HuberLoss()
        criterion1 = inverse_huber_loss()
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[params.num_epoch * 0.75],
            gamma=params.gamma,
        )

        ######################################################################
        anchors = Anchors.from_config(config)
        anchor_labeler = AnchorLabeler(
            anchors.to(params.device), config["num_classes"], match_threshold=0.5
        )

        loss_fn = DetectionLoss(config)

        ######################################################################

        save_weight, save_image = utils.get_path()
        for epoch in range(params.num_epoch):
            write_loss, write_seg, write_depth, write_od = 0, 0, 0, 0
            for idx, output in enumerate(data_loader_train):
                img = output.get("img")
                seg = output.get("seg")
                depth = output.get("depth")
                target = output.get("od")
                # if target["bbox"].shape[1] == 0:
                #     continue

                model.train()  # 혹시 모르니까
                model.zero_grad()
                # print("Device Check: ", img.device)

                #######################################################################
                out_seg, out_depth, class_out, box_out = model(img)
                cls_targets, box_targets, num_positives = anchor_labeler.batch_label_anchors(
                    target["bbox"], target["cls"]
                )
                # print(class_out.device(), box_out.device(), cls_targets.device(), box_targets.device(), num_positives.device())
                Loss_od, class_loss, box_loss = loss_fn(
                    class_out, box_out, cls_targets, box_targets, num_positives
                )
                Loss_seg = criterion(
                    out_seg, seg.long()
                )  # out_seg = [1,34,1024,2048], seg = [1,1024,2048] (0~33)

                # Loss_depth = criterion1(out_depth, depth)
                Loss_depth = criterion1.forward(out_depth, depth)
                Loss = (
                    Loss_seg
                    + params.depth_weight * Loss_depth
                    + params.od_weight * Loss_od
                )

                ###########################################################################

                ################################ Old #############################################
                # Loss_seg = criterion(
                #     out_seg, seg.long()
                # )  # out_seg = [1,34,1024,2048], seg = [1,1024,2048] (0~33)

                # Loss_depth = criterion1(out_depth, depth)

                # Loss = Loss_seg + 10 * Loss_depth
                ################################ Old #############################################

                # Loss = Loss_seg
                Loss.backward()
                optimizer.step()

                write_loss += Loss
                write_seg += Loss_seg
                write_depth += Loss_depth
                write_od += Loss_od

                if idx % 100 == 0 or idx == (len(data_loader_train) - 1):
                    bs = img.size(0)
                    # print("model output", box_out[])
                    # out_depth *= 126
                    img *= 255
                    # depth *= 126
                    out_seg = torch.argmax(out_seg, dim=1)  # size = (1024, 2048)
                    out_seg = utils.decode_segmap(out_seg[0])
                    seg = utils.decode_segmap(seg[0])
                    out_seg = ToPILImage()(out_seg.detach())
                    seg = ToPILImage()(seg.detach())  # size = (1024, 2048, 3)
                    out_depth = ToPILImage()(out_depth[0].detach())
                    img = F.interpolate(img, size=(128, 256))
                    img = img[0].permute(1, 2, 0).contiguous().int().to("cpu")
                    img = img.numpy()[:, :, ::-1].copy()
                    depth = ToPILImage()(depth[0].detach())
                    #########################################################################################
                    class_out, box_out, indices, classes = utils.post_process(
                        class_out,
                        box_out,
                        num_levels=config["num_levels"],
                        num_classes=config["num_classes"],
                        max_detection_points=config["max_detection_points"],
                    )

                    img_scale, img_size = (
                        [torch.tensor(1).to(params.device) for _ in range(bs)],
                        [torch.tensor([256, 128]).to(params.device) for _ in range(bs)],
                    )
                    res = utils.batch_detection(
                        bs,
                        class_out,
                        box_out,
                        anchors.boxes,
                        indices,
                        classes,
                        img_scale,
                        img_size,
                        max_det_per_image=config["max_det_per_image"],
                        soft_nms=config["soft_nms"],
                    )
                    bboxes = utils.decode_det(res[0].unsqueeze(0))
                    # print("resshape", res.shape)
                    #########################################################################################

                    imgs = [
                        # out_seg.detach(),
                        img,
                        out_seg,
                        seg,
                        out_depth,
                        depth,
                        img,
                        img,
                    ]

                    utils.visualize(imgs, save_image, bboxes, target, epoch, idx)
                    print(
                        "epoch : {}, index: {},  Loss : {:.4f}, Loss_seg : {:.4f}, Loss_depth : {:.4f}, Loss_cls : {:.4f}, Loss_box : {:.4f}".format(
                            epoch, idx, Loss, Loss_seg, Loss_depth, class_loss, box_loss
                        )
                    )

            write_loss /= len(data_loader_train)
            write_seg /= len(data_loader_train)
            write_depth /= len(data_loader_train)
            write_od /= len(data_loader_train)

            writer.add_scalar("Loss/train", write_loss, epoch)
            writer.add_scalar("Loss/train/od", write_od, epoch)
            writer.add_scalar("Loss/train/seg", write_seg, epoch)
            writer.add_scalar("Loss/train/depth", write_depth, epoch)

            ################################ Validate ######################################
            with torch.set_grad_enabled(False):
                model.eval()
                val_loss = 0
                for i, output in enumerate(data_loader_val):
                    img = output.get("img")
                    seg = output.get("seg")
                    depth = output.get("depth")
                    target = output.get("od")
                    # if target["bbox"].shape[1] == 0:
                    #     continue
                    out_seg, out_depth, class_out, box_out = model(img)

                    cls_targets, box_targets, num_positives = anchor_labeler.batch_label_anchors(
                        target["bbox"], target["cls"]
                    )
                    Loss_od, class_loss, box_loss = loss_fn(
                        class_out, box_out, cls_targets, box_targets, num_positives
                    )
                    Loss_seg = criterion(out_seg, seg.long())
                    # Loss_depth = criterion1(out_depth, depth)
                    Loss_depth = criterion1.forward(out_depth, depth)
                    Loss = Loss_seg + 10 * Loss_depth + 0.005 * Loss_od
                    val_loss += Loss
                val_loss = val_loss / len(data_loader_val)

                writer.add_scalar("Loss/val", val_loss, epoch)

            ################################ Validate ######################################

            # writer.add_scalar("Loss/")
            scheduler.step()
            utils.save_network(model.eval(), save_weight, epoch)
            # scheduler_s.step()
            # scheduler_d.step()

    def init_model(self, name):
        config = get_efficientdet_config(name)
        model = EfficientDet(
            config,
            params.num_classes_seg,
            params.num_classes_depth,
            pretrained_backbone=False,
        ).to(params.device)
        # model = FSUNet(params.num_classes_seg).to(params.device)
        # model.apply(self.weights_init)
        model.train()
        return model

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def init_optimizer(self, model):
        optimizer = optim.Adam(
            model.parameters(), params.lr, betas=(params.beta1, 0.999)
        )
        return optimizer

