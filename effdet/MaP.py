""" PyTorch Soft-NMS

This code was adapted from a PR for detectron2 submitted by https://github.com/alekseynp
https://github.com/facebookresearch/detectron2/pull/1183/files

Detectron2 is licensed Apache 2.0, Copyright Facebook Inc.
"""
import torch
from typing import List
import numpy as np
from collections import Counter


def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: IoU, sized [N,M].
    """

    boxes1 = 8*boxes1
    area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])  # [N,]
    area2 = (boxes2[3] - boxes2[1]) * (boxes2[2] - boxes2[0])  # [M,]

    idx2 = [2,1]
    idx1 = [3,2]
    width_height = torch.min(boxes1[None, 2:], boxes2[idx1]) - torch.max(
        boxes1[None,:2], boxes2[idx2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=1)  # [N,M]

    # print("boxes1",[boxes1, boxes2])
    
    # print("inter : ",inter)

    # handle empty boxes
    # iou = torch.where(
    #     inter > 0,
    #     inter / (area1 + area2 - inter),
    #     torch.zeros(1, dtype=inter.dtype, device=inter.device),
    # )
    iou = torch.max(inter / (area1 + area2 - inter), torch.tensor(0).to(inter.device))

    # print("iou : ", iou)
    return iou


def Map(
    prediction,
    GT,
    num_classes = 34,
    iou_threshold: float = .5,
):

    device = prediction[0].device
    average_precisions = []
    epsilon = 1e-6


    for c in range(1,num_classes+1):

        detections = []
        ground_truths = []

        print("========================================",c)
        for i in range(len(prediction)):

            detection_list = prediction[i]
            for detection in detection_list:
                
                if detection[5]==c:
                    detections.append(torch.cat([torch.tensor([i]).to(device), detection], dim=-1))
                    
                    # detections_img_idx.append(i)
        
        for i in range(len(GT)):

            true_list = GT[i]
            
            
            for j in range(len(true_list['cls'][0])):
                if true_list["cls"][0][j] == c:

                    ground_truths.append(torch.cat([torch.tensor([i]).to(device), true_list["bbox"][0][j],true_list["cls"][0][j].unsqueeze(0)], dim=-1))
                    
                   
                    # ground_truths_img_idx.append(i)
        
        amount_bboxes = Counter([gt[0].item() for gt in ground_truths])

        # for k in range(len(GT)):
        #     print("test1", amount_bboxes[k])
        #     print("test1", GT[k]['cls'])
            

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)


        detections.sort(key=lambda x: x[5], reverse=True)
        
       
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            print("dt", detection[1:5]*8)
            print("gt", ground_truth_img)


            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = pairwise_iou(detection[1:5], gt[1:5])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes)
        precisions = torch.divide(TP_cumsum, (TP_cumsum+FP_cumsum+epsilon))

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

                        
        # for img_idx, detection_list in enumerate(prediction):

        #     detection_end +=len(detection_list)

        #     ground_truth_img = GT[img_idx]
        #     num_gts = len(ground_truth_img)

        #     check_idx = np.zeros([num_gts,1])

        #     for det_idx, detection in enumerate(detections[detection_start:detection_end]):
                
        #         best_iou = 0

        #         for idx, gt in enumerate(ground_truth_img):
        #             iou = pairwise_iou(torch.tensor(detection[0:4], torch.tensor(gt["bbox"])))

        #             if iou > best_iou:
        #                 best_iou = iou
        #                 best_gt_idx = idx

        #         if best_iou > iou_threshold:
        #             if check_idx[best_gt_idx] == 0:
        #                 TP[det_idx+detection_start] = 1
        #                 check_idx[best_gt_idx] = 1
        #             else:
        #                 FP[det_idx+detection_start] = 1

        #         else:
        #             FP[det_idx+detection_start] = 1

        #         TP_cumsum = torch.cumsum(TP, dim=0)
        #         FP_cumsum = torch.cumsum(FP, dim=0)

        #         recalls = TP_cumsum / (total_true_bboxes)
        #         precisions = torch.divide(TP_cumsum, (TP_cumsum+FP_cumsum+epsilon))

        #         precisions = torch.cat((torch.tensor([1]), precisions))
        #         recalls = torch.cat((torch.tensor([0]), recalls))

        #         average_precisions.append(torch.trapz(precisions, recalls))

        #     detection_start = detection_end
            
    return sum(average_precisions) / len(average_precisions)

