import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# def evaluate_depth(pred, gt):
#     thresh = torch.maximum((gt / pred), (pred / gt))

#     a1 = torch.FloatTensor((thresh < 1.25)).mean()
#     a2 = torch.FloatTensor((thresh < 1.25 ** 2)).mean()
#     a3 = torch.FloatTensor((thresh < 1.25 ** 3)).mean()

#     rmse = (gt - pred) ** 2
#     rmse = torch.sqrt(torch.mean(rmse))

#     rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
#     rmse_log = torch.sqrt(torch.mean(rmse_log))

#     abs_rel = torch.mean(torch.abs(gt - pred) / gt)

#     sq_rel = torch.mean(((gt - pred)**2) / gt)

#     return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate_depth(pred, gt):
    gt = gt.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    # abs_rel = torch.from_numpy(abs_rel)
    # sq_rel = torch.from_numpy(sq_rel)
    # rmse = torch.from_numpy(rmse)
    # rmse_log = torch.from_numpy(rmse_log)
    # a1 = torch.from_numpy(a1)
    # a2 = torch.from_numpy(a2)
    # a3 = torch.from_numpy(a3)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
