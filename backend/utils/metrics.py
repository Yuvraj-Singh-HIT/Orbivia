import torch

def iou_score(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    iou = 0.0

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union != 0:
            iou += intersection / union

    return iou / num_classes
