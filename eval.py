import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from dice_loss import dice_coeff


def eval_net(net, dataset, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    for i, b in tqdm(enumerate(dataset), total=n_val, desc='Validation round', unit='img'):
        with torch.no_grad():
            img = b[0].astype(np.float32)
            true_mask = b[1].astype(np.float32)

            img = torch.from_numpy(img).unsqueeze(0)
            true_mask = torch.from_numpy(true_mask)

            img = img.to(device=device)
            true_mask = true_mask.to(device=device)

            mask_pred = net(img).squeeze(dim=0)
            
            mask_pred = mask_pred[true_mask!=255]
            true_mask = true_mask[true_mask!=255]

            mask_pred = (mask_pred > 0.5).float()
            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_mask).item()
            else:
                tot += torch.sum(mask_pred==true_mask).item()/mask_pred.size().numel() # 2*torch.sum(mask_pred==true_mask).item()/(mask_pred.size().numel() + true_mask.size().numel()) or dice_coeff(mask_pred, true_mask).item()

    return tot / n_val
