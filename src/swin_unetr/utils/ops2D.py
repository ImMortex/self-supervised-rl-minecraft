"""
Override by Christion Gurski
Copy of MONAI ops.py for 2D image tensors instead of 3D image tensors accordingly without rotation augmentation
"""

import numpy as np
import torch
from numpy.random import randint



def patch_rand_drop2D(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w
    mx_blk_height = max(int(h * max_block_sz), 1)  # stcngurs bugfix: added max to ensure min value of 1
    mx_blk_width = max(int(w * max_block_sz), 1)  # stcngurs bugfix: added max to ensure min value of 1

    tolr = (int(tolr * h), int(tolr * w))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_h = max(min(randint(tolr[0], mx_blk_height) + rnd_r, h), 1)  # stcngurs bugfix: added max to ensure min value of 1
        rnd_w = max(min(randint(tolr[1], mx_blk_width) + rnd_c, w), 1)  # stcngurs bugfix: added max to ensure min value of 1

        if rnd_h - rnd_r > 0 and rnd_w - rnd_c > 0: # stcngurs bugfix: added condi

            if x_rep is None:
                x_uninitialized = torch.empty(
                    (c, rnd_h - rnd_r, rnd_w - rnd_c), dtype=x.dtype, device=args.local_rank
                ).normal_()
                x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                        torch.max(x_uninitialized) - torch.min(x_uninitialized)
                )
                x[:, rnd_r:rnd_h, rnd_c:rnd_w] = x_uninitialized
            else:
                x[:, rnd_r:rnd_h, rnd_c:rnd_w] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w]
            total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c)
    return x


def aug_rand2D(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop2D(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop2D(args, x_aug[i], x_aug[idx_rnd])
    return x_aug
