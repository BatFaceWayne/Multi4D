#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def build_rotation_4d(l, r):
    l_norm = torch.norm(l, dim=-1, keepdim=True)
    r_norm = torch.norm(r, dim=-1, keepdim=True)

    q_l = l / l_norm
    q_r = r / r_norm

    a, b, c, d = q_l.unbind(-1)
    p, q, r, s = q_r.unbind(-1)

    M_l = torch.stack([a,-b,-c,-d,
                       b, a,-d, c,
                       c, d, a,-b,
                       d,-c, b, a]).view(4,4,-1).permute(2,0,1)
    M_r = torch.stack([ p, q, r, s,
                       -q, p,-s, r,
                       -r, s, p,-q,
                       -s,-r, q, p]).view(4,4,-1).permute(2,0,1)
    A = M_l @ M_r
    A = A.flip(1,2)
    return A

def inverse_sigmoid(x):
    return torch.log(x/(1-x))




def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper



def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def velocity_to_rotation_r(velocity, rotation_l):
    """
    Compute the right-isoclinic quaternion `rotation_r` such that, combined
    with `rotation_l` via build_rotation_4d, the resulting 4D rotation maps
    the canonical time axis (0, 0, 0, 1) onto the spacetime motion direction
    normalize((vx, vy, vz, 1)).

    Args:
        velocity:   (N, 3) tensor — 3D velocity in TR time units.
        rotation_l: (N, 4) tensor of unit quaternions for the left-isoclinic
                    rotation. For TR points transferred from FG, this is the
                    FG's deformed rotation (= the TR's `_rotation`), since
                    build_rotation_4d uses both quaternions at render time.

    Returns:
        (N, 4) tensor — unit quaternion rotation_r.

    Derivation:
        Let q_l = (a, b, c, d), q_r = (p, q, r, s), and let
        target = (tx, ty, tz, tt) = normalize((vx, vy, vz, 1)).
        The constraint R @ (0,0,0,1)^T = target reduces to a linear system
        M @ q_r^T = t^T with M orthogonal under unit q_l, so q_r = M^T @ t.
    """
    N = velocity.shape[0]
    device = velocity.device

    target = torch.cat([velocity, torch.ones((N, 1), device=device)], dim=1)
    target = target / torch.norm(target, dim=1, keepdim=True)
    tx, ty, tz, tt = target.unbind(-1)

    # q_r = M^T @ target.
    ql = rotation_l / torch.norm(rotation_l, dim=-1, keepdim=True)
    a, b, c, d = ql.unbind(-1)
    p =  d * tx + c * ty + b * tz + a * tt
    q =  c * tx - d * ty - a * tz + b * tt
    r = -b * tx - a * ty + d * tz + c * tt
    s = -a * tx + b * ty - c * tz + d * tt
    rotation_r = torch.stack([p, q, r, s], dim=1)

    return rotation_r / torch.norm(rotation_r, dim=1, keepdim=True)
