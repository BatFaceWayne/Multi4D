
# Utility functions extracted from TRASE repository
# https://github.com/yunjinli/TRASE

import torch

# --- feature_utils.py ---

@torch.no_grad()
def get_sample_pixel_and_mask(sam_masks, num_sampled_pixels, num_sampled_masks):
    mask_sample_rate = num_sampled_masks / (sam_masks.shape[0])
    sampled_mask = torch.rand(sam_masks.shape[0]).cuda() < mask_sample_rate
    pixel_sample_rate = num_sampled_pixels / (sam_masks.shape[-1] * sam_masks.shape[-2])
    sampled_pixel = torch.rand(sam_masks.shape[-2], sam_masks.shape[-1]).cuda() < pixel_sample_rate
    non_mask_region = sam_masks.sum(dim = 0) == 0
    sampled_pixel = torch.logical_and(sampled_pixel, ~non_mask_region)

    return sampled_pixel, sampled_mask

@torch.no_grad()
def get_pixel_weights(sam_masks, sampled_pixel):
    per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:, None, None]
    per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim = 0) / (sam_masks.sum(dim = 0) + 1e-9)
    per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_pixel]
    pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1)
    ptp_max_size = pixel_to_pixel_mask_size.max()
    pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
    per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)
    per_pixel_weight = (per_pixel_weight - per_pixel_weight.min()) / (per_pixel_weight.max() - per_pixel_weight.min()) * 9. + 1.
    return per_pixel_weight

@torch.no_grad()
def get_pixel_mask_correspondence_matrix(sam_masks, sampled_pixel, sampled_mask):
    sam_masks_sampled_pixel = sam_masks[:, sampled_pixel]
    ## Calculate the pixel-mask correspondence vector based on SAM masks
    pixel_mask_correspondence_vector = sam_masks_sampled_pixel[sampled_mask, :]
    mask_corr_matrix = torch.einsum('nh,nj->hj', pixel_mask_correspondence_vector.float(), pixel_mask_correspondence_vector.float())
    mask_corr_matrix[mask_corr_matrix != 0] = 1

    return mask_corr_matrix

def get_features_correspondence_matrix(rendered_features, sampled_pixel):
    sampled_rendered_features = rendered_features[:, sampled_pixel]
    sampled_rendered_features = sampled_rendered_features.permute([1, 0])
    sampled_rendered_features = torch.nn.functional.normalize(sampled_rendered_features, dim=-1, p=2)
    feature_corr_matrix = torch.einsum('hc,jc->hj', sampled_rendered_features, sampled_rendered_features)

    return feature_corr_matrix

# --- loss_utils.py (Contrastive Part) ---
# Multi4D paper (Sec 3.8) L_sem = L_pos + L_neg maps onto these TRASE-derived
# helpers: the positive_pixel_pair_loss[...] family is the paper's L_pos, and
# negative_pixel_pair_loss[...] is L_neg. Names are kept as in TRASE for
# provenance; the paper's soft-mined objective corresponds to the 'soft' mode.

def pixel_mask_correspondence_loss_positive(C, C_F, weights=None):
    # 'all' mode: every positive pair contributes; no threshold by design.
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)

    positive_mask = torch.any(C == 1, dim = 0)
    positive_mask = torch.logical_and(positive_mask, ~diag_mask)
    positive_mask = torch.triu(positive_mask, diagonal=0) ## set the symmetric part to false
    number_of_all_pixel_pair = torch.nonzero(positive_mask).shape[0]
    positive_mask = torch.logical_and(positive_mask, C == 1)

    positive_mask = positive_mask.bool()

    if weights is not None:
        return (-weights[positive_mask]* C_F[positive_mask]).sum() / number_of_all_pixel_pair
    else:
        return (-C_F[positive_mask]).sum() / number_of_all_pixel_pair

def pixel_mask_correspondence_loss_negative(C, C_F, weights=None):
    # 'all' mode: every negative pair contributes; no threshold by design.
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)
    negative_mask = torch.any(C == 0, dim = 0)
    negative_mask = torch.logical_and(negative_mask, ~diag_mask)
    negative_mask = torch.triu(negative_mask, diagonal=0) ## set the symmetric part to false
    number_of_all_pixel_pair = torch.nonzero(negative_mask).shape[0]
    negative_mask = torch.logical_and(negative_mask, C == 0)
    negative_mask = negative_mask.bool()

    if weights is not None:
        return (weights[negative_mask] * torch.relu(C_F[negative_mask])).sum() / number_of_all_pixel_pair
    else:
        return (torch.relu(C_F[negative_mask])).sum() / number_of_all_pixel_pair

def pixel_mask_correspondence_loss_soft_hard_positive(C, C_F, positive_th=0.75, weights=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)
    soft_hard_positive_mask = torch.any(torch.logical_and(C_F < positive_th, C == 1), dim = 0)
    soft_hard_positive_mask = torch.logical_and(soft_hard_positive_mask, ~diag_mask)
    soft_hard_positive_mask = torch.triu(soft_hard_positive_mask, diagonal=0) ## set the symmetric part to false

    number_of_all_pixel_pair = torch.nonzero(soft_hard_positive_mask).shape[0]
    soft_hard_positive_mask = torch.logical_and(soft_hard_positive_mask, C == 1)

    soft_hard_positive_mask = soft_hard_positive_mask.bool()

    if soft_hard_positive_mask.sum() == 0: ## No positvie sample found
        return torch.tensor(0.0, device=C_F.device)
    else:
        if weights is not None:
            loss = (-weights[soft_hard_positive_mask] * C_F[soft_hard_positive_mask]).sum() / number_of_all_pixel_pair

        else:
            loss = (-C_F[soft_hard_positive_mask]).sum() / number_of_all_pixel_pair

        return loss

def pixel_mask_correspondence_loss_soft_negative(C, C_F, negative_th=0.5, weights=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)
    soft_hard_negative_mask = torch.any(torch.logical_and(C_F > negative_th, C == 0), dim = 0)

    soft_hard_negative_mask = torch.logical_and(soft_hard_negative_mask, ~diag_mask)

    soft_hard_negative_mask = torch.triu(soft_hard_negative_mask, diagonal=0) ## set the symmetric part to false

    number_of_all_pixel_pair = torch.nonzero(soft_hard_negative_mask).shape[0]

    soft_hard_negative_mask = torch.logical_and(soft_hard_negative_mask, C == 0)
    soft_hard_negative_mask = soft_hard_negative_mask.bool()

    if soft_hard_negative_mask.sum() == 0:
        return torch.tensor(0.0, device=C_F.device)
    else:
        if weights is not None:
            loss = (weights[soft_hard_negative_mask] * torch.relu(C_F[soft_hard_negative_mask])).sum() / number_of_all_pixel_pair
        else:
            loss = (torch.relu(C_F[soft_hard_negative_mask])).sum() / number_of_all_pixel_pair

        return loss

def pixel_mask_correspondence_loss_hard_positive(C, C_F, positive_th=0.75, weights=None):
    diag_mask = torch.eye(C.shape[0], dtype=bool, device=C_F.device)

    # Find hard positive indices (i, j)
    hard_positive_mask = torch.triu((C_F < positive_th) & (C == 1) & (~diag_mask), diagonal=0)
    hard_positive_indices = torch.nonzero(hard_positive_mask, as_tuple=False)

    if hard_positive_indices.shape[0] == 0:
        return torch.tensor(0.0, device=C_F.device)

    i, j = hard_positive_indices[:, 0], hard_positive_indices[:, 1]

    # Compute loss
    C_F_hard = C_F[i, j]  # Use indexed values instead of masked tensor

    if weights is not None:
        loss = (-weights[i, j] * C_F_hard).mean()
    else:
        loss = (-C_F_hard).mean()

    return loss

def pixel_mask_correspondence_loss_hard_negative(C, C_F, negative_th=0.5, weights=None):
    diag_mask = torch.eye(C.shape[0], dtype=bool, device=C_F.device)

    # Find hard negative indices (i, j)
    hard_negative_mask = torch.triu((C_F > negative_th) & (C == 0) & (~diag_mask), diagonal=0)
    hard_negative_indices = torch.nonzero(hard_negative_mask, as_tuple=False)
    if hard_negative_indices.shape[0] == 0:
        return torch.tensor(0.0, device=C_F.device)

    i, j = hard_negative_indices[:, 0], hard_negative_indices[:, 1]

    # Compute loss
    C_F_hard = C_F[i, j]

    if weights is not None:
        loss = (weights[i, j] * torch.relu(C_F_hard)).mean()
    else:
        loss = (torch.relu(C_F_hard)).mean()

    return loss

positive_pixel_pair_loss = {
    'hard': pixel_mask_correspondence_loss_hard_positive,
    'all': pixel_mask_correspondence_loss_positive,
    'soft': pixel_mask_correspondence_loss_soft_hard_positive
}

negative_pixel_pair_loss = {
    'hard': pixel_mask_correspondence_loss_hard_negative,
    'all': pixel_mask_correspondence_loss_negative,
    'soft': pixel_mask_correspondence_loss_soft_negative
}
