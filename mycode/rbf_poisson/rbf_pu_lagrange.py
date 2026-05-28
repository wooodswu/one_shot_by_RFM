import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-12
# -------------------------
# generate patch points
# -------------------------



# -------------------------
# Helpers: geometry / h
# -------------------------
def compute_fill_distance(X: torch.Tensor) -> float:
    """
    X: (N, d) CPU or GPU tensor
    returns scalar h = max_i min_{j != i} ||x_i - x_j||
    """
    with torch.no_grad():
        if X.shape[0] < 2:
            return 0.0
        # pairwise distances
        d2 = torch.cdist(X, X, p=2)  # (N,N)
        n = d2.shape[0]
        mask = torch.eye(n, device=X.device, dtype=torch.bool)
        d2.masked_fill_(mask, float('inf'))
        min_vals, _ = d2.min(dim=1)   # (N,)
        h = float(min_vals.max().cpu().item())
        return h

# -------------------------
# Wendland C2 and Shepard weights (torch)
# -------------------------
def wendland_C2(r: torch.Tensor) -> torch.Tensor:
    """
    r: scaled radial distances (any shape)
    varphi = (4*r + 1)*(1-r)^4 for r<=1 else 0
    """
    mask = (r <= 1.0).to(r.dtype)
    t = r * mask
    return mask * ((4.0 * t + 1.0) * (1.0 - t)**4)

def shepard_weights_torch(Y: torch.Tensor, patch_centers: torch.Tensor, patch_radii: torch.Tensor) -> torch.Tensor:
    """
    Y: (M, d)
    patch_centers: (P, d)
    patch_radii: (P,) or (P,1)
    returns W: (M, P)
    """
    r_dist = torch.cdist(Y, patch_centers, p=2)  # (M,P)
    pr = patch_radii.view(1, -1)
    r_scaled = r_dist / (pr + _EPS)
    Varphi = wendland_C2(r_scaled)
    S = Varphi.sum(dim=1, keepdim=True)  # (M,1)
    S = torch.where(S == 0.0, torch.full_like(S, _EPS), S)
    return Varphi / S  # (M,P)

def diff_weights_torch(Yin: torch.Tensor, patch_centers: torch.Tensor, patch_radii: torch.Tensor):
    """
    Yin: (M1, 2)  -- interior evaluation points (2D)
    patch_centers: (P,2)
    patch_radii: (P,)
    returns dx_w, dy_w, d2_w each (M1, P)
    """
    M1 = Yin.shape[0]
    P = patch_centers.shape[0]
    r_dist = torch.cdist(Yin, patch_centers, p=2)  # (M1,P)
    pr = patch_radii.view(1, -1)
    r_scaled = r_dist / (pr + _EPS)
    Varphi = wendland_C2(r_scaled)  # (M1,P)
    S = Varphi.sum(dim=1, keepdim=True)
    S = torch.where(S == 0.0, torch.full_like(S, _EPS), S)
    mask = (r_scaled <= 1.0).to(r_scaled.dtype)

    dx = Yin[:, 0].unsqueeze(1) - patch_centers[:, 0].unsqueeze(0)  # (M1,P)
    dy = Yin[:, 1].unsqueeze(1) - patch_centers[:, 1].unsqueeze(0)

    invR2 = 1.0 / (patch_radii**2 + _EPS)  # (P,)
    dx_varphi = (-20.0) * (1.0 - r_scaled)**3 * (dx * mask) * invR2.unsqueeze(0)
    dy_varphi = (-20.0) * (1.0 - r_scaled)**3 * (dy * mask) * invR2.unsqueeze(0)
    d2_varphi  = (1.0 - r_scaled)**2 * (20.0 * (5.0 * r_scaled - 2.0) * invR2.unsqueeze(0)) * mask

    sum_dx = dx_varphi.sum(dim=1, keepdim=True)
    sum_dy = dy_varphi.sum(dim=1, keepdim=True)
    sum_d2 = d2_varphi.sum(dim=1, keepdim=True)

    dx_w = (S * dx_varphi - Varphi * sum_dx) / (S**2 + _EPS)
    dy_w = (S * dy_varphi - Varphi * sum_dy) / (S**2 + _EPS)

    d2_w = (S * d2_varphi - Varphi * sum_d2) / (S**2 + _EPS) \
           - 2.0 / (S + _EPS) * (dx_w * sum_dx + dy_w * sum_dy)

    return dx_w, dy_w, d2_w  # each (M1,P)

# -------------------------
# Gaussian phi and derivatives (torch)
# -------------------------
def phi_gauss_torch(eps: torch.Tensor, X: torch.Tensor, Xs: torch.Tensor) -> torch.Tensor:
    """
    eps: scalar tensor or python float
    X: (M, d)
    Xs: (n_s, d)
    returns: (M, n_s)
    """
    r2 = torch.cdist(X, Xs, p=2)**2
    return torch.exp(-(eps**2) * r2)

def laplace_phi_gauss_torch(eps: torch.Tensor, X: torch.Tensor, Xs: torch.Tensor) -> torch.Tensor:
    r2 = torch.cdist(X, Xs, p=2)**2
    Phi = torch.exp(-(eps**2) * r2)
    return Phi * (4.0 * (eps**4) * r2 - 4.0 * (eps**2))

def dx1_phi_gauss_torch(eps: torch.Tensor, X: torch.Tensor, Xs: torch.Tensor) -> torch.Tensor:
    Phi = phi_gauss_torch(eps, X, Xs)
    diff = X[:, 0].unsqueeze(1) - Xs[:, 0].unsqueeze(0)  # (M, n_s)
    return Phi * (-2.0 * (eps**2) * diff)

def dx2_phi_gauss_torch(eps: torch.Tensor, X: torch.Tensor, Xs: torch.Tensor) -> torch.Tensor:
    Phi = phi_gauss_torch(eps, X, Xs)
    diff = X[:, 1].unsqueeze(1) - Xs[:, 1].unsqueeze(0)
    return Phi * (-2.0 * (eps**2) * diff)


