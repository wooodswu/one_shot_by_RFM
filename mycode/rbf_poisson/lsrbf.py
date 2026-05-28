import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

# =============================================
# 1. 计算填充距离 (fill distance)
# =============================================
def fill_distance(node):
    N = node.shape[0]
    h_list = np.zeros(N)
    for i in range(N):
        point = node[i, :]
        c = np.delete(node, i, axis=0)
        h_list[i] = np.min(np.sqrt(np.sum((point - c)**2, axis=1)))
    h = np.max(h_list)
    return h

# =============================================
# 2. 星形区域节点生成 + Patch 覆盖
# =============================================
def generator_points(N_patch_grid, N_local_per_patch, N_boundary, R_ptch):
    
    # 星形区域极径函数
    r_max_func = lambda th: 2 * (0.7 + 0.12 * (np.sin(6 * th) + np.sin(3 * th)))

    # ----- 生成真实区域边界 -----
    theta_plot = np.linspace(0, 2*np.pi, 1000)
    r_plot = r_max_func(theta_plot)
    x_plot = r_plot * np.cos(theta_plot)
    y_plot = r_plot * np.sin(theta_plot)

    x_min, x_max = x_plot.min(), x_plot.max()
    y_min, y_max = y_plot.min(), y_plot.max()


    # 判断点是否在星形域内
    def isInsideStar(x, y):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2*np.pi
        r_limit = r_max_func(theta)
        return r <= r_limit + 1e-6

    # ---------------------------
    # Patch 覆盖生成
    # ---------------------------
    Domain_size = max(x_max - x_min, y_max - y_min)
    del_val = (N_patch_grid - Domain_size / R_ptch) / (N_patch_grid - 1)
    patch_spacing = 2 * R_ptch * (1 - del_val)

    Grid_min = min(x_min, y_min)
    Grid_max = max(x_max, y_max)
    xs = np.linspace(Grid_min, Grid_max, N_patch_grid)
    ys = np.linspace(Grid_min, Grid_max, N_patch_grid)

    Cx, Cy = np.meshgrid(xs, ys)
    C_all = np.vstack([Cx.flatten(), Cy.flatten()]).T
    R_all = R_ptch * np.ones((C_all.shape[0], 1))

    # 判断 Patch 是否与星形边界有交集
    all_boundary_theta = theta_plot
    bc_x = Cx.flatten()[:, None] + R_ptch * np.cos(all_boundary_theta)
    bc_y = Cy.flatten()[:, None] + R_ptch * np.sin(all_boundary_theta)
    inside_flag = np.sum(isInsideStar(bc_x, bc_y), axis=1) > 0

    ptch_C = C_all[inside_flag]
    ptch_R = R_all[inside_flag].flatten()
    P = ptch_C.shape[0]

    print("保留 Patch 数量 P =", P)

    # ---------------------------
    # 局部 RBF 中心点生成
    # ---------------------------
    X_inner_list = []
    N_generate = int(1.5 * N_local_per_patch)

    for i in range(P):
        C = ptch_C[i]
        R = ptch_R[i]

        # 均匀极坐标采样
        theta_local = 2 * np.pi * np.random.rand(N_generate)
        r_local = np.sqrt(np.random.rand(N_generate)) * R

        x_local = r_local * np.cos(theta_local) + C[0]
        y_local = r_local * np.sin(theta_local) + C[1]

        inside = isInsideStar(x_local, y_local)
        Xcand = np.vstack([x_local[inside], y_local[inside]]).T

        if Xcand.shape[0] > N_local_per_patch:
            Xcand = Xcand[:N_local_per_patch]

        X_inner_list.append(Xcand)

    X_inner = np.vstack(X_inner_list)
    print("生成内部点数量 =", X_inner.shape[0])

    # ---------------------------
    # 生成边界点
    # ---------------------------
    theta_b = np.linspace(0, 2*np.pi, N_boundary, endpoint=False)
    r_b = r_max_func(theta_b)
    X_boundary = np.vstack([r_b * np.cos(theta_b), r_b * np.sin(theta_b)]).T

    print("边界点数量 =", X_boundary.shape[0])

    ptch = {"C": ptch_C, "R": ptch_R}

    return X_inner, X_boundary, ptch


# =============================================
# 3. Wendland C2 权函数
# =============================================
def wendland_C2(r):
    mask = (r <= 1)
    phi = np.zeros_like(r)
    phi[mask] = (4*r[mask] + 1) * (1 - r[mask])**4
    return phi


# =============================================
# 4. Shepard 权重
# =============================================
def shepard_weights(x, ptch_C, ptch_R):
    r_dist = cdist(x, ptch_C)
    r_scaled = r_dist / ptch_R[None, :]
    Varphi = wendland_C2(r_scaled)
    S = Varphi.sum(axis=1, keepdims=True)
    S[S == 0] = 1e-12
    return Varphi / S


# =============================================
# 5. 权函数导数（dx, dy, laplace）
# =============================================
def diff_weights(x, ptch_C, ptch_R):
    r_dist = cdist(x, ptch_C)
    r_scaled = r_dist / ptch_R[None, :]

    Varphi = wendland_C2(r_scaled)
    S = Varphi.sum(axis=1, keepdims=True)
    S[S == 0] = 1e-12

    mask = (r_scaled <= 1)

    dx = x[:, 0:1] - ptch_C[:, 0].reshape(1, -1)
    dy = x[:, 1:1+1] - ptch_C[:, 1].reshape(1, -1)

    dx_varphi = -20 / (ptch_R**2) * (1 - r_scaled)**3 * dx * mask
    dy_varphi = -20 / (ptch_R**2) * (1 - r_scaled)**3 * dy * mask
    d2_varphi = (1 - r_scaled)**2 * (20 / (ptch_R**2)) * (5*r_scaled - 2) * mask

    sum_dx = dx_varphi.sum(axis=1, keepdims=True)
    sum_dy = dy_varphi.sum(axis=1, keepdims=True)
    sum_d2 = d2_varphi.sum(axis=1, keepdims=True)

    dx_w = (S*dx_varphi - Varphi*sum_dx) / (S**2)
    dy_w = (S*dy_varphi - Varphi*sum_dy) / (S**2)
    d2_w = (S*d2_varphi - Varphi*sum_d2) / (S**2) - (2/S)*(dx_w*sum_dx + dy_w*sum_dy)

    return dx_w, dy_w, d2_w


# =============================================
# 6. Gaussian RBF 及其导数
# =============================================
def phi_gauss(eps, x, xs):
    r2 = cdist(x, xs)**2
    return np.exp(-eps**2 * r2)

def dx1_phi_gauss(eps, x, xs):
    Phi = phi_gauss(eps, x, xs)
    diff = x[:, 0:1] - xs[:, 0].reshape(1, -1)
    return Phi * (-2*eps**2 * diff)

def dx2_phi_gauss(eps, x, xs):
    Phi = phi_gauss(eps, x, xs)
    diff = x[:, 1:1+1] - xs[:, 1].reshape(1, -1)
    return Phi * (-2*eps**2 * diff)

def laplace_phi_gauss(eps, x, xs):
    r2 = cdist(x, xs)**2
    Phi = phi_gauss(eps, x, xs)
    return Phi * (4*eps**4 * r2 - 4*eps**2)


# =============================================
# 7. RBF–PU 求解 Poisson
# =============================================
def solve_poisson_lsrbf(ptch, eps, X, Yin, Yb, f_func, g_func):
    patch_C = ptch["C"]
    patch_R = ptch["R"]
    P = patch_C.shape[0]

    M1 = Yin.shape[0]
    M2 = Yb.shape[0]
    M = M1 + M2
    N = X.shape[0]

    Y = np.vstack([Yin, Yb])
    RHS = np.hstack([f_func(Yin[:, 0], Yin[:, 1]),
                     g_func(Yb[:, 0], Yb[:, 1])])

    # 全局权函数
    W = shepard_weights(Y, patch_C, patch_R)
    dx_w, dy_w, d2_w = diff_weights(Yin, patch_C, patch_R)

    L = np.zeros((M, N))

    # -----------------------------
    # 组装 L
    # -----------------------------
    for j in range(P):
        idx_X = np.where(cdist(X, patch_C[j:j+1])[:, 0] <= patch_R[j])[0]
        idx_Yi = np.where(cdist(Yin, patch_C[j:j+1])[:, 0] <= patch_R[j])[0]
        idx_Yb = np.where(cdist(Yb,  patch_C[j:j+1])[:, 0] <= patch_R[j])[0]

        Xj = X[idx_X]
        Yj_in = Yin[idx_Yi]
        Yj_b  = Yb[idx_Yb]

        # interior
        Phi = phi_gauss(eps, Yj_in, Xj)
        dx_Phi = dx1_phi_gauss(eps, Yj_in, Xj)
        dy_Phi = dx2_phi_gauss(eps, Yj_in, Xj)
        lap_Phi = laplace_phi_gauss(eps, Yj_in, Xj)

        wj = W[idx_Yi, j][:, None]
        dwjx = dx_w[idx_Yi, j][:, None]
        dwjy = dy_w[idx_Yi, j][:, None]
        dwj2 = d2_w[idx_Yi, j][:, None]

        Lj = dwj2 * Phi + 2*(dx_Phi * dwjx + dy_Phi * dwjy) + wj * lap_Phi

        Phi_jj = phi_gauss(eps, Xj, Xj)
        # cond check
        reg = 0
        if np.linalg.cond(Phi_jj) > 1e8:
            reg = 1e-8

        R = cho_factor(Phi_jj + reg*np.eye(Phi_jj.shape[0]))
        Cj = cho_solve(R, Lj.T).T

        L[idx_Yi[:, None], idx_X] -= Cj

        # boundary
        Phi_bc = phi_gauss(eps, Yj_b, Xj)
        Cj_bc = cho_solve(R, (W[M1 + idx_Yb, j][:, None] * Phi_bc).T).T

        L[M1 + idx_Yb[:, None], idx_X] += Cj_bc

    # Solve
    UjXj = np.linalg.lstsq(L, RHS, rcond=None)[0] # local solution on X

    # -----------------------------
    # 预测 U(X)
    # -----------------------------
    W_X = shepard_weights(X, patch_C, patch_R); # M \times P
    UX = np.zeros(N)
    for j in range(P):
        idx = np.where(cdist(X, patch_C[j:j+1])[:, 0] <= patch_R[j])[0]
        Xj = X[idx]
        Phi_jj = phi_gauss(eps, Xj, Xj)
        R = cho_factor(Phi_jj + 1e-8*np.eye(Phi_jj.shape[0]))
        Cj = cho_solve(R, UjXj[idx])

        Phi_XXj = phi_gauss(eps, X, Xj)
        UX += W_X[:, j] * (Phi_XXj @ Cj)

    # -----------------------------
    # 预测 U(Y)
    # -----------------------------
    U_Y = np.zeros(M)
    for j in range(P):
        idx = np.where(cdist(X, patch_C[j:j+1])[:, 0] <= patch_R[j])[0]
        Xj = X[idx]
        Phi_jj = phi_gauss(eps, Xj, Xj)
        R = cho_factor(Phi_jj + 1e-8*np.eye(Phi_jj.shape[0]))
        Cj = cho_solve(R, UjXj[idx])

        Phi_YXj = phi_gauss(eps, Y, Xj)
        U_Y += W[:, j] * (Phi_YXj @ Cj)


    return UX, U_Y



u = lambda x,y: np.sin(x)*np.cos(y)
f = lambda x,y: 2*np.sin(x)*np.cos(y)
g = lambda x,y: np.sin(x)*np.cos(y)


np.random.seed(123)
N_patch_grid = 5
N_local_per_patch = 100
N_boundary = 100
R_ptch = 0.8

X_inner, X_boundary, ptch = generator_points(
    N_patch_grid, N_local_per_patch, N_boundary, R_ptch
)

# 另一套点作为 Yin, Yb
X_inner1, X_boundary1, _ = generator_points(
    N_patch_grid, N_local_per_patch, N_boundary, R_ptch
)

X = np.vstack([X_inner, X_boundary])
Yin = X_inner1
Yb  = X_boundary1
Y = np.vstack([Yin, Yb])

h = fill_distance(X)
rho = 40 * h
eps = np.sqrt(1/rho)

UX, UY = solve_poisson_lsrbf(ptch, eps, X, Yin, Yb, f, g)


u_x_true = u(X[:,0], X[:,1])
u_y_true = u(Y[:,0], Y[:,1])

error_X = np.max(np.abs(UX - u_x_true))
error_Y = np.max(np.abs(UY - u_y_true))

print(f"Max Error at Nodes X: {error_X:.5e}")
print(f"Max Error at Eval Y: {error_Y:.5e}")


# =========================================================================
# 结果可视化
# =========================================================================
fig = plt.figure(figsize=(15, 5))

# Subplot 1: Prediction
ax1 = fig.add_subplot(131, projection='3d')
sc1 = ax1.scatter(Y[:, 0], Y[:, 1], UY.flatten(), c=UY.flatten(), cmap='turbo', s=10)
ax1.set_title('Predicted $u_{pred}$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')
plt.colorbar(sc1, ax=ax1, shrink=0.5, label='u')

# Subplot 2: True Solution
ax2 = fig.add_subplot(132, projection='3d')
sc2 = ax2.scatter(Y[:, 0], Y[:, 1], u_y_true.flatten(), c=u_y_true.flatten(), cmap='turbo', s=10)
ax2.set_title('True $u_{true}$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u')
plt.colorbar(sc2, ax=ax2, shrink=0.5, label='u')

# Subplot 3: Error
error_vals = np.abs(u_y_true - UY).flatten()
ax3 = fig.add_subplot(133, projection='3d')
sc3 = ax3.scatter(Y[:, 0], Y[:, 1], error_vals, c=error_vals, cmap='turbo', s=10)
ax3.set_title('Error $|u_{pred} - u_{true}|$')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Error')
plt.colorbar(sc3, ax=ax3, shrink=0.5, label='Error')

plt.suptitle('2D RBF-PU Poisson Equation Solver')
plt.tight_layout()
plt.show()