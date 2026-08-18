"""
Kernelized Schrodinger Bridge between two arbitrary distributions.

Combines:
  - Iterative Markovian Fitting (IMF / DSBM) to obtain the Schrodinger bridge,
  - the kernel trick of Coeurdoux et al. (arXiv:2602.20070): the drift regression
    at each time step is a P x P linear system, NOT a neural network.

Two demos:
  1D : N(0,1)  <->  Uniform(-3,3)          ("gaussian" <-> "square")
  2D : 2 Gaussian blobs <-> 2 uniform squares
"""

import numpy as np

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------
# 1. Feature maps  psi : R^d -> R^{P x d}   (a basis of vector fields)
# ----------------------------------------------------------------------
class RBFFeatures:
    """Scalar RBFs on a grid; used componentwise as a vector-field basis.

    We regress the *denoiser* E[a_1 | X_t = x] (a R^d-valued function), so we
    need a basis of R^d-valued functions. We take
        psi_{i,c}(x) = rbf_i(x) e_c ,   plus affine terms  {e_c, x_c e_c ...}
    which makes the Gram matrix block-structured; we simply build the design
    matrix Phi(x) in R^{N x P} of scalar features and solve d independent
    ridge regressions sharing the same Gram matrix K_t.
    """

    def __init__(self, centers, width):
        self.centers = centers          # (M, d)
        self.width = width
        self.d = centers.shape[1]
        self.P = centers.shape[0] + 1 + self.d

    def __call__(self, x):              # x : (N, d) -> (N, P)
        d2 = ((x[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        g = np.exp(-0.5 * d2 / self.width ** 2)
        return np.concatenate([g, np.ones((len(x), 1)), x], axis=1)


# ----------------------------------------------------------------------
# 2. One Markovian projection = one ridge solve per time step
# ----------------------------------------------------------------------
def fit_denoiser(pairs, feats, ts, eps, ridge=1e-6):
    """pairs = (a0, a1) coupling, shape (N,d) each.

    For each t in ts, build the Brownian-bridge interpolant
        I_t = (1-t) a0 + t a1 + sqrt(eps t(1-t)) z
    and solve the P x P system
        K_t eta_t = r_t ,  K_t = E[psi(I_t) psi(I_t)^T],  r_t = E[psi(I_t) a1^T]
    so that  ahat_1(x) = psi(x)^T eta_t  ~=  E[a1 | I_t = x].
    """
    a0, a1 = pairs
    N, d = a0.shape
    etas = []
    for t in ts:
        z = rng.standard_normal((N, d))
        It = (1 - t) * a0 + t * a1 + np.sqrt(eps * t * (1 - t)) * z
        Phi = feats(It)                                   # (N, P)
        K = Phi.T @ Phi / N + ridge * np.eye(feats.P)     # (P, P)
        r = Phi.T @ a1 / N                                # (P, d)
        etas.append(np.linalg.solve(K, r))
    return etas


# ----------------------------------------------------------------------
# 3. Exact Brownian-bridge integrator (no blow-up at t -> 1)
# ----------------------------------------------------------------------
def simulate(x0, feats, etas, ts, eps):
    """dX = (ahat_1(X) - X)/(1-t) dt + sqrt(eps) dW, integrated with the exact
    bridge transition kernel given the frozen endpoint estimate ahat_1(X_t):
        X_{t+h} ~ N( X_t + (h/(1-t))(y - X_t),  eps*h*(1-t-h)/(1-t) )
    The variance vanishes as t+h -> 1, so the last step lands on ahat_1 exactly.
    """
    x = x0.copy()
    for k, t in enumerate(ts):
        h = (ts[k + 1] - t) if k + 1 < len(ts) else (1.0 - t)
        y = feats(x) @ etas[k]
        frac = h / (1 - t)
        var = eps * h * (1 - t - h) / (1 - t)
        x = x + frac * (y - x)
        if var > 0:
            x = x + np.sqrt(var) * rng.standard_normal(x.shape)
    return x


# ----------------------------------------------------------------------
# 4. IMF outer loop
# ----------------------------------------------------------------------
def imf(s0, s1, feats, eps, n_steps=100, n_iter=8, verbose=True, metric=None):
    ts = np.linspace(0, 1, n_steps, endpoint=False)
    pairs = (s0, s1)                       # start from the independent coupling
    fwd = bwd = None
    for it in range(n_iter):
        # --- forward Markovian projection, then resample the coupling
        fwd = fit_denoiser(pairs, feats, ts, eps)
        x1 = simulate(s0, feats, fwd, ts, eps)
        pairs = (s0, x1)
        # --- backward Markovian projection (roles swapped), resample
        bwd = fit_denoiser((pairs[1], pairs[0]), feats, ts, eps)
        x0 = simulate(s1, feats, bwd, ts, eps)
        pairs = (x0, s1)
        if verbose and metric is not None:
            print(f"  iter {it+1:2d}   W1(fwd(mu0), mu1) = {metric(x1, s1):.4f}")
    return fwd, bwd, ts


def w1_1d(x, y):
    return np.abs(np.sort(x.ravel()) - np.sort(y.ravel())).mean()


# ----------------------------------------------------------------------
# DEMO 1 : 1D  N(0,1) <-> Uniform(-3,3)
# ----------------------------------------------------------------------
def demo_1d():
    N, eps = 20000, 0.25
    s0 = rng.standard_normal((N, 1))
    s1 = rng.uniform(-3, 3, (N, 1))

    centers = np.linspace(-6, 6, 48)[:, None]
    feats = RBFFeatures(centers, width=12 / 47 * 1.5)
    print(f"1D demo:  P = {feats.P} features, eps = {eps}")

    fwd, bwd, ts = imf(s0, s1, feats, eps, n_steps=100, n_iter=6, metric=w1_1d)

    gen1 = simulate(s0, feats, fwd, ts, eps)
    gen0 = simulate(s1, feats, bwd, ts, eps)
    print(f"  final  W1(gen -> uniform) = {w1_1d(gen1, s1):.4f}")
    print(f"  final  W1(gen -> normal ) = {w1_1d(gen0, s0):.4f}")

    # sanity check: for small eps the SB map should approach the OT map,
    # which in 1D is the CDF-matching map F1^{-1} o F0.
    from scipy.stats import norm
    xs = np.linspace(-2.5, 2.5, 11)[:, None]
    ot = 6 * norm.cdf(xs) - 3
    traj = xs.copy()
    for k, t in enumerate(ts):                    # noiseless (ODE-like) push
        h = (ts[k + 1] - t) if k + 1 < len(ts) else (1.0 - t)
        traj = traj + (h / (1 - t)) * (feats(traj) @ fwd[k] - traj)
    print("  mean |learned map - exact OT map| on [-2.5,2.5]:",
          f"{np.abs(traj - ot).mean():.4f}")
    return s0, s1, gen1, gen0, fwd, feats, ts


# ----------------------------------------------------------------------
# DEMO 2 : 2D  two gaussian blobs <-> two uniform squares
# ----------------------------------------------------------------------
def demo_2d():
    N, eps = 5000, 0.15
    c = rng.integers(0, 2, N)
    mu = np.where(c[:, None] == 0, np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    s0 = mu + 0.4 * rng.standard_normal((N, 2))

    c = rng.integers(0, 2, N)
    ctr = np.where(c[:, None] == 0, np.array([-2.0, 2.0]), np.array([2.0, -2.0]))
    s1 = ctr + rng.uniform(-1.0, 1.0, (N, 2))

    g = np.linspace(-4.5, 4.5, 15)
    centers = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    feats = RBFFeatures(centers, width=(g[1] - g[0]) * 1.3)
    print(f"\n2D demo:  P = {feats.P} features, eps = {eps}")

    fwd, bwd, ts = imf(s0, s1, feats, eps, n_steps=80, n_iter=6, verbose=False)
    gen1 = simulate(s0, feats, fwd, ts, eps)
    gen0 = simulate(s1, feats, bwd, ts, eps)

    # crude marginal check
    for name, a, b in [("mu1", gen1, s1), ("mu0", gen0, s0)]:
        print(f"  {name}: mean {a.mean(0).round(3)} vs {b.mean(0).round(3)} | "
              f"std {a.std(0).round(3)} vs {b.std(0).round(3)}")

    # intermediate marginals of the bridge
    snaps = {}
    x = s0.copy()
    for k, t in enumerate(ts):
        h = (ts[k + 1] - t) if k + 1 < len(ts) else (1.0 - t)
        y = feats(x) @ fwd[k]
        var = eps * h * (1 - t - h) / (1 - t)
        x = x + (h / (1 - t)) * (y - x) + np.sqrt(max(var, 0)) * rng.standard_normal(x.shape)
        if k in (0, 20, 40, 60, 79):
            snaps[round(t + h, 2)] = x.copy()
    return s0, s1, gen1, snaps


if __name__ == "__main__":
    demo_1d()
    demo_2d()