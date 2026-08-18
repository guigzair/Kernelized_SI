"""
1D toy version of "Generative Modeling via Kernelized Stochastic Interpolants"
(Coeurdoux, Lempereur, Cuvelle-Magar, Mallat, Vanden-Eijnden, 2026).

Feature map phi: R -> R^P is a fixed bank of Gaussian bumps or triangular
"hat" functions centered on a grid. The drift b_t(x) = phi'(x)^T eta_t is
estimated by solving the P x P linear system of Proposition 2.1 (eq. 7) on
a time grid, then samples are drawn with the optimal-diffusion integrator
of eq. (14) / Algorithm 1, which needs no clamping as D*_t -> infinity at t=0.
"""

import os

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

P = 25
CENTERS = np.linspace(-6.0, 6.0, P)
WIDTH = CENTERS[1] - CENTERS[0]


# ----------------------------------------------------------------------
# Target distribution mu on R, known only through samples (a bimodal mix)
# ----------------------------------------------------------------------

def sample_target(n):
    comp = rng.integers(0, 2, size=n)
    means = np.array([-2.0, 2.5])[comp]
    stds = np.array([0.6, 0.9])[comp]
    return rng.normal(means, stds)


def target_density(x):
    def normal_pdf(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return 0.5 * normal_pdf(x, -2.0, 0.6) + 0.5 * normal_pdf(x, 2.5, 0.9)

x_grid = np.linspace(-6, 6, 400)
y = target_density(x_grid)
plt.plot(x_grid, y, "k--", label="true density")
plt.show()

# ----------------------------------------------------------------------
# Feature gradients grad phi(x) in R^P (Section 2.5 asks for grad phi, not
# phi itself, since the drift ansatz is b_hat_t(x) = grad phi(x)^T eta_t)
# ----------------------------------------------------------------------

def phi_grad(x, feature):
    d = x[:, None] - CENTERS[None, :]
    if feature == "gaussian":
        bump = np.exp(-0.5 * (d / WIDTH) ** 2)
        return -d / WIDTH**2 * bump
    if feature == "hat":
        active = np.abs(d) < WIDTH
        return np.where(active, -np.sign(d) / WIDTH, 0.0)
    raise ValueError(feature)


# ----------------------------------------------------------------------
# Linear interpolant schedule: alpha_t = 1-t, beta_t = t (Section 2.1)
# ----------------------------------------------------------------------

def alpha(t):
    return 1.0 - t


def beta(t):
    return t


ALPHA_DOT, BETA_DOT = -1.0, 1.0


def gamma(t):
    return alpha(t) * BETA_DOT - ALPHA_DOT * beta(t)


# ----------------------------------------------------------------------
# Proposition 2.1: fit eta_t on a time grid from the empirical system (eq. 7)
# ----------------------------------------------------------------------

def fit_drift_coefficients(a_samples, K, feature, ridge=1e-6):
    n = len(a_samples)
    z = rng.normal(size=n)
    etas = np.zeros((K, P))
    for k in range(K):
        t = k / K
        I = alpha(t) * z + beta(t) * a_samples
        I_dot = ALPHA_DOT * z + BETA_DOT * a_samples
        G = phi_grad(I, feature)                      # (n, P)
        K_t = (G.T @ G) / n + ridge * np.eye(P)
        r_t = (G.T @ I_dot) / n
        etas[k] = np.linalg.solve(K_t, r_t)
    return etas


# ----------------------------------------------------------------------
# Algorithm 1 / eq. (14): optimal-diffusion integrator, singular at t=0 but
# never divides by beta_t (only by beta_{t+h} > 0), so no clamping needed.
# ----------------------------------------------------------------------

def generate(etas, K, n_samples, feature):
    X = rng.normal(size=n_samples)
    h = 1.0 / K
    for k in range(K):
        t, t_next = k / K, (k + 1) / K
        a_t, b_t, g_t = alpha(t), beta(t), gamma(t)
        a_n, b_n, g_n = alpha(t_next), beta(t_next), gamma(t_next)

        drift = phi_grad(X, feature) @ etas[k]
        ratio = b_t / b_n
        noise_scale = np.sqrt(h * (a_t * b_t * g_t + a_n * b_n * g_n)) / b_n

        X = ratio * X + h * (1.0 + ratio) * drift + noise_scale * rng.normal(size=n_samples)
    return X


if __name__ == "__main__":
    N_TRAIN, K_STEPS, N_GEN = 4000, 200, 8000
    a_samples = sample_target(N_TRAIN)
    x_grid = np.linspace(-6, 6, 400)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, feature in zip(axes, ["gaussian", "hat"]):
        etas = fit_drift_coefficients(a_samples, K_STEPS, feature)
        samples = generate(etas, K_STEPS, N_GEN, feature)

        ax.hist(samples, bins=60, density=True, alpha=0.6, label="generated")
        ax.plot(x_grid, target_density(x_grid), "k--", label="true density")
        ax.set_title(f"{feature} features (P={P})")
        ax.set_xlabel("x")
        ax.legend()
    axes[0].set_ylabel("density")

    fig.suptitle("Kernelized stochastic interpolant, 1D toy example")
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_si_1d.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
