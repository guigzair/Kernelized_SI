"""
Schrodinger bridge between two squares WITH a reference drift, on a grid.

Reference process   (R)  dX = beta_t(X) dt + sqrt(2 gamma) dW
Bridge drift             b_t = beta_t + grad phi_t,   phi_t = 2 gamma log eta*_t

IPFP acts on the transport-diffusion semigroups of R (notes, "IPFP with a drift"):
    forward     d_t eta  + div(beta eta) - gamma lap eta  = 0     -> Qf
    backward    d_t eta* + beta.grad eta* + gamma lap eta* = 0     -> Qb   (adjoint of Qf)
    g = log nu - log Qf_1(e^f),      f = log mu - log Qb_1(e^g)

rho_t = eta_t eta*_t,  and the probability-flow velocity is
    v_t = beta_t + gamma grad( log eta*_t - log eta_t ).

beta is taken constant (= c_nu - c_mu), so it is divergence free and both
semigroups are exact Fourier multipliers.  Point of the exercise: beta absorbs
the bulk translation, the residual Sinkhorn potentials become O(1) instead of
O(10^300), and gamma can be pushed ~10x lower -> the interpolation keeps the
square structure instead of melting into a blob.

Then: recover v_t from {rho_t} alone by the P x P Galerkin system, and integrate
the Fokker-Planck equation with the recovered drift from mu to check it lands on nu.
"""

import jax, jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ------------------------------------------------------------------ grid
M, L = 128, 8.0
dx = L / M
xs = (jnp.arange(M) + 0.5) * dx - L / 2
X, Y = jnp.meshgrid(xs, xs, indexing="ij")
GRID = jnp.stack([X, Y], -1).reshape(-1, 2)
KX = 2 * jnp.pi * jnp.fft.fftfreq(M, d=dx)
K1, K2c = KX[:, None], KX[None, :]
K2 = K1**2 + K2c**2

CA, HA = jnp.array([-1.1, -1.1]), 0.80
CB, HB = jnp.array([+1.1, +0.7]), 0.50
FLOOR, SIG = 1e-8, 2.5 * (8.0 / 128)     # mollification width

def center_t(Ca, Cb, t):
    """Linear interpolation of the square centers."""
    return (1 - t) * Ca + t * Cb

def square(c, h):
    d = (jnp.abs(X - c[0]) <= h) & (jnp.abs(Y - c[1]) <= h)
    r = d.astype(jnp.float64)
    # mollify: a hard edge makes the bridge velocity singular at t=0,1
    r = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(r) * jnp.exp(-0.5 * SIG**2 * K2)))
    r = jnp.maximum(r, 0.0) + FLOOR
    return r / (r.sum() * dx**2)

MU, NU = square(CA, HA), square(CB, HB)

def grad2(u):
    return jnp.stack([jnp.gradient(u, dx, axis=-2),
                      jnp.gradient(u, dx, axis=-1)], -1)

# --------------------------------- reference semigroups (constant beta => exact)
def semigroups(beta, gamma):
    """Qf: forward Fokker-Planck of R.   Qb: its adjoint (backward Kolmogorov)."""
    phase = K1 * beta[0] + K2c * beta[1]
    def Qf(u, s):
        m = jnp.exp(-1j * s * phase - gamma * s * K2)
        return jnp.maximum(jnp.real(jnp.fft.ifft2(jnp.fft.fft2(u) * m)), 1e-280)
    def Qb(u, s):
        m = jnp.exp(+1j * s * phase - gamma * s * K2)
        return jnp.maximum(jnp.real(jnp.fft.ifft2(jnp.fft.fft2(u) * m)), 1e-280)
    return Qf, Qb

# ------------------------------------------------------------------ IPFP
NT = 41
TS = jnp.linspace(0.0, 1.0, NT)

def bridge(beta, gamma, iters=400):
    Qf, Qb = semigroups(beta, gamma)
    a = jnp.ones_like(MU)
    rng = []
    for _ in range(iters):
        a = a / a.max()                       # gauge freedom a->a/c, b->b*c
        b = NU / Qf(a, 1.0)
        a = MU / Qb(b, 1.0)
        rng.append(float(jnp.log10(a.max() / a.min())))
    e0 = float(jnp.abs(a * Qb(b, 1.0) - MU).sum() * dx**2)
    e1 = float(jnp.abs(b * Qf(a, 1.0) - NU).sum() * dx**2)
    ETA = jnp.stack([Qf(a, float(t)) for t in TS])          # eta_t
    ETAS = jnp.stack([Qb(b, float(1 - t)) for t in TS])     # eta*_t
    RHO = ETA * ETAS
    V = beta + gamma * (grad2(jnp.log(ETAS)) - grad2(jnp.log(ETA)))
    return dict(eta=ETA, etas=ETAS, rho=RHO, v=V, e0=e0, e1=e1,
                rng=rng[-1], beta=beta, gamma=gamma)

# --------------------------------------------------------------- feature map
KMAX = 3
_ks = [(p, q) for p in range(-KMAX, KMAX + 1) for q in range(-KMAX, KMAX + 1)
       if (p, q) != (0, 0) and (p > 0 or (p == 0 and q > 0))]
OM = (2 * jnp.pi / L) * jnp.array(_ks, dtype=jnp.float64)
P = 2 + 2 * OM.shape[0]

def phi(x):
    w = OM @ x
    return jnp.concatenate([x, jnp.cos(w), jnp.sin(w)])

def dphi(x):
    w = OM @ x
    return jnp.concatenate([jnp.eye(2), -jnp.sin(w)[:, None] * OM,
                            +jnp.cos(w)[:, None] * OM], axis=0)

PHI, DPHI = jit(vmap(phi))(GRID), jit(vmap(dphi))(GRID)
W = dx**2

# NOTE: the two linear features have constant gradients e1,e2, so a *constant*
# reference drift beta lies exactly in span{grad phi_i}: nothing is lost.

@jit
def gram(rho):
    return jnp.einsum("mid,mjd,m->ij", DPHI, DPHI, rho.ravel()) * W

@jit
def means(rho):
    return PHI.T @ rho.ravel() * W

@jit
def flux_rhs(eta, etas, beta, gamma):
    rho = eta * etas
    J = rho[..., None] * beta + gamma * (eta[..., None] * grad2(etas)
                                         - etas[..., None] * grad2(eta))
    return jnp.einsum("mid,md->i", DPHI, J.reshape(-1, 2)) * W

@jit
def ref_rhs(rho, v):
    return jnp.einsum("mid,md->i", DPHI, (rho[..., None] * v).reshape(-1, 2)) * W

def galerkin(B):
    """Recover v_t from the densities only."""
    Mt = jnp.stack([means(B["rho"][k]) for k in range(NT)])
    R_FD = jnp.gradient(Mt, TS[1] - TS[0], axis=0)
    R_FLUX = jnp.stack([flux_rhs(B["eta"][k], B["etas"][k],
                                 B["beta"], B["gamma"]) for k in range(NT)])
    R_REF = jnp.stack([ref_rhs(B["rho"][k], B["v"][k]) for k in range(NT)])
    C = []
    for k in range(NT):
        K = gram(B["rho"][k])
        C.append(jnp.linalg.solve(K + 1e-10 * jnp.trace(K) / P * jnp.eye(P), R_FD[k]))
    C = jnp.stack(C)
    VH = jnp.stack([jnp.einsum("mid,i->md", DPHI, c).reshape(M, M, 2) for c in C])
    err = jnp.array([jnp.sqrt((((VH[k]-B["v"][k])**2).sum(-1)*B["rho"][k]).sum()
                              / ((B["v"][k]**2).sum(-1)*B["rho"][k]).sum())
                     for k in range(NT)])
    return dict(vhat=VH, err=err, R_FD=R_FD, R_FLUX=R_FLUX, R_REF=R_REF)

# ------------------------ integrate the continuity equation with the recovered v
def integrate(B, G, cfl=0.85, use_true=False):
    """d_t rho + div(rho vhat) = 0  from mu.  Conservative upwind, no diffusion
    needed since vhat is already the probability-flow velocity."""
    RHO = B["rho"]
    # zero the velocity where there is no mass: outside the support the Fourier
    # features extrapolate to nonsense and would wreck the CFL condition.
    msk = RHO > 1e-4 * RHO.max(axis=(1, 2), keepdims=True)
    V = jnp.where(msk[..., None], B["v"] if use_true else G["vhat"], 0.0)
    vmax = float(jnp.abs(V).max())
    nsub = int(np.ceil((TS[1] - TS[0]) * vmax / (cfl * dx)))
    h = (TS[1] - TS[0]) / nsub

    @jit
    def advect(r, b):
        u, v = b[..., 0], b[..., 1]
        fx = jnp.maximum(u, 0) * r + jnp.minimum(u, 0) * jnp.roll(r, -1, 0)
        fy = jnp.maximum(v, 0) * r + jnp.minimum(v, 0) * jnp.roll(r, -1, 1)
        return r - h / dx * (fx - jnp.roll(fx, 1, 0) + fy - jnp.roll(fy, 1, 1))

    r = MU
    for k in range(NT - 1):
        for j in range(nsub):
            s = (j + 0.5) / nsub
            r = advect(r, (1 - s) * V[k] + s * V[k + 1])
    return r, nsub

# ------------------------------------------------------------------- run both
BETA0 = jnp.array([0.0, 0.0])
BETA1 = CB - CA                                  # simple constant reference drift

configs = [("no drift, gamma=0.20", BETA0, 0.20),
           ("beta=dc, gamma=0.15", BETA1, 0.15),
           ("beta=dc, gamma=0.07", BETA1, 0.07)]

results = []
for name, beta, gamma in configs:
    B = bridge(beta, gamma)
    G = galerkin(B)
    rf, nsub = integrate(B, G)
    rt, _ = integrate(B, G, use_true=True)
    l1 = float(jnp.abs(rf - NU).sum() * dx**2)
    l1t = float(jnp.abs(rt - NU).sum() * dx**2)
    sl = slice(3, NT - 3)
    print(f"{name:28s} | log10 range(a) {B['rng']:5.1f} | IPFP L1 "
          f"{B['e0']:.1e}/{B['e1']:.1e} | v err {float(jnp.median(G['err'][sl])):.4f}"
          f" | transport L1 to nu: exact v {l1t:.3f}, est v {l1:.3f}")
    results.append((name, B, G, rf))

# rdot cross-check on the drift-corrected case
_, B, G, _ = results[2]
sl = slice(3, NT - 3)
n = jnp.linalg.norm(G["R_REF"][sl])
print(f"\nrdot  FD of m(t)    rel.err {jnp.linalg.norm(G['R_FD'][sl]-G['R_REF'][sl])/n:.2e}")
print(f"rdot  flux formula  rel.err {jnp.linalg.norm(G['R_FLUX'][sl]-G['R_REF'][sl])/n:.2e}")

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.25])
snap = [0, 10, 20, 30, 40]
ext = [-L/2, L/2, -L/2, L/2]

for row, idx in enumerate([0, 2]):
    name, B, G, rf = results[idx]
    for i, kk in enumerate(snap):
        ax = fig.add_subplot(gs[row, i])
        ax.imshow(np.asarray(B["rho"][kk]).T, origin="lower", extent=ext, cmap="magma")
        ax.set_xlim(-2.6, 2.2); ax.set_ylim(-2.6, 1.9)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(name.replace(", ", "\n"), fontsize=8)
        ax.set_title(rf"$t={TS[kk]:.2f}$", fontsize=9)

ax = fig.add_subplot(gs[2, 0:2])
for name, B, G, rf in results:
    ax.semilogy(TS[3:NT-3], G["err"][3:NT-3], "o-", ms=3, label=name)
ax.set_xlabel("t"); ax.set_ylabel(r"rel. $L^2(\rho_t)$ error on $v_t$")
ax.set_title(f"Galerkin velocity error, P={P}", fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=.3)

_, B, G, _ = results[2]
ax = fig.add_subplot(gs[2, 2])
for j, lab in [(0, r"$x_1$"), (2, r"$\cos(\omega_1\!\cdot\!x)$")]:
    l, = ax.plot(TS[3:NT-3], G["R_REF"][3:NT-3, j], lw=3, alpha=.4, label=f"exact {lab}")
    ax.plot(TS[3:NT-3], G["R_FLUX"][3:NT-3, j], "-", lw=1, color=l.get_color())
    ax.plot(TS[3:NT-3:3], G["R_FD"][3:NT-3:3, j], "o", ms=3.5, mfc="none",
            color=l.get_color())
ax.set_xlabel("t"); ax.legend(fontsize=7); ax.grid(alpha=.3)
ax.set_title(r"$\dot r_t$ (flux vs FD), $\gamma=0.08$", fontsize=10)

for i, (ttl, im) in enumerate([(r"$\nu$ (target)", NU),
                               (r"$\rho_1$ from $\hat b_t$", results[2][3])]):
    ax = fig.add_subplot(gs[2, 3 + i])
    ax.imshow(np.asarray(im).T, origin="lower", extent=ext, cmap="magma")
    ax.set_xlim(0.2, 2.0); ax.set_ylim(-0.2, 1.6)
    ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("outputs/sb_squares_w_drift.png", dpi=125)
print("figure saved")


# 1D slice for different t
fig, ax = plt.subplots(figsize=(6, 4))
for name, B, G, rf in results:
    for i, kk in enumerate(snap):
        x, y = center_t(CA, CB, TS[kk])
        N_y = int(np.ceil(0.5 * L / dx))
        ax.plot(xs, B["rho"][kk][:, N_y], "-", lw=1.5, alpha=0.5,
                label=f"{name}, t={TS[kk]:.2f}")
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$\rho_t(x_1,x_c)$")
ax.legend(fontsize=7)
ax.grid(alpha=.3)
plt.tight_layout()