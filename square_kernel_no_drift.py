"""
Schrodinger bridge between two squares, on a grid, with NO sampling.

Stage 1 -- IPFP in the (eta, eta*) variables gives the bridge marginals
           rho_t = eta_t eta*_t   and the exact drift b_t = 2 gamma grad log eta*_t.

Stage 2 -- pretend we only have {rho_t} and recover the velocity by the
           Galerkin / kernel system, everything by quadrature:

              K_t[i,j] = int grad_phi_i . grad_phi_j rho_t dx
              rdot_t[j] = d/dt int phi_j rho_t dx
              K_t eta_t = rdot_t,     v_t = grad_phi^T eta_t

           rdot_t is obtained two ways:
             (FD)   finite differences of the feature means m(t)
             (FLUX) gamma * int grad_phi_j . (eta grad eta* - eta* grad eta) dx
           and both are compared to the true flux int grad_phi_j . rho_t v_t.

Stage 3 -- compare v_t to the exact bridge velocity gamma grad(g_t - f_t).
"""

import jax, jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------- grid & data
M, L, GAMMA = 128, 8.0, 0.20
dx = L / M
xs = (jnp.arange(M) + 0.5) * dx - L / 2
X, Y = jnp.meshgrid(xs, xs, indexing="ij")
GRID = jnp.stack([X, Y], -1).reshape(-1, 2)          # (M*M, 2)

def square(c, h):
    d = (jnp.abs(X - c[0]) <= h) & (jnp.abs(Y - c[1]) <= h)
    r = d.astype(jnp.float64) + 1e-8
    return r / (r.sum() * dx**2)

MU = square((-1.1, -1.1), 0.80)                      # the two "fields"
NU = square((+1.1, +0.7), 0.50)

# ------------------------------------------------- heat semigroup P_s = e^{s g D}
KX = 2 * jnp.pi * jnp.fft.fftfreq(M, d=dx)
K2 = KX[:, None] ** 2 + KX[None, :] ** 2

def heat(u, s):
    if s <= 0.0:
        return u
    v = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(u) * jnp.exp(-GAMMA * s * K2)))
    return jnp.maximum(v, 1e-280)




# ------------------------------------------------------------------ Stage 1: IPFP
a = jnp.ones_like(MU)
for _ in range(400):
    a = a / a.max()                 # gauge: a -> a/c, b -> b*c leaves rho_t invariant
    b = NU / heat(a, 1.0)
    a = MU / heat(b, 1.0)
err0 = jnp.abs(a * heat(b, 1.0) - MU).sum() * dx**2
err1 = jnp.abs(b * heat(a, 1.0) - NU).sum() * dx**2
print(f"IPFP marginal errors (L1):  mu {err0:.2e}   nu {err1:.2e}")

NT = 41
TS = jnp.linspace(0.0, 1.0, NT)

ETA  = jnp.stack([heat(a, float(t))       for t in TS])   # eta_t   (NT,M,M)
ETAS = jnp.stack([heat(b, float(1.0 - t)) for t in TS])   # eta*_t
RHO  = ETA * ETAS                                          # bridge marginals

fig, ax = plt.subplots()
ax.imshow(np.asarray(RHO[20]).T, origin="lower", extent=[-L/2, L/2, -L/2, L/2], cmap="magma")

def grad2(u):
    gx = jnp.gradient(u, dx, axis=-2)
    gy = jnp.gradient(u, dx, axis=-1)
    return jnp.stack([gx, gy], -1)

# exact bridge velocity  v = gamma grad(log eta* - log eta)
V_TRUE = GAMMA * (grad2(jnp.log(ETAS)) - grad2(jnp.log(ETA)))   # (NT,M,M,2)

# ------------------------------------------------------ feature map (analytic)
KMAX = 3
_ks = [(p, q) for p in range(-KMAX, KMAX + 1) for q in range(-KMAX, KMAX + 1)
       if (p, q) != (0, 0) and (p > 0 or (p == 0 and q > 0))]
OM = (2 * jnp.pi / L) * jnp.array(_ks, dtype=jnp.float64)
F = OM.shape[0]; P = 2 + 2 * F

def phi(x):
    w = OM @ x
    return jnp.concatenate([x, jnp.cos(w), jnp.sin(w)])

def dphi(x):
    w = OM @ x
    return jnp.concatenate([jnp.eye(2),
                            -jnp.sin(w)[:, None] * OM,
                            +jnp.cos(w)[:, None] * OM], axis=0)

PHI = jit(vmap(phi))(GRID)                  # (M*M, P)
DPHI = jit(vmap(dphi))(GRID)                # (M*M, P, 2)

# ------------------------------------- Stage 2: everything from the densities
W = dx**2                                   # quadrature weight

@jit
def gram(rho):
    return jnp.einsum("mid,mjd,m->ij", DPHI, DPHI, rho.ravel()) * W

@jit
def means(rho):                             # m_j(t) = int phi_j rho_t
    return PHI.T @ rho.ravel() * W

@jit
def flux_rhs(eta, etas):                    # gamma int grad_phi.(eta grad eta* - eta* grad eta)
    J = GAMMA * (eta[..., None] * grad2(etas) - etas[..., None] * grad2(eta))
    return jnp.einsum("mid,md->i", DPHI, J.reshape(-1, 2)) * W

@jit
def true_rhs(rho, v):                       # int grad_phi . rho v   (reference)
    return jnp.einsum("mid,md->i", DPHI, (rho[..., None] * v).reshape(-1, 2)) * W

Mt = jnp.stack([means(RHO[k]) for k in range(NT)])                 # (NT,P)
R_FD = jnp.gradient(Mt, TS[1] - TS[0], axis=0)                     # finite differences
R_FLUX = jnp.stack([flux_rhs(ETA[k], ETAS[k]) for k in range(NT)])
R_REF = jnp.stack([true_rhs(RHO[k], V_TRUE[k]) for k in range(NT)])

sl = slice(3, NT - 3)                       # drop FD end-effects for the report
nrm = jnp.linalg.norm(R_REF[sl])
print(f"rdot  finite-diff of m(t)   rel.err vs exact flux : "
      f"{jnp.linalg.norm(R_FD[sl]-R_REF[sl])/nrm:.2e}")
print(f"rdot  flux formula          rel.err vs exact flux : "
      f"{jnp.linalg.norm(R_FLUX[sl]-R_REF[sl])/nrm:.2e}")

# solve the P x P system at each time
def solve(k, rhs):
    K = gram(RHO[k])
    lam = 1e-10 * jnp.trace(K) / P
    return jnp.linalg.solve(K + lam * jnp.eye(P), rhs[k])

C_FD = jnp.stack([solve(k, R_FD) for k in range(NT)])

# ------------------------------------------------- Stage 3: velocity accuracy
@jit
def vhat(c):
    return jnp.einsum("mid,i->md", DPHI, c).reshape(M, M, 2)

V_HAT = jnp.stack([vhat(C_FD[k]) for k in range(NT)])

def rel_L2rho(k):
    d = ((V_HAT[k] - V_TRUE[k]) ** 2).sum(-1) * RHO[k]
    n = (V_TRUE[k] ** 2).sum(-1) * RHO[k]
    return jnp.sqrt(d.sum() / n.sum())

errs = jnp.array([rel_L2rho(k) for k in range(3, NT - 3)])
print(f"velocity   rel. L2(rho_t) error   median {jnp.median(errs):.3f}   "
      f"max {errs.max():.3f}")

# reconstructed SB drift  b = v + gamma grad log rho
B_HAT = V_HAT + GAMMA * grad2(jnp.log(RHO))
B_TRUE = 2 * GAMMA * grad2(jnp.log(ETAS))
k = NT // 2
d = ((B_HAT[k] - B_TRUE[k]) ** 2).sum(-1) * RHO[k]
n = (B_TRUE[k] ** 2).sum(-1) * RHO[k]
print(f"drift b_t  rel. L2(rho_t) error at t=0.5 : {jnp.sqrt(d.sum()/n.sum()):.3f}")

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(15, 7.2))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15])

for i, kk in enumerate([0, 10, 20, 30, 40]):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(np.asarray(RHO[kk]).T, origin="lower",
              extent=[-L/2, L/2, -L/2, L/2], cmap="magma")
    ax.set_title(rf"$\rho_t$, $t={TS[kk]:.2f}$", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-2.6, 2.2); ax.set_ylim(-2.6, 1.9)

ax = fig.add_subplot(gs[1, :2])
ii = slice(2, NT - 2)
for j, lab in [(0, r"$\varphi=x_1$"), (2, r"$\cos(\omega_1\!\cdot\!x)$")]:
    l, = ax.plot(TS[ii], R_REF[ii, j], lw=3, alpha=.4, label=f"exact flux  {lab}")
    ax.plot(TS[ii], R_FLUX[ii, j], "-", lw=1, color=l.get_color())
    ax.plot(TS[2:NT-2:3], R_FD[2:NT-2:3, j], "o", ms=4, mfc="none", color=l.get_color())
ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title(r"$\dot r_t$: thick = reference, thin = flux formula, "
             "circles = FD of $m(t)$", fontsize=10)

ax = fig.add_subplot(gs[1, 2])
ax.plot(TS[3:NT-3], errs, "o-", ms=3)
ax.set_xlabel("t"); ax.set_ylabel("rel. $L^2(\\rho_t)$ error")
ax.set_title(f"velocity error, P={P}", fontsize=10); ax.grid(alpha=.3)

sub = slice(None, None, 6)
msk = np.asarray(RHO[k] > 0.02 * RHO[k].max())[sub, sub]
for i, (Vp, ttl) in enumerate([(V_TRUE[k], "exact $v_t$"),
                               (V_HAT[k], r"$\nabla\varphi^\top\eta_t$")]):
    ax = fig.add_subplot(gs[1, 3 + i])
    ax.imshow(np.asarray(RHO[k]).T, origin="lower",
              extent=[-L/2, L/2, -L/2, L/2], cmap="Greys", alpha=.55)
    U = np.where(msk, np.asarray(Vp)[sub, sub, 0], np.nan)
    Vv = np.where(msk, np.asarray(Vp)[sub, sub, 1], np.nan)
    ax.quiver(np.asarray(X)[sub, sub], np.asarray(Y)[sub, sub], U, Vv,
              color="crimson", scale=45, width=.006)
    ax.set_xlim(-2.6, 2.2); ax.set_ylim(-2.6, 1.9)
    ax.set_title(ttl + r"  at $t=0.5$", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("outputs/sb_squares_grid.png", dpi=125)
print("figure saved")