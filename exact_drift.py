"""
Schrodinger bridges between two squares along a PRESCRIBED CURVED PATH.

WHAT CHANGED FROM THE STRAIGHT-LINE VERSION
-------------------------------------------
Previously the reference path was the McCann displacement interpolation: the
square slid along a straight line while its size varied linearly.  Now the path
is prescribed to be a sine arc with a *breathing* size,

    lam(t) = (1-t) + t s + A_lam sin(pi t)          (size, lam(0)=1, lam(1)=s)
    c(t)   = (1-t) c_A + t c_B + A_c sin(pi t) n    (centre, n perp to c_B - c_A)

with endpoints pinned so the problem mu -> nu is unchanged.

The crucial structural point: the map is STILL AFFINE IN SPACE at every instant,

    Phi_t(y) = lam(t) y + b(t),      b(t) = c(t) - lam(t) c_A,

only lam and b are no longer linear in t.  Differentiating along a trajectory,
y = Phi_t^{-1}(x) = (x - b_t)/lam_t, gives the exact drift

    beta_t(x) = (lam'_t / lam_t) (x - b_t) + b'_t,

an isotropic dilation plus a translation.  That is curl-free -- beta_t is the
gradient of  (lam'/2 lam)|x|^2 + const . x  -- so the gradient ansatz used by the
Galerkin recovery can still represent it exactly, provided the feature map
contains quadratics.  A curved *trajectory* does not make the *velocity field*
non-gradient.

Consequently the Lagrangian trick still applies verbatim.  Under x = lam_t y + b_t
the reference SDE becomes dY = sqrt(2 gamma)/lam_t dW, pure Brownian motion with
the time change

    tau(t) = gamma * int_0^t lam_s^{-2} ds

which is now evaluated by quadrature rather than in closed form (that is the ONLY
place the extra generality costs anything).  Both IPFP semigroups remain exact,
self-adjoint heat multipliers in y, and nu still pulls back to exactly mu.

READ THE COMPARISON CAREFULLY
-----------------------------
With a curved reference path, the driftless bridge no longer approximates it even
as gamma -> 0: it converges to the STRAIGHT displacement interpolation, which is a
genuinely different path between the same endpoints.  So its distance to the
reference does not go to zero, and should not.  That is the point -- beta selects
the path, gamma only controls how sharply the path is followed.  Only the
exact-drift runs should be read as converging to the reference.

Outputs three figures to $OUTDIR (default ./outputs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
size = 14
params = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'cm',  # Computer Modern font
	'legend.fontsize':size,
    'axes.labelsize' : size,
	'axes.titlesize' : size +2,
    'xtick.labelsize' : size+1,
    'ytick.labelsize' : size+1
}
plt.rcParams.update(params)
import numpy as np
from jax import jit, vmap
from jax.scipy.ndimage import map_coordinates
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

OUTDIR = Path(os.environ.get("OUTDIR", "outputs"))


# =============================================================================
# 1. Grid
# =============================================================================

@dataclass(frozen=True)
class Grid:
    """A periodic square grid on [-L/2, L/2]^2 with M points per side."""
    M: int
    L: float

    @property
    def dx(self) -> float:
        return self.L / self.M

    @property
    def xs(self) -> jnp.ndarray:
        return (jnp.arange(self.M) + 0.5) * self.dx - self.L / 2

    @property
    def XY(self) -> jnp.ndarray:
        """(M, M, 2) array of grid point coordinates."""
        X, Y = jnp.meshgrid(self.xs, self.xs, indexing="ij")
        return jnp.stack([X, Y], -1)

    @property
    def points(self) -> jnp.ndarray:
        """(M*M, 2) flattened coordinates, for the feature map."""
        return self.XY.reshape(-1, 2)

    @property
    def k2(self) -> jnp.ndarray:
        """|xi|^2 on the FFT frequency grid."""
        kx = 2 * jnp.pi * jnp.fft.fftfreq(self.M, d=self.dx)
        return kx[:, None] ** 2 + kx[None, :] ** 2

    @property
    def kvec(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        kx = 2 * jnp.pi * jnp.fft.fftfreq(self.M, d=self.dx)
        return kx[:, None], kx[None, :]

    # --- differential / integral operators -----------------------------------

    def grad(self, u: jnp.ndarray) -> jnp.ndarray:
        """Central-difference gradient, acting on the LAST TWO axes so it can be
        applied to a stacked (NT, M, M) array of time slices."""
        return jnp.stack([jnp.gradient(u, self.dx, axis=-2),
                          jnp.gradient(u, self.dx, axis=-1)], -1)

    def integrate(self, u: jnp.ndarray) -> jnp.ndarray:
        return u.sum(axis=(-2, -1)) * self.dx ** 2

    def normalize(self, rho: jnp.ndarray) -> jnp.ndarray:
        return rho / (rho.sum(axis=(-2, -1), keepdims=True) * self.dx ** 2)

    def heat(self, u: jnp.ndarray, tau: float, floor: float = 1e-300):
        """Heat semigroup exp(tau * Laplacian): Gaussian blur of VARIANCE 2*tau.
        Clamped from below: the FFT produces small negative values that would
        otherwise poison the divisions in IPFP."""
        if tau <= 0.0:
            return jnp.maximum(u, floor)
        mult = jnp.exp(-tau * self.k2)
        return jnp.maximum(jnp.real(jnp.fft.ifft2(jnp.fft.fft2(u) * mult)), floor)

    def blur(self, u: jnp.ndarray, var: float) -> jnp.ndarray:
        return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(u) * jnp.exp(-0.5 * var * self.k2)))


# =============================================================================
# 2. Marginals
# =============================================================================

FLOOR = 1e-8      # positivity floor: IPFP divides by these densities
MOLLIFY = 0.1     # edge smoothing, in units of sig^2. Larger (0.5) gives smoother
                  # edges and a far better conditioned Galerkin system.


def mollified_square(g: Grid, centre, half_width, sig) -> jnp.ndarray:
    """Indicator of a square, lightly smoothed and floored, then normalized.

    The smoothing is not cosmetic: a hard edge makes the bridge velocity
    genuinely singular at t = 0 and t = 1 (|v| ~ 1e3), which wrecks the
    conditioning of everything downstream.
    """
    X, Y = g.XY[..., 0], g.XY[..., 1]
    inside = ((jnp.abs(X - centre[0]) <= half_width) &
              (jnp.abs(Y - centre[1]) <= half_width)).astype(jnp.float64)
    smooth = g.blur(inside, MOLLIFY * sig ** 2 / 0.5)
    return g.normalize(jnp.maximum(smooth, 0.0) + FLOOR)


# =============================================================================
# 3. The prescribed path: a time-dependent affine map  x = lam(t) y + b(t)
# =============================================================================

class AffineFlow:
    """A path of squares generated by an affine-in-space, arbitrary-in-time map.

    Everything the solvers need is derived from four scalars/vectors per time:
    lam(t), lam'(t), b(t), b'(t).  Subclass or swap in different lam/c to get a
    different trajectory; nothing downstream changes.
    """

    def __init__(self, c_a, h_a, c_b, h_b, amp_centre=0.0, amp_size=0.0):
        self.c_a = jnp.asarray(c_a)
        self.c_b = jnp.asarray(c_b)
        self.s = h_b / h_a                      # terminal dilation
        self.amp_centre = amp_centre            # sine bow, perpendicular to c_b-c_a
        self.amp_size = amp_size                # sine breathing of the size
        d = self.c_b - self.c_a
        self.normal = jnp.array([-d[1], d[0]]) / jnp.linalg.norm(d)

    # --- the two time-dependent generators and their derivatives -------------

    def lam(self, t):
        """Dilation. lam(0)=1, lam(1)=s, with a sine bulge in between."""
        return (1 - t) + t * self.s + self.amp_size * jnp.sin(jnp.pi * t)

    def dlam(self, t):
        return (self.s - 1) + self.amp_size * jnp.pi * jnp.cos(jnp.pi * t)

    def centre(self, t):
        """Centre of the square. Endpoints pinned; sine bow in between."""
        return ((1 - t) * self.c_a + t * self.c_b
                + self.amp_centre * jnp.sin(jnp.pi * t) * self.normal)

    def dcentre(self, t):
        return ((self.c_b - self.c_a)
                + self.amp_centre * jnp.pi * jnp.cos(jnp.pi * t) * self.normal)

    def offset(self, t):
        """b(t), so that Phi_t(y) = lam(t) y + b(t) maps the square correctly."""
        return self.centre(t) - self.lam(t) * self.c_a

    def doffset(self, t):
        return self.dcentre(t) - self.dlam(t) * self.c_a

    # --- derived quantities ---------------------------------------------------

    def velocity(self, g: Grid, t: float) -> jnp.ndarray:
        """Exact drift beta_t(x) = (lam'/lam)(x - b) + b'  -> (M, M, 2).

        Affine in x, and curl-free (isotropic dilation + translation), so it lies
        in the span of gradients of {linear, quadratic} features.
        """
        return (self.dlam(t) / self.lam(t)) * (g.XY - self.offset(t)) + self.doffset(t)

    def pullback(self, g: Grid, field, t: float, cval: float):
        """Evaluate a Lagrangian field at y(x) = (x - b_t)/lam_t (bilinear)."""
        y = (g.XY - self.offset(t)) / self.lam(t)
        coords = (y - g.xs[0]) / g.dx
        return map_coordinates(field, [coords[..., 0], coords[..., 1]],
                               order=1, mode="constant", cval=cval)

    def reference_path(self, g: Grid, mu, ts) -> jnp.ndarray:
        """rho_t^ref = (Phi_t)_# mu -- the noiseless prescribed interpolation."""
        path = jnp.stack([self.pullback(g, mu, float(t), FLOOR) / self.lam(float(t)) ** 2
                          for t in ts])
        return g.normalize(path)

    def time_change(self, gamma: float, n: int = 4001) -> Callable[[float], float]:
        """tau(t) = gamma * int_0^t lam_s^{-2} ds, by trapezoid quadrature.

        This is the ONLY place the curved trajectory costs anything relative to
        the straight-line case, where the integral was gamma * t / lam_t.
        """
        tg = jnp.linspace(0.0, 1.0, n)
        f = 1.0 / self.lam(tg) ** 2
        cum = jnp.concatenate([jnp.zeros(1),
                               jnp.cumsum(0.5 * (f[1:] + f[:-1]) * (tg[1] - tg[0]))])
        return lambda t: float(gamma * jnp.interp(t, tg, cum))


# =============================================================================
# 4. IPFP / Sinkhorn
# =============================================================================

def ipfp(mu, nu, forward: Callable, backward: Callable, iters: int = 300):
    """Iterative proportional fitting for the potentials a = e^f, b = e^g.

    `forward`  propagates a density from t=0 to t=1 under the reference process.
    `backward` is its adjoint (NOT a copy of it once beta != 0: advection flips).

    The `a /= a.max()` line is gauge fixing, not a hack: the bridge is exactly
    invariant under a -> a/c, b -> b*c, so rescaling is free and keeps the
    iterates inside float64.  Without it they overflow within two sweeps.  It must
    come BEFORE the b update so the returned pair stays mutually consistent.
    """
    a = jnp.ones_like(mu)
    b = jnp.ones_like(nu)
    for _ in range(iters):
        a = a / a.max()
        b = nu / forward(a)
        a = mu / backward(b)
    return a, b


@dataclass
class BridgeSolution:
    rho: jnp.ndarray          # (NT, M, M)     marginals rho_t
    v: jnp.ndarray            # (NT, M, M, 2)  probability-flow velocity v_t
    gamma: float
    log_range: float          # log10 dynamic range of the potentials
    marginal_err: float       # L1 error on the terminal marginal


def solve_bridge_constant_drift(g: Grid, mu, nu, beta, gamma, ts, iters=300):
    """Bridge with a CONSTANT reference drift (beta = 0 is the classical case).

    A constant drift is a pure translation, so both semigroups are exact Fourier
    multipliers: advection contributes a phase, diffusion a real decay.  Note the
    sign flip -- Qb is the adjoint of Qf, not a copy.
    """
    k1, k2 = g.kvec
    phase = k1 * beta[0] + k2 * beta[1]

    def _apply(u, s, sign):
        mult = jnp.exp(sign * 1j * s * phase - gamma * s * g.k2)
        return jnp.maximum(jnp.real(jnp.fft.ifft2(jnp.fft.fft2(u) * mult)), 1e-280)

    forward = lambda u, s=1.0: _apply(u, s, -1.0)     # Fokker-Planck
    backward = lambda u, s=1.0: _apply(u, s, +1.0)    # its adjoint

    a, b = ipfp(mu, nu, forward, backward, iters)

    eta = jnp.stack([forward(a, float(t)) for t in ts])
    eta_star = jnp.stack([backward(b, float(1 - t)) for t in ts])
    # v = beta + gamma grad(log eta* - log eta). The bridge DRIFT is
    # b_t = beta + 2 gamma grad log eta*, and v = b_t - gamma grad log rho_t.
    v = beta + gamma * (g.grad(jnp.log(eta_star)) - g.grad(jnp.log(eta)))

    return BridgeSolution(
        rho=eta * eta_star, v=v, gamma=gamma,
        log_range=float(jnp.log10(a.max() / a.min())),
        marginal_err=float(g.integrate(jnp.abs(b * forward(a) - nu))),
    )


def solve_bridge_exact_drift(g: Grid, mu, flow: AffineFlow, gamma, ts, iters=300):
    """Bridge whose reference process carries the EXACT drift of the prescribed flow.

    Solved in the Lagrangian frame, where:
      * the reference is Brownian motion under the time change tau(t), so both
        semigroups are the same self-adjoint heat kernel -- no operator splitting,
        hence no risk of an adjoint-inconsistent pair silently converging to the
        bridge of a slightly different reference process;
      * nu pulls back to exactly mu, so IPFP has no transport left to explain and
        the potentials stay O(1) instead of O(1e17).
    """
    tau = flow.time_change(gamma)
    tau1 = tau(1.0)
    heat_full = lambda u: g.heat(u, tau1)

    a, b = ipfp(mu, mu, heat_full, heat_full, iters)

    rho, vel = [], []
    for t in ts:
        t = float(t)
        eta = g.heat(a, tau(t))                  # eta_t   in y
        eta_star = g.heat(b, tau1 - tau(t))      # eta*_t  in y
        lam = flow.lam(t)

        # Push back to x; lam^-2 is the Jacobian of the affine map.
        rho.append(flow.pullback(g, eta * eta_star, t, FLOOR) / lam ** 2)

        # v = beta + gamma grad_x(log eta* - log eta), with grad_x = grad_y / lam.
        # The Jacobian factors are constant in x, so they drop out of log-gradients.
        gl = g.grad(jnp.log(eta_star)) - g.grad(jnp.log(eta))
        gl_x = jnp.stack([flow.pullback(g, gl[..., 0], t, 0.0),
                          flow.pullback(g, gl[..., 1], t, 0.0)], -1) / lam
        vel.append(flow.velocity(g, t) + gamma * gl_x)

    return BridgeSolution(
        rho=jnp.stack(rho), v=jnp.stack(vel), gamma=gamma,
        log_range=float(jnp.log10(a.max() / a.min())),
        marginal_err=float(g.integrate(jnp.abs(b * heat_full(a) - mu))),
    )


# =============================================================================
# 5. Feature map and the Galerkin recovery of the velocity
# =============================================================================

class FeatureMap:
    """phi: R^2 -> R^P with analytic gradients, evaluated once on the grid.

      * linear     x_1, x_2       -> CONSTANT gradients: a constant drift exactly.
      * quadratic  x_i x_j / 2    -> LINEAR gradients: this is what puts the exact
                                     AFFINE beta in the span. Without them the fit
                                     cannot represent the field it is recovering.
      * Fourier    cos/sin(w.x)   -> everything else.
    """

    def __init__(self, g: Grid, kmax: int = 3):
        ks = [(p, q) for p in range(-kmax, kmax + 1) for q in range(-kmax, kmax + 1)
              if (p, q) != (0, 0) and (p > 0 or (p == 0 and q > 0))]   # half-lattice
        self.omega = (2 * jnp.pi / g.L) * jnp.array(ks, dtype=jnp.float64)
        self.P = 5 + 2 * self.omega.shape[0]
        self.phi = jit(vmap(self._phi))(g.points)      # (M*M, P)
        self.dphi = jit(vmap(self._dphi))(g.points)    # (M*M, P, 2)

    def _phi(self, x):
        w = self.omega @ x
        quad = jnp.array([x[0] ** 2 / 2, x[1] ** 2 / 2, x[0] * x[1]])
        return jnp.concatenate([x, quad, jnp.cos(w), jnp.sin(w)])

    def _dphi(self, x):
        w = self.omega @ x
        quad = jnp.array([[x[0], 0.0], [0.0, x[1]], [x[1], x[0]]])
        return jnp.concatenate([jnp.eye(2), quad,
                                -jnp.sin(w)[:, None] * self.omega,
                                +jnp.cos(w)[:, None] * self.omega], axis=0)

    def gram(self, g: Grid, rho):
        """K_t[i,j] = int grad_phi_i . grad_phi_j rho_t dx."""
        return jnp.einsum("mid,mjd,m->ij", self.dphi, self.dphi,
                          rho.ravel()) * g.dx ** 2

    def means(self, g: Grid, rho):
        """m_j(t) = int phi_j rho_t dx."""
        return self.phi.T @ rho.ravel() * g.dx ** 2

    def field(self, g: Grid, coeff):
        return jnp.einsum("mid,i->md", self.dphi, coeff).reshape(g.M, g.M, 2)


# K_t is ill-conditioned (cond ~ 1e8) once rho_t concentrates on a small support,
# because the features become nearly linearly dependent there.  Real tension:
# SHRINKING GAMMA IMPROVES THE BRIDGE BUT DEGRADES THE RECOVERY.  Ridge 1e-10
# gives a velocity error ~5; 1e-3 brings it back to ~0.015.  The principled fix is
# features matched to the support scale, not more regularization.
RIDGE = 1e-3


def galerkin_velocity(g: Grid, fm: FeatureMap, sol: BridgeSolution, ts,
                      ridge: float = RIDGE):
    """Recover v_t from the marginals {rho_t} ALONE:

        K_t eta_t = rdot_t,   rdot_t = d/dt int phi rho_t dx,   v_t = grad_phi^T eta_t.

    rdot_t differences the P feature MEANS in time, never the density itself:
    m(t) is a smooth low-dimensional summary, so this is far better conditioned
    than differentiating rho_t pointwise.
    """
    means = jnp.stack([fm.means(g, sol.rho[k]) for k in range(len(ts))])
    rdot = jnp.gradient(means, ts[1] - ts[0], axis=0)

    v_hat, err = [], []
    for k in range(len(ts)):
        kt = fm.gram(g, sol.rho[k])
        reg = ridge * jnp.trace(kt) / fm.P * jnp.eye(fm.P)
        coeff = jnp.linalg.solve(kt + reg, rdot[k])
        vh = fm.field(g, coeff)
        v_hat.append(vh)
        num = (((vh - sol.v[k]) ** 2).sum(-1) * sol.rho[k]).sum()   # L2(rho_t)
        den = ((sol.v[k] ** 2).sum(-1) * sol.rho[k]).sum()
        err.append(jnp.sqrt(num / den))
    return jnp.stack(v_hat), jnp.array(err)


# =============================================================================
# 6. Error metrics against the reference path
# =============================================================================

def l1_to_reference(g: Grid, sol: BridgeSolution, ref) -> float:
    return float(jnp.mean(g.integrate(jnp.abs(sol.rho - ref))))


class SinkhornDivergence:
    """Debiased entropic W2 between densities on a COARSENED grid.

    The full grid would need an (M^2 x M^2) cost matrix; coarsening by `factor`
    makes it tractable. Returns the Sinkhorn divergence
        S(p,q) = OT_eps(p,q) - (OT_eps(p,p) + OT_eps(q,q))/2,
    which cancels the O(eps) entropic bias and tends to W2^2 as eps -> 0.

    Cost scales like factor^-4: factor=4 is ~40 s per batch of 41 slices on CPU,
    factor=8 is ~3 s but resolves differences only down to ~0.5 grid units.
    """

    def __init__(self, g: Grid, factor: int = 4, iters: int = 60):
        self.g, self.factor, self.iters = g, factor, iters
        mc = g.M // factor
        self.mc, self.dxc = mc, g.dx * factor
        xsc = (jnp.arange(mc) + 0.5) * self.dxc - g.L / 2
        Xc, Yc = jnp.meshgrid(xsc, xsc, indexing="ij")
        pts = jnp.stack([Xc, Yc], -1).reshape(-1, 2)
        self.cost = jnp.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        self.eps = self.dxc ** 2
        self._batch = jit(vmap(self._ot))
        self._self_batch = jit(vmap(lambda p: self._ot(p, p)))

    def coarsen(self, rho):
        w = rho.reshape(self.mc, self.factor, self.mc, self.factor)
        w = w.mean(axis=(1, 3)).reshape(-1) * self.dxc ** 2
        return w / w.sum()

    def _ot(self, p, q):
        """Entropic OT cost, log-domain Sinkhorn (stable for small eps)."""
        logp, logq = jnp.log(p), jnp.log(q)

        def step(_, fg):
            f, gg = fg
            gg = self.eps * (logq - logsumexp((f[:, None] - self.cost) / self.eps, 0))
            f = self.eps * (logp - logsumexp((gg[None, :] - self.cost) / self.eps, 1))
            return (f, gg)

        f, gg = jax.lax.fori_loop(0, self.iters, step,
                                  (jnp.zeros_like(logp), jnp.zeros_like(logq)))
        pi = jnp.exp((f[:, None] + gg[None, :] - self.cost) / self.eps)
        return jnp.sum(pi * self.cost)

    def curve(self, path, ref_path, ref_self=None):
        p = jnp.stack([self.coarsen(r) for r in path])
        q = jnp.stack([self.coarsen(r) for r in ref_path])
        if ref_self is None:
            ref_self = self._self_batch(q)
        sq = self._batch(p, q) - 0.5 * self._self_batch(p) - 0.5 * ref_self
        return jnp.sqrt(jnp.maximum(sq, 0.0)), ref_self


# =============================================================================
# 7. Plotting
# =============================================================================

XLIM, YLIM = (-2.8, 2.6), (-2.8, 2.8)     # widened: the sine arc bows upwards


def plot_slices(g: Grid, ts, tk, ref, runs, styles, flow, path):
    """1D cuts along x at the moving centre height of the reference square."""
    fig, axes = plt.subplots(1, len(tk), figsize=(19, 3.6), sharey=True)
    for ax, k in zip(axes, tk):
        t = float(ts[k])
        y_c = float(flow.centre(t)[1])
        j = int(jnp.argmin(jnp.abs(g.xs - y_c)))
        ax.plot(np.asarray(g.xs), np.asarray(ref[k][:, j]), color="k", lw=3,
                alpha=.35, label=r"reference ($\gamma\to0$)")
        for name, color, ls in styles:
            ax.plot(np.asarray(g.xs), np.asarray(runs[name].rho[k][:, j]),
                    color=color, ls=ls, lw=1.3, label=name)
        ax.set_title(rf"$t={t:.2f}$   (slice $y={y_c:.2f}$)", fontsize=10)
        ax.set_xlim(*XLIM)
        ax.set_xlabel("x")
        ax.grid(alpha=.3)
    axes[0].set_ylabel(r"$\rho_t(x,\,y_c)$")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_snapshots(g: Grid, ts, tk, ref, runs, styles, flow, path):
    """2D marginals: reference on top, one row per configuration.
    The dashed white curve is the prescribed centre trajectory."""
    ext = [-g.L / 2, g.L / 2, -g.L / 2, g.L / 2]
    tt = jnp.linspace(0, 1, 200)
    traj = np.asarray(jnp.stack([flow.centre(float(s)) for s in tt]))
    nrow = len(styles) + 1
    fig, axes = plt.subplots(nrow, len(tk), figsize=(19, 3.0 * nrow),
                             sharex=True, sharey=True)
    for j, k in enumerate(tk):
        axes[0, j].imshow(np.asarray(ref[k]).T, origin="lower", extent=ext, cmap="magma")
        axes[0, j].set_title(rf"$t={float(ts[k]):.2f}$", fontsize=10)
    axes[0, 0].set_ylabel(r"reference ($\gamma\to0$)", fontsize=8)
    for i, (name, _c, _ls) in enumerate(styles):
        for j, k in enumerate(tk):
            axes[i + 1, j].imshow(np.asarray(runs[name].rho[k]).T, origin="lower",
                                  extent=ext, cmap="magma")
        axes[i + 1, 0].set_ylabel(name.replace(",", "\n"), fontsize=8)
    for ax in axes.ravel():
        ax.plot(traj[:, 0], traj[:, 1], color="w", ls="--", lw=0.8, alpha=.6)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_w2(ts, w2, styles, sl, mc, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, color, ls in styles:
        ax.semilogy(np.asarray(ts[sl]), np.asarray(w2[name][sl]),
                    color=color, ls=ls, lw=1.5, label=name)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$W_2(\rho_t,\,\rho_t^{\rm ref})$")
    ax.set_title(f"entropic $W_2$ to the prescribed path (coarse grid {mc}x{mc})",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# 8. Main
# =============================================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    g = Grid(M=128, L=8.0)
    ts = jnp.linspace(0.0, 1.0, 41)
    nt = len(ts)
    sl = slice(3, nt - 3)     # drop endpoints: one-sided FD stencil there, and the
                              # bridge velocity is near-singular at the marginals

    c_a, h_a = jnp.array([-1.1, -1.1]), 0.80
    c_b, h_b = jnp.array([+1.1, +0.70]), 0.50
    sig = 2.5 * g.dx

    mu = mollified_square(g, c_a, h_a, sig)

    # The curved path. amp_centre bows the trajectory perpendicular to c_b - c_a;
    # amp_size makes the square swell then shrink. Both vanish at t = 0, 1, so the
    # endpoint problem mu -> nu is identical to the straight-line case.
    flow = AffineFlow(c_a, h_a, c_b, h_b, amp_centre=1.20, amp_size=0.35)
    ref = flow.reference_path(g, mu, ts)

    # nu is the EXACT pushforward Phi_1 # mu, so every solver shares endpoints and
    # the reference is exactly the noiseless path. (Independently mollifying nu
    # makes it blunter than the transported mollification of mu, and every run
    # then disagrees with the reference at t = 1 for a purely cosmetic reason.)
    nu = ref[-1]

    print(f"path: centre bow {flow.amp_centre:.2f}, size bow {flow.amp_size:.2f}; "
          f"lam in [{float(flow.lam(jnp.linspace(0,1,101)).min()):.2f}, "
          f"{float(flow.lam(jnp.linspace(0,1,101)).max()):.2f}]")

    configs = [
        ("no drift,      gamma=0.20",
         lambda: solve_bridge_constant_drift(g, mu, nu, jnp.zeros(2), 0.20, ts)),
        ("const drift,   gamma=0.10",
         lambda: solve_bridge_constant_drift(g, mu, nu, c_b - c_a, 0.10, ts)),
        ("exact drift,   gamma=0.08",
         lambda: solve_bridge_exact_drift(g, mu, flow, 0.08, ts)),
        ("exact drift,   gamma=0.05",
         lambda: solve_bridge_exact_drift(g, mu, flow, 0.05, ts)),
        ("exact drift,   gamma=0.01",
         lambda: solve_bridge_exact_drift(g, mu, flow, 0.01, ts)),
        ("exact drift,   gamma=0.005",
         lambda: solve_bridge_exact_drift(g, mu, flow, 0.005, ts)),
    ]

    fm = FeatureMap(g, kmax=3)
    runs: dict[str, BridgeSolution] = {}

    print(f"\n{'configuration':30s} {'log10 rng':>10s} {'marg err':>10s} {'v err':>9s}")
    for name, build in configs:
        sol = build()
        _, err = galerkin_velocity(g, fm, sol, ts)
        runs[name] = sol
        print(f"{name:30s} {sol.log_range:10.1f} {sol.marginal_err:10.1e} "
              f"{float(jnp.median(err[sl])):9.4f}")

    print("\nmean_t L1( rho_t , rho_t^ref ):")
    for name, sol in runs.items():
        print(f"  {name:30s} {l1_to_reference(g, sol, ref):.4f}")

    sd = SinkhornDivergence(g, factor=4, iters=60)
    w2, ref_self = {}, None
    print("\nmedian_t W2( rho_t , rho_t^ref )  (debiased entropic):")
    for name, sol in runs.items():
        w2[name], ref_self = sd.curve(sol.rho, ref, ref_self)
        print(f"  {name:30s} {float(jnp.median(w2[name][sl])):.4f}")
    print("  (the driftless run is NOT expected to converge to the reference: it "
          "follows the straight path)")

    styles = [("no drift,      gamma=0.20", "tab:blue", "-"),
              ("const drift,   gamma=0.10", "tab:orange", "-"),
              ("exact drift,   gamma=0.08", "tab:green", "-"),
              ("exact drift,   gamma=0.01", "tab:red", "-"),
              ("exact drift,   gamma=0.005", "tab:purple", "--")]
    tk = [0, 8, 16, 24, 32, 40]

    plot_slices(g, ts, tk, ref, runs, styles, flow, OUTDIR / "slices.png")
    plot_snapshots(g, ts, tk, ref, runs, styles, flow, OUTDIR / "rho2d.png")
    plot_w2(ts, w2, styles, sl, sd.mc, OUTDIR / "wasserstein.png")
    print(f"\nfigures written to {OUTDIR}/")


if __name__ == "__main__":
    main()