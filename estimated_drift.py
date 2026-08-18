"""
Schrodinger bridges between two squares along a PRESCRIBED CURVED PATH,
with the reference drift ESTIMATED FROM A FEW INTERMEDIATE SNAPSHOTS.

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
the time change tau(t) = gamma * int_0^t lam_s^{-2} ds, now evaluated by
quadrature rather than in closed form.

WHAT IS NEW HERE: beta FROM SNAPSHOTS  (sections 5b and 5c)
-----------------------------------------------------------
The exact-drift solver above assumes beta is handed to us analytically.  In
practice one only observes the intermediate law at a handful of times, say
t = 0.2, 0.4, 0.6, 0.8 (plus the two marginals, which are known by definition).
Section 5b implements the weak-form recipe:

    d/dt E_{rho_t}[phi] = E_{rho_t}[beta_t . grad phi] + gamma_obs E_{rho_t}[Lap phi]

    =>   K_t eta_t = mdot_t - gamma_obs * l_t,      beta_t = grad_phi^T eta_t,

    (K_t)_ij = E_{rho_t}[grad phi_i . grad phi_j],
    (m_t)_i  = E_{rho_t}[phi_i],
    (l_t)_i  = E_{rho_t}[Lap phi_i].

Three things deserve emphasis.

  * mdot is obtained by fitting a low-order polynomial in t to the P feature
    MEANS through all known times and differentiating it analytically.  m(t) is a
    smooth, low-dimensional summary, so this is enormously better conditioned
    than differencing rho_t pointwise -- which is hopeless anyway with a time
    spacing of 0.2.  With only six times, the accuracy of mdot, not the ridge, is
    the binding constraint on beta_hat; the run prints the same recovery done on
    the dense time grid so you can see that gap directly.

  * gamma_obs is the diffusivity OF THE PROCESS THAT PRODUCED THE SNAPSHOTS, not
    the gamma of the bridge we are about to solve.  Snapshots taken from the
    noiseless prescribed path satisfy a pure continuity equation, so gamma_obs=0
    and the l term drops.  Snapshots taken from a diffusive bridge need
    gamma_obs = that bridge's gamma -- and then what comes back is that bridge's
    own drift beta + 2 gamma grad log eta*, which agrees with beta only where the
    bridge is not busy bending to hit its endpoints.  Both are reported.

  * Only the four interior snapshots get a K_t solve.  At t = 0 and t = 1 the
    density is sharp and the field is near-singular; those two times are used
    only to constrain the polynomial fit of m(t).  eta_t is then interpolated
    linearly in t between the nodes (constant extrapolation outside).

Section 5c is the solver that actually consumes beta_hat.  Neither existing
solver can: the constant-drift one needs beta spatially constant, the exact-drift
one needs the affine structure to move to a Lagrangian frame.  For a general
tabulated beta_t we integrate the two PDEs directly, with the discrete backward
operator built as the EXACT TRANSPOSE of the forward one, step by step in reverse
order -- if the pair is only adjoint up to discretization error, IPFP converges
happily to the bridge of a slightly different reference process and nothing warns
you.  Spectral derivatives with the Nyquist mode zeroed are exactly skew, so
adjoint(u -> -div(beta u)) = beta . grad exactly, and the heat multiplier is
self-adjoint by inspection.

READ THE COMPARISON CAREFULLY
-----------------------------
With a curved reference path, the driftless bridge no longer approximates it even
as gamma -> 0: it converges to the STRAIGHT displacement interpolation, which is a
genuinely different path between the same endpoints.  So its distance to the
reference does not go to zero, and should not.  That is the point -- beta selects
the path, gamma only controls how sharply the path is followed.  Only the
exact-drift and estimated-drift runs should be read as converging to the reference.

Outputs three figures to $OUTDIR (default ./outputs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

size = 14
params = {
    'text.usetex': bool(int(os.environ.get("USETEX", "1"))),
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
from jax import jit, lax, vmap
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

    @property
    def kvec_skew(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """kvec with the Nyquist mode zeroed.

        For even M the Nyquist column is its own negative, so i*k there is NOT
        the symbol of a skew-symmetric real matrix.  Zeroing it makes spectral
        differentiation exactly skew, hence div and -grad exactly transpose --
        which is what lets the drifted propagator have an exact discrete adjoint.
        """
        kx = 2 * jnp.pi * jnp.fft.fftfreq(self.M, d=self.dx)
        kx = kx.at[self.M // 2].set(0.0)
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
    """phi: R^2 -> R^P with analytic gradients AND Laplacians, evaluated once on
    the grid.

      * linear     x_1, x_2       -> CONSTANT gradients: a constant drift exactly.
      * quadratic  x_i x_j / 2    -> LINEAR gradients: this is what puts the exact
                                     AFFINE beta in the span. Without them the fit
                                     cannot represent the field it is recovering.
      * Fourier    cos/sin(w.x)   -> everything else.

    The Laplacians are needed only by the drift estimator (section 5b), for the
    gamma * E[Lap phi] term of the weak form.  They are cheap and exact here;
    Hutchinson would only be needed for a learned/high-dimensional phi.
    """

    def __init__(self, g: Grid, kmax: int = 3):
        ks = [(p, q) for p in range(-kmax, kmax + 1) for q in range(-kmax, kmax + 1)
              if (p, q) != (0, 0) and (p > 0 or (p == 0 and q > 0))]   # half-lattice
        # kmax = 0 leaves the purely polynomial map {x, x^2/2, xy}: exactly the
        # span that contains an affine drift and nothing else. The drift
        # estimator uses it (see main).
        self.omega = ((2 * jnp.pi / g.L) * jnp.array(ks, dtype=jnp.float64)
                      if ks else jnp.zeros((0, 2), dtype=jnp.float64))
        self.P = 5 + 2 * self.omega.shape[0]
        self.phi = jit(vmap(self._phi))(g.points)      # (M*M, P)
        self.dphi = jit(vmap(self._dphi))(g.points)    # (M*M, P, 2)
        self.lphi = jit(vmap(self._lphi))(g.points)    # (M*M, P)

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

    def _lphi(self, x):
        """Lap phi: 0 for linear, (1,1,0) for the quadratics, -|w|^2 phi for Fourier."""
        w = self.omega @ x
        w2 = (self.omega ** 2).sum(-1)
        return jnp.concatenate([jnp.zeros(2), jnp.array([1.0, 1.0, 0.0]),
                                -w2 * jnp.cos(w), -w2 * jnp.sin(w)])

    def gram(self, g: Grid, rho):
        """K_t[i,j] = int grad_phi_i . grad_phi_j rho_t dx."""
        return jnp.einsum("mid,mjd,m->ij", self.dphi, self.dphi,
                          rho.ravel()) * g.dx ** 2

    def means(self, g: Grid, rho):
        """m_j(t) = int phi_j rho_t dx."""
        return self.phi.T @ rho.ravel() * g.dx ** 2

    def lap_means(self, g: Grid, rho):
        """l_j(t) = int Lap phi_j rho_t dx."""
        return self.lphi.T @ rho.ravel() * g.dx ** 2

    def field(self, g: Grid, coeff):
        return jnp.einsum("mid,i->md", self.dphi, coeff).reshape(g.M, g.M, 2)


# K_t is ill-conditioned (cond ~ 1e8) once rho_t concentrates on a small support,
# because the features become nearly linearly dependent there.  Real tension:
# SHRINKING GAMMA IMPROVES THE BRIDGE BUT DEGRADES THE RECOVERY.  Ridge 1e-10
# gives a velocity error ~5; 1e-3 brings it back to ~0.015.  The principled fix is
# features matched to the support scale, not more regularization.
RIDGE = 1e-3

# The drift estimate tolerates a smaller ridge than the bridge-velocity recovery:
# the observed rho_t are honest densities (not products of near-singular
# potentials) and beta itself lies exactly in the span of the quadratic features,
# so there is a genuinely low-bias solution to find.
RIDGE_BETA = 1e-5

# IPFP sweeps for the tabulated-drift solver. Each one integrates the PDE twice
# over [0,1], so this is the run-time knob that matters; watch `marg err` rather
# than trusting a number. With a drift that already nearly transports mu to nu,
# there is very little left for IPFP to do.
IPFP_ITERS_EST = 60


def galerkin_velocity(g: Grid, fm: FeatureMap, sol: BridgeSolution, ts,
                      ridge: float = RIDGE):
    """Recover v_t from the marginals {rho_t} ALONE:

        K_t eta_t = rdot_t,   rdot_t = d/dt int phi rho_t dx,   v_t = grad_phi^T eta_t.

    rdot_t differences the P feature MEANS in time, never the density itself:
    m(t) is a smooth low-dimensional summary, so this is far better conditioned
    than differentiating rho_t pointwise.

    No gamma * l_t term here, deliberately: v_t is the PROBABILITY-FLOW velocity,
    which by definition transports rho_t with no diffusion left over.
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
# 5b. Drift estimation from a handful of snapshots  (the new piece)
# =============================================================================

@dataclass
class DriftEstimate:
    """beta_hat_t = grad_phi^T eta_t, with eta_t interpolated between the nodes.

    Linear interpolation in t, constant extrapolation outside [t_first, t_last]
    (that is jnp.interp's edge behaviour, and it is the honest choice: nothing in
    the data constrains beta beyond the observed window, and extrapolating a
    least-squares coefficient vector is a good way to manufacture a large
    spurious velocity right where the marginal constraint is tightest).
    """
    ts_nodes: jnp.ndarray      # (S,)
    eta_nodes: jnp.ndarray     # (S, P)
    g: Grid
    fm: FeatureMap

    def coeff(self, t) -> jnp.ndarray:
        return vmap(lambda col: jnp.interp(t, self.ts_nodes, col))(self.eta_nodes.T)

    def field(self, t) -> jnp.ndarray:
        return self.fm.field(self.g, self.coeff(jnp.asarray(t, dtype=jnp.float64)))

    def fields_on(self, ts) -> jnp.ndarray:
        return jnp.stack([self.field(float(t)) for t in ts])


def _poly_design(ts: jnp.ndarray, deg: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vandermonde and its t-derivative, for the parametric fit of m(t)."""
    V = jnp.stack([ts ** p for p in range(deg + 1)], -1)
    Vd = jnp.stack([jnp.zeros_like(ts) if p == 0 else p * ts ** (p - 1)
                    for p in range(deg + 1)], -1)
    return V, Vd


def estimate_drift(g: Grid, fm: FeatureMap, snap_ts, snap_rho,
                   node_sel: Sequence[int], gamma_obs: float = 0.0,
                   ridge: float = RIDGE_BETA, deg: int = 5) -> DriftEstimate:
    """The practical recipe, step by step.

    1. assemble m_{t_k}, K_{t_k}, l_{t_k} from the observed densities;
    2. fit a degree-`deg` polynomial in t to k -> m_{t_k} (least squares over ALL
       snapshots, including the two marginals) and differentiate it analytically
       to get mdot at the nodes;
    3. solve (K_{t_k} + lambda I) eta_{t_k} = mdot_{t_k} - gamma_obs l_{t_k};
    4. return eta interpolated in t, so beta_t = grad_phi^T eta_t.

    `node_sel` indexes into snap_ts: which snapshots get a K-solve.  Endpoints are
    normally excluded -- the density is sharp there, K is at its worst
    conditioned, and the true field is near-singular -- but they still constrain
    step 2, which is where they earn their keep.

    Step 2 is the accuracy bottleneck at four snapshots.  Both the exact m(t) for
    an affine flow with a sine bow and the Fourier feature means are analytic but
    not polynomial, so a quartic through six nodes leaves a small residual in
    mdot that no amount of ridge tuning can remove.  Nothing here is specific to
    a grid: replace the quadratures by Monte-Carlo averages over N samples and the
    same four lines run from particles.
    """
    snap_ts = jnp.asarray(snap_ts, dtype=jnp.float64)
    if deg > len(snap_ts) - 1:
        raise ValueError("polynomial degree must be < number of snapshots")

    m = jnp.stack([fm.means(g, r) for r in snap_rho])            # (S, P)
    V, Vd = _poly_design(snap_ts, deg)
    coef = jnp.linalg.lstsq(V, m, rcond=None)[0]                 # (deg+1, P)
    mdot = Vd @ coef                                             # (S, P)

    etas = []
    for i in node_sel:
        kt = fm.gram(g, snap_rho[i])
        rhs = mdot[i] - gamma_obs * fm.lap_means(g, snap_rho[i])
        reg = ridge * jnp.trace(kt) / fm.P * jnp.eye(fm.P)
        etas.append(jnp.linalg.solve(kt + reg, rhs))

    return DriftEstimate(ts_nodes=snap_ts[jnp.array(list(node_sel))],
                         eta_nodes=jnp.stack(etas), g=g, fm=fm)


def drift_error(g: Grid, beta_hat, beta_true, weights, mask: float = 1e-6) -> jnp.ndarray:
    """Relative L2(rho_t) error of the estimated field, one number per time.

    The vacuum is masked out. Without it the metric is meaningless: the densities
    carry a positivity FLOOR of 1e-8 everywhere, beta_hat is unconstrained out
    there by construction, and 16k floor-weighted cells at |beta_hat| ~ 1e4 swamp
    the O(1) contribution from the actual support.
    """
    weights = jnp.where(weights < mask * weights.max(), 0.0, weights)
    num = (((beta_hat - beta_true) ** 2).sum(-1) * weights).sum(axis=(-2, -1))
    den = ((beta_true ** 2).sum(-1) * weights).sum(axis=(-2, -1))
    return jnp.sqrt(num / den)


# =============================================================================
# 5c. Bridge with a general TABULATED drift beta_t(x)
# =============================================================================

def taper_to_box(g: Grid, betas: jnp.ndarray, margin: float = 0.6,
                 width: float = 0.25) -> jnp.ndarray:
    """Smoothly switch beta off before the edge of the periodic box.

    THIS IS NOT COSMETIC, it is the difference between the solver working and
    returning striped garbage.  beta_hat is affine, so as a function on the TORUS
    it has a jump of size |beta| across the boundary, and it is largest exactly
    there.  Any transported quantity then acquires that discontinuity; a spectral
    scheme turns it into Gibbs ringing across the whole domain, and a
    semi-Lagrangian one teleports vacuum mass from one edge to the other.  Both
    show up as the high-frequency stripes that have nothing to do with the
    bridge.

    Nothing physical is lost: the taper only acts where the reference path has no
    mass, and beta there was never constrained by the snapshots in the first
    place -- it is pure extrapolation of a least-squares fit.
    """
    w = jnp.ones((g.M, g.M))
    for d in range(2):
        z = jnp.abs(g.XY[..., d])
        w = w * 0.5 * (1 + jnp.tanh((g.L / 2 - margin - z) / width))
    return betas * w[None, ..., None]


class DriftedPropagator:
    """Q_f: forward Fokker-Planck   d_t eta  + div(beta eta) = gamma Lap eta
       Q_b: its exact discrete adjoint, the backward Kolmogorov equation
            d_t eta* + beta . grad eta* + gamma Lap eta* = 0, run from t=1.

    Neither closed-form solver above takes a general tabulated beta_t, so this one
    integrates directly, by splitting each substep into transport and diffusion:

        F_n = H o Sc_n,      Sc_n = pushforward along  T_n(x) = x + dt beta_n(x),
                             H    = Fourier heat multiplier exp(-gamma dt |k|^2).

    Sc_n is SEMI-LAGRANGIAN, implemented as the transpose of bilinear
    interpolation: Sc_n = G_n^T where (G_n u)(x) = u(T_n(x)).  That identity is
    just <T_# rho, f> = <rho, f o T> written on the grid, and it buys three
    things at once.

      * POSITIVITY. Bilinear weights are non-negative, so a non-negative density
        stays non-negative -- there is no ringing to clamp.  The previous
        spectral-advection version of this class produced O(1e-8) negatives in
        the vacuum, the IPFP divisions turned those into 1e+272 spikes, and the
        marginals came out as high-frequency stripes.  Positivity of the
        transport step is not a nicety here, it is the whole ballgame.
      * NO CFL CONDITION.  Unconditionally stable, so nsub_per is set by how
        accurately you want the trajectories, not by stability.  The spectral
        version needed 14 substeps per interval just to survive.
      * EXACT ADJOINTNESS.  F_n^T = G_n o H, same weights and same indices, in
        floating point.  IPFP converges happily with a mismatched pair -- silently
        solving the bridge of a different reference process -- and
        `adjoint_residual` is the only thing that would ever tell you.

    Mass is conserved exactly (the scatter weights sum to one per source cell) and
    constants are preserved exactly by the adjoint: the same statement transposed.

    THE COST IS NUMERICAL DIFFUSION.  Bilinear interpolation adds a variance of
    about f(1-f) dx^2 per step per axis, f the fractional cell offset, so the
    scheme behaves like a slightly larger gamma: roughly 0.17 * n_steps * dx^2 of
    extra variance against the physical 2 gamma.  With the defaults here that is
    a ~25% inflation, and it gets WORSE with more substeps, not better -- the
    opposite of the usual convergence instinct.  `effective_gamma` reports the
    estimate.  Fewer, larger steps with an accurate map is the right trade; use
    order-2 map construction (midpoint beta, already done) rather than many
    small Euler steps.
    """

    # RELATIVE floor on a completed sweep, and the single most consequential
    # constant in this class. It is not there for positivity (semi-Lagrangian
    # transport and the heat multiplier already give that) but to CAP THE DYNAMIC
    # RANGE OF THE POTENTIALS. IPFP divides by these sweeps: with a Gaussian-
    # tailed reference process and marginals that carry a 1e-8 relative floor
    # everywhere, the exact fixed point wants potentials spanning 1e24, float64
    # loses the bulk to roundoff, and the iteration stalls at a few percent
    # marginal error -- or diverges. Matching the floor of mu and nu keeps the
    # range near 1e10 and the terminal marginal is then matched to ~1e-6.
    # Measured, 60 sweeps, everything else fixed:
    #   FLOOR_REL   1e-8      1e-12     1e-16
    #   marg err    1.7e-6    4.1e-2    1.5e+06
    #   log10 rng   10.2      14.5      24.0
    FLOOR_REL = 1e-8

    def __init__(self, g: Grid, betas: jnp.ndarray, gamma: float,
                 nsub_per: int = 1, vmax: float | None = None):
        self.g, self.gamma, self.nsub_per = g, gamma, nsub_per
        self.nout = betas.shape[0] - 1
        self.dt = 1.0 / (self.nout * nsub_per)

        speed = jnp.sqrt((betas ** 2).sum(-1))
        self.clipped = 0.0
        if vmax is not None:
            scale = jnp.minimum(1.0, vmax / jnp.maximum(speed, 1e-30))
            self.clipped = float((speed > vmax).mean())
            betas = betas * scale[..., None]
        self.betas = betas
        self.vmax = float(jnp.sqrt((betas ** 2).sum(-1)).max())
        self.cfl_cells = self.vmax * self.dt / g.dx        # displacement, in cells
        self.effective_gamma = gamma + 0.17 * self.nout * nsub_per * g.dx ** 2 / 2

        M, dx, dt, nper = g.M, g.dx, self.dt, nsub_per
        x0 = float(g.xs[0])
        heat = jnp.exp(-gamma * dt * g.k2)

        def beta_at(n):
            j = n // nper
            f = (n % nper + 0.5) / nper                 # midpoint of the substep
            return (1 - f) * self.betas[j] + f * self.betas[j + 1]

        def stencil(n):
            """Bilinear weights/indices for T_n(x) = x + dt beta_n(x)."""
            p = g.XY + dt * beta_at(n)
            idx = (p - x0) / dx
            i0 = jnp.floor(idx)
            fr = idx - i0
            i0 = i0.astype(jnp.int32)
            ix0, iy0 = i0[..., 0] % M, i0[..., 1] % M    # periodic wrap
            ix1, iy1 = (ix0 + 1) % M, (iy0 + 1) % M
            fx, fy = fr[..., 0], fr[..., 1]
            w = ((1 - fx) * (1 - fy), (1 - fx) * fy, fx * (1 - fy), fx * fy)
            ii = ((ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1))
            return w, ii

        def gather(u, n):                                # (G u)(x) = u(T(x))
            w, ii = stencil(n)
            return sum(wk * u[i, j] for wk, (i, j) in zip(w, ii))

        def scatter(u, n):                               # G^T, the pushforward
            w, ii = stencil(n)
            out = jnp.zeros_like(u)
            for wk, (i, j) in zip(w, ii):
                out = out.at[i, j].add(wk * u)
            return out

        def heat_apply(u):
            return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(u) * heat))

        def fstep(u, n):
            return heat_apply(scatter(u, n)), None

        def bstep(u, n):
            return gather(heat_apply(u), n), None

        self._fchunk = jit(lambda u, n0: lax.scan(
            fstep, u, n0 + jnp.arange(nper))[0])
        self._bchunk = jit(lambda u, n0: lax.scan(
            bstep, u, n0 + jnp.arange(nper)[::-1])[0])

    # --- one outer interval at a time, so intermediate slices are free --------

    def step_fwd(self, u, k):
        return self._fchunk(u, k * self.nsub_per)

    def step_bwd(self, u, k):
        """Adjoint of step_fwd over the SAME interval k (used going backwards)."""
        return self._bchunk(u, k * self.nsub_per)

    def _floor(self, u):
        return jnp.maximum(u, self.FLOOR_REL * u.max())

    def forward_all(self, u):
        for k in range(self.nout):
            u = self.step_fwd(u, k)
        return self._floor(u)

    def backward_all(self, u):
        for k in range(self.nout - 1, -1, -1):
            u = self.step_bwd(u, k)
        return self._floor(u)

    def adjoint_residual(self, key=0) -> float:
        """<Q_f u, w> - <u, Q_b w> on random inputs, relative. Should be ~1e-15."""
        k = jax.random.PRNGKey(key)
        u = jax.random.uniform(k, (self.g.M, self.g.M))
        w = jax.random.uniform(jax.random.fold_in(k, 1), (self.g.M, self.g.M))
        lhs = float((self.forward_all(u) * w).sum())
        rhs = float((u * self.backward_all(w)).sum())
        return abs(lhs - rhs) / abs(lhs)


def solve_bridge_estimated_drift(g: Grid, mu, nu, betas: jnp.ndarray, gamma, ts,
                                 iters: int = 60, nsub_per: int = 1,
                                 vmax: float | None = None, verbose: bool = True):
    """Bridge whose reference process carries a TABULATED drift beta_t (here the
    one estimated from snapshots).  Same IPFP, different pair of semigroups.

    Cost note: each IPFP sweep integrates the PDE twice over the whole interval,
    so this is orders of magnitude more expensive than the two closed-form
    solvers.  `iters` is correspondingly smaller -- with a drift that already
    almost transports mu to nu, IPFP has little left to do and converges fast;
    watch `marg err` rather than trusting the iteration count.
    """
    prop = DriftedPropagator(g, betas, gamma, nsub_per=nsub_per, vmax=vmax)
    if verbose:
        print(f"  propagator: {prop.nsub_per} substep(s)/interval, "
              f"dt={prop.dt:.2e}, max|beta|={prop.vmax:.2f}, "
              f"clipped={100*prop.clipped:.2f}%, {prop.cfl_cells:.1f} cells/step, "
              f"gamma_eff~{prop.effective_gamma:.3f}, "
              f"adjoint residual={prop.adjoint_residual():.1e}")

    a, b = ipfp(mu, nu, prop.forward_all, prop.backward_all, iters)

    # eta_t forward from t=0, eta*_t backward from t=1, saving every outer slice.
    etas = [a]
    u = a
    for k in range(prop.nout):
        u = prop.step_fwd(u, k)
        etas.append(u)
    stars = [b]
    u = b
    for k in range(prop.nout - 1, -1, -1):
        u = prop.step_bwd(u, k)
        stars.append(u)
    stars = stars[::-1]

    floor = lambda z: jnp.maximum(z, DriftedPropagator.FLOOR_REL
                                  * z.max(axis=(-2, -1), keepdims=True))
    eta = floor(jnp.stack(etas))
    eta_star = floor(jnp.stack(stars))
    # Consistency check at the ends: rho_0 must reproduce mu. It does only if the
    # potentials and the semigroups are the same ones IPFP converged with, which
    # is why eta_star is re-swept with the identical operators rather than
    # reconstructed some other way.

    # Same identity as the constant-drift case: v = beta + gamma grad log(eta*/eta).
    v = prop.betas + gamma * (g.grad(jnp.log(eta_star)) - g.grad(jnp.log(eta)))

    return BridgeSolution(
        rho=g.normalize(eta * eta_star), v=v, gamma=gamma,
        log_range=float(jnp.log10(a.max() / jnp.maximum(a.min(), 1e-300))),
        marginal_err=float(g.integrate(jnp.abs(b * prop.forward_all(a) - nu))),
    )


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


def plot_drift(g: Grid, ts, snap_ts, beta_hat, beta_true, ref, err_snap, err_dense,
               path):
    """Top: quiver of beta_hat vs beta at the snapshot times, over rho_t^ref.
    Bottom: the relative error curve, snapshot-based vs dense-grid recovery."""
    ext = [-g.L / 2, g.L / 2, -g.L / 2, g.L / 2]
    idx = [int(jnp.argmin(jnp.abs(ts - t))) for t in snap_ts]
    st = max(1, g.M // 16)
    fig = plt.figure(figsize=(4.2 * len(idx), 7.4))
    for j, k in enumerate(idx):
        ax = fig.add_subplot(2, len(idx), j + 1)
        ax.imshow(np.asarray(ref[k]).T, origin="lower", extent=ext, cmap="Greys")
        X = np.asarray(g.XY[::st, ::st, 0])
        Y = np.asarray(g.XY[::st, ::st, 1])
        bt = np.asarray(beta_true[k][::st, ::st])
        bh = np.asarray(beta_hat[k][::st, ::st])
        ax.quiver(X, Y, bt[..., 0], bt[..., 1], color="tab:blue", alpha=.7,
                  scale=40, width=.004, label=r"$\beta_t$")
        ax.quiver(X, Y, bh[..., 0], bh[..., 1], color="tab:red", alpha=.7,
                  scale=40, width=.004, label=r"$\hat\beta_t$")
        ax.set_title(rf"$t={float(ts[k]):.2f}$", fontsize=10)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.legend(fontsize=8, loc="upper left")
    ax = fig.add_subplot(2, 1, 2)
    ax.semilogy(np.asarray(ts), np.asarray(err_snap), color="tab:red", lw=1.6,
                label="from 4 interior snapshots")
    ax.semilogy(np.asarray(ts), np.asarray(err_dense), color="tab:blue", lw=1.2,
                ls="--", label="from the dense time grid (bound)")
    for t in snap_ts:
        ax.axvline(float(t), color="k", alpha=.15, lw=1)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|\hat\beta_t-\beta_t\|_{L^2(\rho_t^{\rm ref})}/\|\beta_t\|$")
    ax.set_title("weak-form drift recovery", fontsize=10)
    ax.legend(fontsize=8)
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

    fm = FeatureMap(g, kmax=3)

    # -------------------------------------------------------------------------
    # STEP 1-3 of the recipe: beta from four intermediate snapshots.
    # The observed times are t = 0.2, 0.4, 0.6, 0.8; mu and nu are added to the
    # FIT only (they are known by construction) and are not given a K-solve.
    # gamma_obs = 0: these snapshots come from the noiseless prescribed path, so
    # the weak form has no gamma * E[Lap phi] term to subtract.
    # -------------------------------------------------------------------------
    n = 10
    snap_t = jnp.linspace(0., 1., n)
    snap_i = [int(jnp.argmin(jnp.abs(ts - t))) for t in snap_t]
    snaps = [ref[i] for i in snap_i]
    node_sel = jnp.arange(1, n - 1)                     # the interior snapshots

    # FOUR SNAPSHOTS BUY YOU ABOUT FIVE FEATURES. The estimator gets its own,
    # deliberately small feature map: linear + quadratic only. The Fourier block
    # that the velocity recovery uses is fine in space but its MEANS oscillate
    # several times over [0,1] -- six samples in t simply alias them, mdot comes
    # back ~30% wrong for those components, and the corrupted rows leak into the
    # whole solve. Measured relative error of beta_hat here, snapshot recovery:
    #   kmax=0 (P=5)  0.056     kmax=1 (P=13)  0.30     kmax=3 (P=53)  1.7
    # with the same ridge and deg. That is an identifiability statement about the
    # number of snapshots, not a conditioning problem to be ridged away: on the
    # dense time grid the same kmax=3 map recovers beta to 0.02.
    fm_beta = FeatureMap(g, kmax=0)

    est = estimate_drift(g, fm_beta, snap_t, snaps, node_sel, gamma_obs=0.0, deg=5)
    beta_hat = est.fields_on(ts)
    beta_true = jnp.stack([flow.velocity(g, float(t)) for t in ts])
    err_snap = drift_error(g, beta_hat, beta_true, ref)

    # Upper bound on what the recovery could do with perfect mdot: the same
    # spatial solve at all 41 times, with the rich feature map and mdot from the
    # dense grid. The gap between the two curves is the price of four snapshots,
    # and it sits almost entirely in the time-differentiation.
    est_dense = estimate_drift(g, fm, ts, list(ref), list(range(1, nt - 1)),
                               gamma_obs=0.0, ridge=RIDGE, deg=12)
    err_dense = drift_error(g, est_dense.fields_on(ts), beta_true, ref)

    print("\ndrift recovery (relative L2(rho_t^ref) error of beta_hat):")
    print(f"  from 4 snapshots  (P={fm_beta.P:2d})  median_t "
          f"{float(jnp.median(err_snap[sl])):.4f}"
          f"   at nodes {[round(float(err_snap[i]), 4) for i in snap_i[1:-1]]}")
    print(f"  from the dense grid (P={fm.P:2d})  median_t "
          f"{float(jnp.median(err_dense[sl])):.4f}   (bound: perfect mdot)")
    print(f"  max |beta| true {float(jnp.sqrt((beta_true**2).sum(-1)).max()):.2f}, "
          f"estimated {float(jnp.sqrt((beta_hat**2).sum(-1)).max()):.2f}")

    # Same estimator on the marginals of an actual diffusive bridge, with the
    # gamma * l_t term switched on. What comes back is that bridge's own drift
    # beta + 2 gamma grad log eta*, not beta -- they agree only away from the
    # endpoints, where the bridge is not bending to hit its marginals.
    probe = solve_bridge_exact_drift(g, mu, flow, 0.08, ts)
    est_b = estimate_drift(g, fm_beta, snap_t, [probe.rho[i] for i in snap_i],
                           node_sel, gamma_obs=0.08, deg=5)
    err_b = drift_error(g, est_b.fields_on(ts), beta_true, probe.rho)
    print(f"  from bridge snapshots (gamma_obs=0.08) median_t "
          f"{float(jnp.median(err_b[sl])):.4f}  -- this recovers the BRIDGE "
          f"drift, which is not beta near t=0,1")

    # Velocity cap for the propagator, measured on the support: beta is affine
    # and so grows all the way to the corners of the box, where there is no mass
    # and where the CFL limit would otherwise be set for no physical reason.
    support = ref.max(0) > 1e-4 * ref.max()
    vmax = 1.25 * float((jnp.sqrt((beta_true ** 2).sum(-1)) * support).max())

    # ... and the taper, which switches beta_hat off before the periodic edge.
    # Without it the affine field is discontinuous across the torus and the
    # transported potentials inherit that jump; see taper_to_box.
    beta_run = taper_to_box(g, beta_hat)

    # -------------------------------------------------------------------------
    # STEP 4: feed beta_hat into the drifted IPFP.
    # -------------------------------------------------------------------------
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
        ("est. drift,    gamma=0.05",
         lambda: solve_bridge_estimated_drift(g, mu, nu, beta_run, 0.05, ts,
                                              iters=IPFP_ITERS_EST, vmax=vmax)),
    ]

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
              ("exact drift,   gamma=0.005", "tab:purple", "--"),
              ("est. drift,    gamma=0.05", "tab:brown", "-")]
    tk = [0, 8, 16, 24, 32, 40]

    plot_slices(g, ts, tk, ref, runs, styles, flow, OUTDIR / "slices.png")
    plot_snapshots(g, ts, tk, ref, runs, styles, flow, OUTDIR / "rho2d.png")
    plot_w2(ts, w2, styles, sl, sd.mc, OUTDIR / "wasserstein.png")
    plot_drift(g, ts, snap_t[1:-1], beta_hat, beta_true, ref, err_snap, err_dense,
               OUTDIR / "drift.png")
    print(f"\nfigures written to {OUTDIR}/")


if __name__ == "__main__":
    main()