# Regression tests for the coordinate frame used by the regularised S-FISTA subproblem
# (dfols/trust_region.py::ctrsbox_sfista).
#
# With scaling_changes set, h and prox_uh act on the original (unscaled) coordinates while
# the iterate d lives in the internally scaled coordinates.  The smoothed gradient must
# therefore carry the chain-rule factor scale, and the gradient's Lipschitz constant is
# ||H||_2 + max(scale**2)/u.
#
# TestScaledGradient and TestScaledStepSize fail on 89ac51b and pass on the patched code;
# TestScaledStepSize also fails if only the gradient formula is fixed.
# TestScaledGradientUnscaledControl passes on every tree and guards against over-correcting
#   the unscaled path.
from __future__ import absolute_import, division, print_function, unicode_literals

import math
import unittest

import numpy as np

from dfols.trust_region import ctrsbox_sfista
from dfols.util import remove_scaling


def prox_l1_scaled(lam):
    return lambda x, u: np.sign(x) * np.maximum(np.abs(x) - lam * u, 0.0)


def moreau_l1(z, u, lam):
    """h_u(z) for h(z) = lam*||z||_1 (Moreau envelope)."""
    p = np.sign(z) * np.maximum(np.abs(z) - lam * u, 0.0)
    return lam * np.abs(p).sum() + np.sum((z - p) ** 2) / (2.0 * u)


def smooth_model(g, H, d, xopt, u, lam, scaling_changes=None):
    """F_u(d) = g'd + 0.5 d'Hd + h_u(remove_scaling(xopt + d))."""
    z = remove_scaling(xopt + d, scaling_changes)
    return float(g @ d + 0.5 * d @ H @ d + moreau_l1(z, u, lam))


def sfista_iters(delta, L_h, k_H, func_tol, max_iters, scale=2.0):
    it = math.ceil(scale * delta * (L_h + math.sqrt(L_h * L_h + 2.0 * k_H * func_tol)) / func_tol)
    return min(it, max_iters)


class TestScaledGradient(unittest.TestCase):
    """gnew must be the chain-rule gradient of F_u when scaling_changes is set."""

    def runTest(self):
        shift = np.array([1.0, -2.0])
        scale = np.array([2.0, 0.5])
        xopt = np.array([0.5, -0.3])
        g = np.array([-0.7, 0.4])
        H = np.array([[0.3, 0.1], [0.1, -0.2]])
        lam = 1.0
        L_h = lam * math.sqrt(2.0)
        delta = 1.0
        h = lambda x: lam * np.linalg.norm(x, 1)
        prox = prox_l1_scaled(lam)
        d, gnew, _ = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox,
                                    func_tol=1e-3, max_iters=500, sfista_iters_scale=2.0,
                                    scaling_changes=(shift, scale))
        N = sfista_iters(delta, L_h, np.linalg.norm(H, 2), 1e-3, 500)
        u = 2.0 * delta / (N * L_h)
        v = np.array([0.6, -0.8])
        eps = 1e-6
        fd = (smooth_model(g, H, d + eps * v, xopt, u, lam, (shift, scale)) -
              smooth_model(g, H, d - eps * v, xopt, u, lam, (shift, scale))) / (2 * eps)
        self.assertAlmostEqual(fd, float(gnew @ v), delta=2e-4,
                               msg="gnew ignores the chain-rule factor dx_raw/dx_scaled")


class TestScaledGradientUnscaledControl(unittest.TestCase):
    """Control: the same check without scaling must pass on the unpatched code."""

    def runTest(self):
        xopt = np.array([0.5, -0.3])
        g = np.array([-0.7, 0.4])
        H = np.array([[0.3, 0.1], [0.1, -0.2]])
        lam = 1.0
        L_h = lam * math.sqrt(2.0)
        delta = 1.0
        h = lambda x: lam * np.linalg.norm(x, 1)
        prox = prox_l1_scaled(lam)
        d, gnew, _ = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox,
                                    func_tol=1e-3, max_iters=500, sfista_iters_scale=2.0)
        N = sfista_iters(delta, L_h, np.linalg.norm(H, 2), 1e-3, 500)
        u = 2.0 * delta / (N * L_h)
        v = np.array([0.6, -0.8])
        eps = 1e-6
        fd = (smooth_model(g, H, d + eps * v, xopt, u, lam) -
              smooth_model(g, H, d - eps * v, xopt, u, lam)) / (2 * eps)
        self.assertAlmostEqual(fd, float(gnew @ v), delta=2e-4)


class TestScaledStepSize(unittest.TestCase):
    """A single S-FISTA step from d=0 must respect the scaled Lipschitz constant.

    With max_iters=1 the returned step is d_1 = proj(-grad F_u(0)/l).  The projection is
    non-expansive and fixes 0, and here it does not clip, so ||d_1|| * l == ||grad F_u(0)||.
    Using l = ||H||+1/u instead of ||H||+max(scale**2)/u makes the step max(scale**2) times
    too long and violates the bound; the ball then clips, so ||d_1|| * l > ||grad F_u(0)||.
    This test therefore fails if only the gradient formula is fixed.
    """

    def runTest(self):
        shift = np.array([0.0, 0.0])
        scale = np.array([20.0, 0.1])
        xopt = np.array([0.3, 0.06])
        g = np.array([-1.0, 0.5])
        H = np.zeros((2, 2))
        lam = 1.0
        L_h = lam * math.sqrt(2.0)
        delta = 1.0
        h = lambda x: lam * np.linalg.norm(x, 1)
        prox = prox_l1_scaled(lam)
        d, gnew, _ = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox,
                                    func_tol=1e-3, max_iters=1, sfista_iters_scale=2.0,
                                    scaling_changes=(shift, scale))
        u = 2.0 * delta / (1 * L_h)
        z0 = remove_scaling(xopt, (shift, scale))
        grad0 = g + scale * (z0 - prox(z0, u)) / u
        L = np.linalg.norm(H, 2) + np.max(scale ** 2) / u
        lhs = np.linalg.norm(d) * L
        rhs = np.linalg.norm(grad0)
        self.assertLessEqual(lhs, rhs * (1.0 + 1e-12),
                             "step is longer than grad/L allows: ||d||*L=%r > ||grad||=%r"
                             % (lhs, rhs))
