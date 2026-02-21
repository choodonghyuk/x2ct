"""
Unit tests for X-ray Beer-Lambert rendering and dataset adapter.

Tests:
  1. Beer-Lambert law with uniform attenuation
  2. Gradient flow through batch_composite_rays_xray
  3. Variable attenuation (linearly increasing mu)
  4. Comparison with NAF raw2outputs
  5. XrayDataset angle2pose consistency
  6. Multi-scene batching
"""

import sys
import os
import math
import torch
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.models.decoders.base_volume_renderer import batch_composite_rays_xray
from lib.datasets.xray_dataset import XrayDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uniform_inputs(n_rays, n_samples, mu_value, t_start=0.0, t_end=1.0):
    """Create uniform attenuation test inputs for batch_composite_rays_xray.

    Uses CUDA ray marcher ts format: ts[:, 0] = t (position), ts[:, 1] = dt (step size).

    Returns:
        sigmas: (M,)  flat attenuation coefficients
        ts:     list[(M_i, 2)]  CUDA format [t_position, dt] per scene
        rays:   list[(N_i, 2)]  [point_offset, point_count] per scene
        num_points: list[int]
    """
    M = n_rays * n_samples
    sigmas = torch.ones(M) * mu_value

    # Evenly spaced intervals along [t_start, t_end]
    t_edges = torch.linspace(t_start, t_end, n_samples + 1)
    t_starts = t_edges[:-1]
    t_ends = t_edges[1:]
    dt = t_ends - t_starts  # uniform step size
    # CUDA format: [t_position (end of interval), dt (step size)]
    ts_single = torch.stack([t_ends, dt], dim=-1)  # (n_samples, 2)
    ts_repeated = ts_single.repeat(n_rays, 1)       # (M, 2)

    # rays: each ray owns n_samples contiguous points
    offsets = torch.arange(n_rays) * n_samples
    counts = torch.full((n_rays,), n_samples, dtype=torch.long)
    rays_single = torch.stack([offsets, counts], dim=-1)  # (n_rays, 2)

    return sigmas, [ts_repeated], [rays_single], [M]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBeerLambert:

    def test_uniform_attenuation(self):
        """Uniform μ=0.5 over [0,1] → projection = μ * length = 0.5."""
        n_rays, n_samples, mu_val = 10, 100, 0.5
        sigmas, ts, rays, np_ = _make_uniform_inputs(n_rays, n_samples, mu_val)
        _, _, _, image = batch_composite_rays_xray(sigmas, ts, rays, np_)

        expected = mu_val * 1.0  # ∫μ dx from 0 to 1
        proj = image.squeeze()   # (n_rays,)
        assert proj.shape == (n_rays,), f"Unexpected shape {proj.shape}"
        assert torch.allclose(proj, torch.tensor(expected), atol=1e-4), \
            f"Expected {expected}, got {proj[0].item():.6f}"

    def test_zero_attenuation(self):
        """μ=0 everywhere → projection = 0."""
        n_rays, n_samples = 5, 64
        sigmas, ts, rays, np_ = _make_uniform_inputs(n_rays, n_samples, 0.0)
        _, _, _, image = batch_composite_rays_xray(sigmas, ts, rays, np_)
        assert torch.allclose(image, torch.zeros_like(image), atol=1e-8)

    def test_varying_attenuation(self):
        """Linearly increasing μ(x)=x over [0,1] → ∫x dx = 0.5."""
        n_rays, n_samples = 5, 200
        M = n_rays * n_samples

        # Evenly spaced intervals
        t_edges = torch.linspace(0, 1, n_samples + 1)
        t_starts = t_edges[:-1]
        t_ends = t_edges[1:]
        dt = t_ends - t_starts  # step sizes
        t_mids = (t_starts + t_ends) / 2  # sample positions

        # μ evaluated at midpoints
        mu_per_sample = t_mids.repeat(n_rays)  # (M,)
        # CUDA format: [t_position, dt]
        ts_single = torch.stack([t_ends, dt], dim=-1).repeat(n_rays, 1)

        offsets = torch.arange(n_rays) * n_samples
        counts = torch.full((n_rays,), n_samples, dtype=torch.long)
        rays_single = torch.stack([offsets, counts], dim=-1)

        _, _, _, image = batch_composite_rays_xray(
            mu_per_sample, [ts_single], [rays_single], [M])
        proj = image.squeeze()
        expected = 0.5

        assert torch.allclose(proj, torch.tensor(expected), atol=2e-2), \
            f"Expected ~{expected}, got {proj[0].item():.4f}"


class TestGradient:

    def test_gradient_flow(self):
        """Gradient must propagate through batch_composite_rays_xray."""
        n_rays, n_samples = 4, 50
        M = n_rays * n_samples

        sigmas = torch.ones(M, requires_grad=True) * 0.3

        t_edges = torch.linspace(0, 1, n_samples + 1)
        dt = t_edges[1:] - t_edges[:-1]  # step sizes
        # CUDA format: [t_position, dt]
        ts_single = torch.stack([t_edges[1:], dt], dim=-1).repeat(n_rays, 1)
        offsets = torch.arange(n_rays) * n_samples
        counts = torch.full((n_rays,), n_samples, dtype=torch.long)
        rays_single = torch.stack([offsets, counts], dim=-1)

        _, _, _, image = batch_composite_rays_xray(
            sigmas, [ts_single], [rays_single], [M])

        loss = image.sum()
        loss.backward()

        assert sigmas.grad is not None, "Gradient is None"
        assert not torch.isnan(sigmas.grad).any(), "Gradient contains NaN"
        assert (sigmas.grad > 0).all(), "All gradients should be positive for sum loss"

    def test_gradient_magnitude(self):
        """Gradient of each sigma should equal its dt contribution."""
        n_rays, n_samples = 1, 10
        M = n_samples

        sigmas = torch.ones(M, requires_grad=True) * 0.5

        t_edges = torch.linspace(0, 2, n_samples + 1)  # total length = 2
        dt = (t_edges[1] - t_edges[0]).item()
        # CUDA format: [t_position, dt]
        ts_single = torch.stack([t_edges[1:], t_edges[1:] - t_edges[:-1]], dim=-1)

        offsets = torch.tensor([0])
        counts = torch.tensor([n_samples])
        rays_single = torch.stack([offsets, counts], dim=-1)

        _, _, _, image = batch_composite_rays_xray(
            sigmas, [ts_single], [rays_single], [M])

        loss = image.sum()
        loss.backward()

        # d(projection)/d(sigma_i) = dt_i
        expected_grad = dt
        assert torch.allclose(sigmas.grad, torch.tensor(expected_grad), atol=1e-5), \
            f"Expected grad={expected_grad}, got {sigmas.grad[0].item():.6f}"


class TestMultiScene:

    def test_two_scenes_equal(self):
        """Two identical scenes should produce identical projections."""
        n_rays, n_samples, mu = 8, 32, 0.4
        M_single = n_rays * n_samples

        # Scene data — CUDA format: [t_position, dt]
        t_edges = torch.linspace(0, 1, n_samples + 1)
        dt = t_edges[1:] - t_edges[:-1]
        ts_single = torch.stack([t_edges[1:], dt], dim=-1).repeat(n_rays, 1)
        offsets = torch.arange(n_rays) * n_samples
        counts = torch.full((n_rays,), n_samples, dtype=torch.long)
        rays_single = torch.stack([offsets, counts], dim=-1)

        # Duplicate for 2 scenes
        sigmas = torch.ones(M_single * 2) * mu
        ts = [ts_single, ts_single.clone()]
        rays = [rays_single, rays_single.clone()]
        np_ = [M_single, M_single]

        _, _, _, image = batch_composite_rays_xray(sigmas, ts, rays, np_)

        # image should be (2, n_rays, 1)
        assert image.shape == (2, n_rays, 1)
        assert torch.allclose(image[0], image[1], atol=1e-6)


class TestOutputFormat:

    def test_return_shapes(self):
        """Verify shapes match batch_composite_rays_train interface."""
        n_rays, n_samples = 16, 64
        sigmas, ts, rays, np_ = _make_uniform_inputs(n_rays, n_samples, 1.0)
        weights, weights_sum, depth, image = batch_composite_rays_xray(
            sigmas, ts, rays, np_)

        assert weights is None
        assert weights_sum.shape == (1, n_rays)
        assert depth.shape == (1, n_rays)
        assert image.shape == (1, n_rays, 1)

    def test_weights_sum_and_depth_are_zero(self):
        """weights_sum and depth should be zeros (unused in X-ray mode)."""
        n_rays, n_samples = 8, 32
        sigmas, ts, rays, np_ = _make_uniform_inputs(n_rays, n_samples, 0.7)
        _, weights_sum, depth, _ = batch_composite_rays_xray(
            sigmas, ts, rays, np_)
        assert (weights_sum == 0).all()
        assert (depth == 0).all()


class TestCompareWithNAF:

    def test_naf_equivalence(self):
        """Compare with NAF raw2outputs Beer-Lambert computation."""
        n_rays, n_samples = 20, 64

        # Random attenuation
        torch.manual_seed(42)
        raw = torch.rand(n_rays, n_samples, 1)  # NAF format: (N, S, 1)

        # Uniform z_vals in [0, 1]
        z_vals = torch.linspace(0, 1, n_samples).unsqueeze(0).expand(n_rays, -1)

        # rays_d along z-axis (unit length, so dists scaling = 1)
        rays_d = torch.zeros(n_rays, 3)
        rays_d[:, 2] = 1.0

        # --- NAF computation (inlined from raw2outputs) ---
        dists = z_vals[:, 1:] - z_vals[:, :-1]  # (N, S-1)
        dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e-10)], dim=-1)  # (N, S)
        dists = dists * torch.norm(rays_d[:, None, :], dim=-1)  # (N, S)
        naf_acc = torch.sum(raw[..., 0] * dists, dim=-1)  # (N,)

        # --- Our computation ---
        # Build ts in CUDA format [t_position, dt] from z_vals
        # To match NAF: last interval has width 1e-10
        z_ends = torch.cat([z_vals[:, 1:], z_vals[:, -1:] + 1e-10], dim=-1)
        # dt = z_ends - z_starts
        dt_vals = z_ends - z_vals
        # CUDA format: [t_position (= z_end), dt]
        ts_flat = torch.stack([z_ends.reshape(-1), dt_vals.reshape(-1)], dim=-1)  # (M, 2)

        sigmas_flat = raw[..., 0].reshape(-1)  # (M,)

        offsets = torch.arange(n_rays) * n_samples
        counts = torch.full((n_rays,), n_samples, dtype=torch.long)
        rays_arr = torch.stack([offsets, counts], dim=-1)

        _, _, _, our_image = batch_composite_rays_xray(
            sigmas_flat, [ts_flat], [rays_arr], [n_rays * n_samples])
        our_proj = our_image.squeeze()  # (n_rays,)

        assert torch.allclose(naf_acc, our_proj, atol=1e-5), \
            f"Max diff: {(naf_acc - our_proj).abs().max().item():.8f}"


class TestXrayDataset:

    def test_angle2pose_rotation_properties(self):
        """Rotation submatrix should be orthogonal (det=1)."""
        from lib.datasets.xray_dataset import XrayDataset

        DSO = 1.0  # 1 meter
        for angle in [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            pose = XrayDataset._angle2pose(DSO, angle)
            R = pose[:3, :3]
            # Orthogonality: R @ R^T = I
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), \
                f"R not orthogonal at angle={angle}"
            # det(R) = 1 (proper rotation)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), \
                f"det(R)={np.linalg.det(R)} at angle={angle}"

    def test_angle2pose_translation_distance(self):
        """Translation vector magnitude should equal DSO."""
        DSO = 0.575  # typical CT distance
        for angle in np.linspace(0, 2 * np.pi, 12):
            pose = XrayDataset._angle2pose(DSO, angle)
            t = pose[:3, 3]
            dist = np.linalg.norm(t)
            assert np.isclose(dist, DSO, atol=1e-6), \
                f"||t||={dist:.6f} != DSO={DSO} at angle={angle}"

    def test_angle2pose_zero_angle(self):
        """At angle=0, source should be at (DSO, 0, 0)."""
        DSO = 1.0
        pose = XrayDataset._angle2pose(DSO, 0.0)
        t = pose[:3, 3]
        assert np.isclose(t[0], DSO, atol=1e-6), f"t[0]={t[0]}, expected {DSO}"
        assert np.isclose(t[1], 0.0, atol=1e-6), f"t[1]={t[1]}, expected 0"
        assert np.isclose(t[2], 0.0, atol=1e-6), f"t[2]={t[2]}, expected 0"

    def test_angle2pose_quarter_turn(self):
        """At angle=π/2, source should be at (0, DSO, 0)."""
        DSO = 1.0
        pose = XrayDataset._angle2pose(DSO, np.pi / 2)
        t = pose[:3, 3]
        assert np.isclose(t[0], 0.0, atol=1e-6)
        assert np.isclose(t[1], DSO, atol=1e-6)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Beer-Lambert X-ray Rendering Tests")
    print("=" * 60)

    # Beer-Lambert
    bl = TestBeerLambert()
    bl.test_uniform_attenuation()
    print("✓ test_uniform_attenuation")
    bl.test_zero_attenuation()
    print("✓ test_zero_attenuation")
    bl.test_varying_attenuation()
    print("✓ test_varying_attenuation")

    # Gradient
    gr = TestGradient()
    gr.test_gradient_flow()
    print("✓ test_gradient_flow")
    gr.test_gradient_magnitude()
    print("✓ test_gradient_magnitude")

    # Multi-scene
    ms = TestMultiScene()
    ms.test_two_scenes_equal()
    print("✓ test_two_scenes_equal")

    # Output format
    of = TestOutputFormat()
    of.test_return_shapes()
    print("✓ test_return_shapes")
    of.test_weights_sum_and_depth_are_zero()
    print("✓ test_weights_sum_and_depth_are_zero")

    # NAF comparison
    nc = TestCompareWithNAF()
    nc.test_naf_equivalence()
    print("✓ test_naf_equivalence")

    # Dataset
    ds = TestXrayDataset()
    ds.test_angle2pose_rotation_properties()
    print("✓ test_angle2pose_rotation_properties")
    ds.test_angle2pose_translation_distance()
    print("✓ test_angle2pose_translation_distance")
    ds.test_angle2pose_zero_angle()
    print("✓ test_angle2pose_zero_angle")
    ds.test_angle2pose_quarter_turn()
    print("✓ test_angle2pose_quarter_turn")

    print()
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
