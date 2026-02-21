import os
import sys
import numpy as np
from torch.utils.data import Dataset

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


class ConeGeometry:
    """
    Cone beam CT geometry. Converts millimeters to meters internally.
    Adapted from naf_cbct/src/dataset/tigre.py
    """

    def __init__(self, data):
        self.DSD = data["DSD"] / 1000  # Distance Source Detector (m)
        self.DSO = data["DSO"] / 1000  # Distance Source Origin   (m)
        self.nDetector = np.array(data["nDetector"])  # [W, H] pixels
        self.dDetector = np.array(data["dDetector"]) / 1000  # pixel size (m)
        self.sDetector = self.nDetector * self.dDetector  # detector size (m)
        self.nVoxel = np.array(data["nVoxel"])  # [X, Y, Z] voxels
        self.dVoxel = np.array(data["dVoxel"]) / 1000  # voxel size (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # volume size (m)
        self.offOrigin = np.array(data["offOrigin"]) / 1000  # (m)
        self.offDetector = np.array(data["offDetector"]) / 1000  # (m)
        self.accuracy = data["accuracy"]
        self.mode = data["mode"]
        self.filter = data.get("filter", None)


class XrayDataset(Dataset):
    """
    X-ray / CBCT dataset adapter for ZeroRF.

    Loads NAF-format pickle data and converts cone-beam geometry
    to ZeroRF-compatible poses and intrinsics.

    GT Projection Format:
        TIGRE's ``tigre.Ax()`` computes forward projections as **line integrals**
        of the attenuation field: projection = ∫ μ(x) dx.
        The network output (Beer-Lambert sum Σ μ_i · Δt_i) is also a line integral,
        so GT and prediction are in the same space. No ``-log(I/I_0)`` conversion
        is needed.

        Note: The CT volume in the pickle is typically already converted from HU
        to attenuation and normalized to [0, 1] by NAF's ``generateData.py``.

    Output format per item:
        cond_imgs:       (N, H, W, 1)   X-ray projections (line integrals)
        cond_poses:      (N, 4, 4)      Camera-to-world pose matrices
        cond_intrinsics: (N, 4)         [fx, fy, cx, cy] per view
    """

    def __init__(self, path, split='train'):
        super().__init__()

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.geo = ConeGeometry(data)
        geo = self.geo

        # --- Auto-detect view count and resolution ---
        if split == 'train':
            projs = data["train"]["projections"]  # (N, H, W)
            angles = data["train"]["angles"]
        elif split == 'val':
            projs = data["val"]["projections"]  # (N, H, W)
            angles = data["val"]["angles"]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.n_views = len(angles)
        # projections shape: (N, H, W) from TIGRE
        self.image_h = projs.shape[1]
        self.image_w = projs.shape[2]

        # --- Compute world_scale so volume fits within [-1, 1]^3 ---
        self.world_scale = 1.8 / np.max(geo.sVoxel)
        


        # --- Compute intrinsics from cone-beam geometry ---
        # Mapping: pinhole focal length = DSD / dDetector (in pixels)
        W, H = geo.nDetector  # W = nDetector[0], H = nDetector[1]
        fx = geo.DSD / geo.dDetector[0]
        fy = geo.DSD / geo.dDetector[1]
        cx = W / 2.0 - geo.offDetector[0] / geo.dDetector[0]
        cy = H / 2.0 - geo.offDetector[1] / geo.dDetector[1]

        # --- Compute poses (4x4 camera-to-world matrices) ---
        poses = []
        for angle in angles:
            pose = self._angle2pose(geo.DSO, angle)
            # Scale translation by world_scale (rotation stays the same)
            pose[:3, 3] *= self.world_scale
            poses.append(pose)
        poses = np.array(poses, dtype=np.float32)  # (N, 4, 4)

        # --- Store data ---
        # Projections: add channel dim -> (N, H, W, 1)
        self.cond_imgs = np.array(projs, dtype=np.float32)[..., np.newaxis]
        self.cond_poses = poses
        self.cond_intrinsics = np.tile(
            np.array([fx, fy, cx, cy], dtype=np.float32),
            (self.n_views, 1)
        )  # (N, 4)

        # Ground truth 3D volume (for evaluation)
        self.gt_volume = data.get("image", None)

        print(f"[XrayDataset] Loaded {split}: {self.n_views} views, "
              f"{self.image_h}x{self.image_w} resolution, "
              f"world_scale={self.world_scale:.4f}")

    def __len__(self):
        return 1  # single scene

    def __getitem__(self, index):
        return dict(
            cond_imgs=self.cond_imgs,           # (N, H, W, 1)
            cond_poses=self.cond_poses,          # (N, 4, 4)
            cond_intrinsics=self.cond_intrinsics  # (N, 4)
        )

    @staticmethod
    def _angle2pose(DSO, angle):
        """
        Compute camera-to-world pose matrix from gantry angle.
        Adapted from naf_cbct/src/dataset/tigre.py
        """
        phi1 = -np.pi / 2
        R1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi1), -np.sin(phi1)],
            [0.0, np.sin(phi1), np.cos(phi1)],
        ])
        phi2 = np.pi / 2
        R2 = np.array([
            [np.cos(phi2), -np.sin(phi2), 0.0],
            [np.sin(phi2), np.cos(phi2), 0.0],
            [0.0, 0.0, 1.0],
        ])
        R3 = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        rot = np.dot(np.dot(R3, R2), R1)
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0.0])
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rot
        T[:3, 3] = trans
        return T
