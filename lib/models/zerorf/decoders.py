import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...ops import SHEncoder, TruncExp
from ..decoders.base_volume_renderer import PointBasedVolumeRenderer
from .generators import TensorialGenerator, CubemapGenerator
from mmgen.models import MODULES, build_module
from mmcv.cnn import xavier_init, constant_init
import mcubes
import trimesh
import matplotlib.pyplot as plotlib


class DepthRegularizer(nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch, n_images, target_h, target_w, loss_weight=1.0) -> None:
        super().__init__()
        self.noise_ch, self.target_w, self.target_h = noise_ch, target_w, target_h
        self.n_images = n_images
        self.upx = 24
        self.register_buffer("noise", torch.randn(n_images, noise_ch, target_h // self.upx, target_w // self.upx))
        self.nets = nn.ModuleList([build_module(dict(
            type='VAEDecoder',
            in_channels=noise_ch,
            out_channels=8,
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'),
            block_out_channels=(32, 64, 128, 128),
            layers_per_block=0                                                                                                                                                                                                                                                  
        )) for _ in range(n_images)])
        self.net_needs_jit = True
        self.ups = nn.UpsamplingBilinear2d((target_h, target_w))
        self.head = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU(True),
            nn.Linear(32, 1)
        )
        self.crit = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, results):
        sample_inds = results['sample_inds']
        if sample_inds is None:
            return results['depth'].new_zeros(1)
        if self.net_needs_jit:
            for i in range(self.n_images):
                self.nets[i] = torch.jit.trace(self.nets[i], self.noise[i: i + 1])
            self.net_needs_jit = False
        with torch.jit.optimized_execution(False):
            depth_features = torch.cat([self.nets[i](self.noise[i: i + 1]) for i in range(self.n_images)])
        ups_features = self.ups(depth_features).reshape(self.n_images, 8, self.target_h * self.target_w).permute(0, 2, 1).flatten(0, 1)
        ups_features = ups_features[sample_inds]
        depths = self.head(ups_features)
        return self.loss_weight * self.crit(results['depth'].reshape_as(depths), depths)


def positional_encoding(positions, freqs):    
    freq_bands = (2 ** torch.arange(freqs, device=positions.device, dtype=torch.float32))  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(*positions.shape[:-1], freqs * positions.shape[-1])  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class CommonDecoder(nn.Module):
    """
    X-ray Attenuation 전용 디코더.
    시점 의존적인 RGB 및 SH 연산을 제거하고, Softplus를 적용하여 
    의료 영상(CT) 내부의 연속적인 감쇠 계수(Attenuation) 학습에 최적화됨.
    """
    def __init__(self, point_channels, sh_coef_only=False, dir_pe=False, sdf_mode=False, xray_mode=True):
        super().__init__()
        
        # 1. Point Feature Base Network
        self.base_net = nn.Linear(point_channels, 64)
        self.base_activation = nn.SiLU()
        
        # 2. 추가적인 MLP 레이어 정의 (ModuleList 사용)
        # 64 -> 64 크기의 선형 계층 3개를 생성합니다.
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(128, 64)
        self.mlp_activation = nn.ReLU()
        
        # 3. Attenuation (Sigma) Network
        self.density_net = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus(beta=1.0)
        )
        
        # 3. X-ray 태스크에서 불필요한 NVS 관련 분기 완전 제거
        self.dir_encoder = None
        self.dir_net = None
        self.color_net = None
        
        self.init_weights()

    def init_weights(self):
        """남아있는 선형 계층(Linear)에 대한 가중치 초기화만 수행합니다."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, point_code, dirs=None, out_sdf=False):
        """
        Args:
            point_code: [M, point_channels] 형태의 3D Feature 좌표 값
            dirs: 파이프라인 호환용 (사용되지 않음)
            out_sdf: 파이프라인 호환용 (사용되지 않음)
            
        Returns:
            sigmas: [M] 형태의 1채널 X-ray Attenuation 값
            rgbs: 항상 None을 반환
        """
        # Feature 디코딩
        base_x = self.base_net(point_code)
        x = self.base_activation(base_x)

        # Attenuation(mu) 계산
        sigmas = self.density_net(x).squeeze(-1)
        
        # X-ray 렌더링 방식에 맞게 RGB 채널은 None 처리
        rgbs = None
        
        return sigmas, rgbs


@MODULES.register_module()
class TensorialDecoder(PointBasedVolumeRenderer):
    preprocessor: TensorialGenerator

    def __init__(self, *args, preprocessor: dict, separate_density_and_color: bool, subreduce, reduce, n_images, image_h, image_w, sh_coef_only=False, sdf_mode=False, xray_mode=False, visualize_mesh=False, **kwargs):
        super().__init__(*args, preprocessor=preprocessor, xray_mode=xray_mode, **kwargs)
        assert isinstance(self.preprocessor, TensorialGenerator)
        if reduce == 'cat':
            in_chs = self.preprocessor.out_ch * len(self.preprocessor.tensor_config) // subreduce
        else:
            in_chs = self.preprocessor.out_ch
        self.in_chs = in_chs
        self.separate_density_and_color = separate_density_and_color
        self.pe = 5
        if separate_density_and_color:
            self.density_decoder = CommonDecoder(in_chs // 2, sh_coef_only, sdf_mode=sdf_mode, xray_mode=xray_mode)
            self.color_decoder = CommonDecoder(in_chs // 2, sh_coef_only, sdf_mode=sdf_mode, xray_mode=xray_mode)
        else:
            if reduce == 'cat':
                self.common_decoder = CommonDecoder(in_chs, sh_coef_only, sdf_mode=sdf_mode, xray_mode=xray_mode)
            else:
                self.common_decoder = CommonDecoder(in_chs, sh_coef_only, sdf_mode=sdf_mode, xray_mode=xray_mode)
        self.subreduce = subreduce
        self.reduce = reduce
        self.sdf_mode = sdf_mode
        self.visualize_mesh = visualize_mesh
        self.preprocessor_needs_jit = True
        self.preprocessor_configured = self.preprocessor
        # self.bg_gen = CubemapGenerator(self.preprocessor.in_ch, self.preprocessor.out_ch, self.preprocessor.noise_res)
        # self.bg_needs_jit = True
        # self.depth_reg = DepthRegularizer(4, n_images, image_h, image_w, 0.06)

    # def loss(self, results):
    #     return self.depth_reg(results)

    # def forward(self, rays_o, rays_d, code, density_bitfield, grid_size, dt_gamma=0, perturb=False, return_loss=False, update_extra_state=0, extra_args=None, extra_kwargs=None, sample_inds=None):
    #     outputs = super().forward(rays_o, rays_d, code, density_bitfield, grid_size, dt_gamma, perturb, return_loss, update_extra_state, extra_args, extra_kwargs, sample_inds)
    #     assert isinstance(rays_d, torch.Tensor)
    #     if self.bg_needs_jit:
    #         self.bg_gen = torch.jit.trace(self.bg_gen, rays_d)
    #         self.bg_needs_jit = False
    #     with torch.jit.optimized_execution(False):
    #         outputs['bg'] = self.bg_gen(rays_d)
    #     return outputs

    def preproc(self, code):
        with torch.jit.optimized_execution(False):
            return super().preproc(code)
    
    def get_point_code(self, code, xyzs):
        preprocessor = self.preprocessor_configured
        if self.preprocessor_needs_jit:
            self.preprocessor_needs_jit = False
            self.preprocessor = torch.jit.trace(self.preprocessor, self.code_buffer)
        codes = []
        for i, cfg in enumerate(preprocessor.tensor_config):
            start = sum(map(numpy.prod, preprocessor.sub_shapes[:i]))
            end = sum(map(numpy.prod, preprocessor.sub_shapes[:i + 1]))
            got: torch.Tensor = code[..., start: end].reshape(code.shape[0], *preprocessor.sub_shapes[i])
            assert len(cfg) + 2 == got.ndim, [len(cfg), got.ndim]
            if got.ndim == 3:
                got = got.unsqueeze(-1)
                cfg += 'x'
            coords = xyzs[..., ['xyzt'.index(axis) for axis in cfg]]
            coords = coords.reshape(code.shape[0], 1, xyzs.shape[-2], 2)
            codes.append(
                F.grid_sample(got, coords, mode='bilinear', padding_mode='zeros', align_corners=False)
                .reshape(code.shape[0], got.shape[1], xyzs.shape[-2]).transpose(1, 2)
            )
        codes_subred = []
        codes_stage = None
        for i, c in enumerate(codes):
            if codes_stage is None:
                codes_stage = c
            else:
                codes_stage = codes_stage * c
            if i % self.subreduce == self.subreduce - 1:
                codes_subred.append(codes_stage)
                codes_stage = None
        codes = codes_subred
        # if not self.separate_density_and_color and self.reduce == 'cat':
        #     codes.append(positional_encoding(xyzs.reshape(code.shape[0], xyzs.shape[-2], 3), self.pe))
        if self.reduce == 'cat':
            return torch.cat(codes, dim=-1).reshape(code.shape[0] * xyzs.shape[-2], self.in_chs)
        else:
            assert self.reduce == 'sum'
            return sum(codes).reshape(code.shape[0] * xyzs.shape[-2], self.in_chs)

    def point_code_render(self, point_code, dirs):
        if self.separate_density_and_color:
            density_code, color_code = torch.chunk(point_code, 2, -1)
            sigmas, _ = self.density_decoder(density_code, None)
            if dirs is not None:
                _, rgbs = self.color_decoder(color_code, dirs)
            else:
                rgbs = None
            return sigmas, rgbs
        else:
            return self.common_decoder(point_code, dirs)

    @torch.no_grad()
    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        if 'mean' in str(scene_name):
            return
        buffer = self.code_proc_buffer.to(code)
        preprocessor = self.preprocessor_configured
        for i, cfg in enumerate(preprocessor.tensor_config):
            start = sum(map(numpy.prod, preprocessor.sub_shapes[:i]))
            end = sum(map(numpy.prod, preprocessor.sub_shapes[:i + 1]))
            got: torch.Tensor = buffer[..., start: end].reshape(code.shape[0], *preprocessor.sub_shapes[i])
            assert len(cfg) + 2 == got.ndim, [len(cfg), got.ndim]
            if got.ndim == 3:
                got = got.unsqueeze(-1)
            got = got.permute(0, 2, 3, 1).squeeze(0).transpose(-1, -2).flatten(1)  # HWC -> HW'
            got = got.detach().cpu().numpy()
            plotlib.imsave(os.path.join(viz_dir, "features-%s.png" % cfg), got)
        if not self.visualize_mesh:
            return
        basis = torch.linspace(-1, 1, 500, device=code.device, dtype=code.dtype)
        x, y, z = torch.meshgrid(basis, basis, basis, indexing='ij')
        xyz = torch.stack([x, y, z], dim=-1)
        xyz_f = xyz.flatten(0, -2)
        batches = []
        for batch_split in torch.split(xyz_f, 2 ** 17):
            pc = self.get_point_code(buffer, batch_split)
            if self.separate_density_and_color:
                density_code, color_code = torch.chunk(pc, 2, -1)
                sdf = self.density_decoder(density_code, None, True)[0]
            else:
                sdf = self.common_decoder(pc, None, True)[0]
            if self.sdf_mode:
                batches.append(sdf.cpu())
            else:
                batches.append((sdf > 5).cpu())
        batches = torch.cat(batches, 0).reshape(xyz.shape[:-1])
        if not self.sdf_mode:
            field = mcubes.smooth_constrained(batches.numpy())
            verts, tris = mcubes.marching_cubes(field, 0.5)
        else:
            verts, tris = mcubes.marching_cubes(batches.numpy(), 0.0)
        verts = verts / 250 - 1
        # mcubes.smooth()
        trimesh.Trimesh(verts, tris).export(os.path.join(viz_dir, str(scene_name[0]) + ".glb"))


@MODULES.register_module()
class FreqFactorizedDecoder(TensorialDecoder):

    def __init__(self, *args, freq_bands, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_bands = freq_bands

    def get_point_code(self, code, xyzs):
        preprocessor = self.preprocessor_configured
        if self.preprocessor_needs_jit:
            self.preprocessor_needs_jit = False
            self.preprocessor = torch.jit.trace(self.preprocessor, self.code_buffer)
        codes = []
        for i, (cfg, band) in enumerate(zip(preprocessor.tensor_config, self.freq_bands)):
            start = sum(map(numpy.prod, preprocessor.sub_shapes[:i]))
            end = sum(map(numpy.prod, preprocessor.sub_shapes[:i + 1]))
            got: torch.Tensor = code[..., start: end].reshape(code.shape[0], *preprocessor.sub_shapes[i])
            assert len(cfg) + 2 == got.ndim == 5, [len(cfg), got.ndim]
            coords = xyzs[..., ['xyzt'.index(axis) for axis in cfg]]
            if band is not None:
                coords = ((coords % band) / (band / 2) - 1)
            coords = coords.reshape(code.shape[0], 1, 1, xyzs.shape[-2], 3)
            codes.append(
                F.grid_sample(got, coords, mode='bilinear', padding_mode='border', align_corners=False)
                .reshape(code.shape[0], got.shape[1], xyzs.shape[-2]).transpose(1, 2)
            )
        codes_subred = []
        codes_stage = None
        for i, c in enumerate(codes):
            if codes_stage is None:
                codes_stage = c
            else:
                codes_stage = codes_stage * c
            if i % self.subreduce == self.subreduce - 1:
                codes_subred.append(codes_stage)
                codes_stage = None
        codes = codes_subred
        # if not self.separate_density_and_color and self.reduce == 'cat':
        #     codes.append(positional_encoding(xyzs.reshape(code.shape[0], xyzs.shape[-2], 3), self.pe))
        if self.reduce == 'cat':
            return torch.cat(codes, dim=-1).reshape(code.shape[0] * xyzs.shape[-2], self.in_chs)
        else:
            assert self.reduce == 'sum'
            return sum(codes).reshape(code.shape[0] * xyzs.shape[-2], self.in_chs)

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        if 'mean' in str(scene_name):
            return
        if not self.visualize_mesh:
            return
        buffer = self.code_proc_buffer.to(code)
        basis = torch.linspace(-1, 1, 500, device=code.device, dtype=code.dtype)
        x, y, z = torch.meshgrid(basis, basis, basis, indexing='ij')
        xyz = torch.stack([x, y, z], dim=-1)
        xyz_f = xyz.flatten(0, -2)
        batches = []
        for batch_split in torch.split(xyz_f, 2 ** 17):
            pc = self.get_point_code(buffer, batch_split)
            if self.separate_density_and_color:
                density_code, color_code = torch.chunk(pc, 2, -1)
                sdf = self.density_decoder(density_code, None, True)[0]
            else:
                sdf = self.common_decoder(pc, None, True)[0]
            if self.sdf_mode:
                batches.append(sdf.cpu())
            else:
                batches.append((sdf > 5).cpu())
        batches = torch.cat(batches, 0).reshape(xyz.shape[:-1])
        if not self.sdf_mode:
            field = mcubes.smooth_constrained(batches.numpy())
            verts, tris = mcubes.marching_cubes(field, 0.5)
        else:
            verts, tris = mcubes.marching_cubes(batches.numpy(), 0.0)
        verts = verts / 250 - 1
        # mcubes.smooth()
        trimesh.Trimesh(verts, tris).export(os.path.join(viz_dir, str(scene_name[0]) + ".glb"))
