# X-ray to CT Reconstruction with ZeroRF Deep Image Prior

**프로젝트 목표**: ZeroRF의 TensoRF-VM 아키텍처와 Deep Image Prior를 활용하여 Sparse-view X-ray to CT Reconstruction 모델 개발

**작성일**: 2026년 2월 16일

---

## 📋 프로젝트 개요

### 핵심 아이디어
- **ZeroRF 유지**: TensoRF-VM representation + Deep Image Prior (네트워크 구조가 자연스러운 3D 형상의 prior 역할)
- **NAF 참고**: X-ray 데이터 로딩, Beer-Lambert law 기반 attenuation 렌더링, CBCT geometry
- **작업 변경**: Natural image novel view synthesis → X-ray to CT reconstruction

### 핵심 설계 원칙
1. **View-Agnostic Architecture**: 데이터셋에서 자동으로 view 수와 해상도 감지
2. **Data-Driven Configuration**: 데이터 경로만 변경하면 즉시 실행 (10/50/100 views 모두 가능)
3. **Per-Scene Optimization**: 각 CT scan마다 독립적으로 네트워크 최적화 (ZeroRF 방식)

---

## 🎯 주요 변경사항

### 아키텍처 비교

| 구성 요소 | ZeroRF (원본) | 이 프로젝트 |
|---------|--------------|------------|
| **입력** | RGB 이미지 (자연 영상) | X-ray projections |
| **출력** | RGB + Density | Attenuation coefficient (μ) |
| **Representation** | TensoRF-VM (6 components) | TensoRF-VM (동일 유지) |
| **Rendering** | Volume rendering (T·α·RGB) | Beer-Lambert integration (Σμ·δt) |
| **Camera** | Pinhole (intrinsics) | Cone-beam (source-detector) |
| **학습 방식** | Per-scene from noise | Per-scene from noise (동일) |
| **Encoder** | Deep Image Prior (noise→VAE) | Deep Image Prior (동일 유지) |
| **Ray sampling** | 4K→65K curriculum | 4K→65K curriculum (동일) |
| **Iterations** | 10000 | 10000 (동일) |

### ZeroRF에서 유지할 것 (변경 없음)
- ✅ **TensoRF-VM Representation**: Noise → VAE → VM components (3 planes + 3 lines)
- ✅ **Feature 추출**: `get_point_code()` - 3D 좌표에서 VM feature 샘플링
- ✅ **MLP 학습**: `CommonDecoder` - Point feature → MLP → 출력
- ✅ **Deep Image Prior**: 네트워크 구조가 3D prior 역할
- ✅ **Per-scene optimization**: 각 scan마다 독립 학습
- ✅ **Curriculum learning**: Ray batch 4096 → 65536 점진적 증가 (100 iter 후)

### NAF에서 가져올 것
- ✅ X-ray 데이터 로더 (`src/dataset/tigre.py` - `ConeGeometry`, `get_rays()`)
- ✅ Attenuation 계산 로직 (`src/render/render.py` - Beer-Lambert law)
- ✅ Projection space loss (`src/loss/loss.py` - MSE)
- ❌ Hash encoder 사용하지 않음 (ZeroRF의 Deep Prior 유지)
- ❌ NAF 네트워크 아키텍처 사용하지 않음

### 실제 변경사항은 오직
- 🔄 **MLP 출력 헤드**: RGB (3 channels) → Attenuation (1 channel)
- 🔄 **렌더링 방정식**: Volume rendering → Beer-Lambert integration
- 🔄 **데이터 입력**: Natural images → X-ray projections

### ⚙️ 중요한 학습 전략 (ZeroRF 그대로 유지)

**왜 Ray 수를 증가시키는가?**
- **초기 (4K rays)**: 빠른 global structure 학습
- **후기 (65K rays)**: 세밀한 detail 학습
- **원리**: Coarse-to-fine curriculum learning

**왜 10000 iterations인가?**
- **Deep Prior의 특성**: Random noise부터 시작하여 3D structure를 "발견"해야 함
- **NAF와 차이**: NAF는 explicit encoding (hash grid) 사용 → 빠른 수렴 (1000 epoch)
- **ZeroRF**: Implicit prior (network structure) → 느린 수렴 (10K iters) but 더 강한 일반화

**GPU 메모리 고려사항** (모델 + 학습 포함):
- 65K rays: **24GB+ GPU 필요** (RTX 4090, A6000)
- 32K rays: **20-22GB GPU** (RTX 3090Ti)
- 16K rays: **14-16GB GPU** (RTX 4070Ti, RTX 3080) ⭐ 권장
- 8K rays: **8-12GB GPU** (RTX 3070)

| GPU 메모리 | n_rays_init | n_rays_up | 예상 VRAM | 모델 포함 |
|-----------|-------------|-----------|-----------|----------|
| **24GB+** | 4096 | 65536 | ~22 GB | Safe |
| **22GB** | 4096 | 32768 | ~18 GB | Safe |
| **16GB** ⭐ | **2048** | **16384** | **~12 GB** | **Safe** |
| **12GB** | 2048 | 8192 | ~9 GB | Tight |

---

## 🔧 구현 계획

### Phase 1: 데이터 파이프라인 구축

#### 1.1 X-ray Dataset Adapter 생성
**파일 위치**: `zerorf/lib/datasets/xray_dataset.py` (신규 생성)

**기능**:
- NAF의 `TIGREDataset` 기반 X-ray 데이터 로딩
- **자동 감지**:
  ```python
  self.n_views = len(projection_files)  # view 수 자동 카운트
  self.image_h, self.image_w = projs[0].shape  # 해상도 자동 감지
  ```
- Cone-beam geometry 파라미터 로드 (DSD, DSO, detector size)
- NAF 형식 → ZeroRF 형식 변환

**참고 파일**:
- NAF: `naf_cbct/src/dataset/tigre.py` (L13-L287)
  - `ConeGeometry` class (L13-59)
  - `get_rays()` method (L195-234)

**출력 형식**:
```python
{
    'cond_imgs': (1, N, H, W, 1),      # X-ray projections (N은 자동 감지)
    'cond_poses': (1, N, 4, 4),        # Equivalent pose matrices
    'cond_intrinsics': (1, N, 4),      # Cone-beam으로부터 파생
    'scene_id': [0],
    'scene_name': ['scan_name']
}
```

#### 1.2 Dataset Builder 등록
**파일 수정**: `zerorf/lib/datasets/builder.py`
- `XrayDataset` import 추가
- Dataset type 등록

---

### Phase 2: Beer-Lambert 렌더링 구현

#### 2.1 CUDA 커널 구현: `composite_rays_xray()`
**파일 위치**: `zerorf/lib/ops/raymarching/` (CUDA extension)

**렌더링 방정식 변경**:
```cpp
// 기존 (Volume Rendering):
// alpha_i = 1 - exp(-sigma_i * dt)
// T_i = prod(1 - alpha_j) for j < i
// RGB = sum(T_i * alpha_i * color_i)

// 신규 (X-ray Attenuation - Beer-Lambert Law):
// I = I_0 * exp(-∫μ(x)dx)
// For discrete sampling: projection = sum(mu_i * dt_i)
// 직접 적분, transmittance 계산 없음
```

**Beer-Lambert Law 구현 (Python)**:

NAF 참고: `naf_cbct/src/render/render.py` (L73-96)
```python
def composite_rays_xray(mu, z_vals, rays_d):
    """
    X-ray attenuation line integral (Beer-Lambert law)
    
    Args:
        mu: [M] Attenuation coefficients (μ) from network
        z_vals: [N, n_samples] Sample positions along rays
        rays_d: [N, 3] Ray directions
    
    Returns:
        projection: [N] Accumulated attenuation per ray
    """
    # 1. Calculate path lengths (δt)
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N, n_samples-1]
    
    # Append small value for last segment
    dists = torch.cat([
        dists, 
        torch.ones_like(dists[..., :1]) * 1e-10
    ], dim=-1)  # [N, n_samples]
    
    # Account for ray direction (actual 3D distance)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)  # [N, n_samples]
    
    # 2. Beer-Lambert line integral: ∫μ(x)dx ≈ Σ(μ_i * δt_i)
    # mu shape: [M] where M = N * n_samples (flattened)
    # Reshape to [N, n_samples]
    mu_reshaped = mu.reshape(z_vals.shape[0], z_vals.shape[1])
    
    # 3. Sum along ray: projection = Σ(μ_i * δt_i)
    projection = torch.sum(mu_reshaped * dists, dim=-1)  # [N]
    
    return projection
```

**핵심 차이점**:
```python
# Volume Rendering (ZeroRF 기존):
alpha = 1.0 - torch.exp(-sigma * dists)  # Absorption
T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)  # Transmittance
rgb = torch.sum(T * alpha * color, dim=-1)  # Accumulated color

# X-ray Attenuation (신규):
projection = torch.sum(mu * dists, dim=-1)  # Direct line integral
# No exponential, No transmittance, No color
```

**CUDA 커널 구현 (Pseudo-code)**:

파일: `zerorf/lib/ops/raymarching/raymarching.cu`
```cpp
__global__ void composite_rays_xray_kernel(
    const float* __restrict__ mu,        // [M] Attenuation coefficients
    const float* __restrict__ ts,        // [M, 2] t_start, t_end per sample
    const int* __restrict__ rays,        // [N, 2] ray_idx, n_samples
    float* __restrict__ projection,      // [N] Output projections
    const int M,
    const int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    
    // Get ray index for this sample
    const int ray_idx = rays[i * 2];
    
    // Calculate path length
    const float dt = ts[i * 2 + 1] - ts[i * 2];
    
    // Accumulate: projection += μ * δt
    atomicAdd(&projection[ray_idx], mu[i] * dt);
}

// Wrapper function
void composite_rays_xray(
    at::Tensor mu,           // [M] from network
    at::Tensor ts,           // [M, 2] sample intervals
    at::Tensor rays,         // [N, 2] ray metadata
    at::Tensor projection    // [N] output
) {
    const int M = mu.size(0);
    const int N = projection.size(0);
    
    // Initialize projection to zero
    projection.fill_(0.0f);
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;
    
    composite_rays_xray_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        ts.data_ptr<float>(),
        rays.data_ptr<int>(),
        projection.data_ptr<float>(),
        M, N
    );
}
```

**참고**:
- 기존 구현: `zerorf/lib/ops/raymarching/raymarching.py` - `composite_rays_train()` (L330-380)
- NAF 로직: `naf_cbct/src/render/render.py` - `raw2outputs()` (L96)
  ```python
  acc = torch.sum((raw[..., 0] + noise) * dists, dim=-1)
  ```

#### 2.2 Base Volume Renderer 수정
**파일 수정**: `zerorf/lib/models/decoders/base_volume_renderer.py`

**변경사항**:
- `__init__()`: `xray_mode` 파라미터 추가
- `forward()` (L205-370):
  ```python
  class VolumeRenderer(nn.Module):
      def __init__(self, *args, xray_mode=False, **kwargs):
          super().__init__(*args, **kwargs)
          self.xray_mode = xray_mode
      
      def forward(self, rays_o, rays_d, code, density_bitfield, ...):
          # 1. Ray marching (동일)
          xyzs, dirs, ts, rays = march_rays_train(
              rays_o, rays_d, density_bitfield, ...)
          
          # 2. Network forward: 3D points → attenuation
          point_code = self.get_point_code(code, xyzs)
          
          if self.xray_mode:
              # X-ray: Only density (attenuation)
              mu, _ = self.point_code_render(point_code, dirs=None)
              # mu: [M] attenuation coefficients
          else:
              # RGB: Density + Color
              sigma, rgb = self.point_code_render(point_code, dirs)
          
          # 3. Rendering
          if self.xray_mode:
              # Beer-Lambert line integral
              projection = composite_rays_xray(
                  mu, ts, rays)  # [N]
              outputs = {
                  'projection': projection,  # [N]
                  'depth': None,  # Optional: can compute depth
                  'weights': None  # Optional: for importance sampling
              }
          else:
              # Volume rendering
              weights, depth, rgb_out = composite_rays_train(
                  sigma, rgb, ts, rays, ...)
              outputs = {
                  'image': rgb_out.reshape(H, W, 3),
                  'depth': depth.reshape(H, W),
                  'weights': weights
              }
          
          return outputs
  ```

**통합 예시 (forward 전체)**:
```python
def forward(self, rays_o, rays_d, code, density_bitfield, 
            grid_size, dt_gamma=0, perturb=False):
    """
    Args:
        rays_o: [N, 3] Ray origins
        rays_d: [N, 3] Ray directions
        code: Scene code from TensorialGenerator
        
    Returns (X-ray mode):
        outputs = {
            'projection': [N] Accumulated attenuation,
            'depth': [N] Average depth (optional),
            'weights': [N, n_samples] For importance sampling (optional)
        }
    """
    N = rays_o.shape[0]
    
    # === Step 1: Ray Marching ===
    xyzs, dirs, ts, rays_info = march_rays_train(
        rays_o, rays_d, 
        bound=self.bound,
        density_bitfield=density_bitfield,
        grid_size=grid_size,
        nears=None, fars=None,
        dt_gamma=dt_gamma, perturb=perturb,
        max_steps=self.max_steps
    )
    # xyzs: [M, 3] Sampled 3D points
    # ts: [M, 2] t_start, t_end for each sample
    # rays_info: [N, 2] (ray_idx, n_samples)
    # M = total samples across all rays
    
    # === Step 2: Feature Extraction (TensoRF-VM) ===
    point_code = self.get_point_code(code, xyzs)
    # point_code: [M, in_chs] from VM grids
    
    # === Step 3: MLP Decoding ===
    if self.xray_mode:
        # X-ray: Attenuation only
        mu, _ = self.point_code_render(point_code, dirs=None)
        # mu: [M] attenuation coefficients
    else:
        # RGB: Density + Color
        sigma, rgb = self.point_code_render(point_code, dirs)
    
    # === Step 4: Rendering ===
    if self.xray_mode:
        # Calculate path lengths
        dists = ts[:, 1] - ts[:, 0]  # [M]
        
        # Beer-Lambert integral
        projection = torch.zeros(N, device=mu.device)
        for i in range(M):
            ray_idx = rays_info[i, 0]
            projection[ray_idx] += mu[i] * dists[i]
        
        # Or use scatter_add for efficiency
        # projection = scatter_add(mu * dists, rays_info[:, 0], dim=0, dim_size=N)
        
        outputs = {'projection': projection}
    else:
        # Standard volume rendering
        weights, depth, rgb_final = composite_rays_train(
            sigma, rgb, ts, rays_info, ...)
        outputs = {'image': rgb_final, 'depth': depth, 'weights': weights}
    
    return outputs
```

---

### Phase 3: 네트워크 출력 수정

#### 3.1 CommonDecoder 수정 (Attenuation Only)
**파일 수정**: `zerorf/lib/models/zerorf/decoders.py` (L60-L139)

**변경사항**:
```python
class CommonDecoder(nn.Module):
    def __init__(self, point_channels, sh_coef_only=False, dir_pe=False, 
                 sdf_mode=False, xray_mode=False):
        super().__init__()
        self.xray_mode = xray_mode
        
        # Base network + Density network (유지)
        self.base_net = nn.Linear(point_channels, 64)
        self.density_net = nn.Sequential(nn.Linear(64, 1), TruncExp())
        
        # X-ray 모드: Color network 비활성화
        if xray_mode:
            self.dir_net = None
            self.color_net = None
        else:
            self.dir_encoder = SHEncoder(degree=3)
            self.dir_net = nn.Linear(9, 64)
            self.color_net = nn.Sequential(nn.Linear(64, 3), nn.Sigmoid())
    
    def forward(self, point_code, dirs=None, out_sdf=False):
        base_x_act = self.base_activation(self.base_net(point_code))
        sigmas = self.density_net(base_x_act).squeeze(-1)  # Attenuation
        
        if self.xray_mode:
            return sigmas, None  # RGB 없음
        else:
            # RGB 계산 (기존 로직)
            ...
            return sigmas, rgbs
```

#### 3.2 TensorialDecoder 파라미터 전달
**파일 수정**: `zerorf/lib/models/zerorf/decoders.py` (L145)

**변경사항**:
- `__init__()` 호출 시 `xray_mode=True` 전달
- `n_images`, `image_h`, `image_w`: 데이터셋에서 자동 감지된 값 사용
- `separate_density_and_color=False` 유지

---

### Phase 4: 학습 파이프라인 구축

#### 4.1 Main Script 수정
**파일 수정**: `zerorf/zerorf.py` (또는 신규 `xray_train.py` 생성)

**데이터 로딩 부분** (L60-120 참고):
```python
# Dataset 먼저 로드 (자동 감지)
if args.dataset == "xray":
    dataset = XrayDataset(args.data_dir, split='train')
    val_dataset = XrayDataset(args.data_dir, split='val')
    
    # 자동 감지된 파라미터 출력
    n_views = dataset.n_views
    image_h, image_w = dataset.image_h, dataset.image_w
    print(f"✓ Auto-detected: {n_views} views, {image_h}×{image_w} resolution")

# Data entry 구성
data_entry = dict(
    cond_imgs=dataset[0]['cond_imgs'],      # (1, N, H, W, 1)
    cond_poses=dataset[0]['cond_poses'],    # (1, N, 4, 4)
    cond_intrinsics=dataset[0]['cond_intrinsics'],  # (1, N, 4)
    scene_id=[0],
    scene_name=[args.data_dir.split('/')[-1]]
)
```

**Decoder 구성** (L142-152 참고):
```python
decoder_1 = dict(
    type='TensorialDecoder',
    preprocessor=dict(
        type='TensorialGenerator',
        in_ch=args.model_ch,
        out_ch=16,
        noise_res=args.model_res,
        tensor_config=['xy', 'z', 'yz', 'x', 'zx', 'y']  # TensoRF-VM
    ),
    subreduce=1,
    reduce='cat',
    separate_density_and_color=False,
    sh_coef_only=False,
    sdf_mode=False,
    xray_mode=True,           # ✓ X-ray 모드 활성화
    n_images=n_views,         # ✓ 자동 감지된 값
    image_h=image_h,          # ✓ 자동 감지된 값
    image_w=image_w,          # ✓ 자동 감지된 값
    max_steps=1024,
)
```

#### 4.2 Training Loop
**참고**: `zerorf.py` (L230-275), `naf_cbct/train.py`

**학습 방식**:
- Per-scene optimization (ZeroRF 방식)
- Ray-based sampling (view 수 무관)
- Hierarchical sampling (coarse + fine)

**Hyperparameters** (ZeroRF 기반, 16GB GPU 최적화):
```yaml
train:
  n_iters: 10000            # Total iterations (ZeroRF 사용)
  n_rays_init: 2048         # 2^11: Ray batch size (초기)
  n_rays_up: 16384          # 2^14: Ray batch size (100 iter 후) - 16GB GPU 권장
  ray_upsample_iter: 100    # Ray curriculum learning
  lrate: 0.002              # AdamW learning rate
  lrate_decay: cosine
  
render:
  n_samples: 256            # Coarse sampling
  n_fine: 256               # Fine sampling (hierarchical)
  perturb: True
```

**Note**: 
- ZeroRF는 ray 수를 점진적으로 증가시키는 curriculum learning 사용
- 16GB GPU: 16K rays 권장 (~12GB VRAM, 모델 포함 안전)
- 22GB+ GPU: 32K rays 가능
- 24GB+ GPU: 65K rays 가능 (원본 ZeroRF 설정)

**구현 예시**:
```python
# Ray curriculum learning (ZeroRF 방식, 16GB GPU 최적화)
for iteration in range(args.n_iters):
    # Ray 수 점진적 증가
    if iteration <= args.ray_upsample_iter:
        n_rays = args.n_rays_init  # 2048
    else:
        n_rays = args.n_rays_up     # 16384 (16GB) or 32768 (22GB+)
    
    # Random ray sampling (view 수 무관)
    rays_batch = sample_rays(all_rays, n_rays)
    
    # Hierarchical rendering
    outputs = render(rays_batch, net, net_fine,
                    n_samples=256, n_importance=256)
    
    # Loss & backward
    loss = F.mse_loss(outputs['projection'], target)
    loss.backward()
    optimizer.step()
```

#### 4.3 Loss Function
**파일 수정**: `zerorf/lib/models/autoencoders/multiscene_nerf.py` (L200-258)

**Loss 계산**:
```python
def compute_loss(self, data, outputs):
    """
    X-ray projection loss
    
    Args:
        data: {
            'target_proj': [N, H, W] Ground truth X-ray projections
            'rays': [N, H, W, 8] Ray parameters
        }
        outputs: {
            'projection': [N*H*W] Predicted projections (flattened)
        }
    
    Returns:
        loss_dict: {'loss': total_loss, 'proj_mse': mse, 'psnr': psnr}
    """
    # 1. Reshape predictions
    target_proj = data['target_proj'].reshape(-1)  # [N*H*W]
    pred_proj = outputs['projection']  # [N*H*W]
    
    # 2. Projection space MSE (main loss)
    loss_mse = F.mse_loss(pred_proj, target_proj)
    
    # 3. Optional: L1 loss for sparsity
    # loss_l1 = F.l1_loss(pred_proj, target_proj)
    
    # 4. Optional: Total Variation Regularization (3D volume)
    # if self.use_tv_reg:
    #     volume = self.reconstruct_volume(code)
    #     loss_tv = self.calc_tv_loss(volume)
    # else:
    #     loss_tv = 0.0
    
    # 5. Total loss
    total_loss = loss_mse  # + 0.1 * loss_l1 + 0.01 * loss_tv
    
    # 6. Compute PSNR for monitoring
    with torch.no_grad():
        mse_value = loss_mse.item()
        psnr = -10 * np.log10(mse_value) if mse_value > 0 else 100.0
    
    loss_dict = {
        'loss': total_loss,
        'proj_mse': loss_mse,
        'psnr': torch.tensor(psnr)
    }
    
    return loss_dict


# NAF 참고: src/loss/loss.py
def calc_mse_loss(loss_dict, target, pred):
    """Simple MSE loss (NAF style)"""
    mse = torch.mean((target - pred) ** 2)
    loss_dict['mse'] = mse
    loss_dict['loss'] += mse
    return loss_dict


# Optional: Total Variation Loss
def calc_tv_loss(volume):
    """
    3D Total Variation for smoothness
    
    Args:
        volume: [D, H, W] 3D reconstructed CT volume
    
    Returns:
        tv_loss: Scalar
    """
    # Gradient in x, y, z directions
    diff_x = torch.abs(volume[1:, :, :] - volume[:-1, :, :])
    diff_y = torch.abs(volume[:, 1:, :] - volume[:, :-1, :])
    diff_z = torch.abs(volume[:, :, 1:] - volume[:, :, :-1])
    
    tv_loss = diff_x.mean() + diff_y.mean() + diff_z.mean()
    return tv_loss
```

**Training Loop에서 사용**:
```python
for iteration in range(args.n_iters):
    # Sample rays
    rays_batch = sample_rays(all_rays, n_rays)
    target_batch = sample_projections(all_projs, n_rays)
    
    # Forward pass
    outputs = model(rays_batch['rays_o'], rays_batch['rays_d'], 
                    code, density_bitfield)
    # outputs['projection']: [n_rays]
    
    # Compute loss
    loss_dict = compute_loss(
        data={'target_proj': target_batch},
        outputs=outputs
    )
    
    # Backward
    optimizer.zero_grad()
    loss_dict['loss'].backward()
    optimizer.step()
    
    # Logging
    if iteration % 100 == 0:
        print(f"Iter {iteration}, Loss: {loss_dict['loss'].item():.6f}, "
              f"PSNR: {loss_dict['psnr'].item():.2f} dB")
```

---

### Phase 5: 평가 및 검증

#### 5.1 Evaluation Metrics
**참고**: `naf_cbct/train.py` (L47-73)

**Projection Space**:
- PSNR (2D)
- SSIM (2D)

**Volume Space**:
- 3D PSNR
- 3D SSIM
- Visual comparison (slice views)

#### 5.2 Visualization
- Rendered projections vs Ground truth
- Reconstructed CT slices (axial, coronal, sagittal)
- TensorBoard logging

---

## 📝 구현 체크리스트

### Phase 1: Data Pipeline
- [ ] `lib/datasets/xray_dataset.py` 생성
  - [ ] `ConeGeometry` 클래스 통합
  - [ ] `get_rays()` 메서드 구현
  - [ ] View 수/해상도 자동 감지
  - [ ] ZeroRF 형식 변환
- [ ] `lib/datasets/builder.py` 수정 (dataset 등록)
- [ ] Unit test: 10/50/100 views 데이터 로드 테스트

### Phase 2: Rendering
- [ ] `lib/ops/raymarching/` Beer-Lambert 렌더링 구현
  - [ ] `composite_rays_xray()` Python 함수 작성
    - [ ] Path length 계산: `dists = z_vals[1:] - z_vals[:-1]`
    - [ ] Line integral: `projection = sum(mu * dists)`
    - [ ] Shape 확인: `mu [M]` → `projection [N]`
  - [ ] CUDA 커널 작성 (성능 최적화)
    - [ ] `composite_rays_xray_kernel` 구현
    - [ ] `atomicAdd` 또는 `scatter_add` 사용
  - [ ] 수식 검증: ∫μ(x)dx ≈ Σ(μ_i * δt_i)
  - [ ] Gradient 계산 확인 (backward pass)
  - [ ] Unit test: 예상값과 비교
- [ ] `lib/models/decoders/base_volume_renderer.py` 통합
  - [ ] `__init__()`: `xray_mode` 파라미터 추가
  - [ ] `forward()`:
    - [ ] X-ray mode 분기 처리
    - [ ] `composite_rays_xray()` 호출
    - [ ] 출력 형식: `{'projection': [N]}`
  - [ ] Volume rendering과 비교 테스트
- [ ] Unit test: End-to-end rendering
  - [ ] Single ray → expected projection
  - [ ] Batch rays → consistent results
  - [ ] Gradient flow 확인

### Phase 3: Network
- [ ] `lib/models/zerorf/decoders.py` 수정
  - [ ] `CommonDecoder`: `xray_mode` 추가
  - [ ] Color network 비활성화 로직
  - [ ] Forward pass 수정
- [ ] `TensorialDecoder` 파라미터 전달 확인
- [ ] Unit test: 출력 shape (N, 1) 확인

### Phase 4: Training
- [ ] `zerorf.py` 또는 `xray_train.py` 생성
  - [ ] Dataset 자동 감지 로직
  - [ ] Decoder config 구성
  - [ ] Training loop 구현
- [ ] Config 파일 생성 (`configs/xray.yaml`)
- [ ] Loss function 통합
- [ ] TensorBoard logging 설정

### Phase 5: Evaluation
- [ ] Evaluation script 작성
- [ ] Metrics 계산 (PSNR, SSIM)
- [ ] Visualization 코드
- [ ] 결과 비교 (NAF 단독 vs ZeroRF+X-ray)

---

## 🧪 테스트 계획

### 1. Unit Tests
```bash
# 데이터 로딩
python -c "from lib.datasets.xray_dataset import XrayDataset; \
    ds = XrayDataset('./data/test_10views'); \
    print(f'Views: {ds.n_views}, Size: {ds.image_h}x{ds.image_w}')"

# Beer-Lambert law 검증
pytest tests/test_xray_rendering.py::test_beer_lambert

# 렌더링 (단일 ray)
pytest tests/test_xray_rendering.py::test_single_ray

# 네트워크 forward
pytest tests/test_xray_network.py::test_attenuation_output
```

**Beer-Lambert Law Unit Test 예시**:

파일: `tests/test_xray_rendering.py`
```python
import torch
import pytest
from lib.ops.raymarching import composite_rays_xray

def test_beer_lambert():
    """
    Beer-Lambert law 검증:
    Uniform attenuation → Expected projection
    """
    # Setup: Uniform attenuation coefficient
    n_rays = 10
    n_samples = 100
    mu_value = 0.5  # Uniform μ = 0.5
    
    # Ray samples
    z_vals = torch.linspace(0, 1, n_samples).repeat(n_rays, 1)  # [N, n_samples]
    rays_d = torch.tensor([[0, 0, 1]]).repeat(n_rays, 1).float()  # [N, 3]
    
    # Attenuation coefficients
    mu = torch.ones(n_rays * n_samples) * mu_value  # [M]
    
    # Expected: ∫μdx = μ * length = 0.5 * 1.0 = 0.5
    expected_projection = mu_value * 1.0
    
    # Compute
    projection = composite_rays_xray(mu, z_vals, rays_d)  # [N]
    
    # Verify
    assert projection.shape == (n_rays,)
    assert torch.allclose(projection, torch.tensor(expected_projection), atol=1e-4)
    print(f"✓ Beer-Lambert test passed: {projection[0]:.4f} ≈ {expected_projection:.4f}")


def test_single_ray_gradient():
    """
    Gradient flow 확인
    """
    n_samples = 50
    z_vals = torch.linspace(0, 1, n_samples).unsqueeze(0)  # [1, n_samples]
    rays_d = torch.tensor([[0, 0, 1]]).float()
    
    # Network output (requires_grad=True)
    mu = torch.ones(n_samples, requires_grad=True) * 0.3
    
    # Forward
    projection = composite_rays_xray(mu, z_vals, rays_d)
    
    # Backward
    loss = projection.sum()
    loss.backward()
    
    # Verify gradient exists
    assert mu.grad is not None
    assert not torch.isnan(mu.grad).any()
    print(f"✓ Gradient test passed: grad shape {mu.grad.shape}")


def test_varying_attenuation():
    """
    Variable attenuation 검증
    """
    n_rays = 5
    n_samples = 100
    
    # Linearly increasing attenuation: μ(x) = x
    z_vals = torch.linspace(0, 1, n_samples).repeat(n_rays, 1)
    rays_d = torch.tensor([[0, 0, 1]]).repeat(n_rays, 1).float()
    
    mu = torch.linspace(0, 1, n_samples).repeat(n_rays)  # [M]
    
    # Expected: ∫x dx from 0 to 1 = 0.5
    expected_projection = 0.5
    
    projection = composite_rays_xray(mu, z_vals, rays_d)
    
    assert torch.allclose(projection, torch.tensor(expected_projection), atol=1e-2)
    print(f"✓ Variable attenuation test passed: {projection[0]:.4f} ≈ {expected_projection:.4f}")


def test_compare_with_naf():
    """
    NAF 구현과 비교
    """
    from naf_cbct.src.render import raw2outputs
    
    # Same input
    n_rays = 20
    n_samples = 64
    raw = torch.rand(n_rays, n_samples, 1)  # NAF format
    z_vals = torch.linspace(0, 1, n_samples).repeat(n_rays, 1)
    rays_d = torch.randn(n_rays, 3)
    
    # NAF version
    naf_acc, _ = raw2outputs(raw, z_vals, rays_d)
    
    # Our version
    mu = raw.squeeze(-1).reshape(-1)  # [M]
    our_projection = composite_rays_xray(mu, z_vals, rays_d)
    
    # Should match
    assert torch.allclose(naf_acc, our_projection, atol=1e-5)
    print(f"✓ NAF comparison passed")


if __name__ == "__main__":
    test_beer_lambert()
    test_single_ray_gradient()
    test_varying_attenuation()
    test_compare_with_naf()
    print("\n✅ All Beer-Lambert tests passed!")
```

### 2. Integration Test
```bash
# 10 views - Quick test (1000 iters, 16GB GPU)
python zerorf.py --dataset=xray --data-dir=./data/test_10views \
    --config=configs/xray.yaml --n-iters=1000 \
    --n-rays-init=2048 --n-rays-up=16384

# 50 views - Full training (코드 수정 없음, 16GB GPU)
python zerorf.py --dataset=xray --data-dir=./data/test_50views \
    --config=configs/xray.yaml --n-iters=10000 \
    --n-rays-init=2048 --n-rays-up=16384

# Mixed precision으로 더 많은 rays (20K)
python zerorf.py --dataset=xray --data-dir=./data/test_50views \
    --config=configs/xray.yaml --n-iters=10000 \
    --n-rays-init=2048 --n-rays-up=20480 \
    --use-amp
```

### 3. 예상 출력
```
✓ Auto-detected: 50 views, 512×512 resolution
✓ TensoRF-VM initialized: 6 components (xy, z, yz, x, zx, y)
✓ X-ray mode enabled: Attenuation-only output
✓ Ray batch: 2048 → 16384 (curriculum learning)
✓ GPU Memory: 11.8 GB / 16.0 GB (Mixed Precision: OFF)
Iter 1000/10000, Loss: 0.0028, Proj-PSNR: 27.2 dB
Iter 5000/10000, Loss: 0.0012, Proj-PSNR: 31.5 dB
Iter 10000/10000, Loss: 0.0006, Proj-PSNR: 35.2 dB
```

---

## 🎛️ Configuration

### Example: `configs/xray.yaml`
```yaml
exp:
  datadir: "./data/abdomen_50"
  expname: "xray_ct_recon"

dataset:
  type: "xray"
  auto_detect: true  # View 수/해상도 자동 감지

model:
  type: "TensorialDecoder"
  xray_mode: true
  model_ch: 8
  model_res: 4
  tensor_config: ['xy', 'z', 'yz', 'x', 'zx', 'y']

train:
  n_iters: 10000              # ZeroRF 기본값
  n_rays_init: 2048           # 초기 ray batch (2^11)
  n_rays_up: 16384            # 후기 ray batch (2^14) - 16GB GPU 최적
  ray_upsample_iter: 100      # Ray curriculum
  net_lr: 0.002
  net_lr_decay_to: 0.002
  optimizer: "AdamW"

render:
  n_samples: 256              # Coarse
  n_importance: 256           # Fine
  perturb: true
  max_steps: 1024

loss:
  type: "MSE"
  weight: 1.0

eval:
  val_iter: 1000              # Validate every 1000 iters
```

---

## 🚀 실행 방법

### Training
```bash
# 기본 실행 (50 views, 10000 iterations, 16GB GPU)
python zerorf.py --dataset=xray --data-dir=./data/abdomen_50 \
    --config=configs/xray.yaml --n-iters=10000 \
    --n-rays-init=2048 --n-rays-up=16384

# Sparse views (10 views) - 동일 코드
python zerorf.py --dataset=xray --data-dir=./data/chest_10 \
    --config=configs/xray.yaml --n-iters=10000 \
    --n-rays-init=2048 --n-rays-up=16384

# Quick test (1000 iterations)
python zerorf.py --dataset=xray --data-dir=./data/abdomen_50 \
    --config=configs/xray.yaml --n-iters=1000 \
    --n-rays-init=2048 --n-rays-up=16384

# Mixed precision으로 더 많은 rays (20K rays)
python zerorf.py --dataset=xray --data-dir=./data/abdomen_50 \
    --config=configs/xray.yaml --n-iters=10000 \
    --n-rays-init=2048 --n-rays-up=20480 \
    --use-amp

# 22GB GPU 이상 (더 많은 rays)
python zerorf.py --dataset=xray --data-dir=./data/abdomen_50 \
    --config=configs/xray.yaml --n-iters=10000 \
    --n-rays-init=4096 --n-rays-up=32768
```

### Evaluation
```bash
python evaluate_xray.py --checkpoint=results/xray_ct_recon/ckpt.pth \
    --data-dir=./data/abdomen_50 --output-dir=results/eval
```

---

## 📊 기대 성능

### NAF 대비 장점
- **Sparse-view 성능**: 10-30 views에서 Deep Prior 덕분에 우수한 재구성
- **사전 학습 불필요**: 각 scan마다 최적화
- **구조적 prior**: TensoRF-VM이 해부학적 구조 표현에 유리

### NAF 대비 단점
- **학습 시간**: Per-scene 최적화로 인해 학습 시간 증가
- **Full-view 성능**: 100+ views에서는 NAF의 explicit encoding이 유리할 수 있음

---

## 📚 참고 자료

### ZeroRF
- Paper: "ZeroRF: Sparse View 360° Reconstruction with Zero Pretraining" (CVPR 2024)
- Code: `lib/models/zerorf/decoders.py`, `lib/models/zerorf/generators.py`
- Key: TensoRF-VM, Deep Image Prior, Per-scene optimization

### NAF (참고용)
- Code: `naf_cbct/src/`
- Key: X-ray geometry, Beer-Lambert law, CBCT dataset

### TensoRF
- Paper: "TensoRF: Tensorial Radiance Fields" (ECCV 2022)
- Representation: Vector-Matrix decomposition (3 planes + 3 lines)

---

## 🔍 핵심 기술 결정사항

### 1. Deep Image Prior 유지 (✓)
- **이유**: ZeroRF의 핵심 contribution, sparse-view에서 강력
- **대안**: Hash encoding (NAF) - rejected (사전 학습 필요, 일반화 어려움)

### 2. TensoRF-VM 유지 (✓)
- **이유**: View-independent 3D representation, 해부학적 구조 표현에 적합
- **변경 없음**: RGB든 attenuation이든 feature grid는 동일

### 3. Beer-Lambert Rendering (✓)
- **이유**: X-ray physics에 정확
- **구현**: Volume rendering 대신 직접 적분
- **핵심 수식**:
  ```
  Volume Rendering:  I = Σ T_i · α_i · c_i
                     T_i = exp(-Σ_{j<i} σ_j·δt_j)
                     α_i = 1 - exp(-σ_i·δt_i)
  
  X-ray Attenuation: I = Σ μ_i · δt_i
                     (단순 선적분, no exponential)
  ```
- **Python 구현**:
  ```python
  # Volume rendering (ZeroRF 기존)
  alpha = 1 - torch.exp(-sigma * dists)
  T = torch.cumprod(1 - alpha, dim=-1)
  rgb = torch.sum(T * alpha * color, dim=-1)
  
  # Beer-Lambert (X-ray)
  projection = torch.sum(mu * dists, dim=-1)
  ```
- **장점**: 더 단순, 메모리 효율적, X-ray physics 정확
- **검증**: NAF 구현과 일치 확인 (`raw2outputs()` L96)

### 4. View-Agnostic Design (✓)
- **이유**: 실용성, 실험 편의성
- **구현**: 데이터셋에서 자동 감지

### 5. Hierarchical Sampling (✓)
- **이유**: Sparse-view에서 재구성 품질 향상
- **구현**: ZeroRF 기본 지원 활용

---

## ✅ Success Criteria

### Minimum Viable Product (MVP)
1. ✓ 10/50/100 views 데이터 자동 로드
2. ✓ X-ray projection rendering 정확도 검증
3. ✓ Training convergence (loss 감소)
4. ✓ CT volume reconstruction 시각화

### Performance Target
- **50 views**: Projection PSNR > 30 dB, Volume PSNR > 28 dB
- **10 views**: Projection PSNR > 25 dB (sparse-view challenge)
- **NAF 대비**: Sparse-view (10-30)에서 동등 이상 성능

### Code Quality
- Unit tests 통과
- View 수 변경 시 코드 수정 불필요
- Clear documentation

---

## 🔧 메모리 최적화 팁 (16GB GPU)

### 1. Mixed Precision Training (강력 권장)
```python
# PyTorch AMP (Automatic Mixed Precision)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = render(rays_batch, net, net_fine)
    loss = F.mse_loss(outputs['projection'], target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
**메모리 절감**: ~30-40% (16K rays → 12GB 대신 ~8GB)
**성능**: 속도 10-20% 향상 + 메모리 절약

### 2. Gradient Checkpointing
```python
# 중간 activation 재계산으로 메모리 절약
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(x):
    return checkpoint(self.network, x)
```
**메모리 절감**: ~20-30% (속도 10-15% 감소)

### 3. Ray Batch 동적 조정
```python
# OOM 발생 시 자동으로 batch 감소
try:
    outputs = render(rays_batch, net, net_fine)
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()
        n_rays = n_rays // 2  # Ray 수 절반으로
        print(f"OOM detected, reducing to {n_rays} rays")
```

### 4. 권장 조합 (16GB GPU)
```yaml
# Best practice for RTX 4070Ti / RTX 3080 (16GB)
train:
  n_rays_init: 2048
  n_rays_up: 16384        # Mixed precision 없이 (안전)
  # or
  n_rays_up: 20480        # Mixed precision 사용 시 (20K rays)
  
  use_amp: true           # ⭐ 16GB GPU는 AMP 권장
  gradient_checkpointing: false  # 속도 우선
```

### 5. 초기 메모리 체크 (16GB GPU 필수)
```python
# Training 시작 전 메모리 확인
import torch
torch.cuda.empty_cache()
print(f"Free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
print(f"Total: {torch.cuda.mem_get_info()[1]/1e9:.2f} GB")

# 15GB 이상 여유 있어야 안전
assert torch.cuda.mem_get_info()[0] > 15e9, "Not enough GPU memory!"
```

### 6. 메모리 모니터링 (16GB GPU 중요)
```bash
# Training 중 메모리 실시간 확인
watch -n 1 nvidia-smi

# Python에서 메모리 체크
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# 주의: 16GB GPU는 14GB 이상 사용 시 위험
# 권장: 12GB 이하 유지
```

---

## �📅 Timeline (예상)

- **Week 1**: Phase 1-2 (Data pipeline, Rendering)
- **Week 2**: Phase 3-4 (Network, Training)
- **Week 3**: Phase 5 (Evaluation, Debugging)
- **Week 4**: Experiments, Performance tuning

---

**Last Updated**: 2026년 2월 16일 (버그 수정 반영)
**Hardware Target**: RTX 4070Ti / RTX 3080 (16GB VRAM)
**Ray Configuration**: 2048 → 16384 (curriculum learning)
**Expected Memory**: ~12GB (16K rays) / ~8-9GB (with AMP)
**Status**: Implementation Phase 🔧 (핵심 파이프라인 구현 완료, 버그 수정 완료)

---

## ⚠️ 발견된 문제점 및 수정 필요사항

**검토일**: 2026년 2월 16일
**수정일**: 2026년 2월 16일

### 문제 1: GT Projection 형태 미명시 — ✅ 해결됨

**문제**: Loss 계산 시 ground truth X-ray projection이 어떤 형태인지 명시되어 있지 않았음.

**결론**: NAF의 `generateData.py`에서 `tigre.Ax()`로 생성한 projection은 **line integral** (∫μ dx)이며, 모델 출력(Σ μ_i · δt_i)과 동일한 공간. **변환 불필요**.

**수정**: `xray_dataset.py` docstring에 GT 형태 명시 추가.

---

### 문제 2: `ts` 포맷 불일치 — ✅ 해결됨 (CRITICAL BUG FIX)

**문제**: CUDA ray marcher의 ts 포맷은 `[t_position, dt]`인데, `batch_composite_rays_xray()`에서 `dts = ts[:, 1] - ts[:, 0]` (= dt - t)으로 잘못 계산하고 있었음.

**영향**: 
- Training: Beer-Lambert 적분이 완전히 잘못됨 (t가 증가할수록 dts가 음수)
- Inference: 동일한 버그 + `rays_t` 미업데이트로 같은 구간 반복 샘플링

**수정**:
- `batch_composite_rays_xray()`: `dts = all_ts[:, 1]` (dt 직접 사용)
- Inference path: `dts = ts[:, 1]` + `rays_t` 업데이트 로직 + ray 종료 로직 추가
- 테스트 코드: ts 생성을 CUDA `[t_position, dt]` 포맷으로 통일

---

### 문제 3: Inference path `rays_t` 미업데이트 — ✅ 해결됨 (CRITICAL BUG FIX)

**문제**: `composite_rays()`가 `rays_t` 업데이트와 ray 종료를 담당하는데, X-ray 모드에서 이를 건너뛰어 `rays_t`가 영원히 초기값에 머뭄.

**수정**: X-ray inference 블록 뒤에 수동으로:
- `ts`의 last valid t 값으로 `rays_t` 업데이트
- n_step 미만 샘플링한 ray는 `-1`로 마킹하여 종료

---

### 문제 4: 함수 시그니처 통일 — ✅ 해결됨

**문제**: Plan 내에서 `composite_rays_xray()` 호출 방식이 3가지로 혼용.

**수정**: ZeroRF packed format (`ts: [M, 2]`, `rays: [N, 2]`)으로 통일 완료. 실제 구현은 `batch_composite_rays_xray(sigmas, ts, rays, num_points)` 단일 인터페이스.

---

### 문제 5: `xray_mode` 파라미터 전달 체인 — ✅ 구현 완료

전달 경로 (구현 확인됨):
```
zerorf.py: xray_mode=True
  → TensorialDecoder.__init__(xray_mode=True)  [decoders.py L140]
    → VolumeRenderer.__init__(xray_mode=True)   [base_volume_renderer.py L90]
    → CommonDecoder.__init__(xray_mode=True)     [decoders.py L148-150]
      → forward(): xray_mode일 때 color network skip
  → VolumeRenderer.forward(): xray_mode일 때 batch_composite_rays_xray 사용
  → BaseNeRF.loss(): xray_mode일 때 bg_color 합성 skip
  → BaseNeRF.render(): xray_mode일 때 1채널 출력
  → BaseNeRF.eval_and_viz(): xray_mode일 때 grayscale 시각화
```

---

### 문제 6: Density Bitfield Threshold — 🟡 모니터링 필요

**현재 설정**: `zerorf.py`에서 `density_thresh=0.05` (train), `0.01` (test).

NAF의 데이터는 attenuation을 [0, 1]로 정규화하고, 네트워크 출력은 `TruncExp()`를 사용하므로 RGB와 유사한 스케일. 현재 threshold가 적합할 수 있으나, 학습 결과를 보고 조정 필요.

**완화 처리**: `zerorf.py`에서 `occlusion_culling_th=0.0` (X-ray 모드)으로 설정하여 occlusion culling 비활성화 완료.

---

### 문제 7: Cone-beam → Pinhole 변환 — ✅ 구현 완료, 검증 필요

`xray_dataset.py`에 detector offset 포함한 변환 구현됨:
```python
fx = geo.DSD / geo.dDetector[0]
fy = geo.DSD / geo.dDetector[1]
cx = W / 2.0 - geo.offDetector[0] / geo.dDetector[0]
cy = H / 2.0 - geo.offDetector[1] / geo.dDetector[1]
```

NAF의 `get_rays()`와 ray 일치 검증 테스트는 아직 미작성.

---

### 문제 8: 학습 스크립트 결정 — ✅ 해결됨

`zerorf.py`에 직접 통합 완료 (`--dataset xray` 플래그로 분기).

---

### 📋 문제 상태 요약

| # | 문제 | 상태 | 수정 파일 |
|---|------|------|----------|
| 1 | GT projection 형태 | ✅ 해결 | `xray_dataset.py` (docstring) |
| 2 | ts 포맷 버그 (CRITICAL) | ✅ 수정 | `base_volume_renderer.py`, `test_xray_rendering.py` |
| 3 | Inference rays_t 미업데이트 (CRITICAL) | ✅ 수정 | `base_volume_renderer.py` |
| 4 | 함수 시그니처 통일 | ✅ 해결 | 코드 일치 확인 |
| 5 | xray_mode 전달 체인 | ✅ 구현 완료 | 전체 파이프라인 확인 |
| 6 | Density bitfield threshold | 🟡 모니터링 | `zerorf.py` (occlusion culling off) |
| 7 | Cone-beam → Pinhole 변환 | 🟡 검증 필요 | `xray_dataset.py` |
| 8 | 학습 스크립트 | ✅ 해결 | `zerorf.py` 통합 |
