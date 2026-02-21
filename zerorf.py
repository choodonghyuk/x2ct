import sys
import shutil
import os
import cv2
import tqdm
import json
import numpy
import wandb
import numpy as np
import torch
import torch_redstone as rst
from sklearn.cluster import KMeans
from lib.models.autoencoders import MultiSceneNeRF
from mmgen.models import build_model, build_module
from lib.core.optimizer import build_optimizers
from lib.core.ssdnerf_gui import OrbitCamera
from lib.datasets.nerf_synthetic import NerfSynthetic
from lib.datasets.oppo import OppoDataset
from lib.datasets.xray_dataset import XrayDataset
from PIL import Image
import einops
from opt import config_parser
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")  # 모든 경고 메시지 무시

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["WANDB_MODE"] = "disabled"

def kmeans_downsample(points, n_points_to_sample):
    kmeans = KMeans(n_points_to_sample).fit(points)
    return ((points - kmeans.cluster_centers_[..., None, :]) ** 2).sum(-1).argmin(-1).tolist()

args = config_parser()
pprint(args)

model_scaling_factor = 16
device = args.device

BLENDER_TO_OPENCV_MATRIX = numpy.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
], dtype=numpy.float32)

code_size = (3, args.model_ch, args.model_res, args.model_res)

rst.seed(args.seed)

poses = []
intrinsics = []
if args.load_image:
    image = numpy.array(Image.open(args.load_image)).astype(numpy.float32) / 255.0
    image = torch.tensor(image).cuda()
    images = einops.rearrange(image, '(ph h) (pw w) c -> (ph pw) h w c', ph=3, pw=2)[None]
    meta = json.load(open(os.path.join(os.path.dirname(__file__), "meta.json")))
    poses = numpy.array([
        (numpy.array(frame['transform_matrix']) @ BLENDER_TO_OPENCV_MATRIX) * 2
        for frame in meta['sample_0']['view_frames']
    ])
    _, b, h, w, c = images.shape
    x, y = w / 2, h / 2
    focal_length = y / numpy.tan(meta['fovy'] / 2)
    intrinsics = numpy.array([[focal_length, focal_length, x, y]] * args.n_views)

# Resolve data paths to absolute before changing cwd
if args.data_path:
    args.data_path = os.path.abspath(args.data_path)
if args.data_dir:
    args.data_dir = os.path.abspath(args.data_dir)

if args.data_path:
    data_name = os.path.splitext(os.path.basename(args.data_path))[0]
else:
    data_name = args.obj

# 2. 폴더명 조합: {프로젝트명}_{데이터명}_res{해상도}_ch{채널}
# 예: results/test_chest_res20_ch8
folder_name = f"{args.proj_name}_{data_name}_{args.rep}_res{args.model_res}_ch{args.model_ch}"
work_dir = os.path.join("results", folder_name)

# 3. 폴더 생성 및 이동
print(f">> Working Directory: {work_dir}")
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)

print(f">> Loading X-ray dataset: {args.obj} ...")
data_path = args.data_path or os.path.join(args.data_dir, args.obj + '.pickle')

# Train / Val 데이터셋
dataset = XrayDataset(data_path, split='train')
val = XrayDataset(data_path, split='val')

# GT Volume 로드 (3D PSNR 계산용)
gt_volume = None
if hasattr(dataset, 'gt_volume'):
    gt_volume = dataset.gt_volume
    if isinstance(gt_volume, np.ndarray):
        gt_volume = torch.from_numpy(gt_volume).to(device)
    else:
        gt_volume = gt_volume.to(device)
    print(f">> GT Volume loaded. Shape: {gt_volume.shape}")

entry = dataset[0]
args.n_views = dataset.n_views  # auto-detect view count
selected_idxs = list(range(args.n_views))  # use all views

# 학습용 데이터 엔트리
data_entry = dict(
    cond_imgs=torch.tensor(entry['cond_imgs'][None]).float().to(device),
    cond_poses=torch.tensor(entry['cond_poses'])[None].float().to(device),
    cond_intrinsics=torch.tensor(entry['cond_intrinsics'])[None].float().to(device),
    scene_id=[0],
    scene_name=[args.proj_name]
)
val_entry_data = val[0]
# 검증용 데이터 엔트리 (gt_volume 포함)
val_entry = dict(
    test_imgs=torch.tensor(val_entry_data['cond_imgs'][None]).float().to(device),
    test_poses=torch.tensor(val_entry_data['cond_poses'])[None].float().to(device),
    test_intrinsics=torch.tensor(val_entry_data['cond_intrinsics'])[None].float().to(device),
    scene_id=[0],
    scene_name=[args.proj_name],
    gt_volume=gt_volume  # [중요] 모델 내부 eval_and_viz로 전달됨
    )
test_entry = val_entry 

selected_idxs = list(range(args.n_views))

pic_h = data_entry['cond_imgs'].shape[-3]
pic_w = data_entry['cond_imgs'].shape[-2]

xray_mode = args.dataset == "xray"

decoder_1 = dict(
    type='TensorialDecoder',
    preprocessor=dict(
        type='TensorialGenerator',
        in_ch=args.model_ch, out_ch=16, noise_res=args.model_res,
        tensor_config=(
            ['xy', 'z', 'yz', 'x', 'zx', 'y']
        )
    ),
    subreduce=2,
    reduce='cat',
    separate_density_and_color=False,
    sh_coef_only=False,
    sdf_mode=False,
    xray_mode=xray_mode,
    max_steps=1024 if not args.load_image else 320,
    n_images=args.n_views,
    image_h=pic_h,
    image_w=pic_w,
    has_time_dynamics=False,
    visualize_mesh=False,
    occlusion_culling_th=0.0 if xray_mode else 0.0001,
)
decoder_2 = dict(
    type='FreqFactorizedDecoder',
    preprocessor=dict(
        type='TensorialGenerator',
        in_ch=args.model_ch, out_ch=16, noise_res=args.model_res,
        tensor_config=['xyz', 'xyz']
    ),
    subreduce=1,
    reduce='cat',
    separate_density_and_color=False,
    sh_coef_only=False,
    sdf_mode=False,
    xray_mode=xray_mode,
    max_steps=1024 if not args.load_image else 640,
    n_images=args.n_views,
    image_h=pic_h,
    image_w=pic_w,
    has_time_dynamics=False,
    freq_bands=[None, 0.4],
    visualize_mesh=False,
    occlusion_culling_th=0.0 if xray_mode else 0.0001,
)

patch_reg_loss = build_module(dict(
    type='MaskedTVLoss',
    power=1.5,
    loss_weight=0.00
))
nerf: MultiSceneNeRF = build_model(dict(
    type='MultiSceneNeRF',
    code_size=code_size,
    code_activation=dict(type='IdentityCode'),
    grid_size=64,
    patch_size=32,
    decoder=decoder_2 if args.rep == 'dif' else decoder_1,
    decoder_use_ema=False,
    bg_color=0.0 if xray_mode else 1.0,
    pixel_loss=dict(
        type='MSELoss',
        loss_weight=3.2
    ),
    use_lpips_metric=False if xray_mode else (torch.cuda.mem_get_info()[1] // 1000 ** 3 >= 32),
    cache_size=1,
    cache_16bit=False,
    init_from_mean=False
), train_cfg = dict(
    dt_gamma_scale=0,
    density_thresh=-1.0,
    extra_scene_step=0,
    n_inverse_rays=args.n_rays_init,
    n_decoder_rays=args.n_rays_init,
    loss_coef=0.1 / (pic_h * pic_w),
    optimizer=dict(type='Adam', lr=0, weight_decay=0.),
    lr_scheduler=dict(type='ExponentialLR', gamma=0.99),
    cache_load_from=None,
    viz_dir=None,
    loss_denom=1.0,
    decoder_grad_clip=1.0
),
test_cfg = dict(
    img_size=(pic_h, pic_w),
    density_thresh=0,
    max_render_rays=1024, # [메모리 보호] 검증 시 Chunking
    dt_gamma_scale=0.0,
    n_inverse_rays=args.n_rays_init,
    loss_coef=0.1 / (pic_h * pic_w),
    n_inverse_steps=400,
    optimizer=dict(type='Adam', lr=0.0, weight_decay=0.),
    lr_scheduler=dict(type='ExponentialLR', gamma=0.998),
    return_depth=False
))

nerf.bg_color = nerf.decoder.bg_color = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

nerf.to(device)
nerf.train()

optim = build_optimizers(nerf, dict(decoder=dict(type='AdamW', lr=args.net_lr, foreach=True, weight_decay=0.2, betas=(0.9, 0.98))))
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim['decoder'], args.n_iters, eta_min=args.net_lr_decay_to)

prog = tqdm.trange(args.n_iters)
best_psnr = 0.0
best_psnr_3d = 0.0

# [추가] 로그 파일 생성 및 헤더 작성
log_txt_path = "val_log.txt"
with open(log_txt_path, "w") as f:
    f.write("Step\tVal_PSNR\t3D_PSNR\n")  # 헤더

for j in prog:
    # -------------------------------------------------------------------------
    # 1. 학습 스텝 (Training Step)
    # -------------------------------------------------------------------------
    outputs = nerf.train_step(data_entry, optim)
    lv = outputs['log_vars']
    lr_sched.step()

    # 불필요한 로그 키 제거
    if 'code_rms' in lv: lv.pop('code_rms')
    if 'loss' in lv: lv.pop('loss')
    
    if j == 50:
        nerf.train_cfg['n_inverse_rays'] = round((args.n_rays_init * args.n_rays_up) ** 0.5)
        nerf.train_cfg['n_decoder_rays'] = round((args.n_rays_init * args.n_rays_up) ** 0.5)
    if j == 100:
        nerf.train_cfg['n_inverse_rays'] = args.n_rays_up
        nerf.train_cfg['n_decoder_rays'] = args.n_rays_up

    # -------------------------------------------------------------------------
    # 2. 검증 (Validation) - 1000번마다 실행
    # -------------------------------------------------------------------------
    if j % args.val_iter == args.val_iter - 1:
        cache = nerf.cache[0]
        nerf.eval()
        
        with torch.no_grad():
            # 스텝별 폴더 생성
            current_viz_dir = os.path.join("viz", f"step{j+1:06d}")
            os.makedirs(current_viz_dir, exist_ok=True)
            
            # Validation 수행
            log_vars, _ = nerf.eval_and_viz(
                val_entry, nerf.decoder,
                cache['param']['code_'][None].to(device),
                cache['param']['density_bitfield'][None].to(device),
                current_viz_dir,
                cfg=nerf.test_cfg
            )
        
        # 점수 가져오기
        this_psnr = log_vars.get('test_psnr', 0.0)
        this_psnr_3d = log_vars.get('val_psnr_3d', 0.0)
        
        # [핵심 1] 터미널에 별도 로그 출력 (Progress Bar와 섞이지 않게)
        # 예: [Val] Step 1000 | 2D PSNR: 30.1234 | 3D PSNR: 32.5678
        log_message = f"[Val] Step {j+1:05d} | 2D PSNR: {this_psnr:.4f} | 3D PSNR: {this_psnr_3d:.4f}"
        tqdm.tqdm.write(log_message)

        # [핵심 2] TXT 파일에 저장
        with open(log_txt_path, "a") as f:
            f.write(f"{j+1}\t{this_psnr:.4f}\t{this_psnr_3d:.4f}\n")

        # Best Model 저장
        lv.update(log_vars)
        if this_psnr > best_psnr:
            best_psnr = this_psnr
            torch.save(nerf.state_dict(), open("nerf-zerorf-best.pt", "wb"))
        if this_psnr_3d > best_psnr_3d:
            best_psnr_3d = this_psnr_3d

        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. 로그 업데이트 (Progress Bar)
    # -------------------------------------------------------------------------
    # 여기서는 3D PSNR을 굳이 표시하지 않고, 기본적인 학습 정보만 표시 (깔끔하게)
    # 검증 때 얻은 log_vars가 lv에 업데이트되어 있긴 하지만, 다음 step에서 자연스럽게 사라지거나 유지됨
    # 사용자 요청대로 '평소엔 깔끔하게, 검증 때만 확인' 의도에 맞춤
    log_msg = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in lv.items() 
        if k not in ['test_psnr', 'val_psnr_3d', 'test_ssim', 'test_lpips']} # 진행바에서는 제외
    prog.set_postfix(**log_msg)

# -------------------------------------------------------------------------
# 4. 학습 종료 (Final)
# -------------------------------------------------------------------------
print(f"\n>> Training Complete. Saving final model...")
torch.save(nerf.state_dict(), open("nerf-zerorf-final.pt", "wb"))