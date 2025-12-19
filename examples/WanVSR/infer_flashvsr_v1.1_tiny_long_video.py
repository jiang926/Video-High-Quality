#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, subprocess
import argparse
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

from diffsynth import ModelManager, FlashVSRTinyLongPipeline
from utils.utils import Causal_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder

def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def smallest_8n1_geq(n):  # 最小的 8n+1 且 >= n
    if n <= 1:
        return 1
    rem = (n - 1) % 8
    return n if rem == 0 else n + (8 - rem)

def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def has_audio_ffprobe(path: str) -> bool:
    # 使用 ffprobe 检测源视频是否含音频流
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                path,
            ],
            stderr=subprocess.STDOUT,
        )
        return bool(out.strip())
    except FileNotFoundError:
        print("[Warn] 未检测到 ffprobe，跳过音频检测与合并。")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[Warn] ffprobe 检测音频失败：{e.output.decode(errors='ignore')}")
        return False

def mux_audio_from_src(src_video: str, dst_video: str):
    """
    将源视频音频复用到已生成的无声视频文件。
    若失败，不影响原始无声视频。
    """
    tmp_out = dst_video + ".tmp_mux.mp4"
    try:
        subprocess.check_call(
            [
                "ffmpeg", "-y",
                "-i", dst_video,    # 0: 目标视频（无声）
                "-i", src_video,    # 1: 源视频（带音频）
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "copy",
                tmp_out,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        os.replace(tmp_out, dst_video)
        print(f"[Audio] 已复用源视频音频到输出：{dst_video}")
    except FileNotFoundError:
        print("[Warn] 未检测到 ffmpeg，无法自动合并音频。")
    except subprocess.CalledProcessError as e:
        print(f"[Warn] ffmpeg 合并音频失败：{e}. 已保留无声版本：{dst_video}")
        try: os.remove(tmp_out)
        except Exception: pass

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    # 使用 .copy() 确保 numpy 数组可写，避免 PyTorch 关于只读数组的警告
    t = torch.from_numpy(np.asarray(img, np.uint8).copy()).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0                                              # CHW in [-1,1]
    return t.to(dtype)

def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def prepare_input_tensor(path: str, scale: float = 4, dtype=torch.bfloat16, device='cuda', max_frames: int = None):
    if max_frames is not None and max_frames < 5:
        raise ValueError("max_frames 应该至少为 5（模型需要 8n+1 帧）。")
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {N0}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        if max_frames is not None:
            paths = paths[:max_frames + 4]
        # 向上补齐到最近的 8n+1，避免截短视频
        F = smallest_8n1_geq(len(paths))
        pad = F - len(paths)
        if pad > 0:
            paths = paths + [paths[-1]] * pad
        paths = paths[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        count_img = 0
        for p in paths:
            with Image.open(p).convert('RGB') as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, 'cpu'))
            print(count_img, len(paths), end = '\r')
            count_img+=1
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)  # 1 C F H W
        fps = 30
        return vid, tH, tW, F, fps

    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try: meta = rdr.get_meta_data()
        except Exception: pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf,int) and nf>0: return nf
            except Exception: pass
            try: return r.count_frames()
            except Exception:
                n=0
                try:
                    while True: r.get_data(n); n+=1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        idx = list(range(total)) + [total-1]*4
        if max_frames is not None:
            idx = idx[:max_frames + 4]
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(idx)}.")
        idx = idx[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, 'cpu'))
                print(i, len(idx), end = '\r')
        finally:
            try: rdr.close()
            except Exception: pass

        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)  # 1 C F H W
        return vid, tH, tW, F, fps

    raise ValueError(f"Unsupported input: {path}")

def init_pipeline():
    print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([
        "./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors",
    ])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    LQ_proj_in_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to('cuda')

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
    mis = pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR-v1.1/TCDecoder.ckpt"), strict=False)
    print(mis)

    pipe.to('cuda'); pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(); pipe.load_models_to_device(["dit","vae"])
    return pipe

def parse_args():
    parser = argparse.ArgumentParser(description="FlashVSR v1.1 tiny long video 推理脚本")
    parser.add_argument("--input", type=str, default="./inputs/example4.mp4", help="输入视频或帧目录路径")
    parser.add_argument("--output", type=str, default=None, help="输出视频文件路径（默认自动命名到 ./results）")
    parser.add_argument("--scale", type=float, default=4.0, help="放大倍数")
    parser.add_argument("--sparse_ratio", type=float, default=2.0, help="稀疏比，推荐 1.5 或 2.0")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--max_frames", type=int, default=None, help="最多处理的原始帧数（不含末尾 padding），需 >=5")
    parser.add_argument("--device", type=str, default="cuda", help="设备：cuda 或 cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="推理精度")
    parser.add_argument("--keep_audio", action="store_true", dest="keep_audio", help="尝试将源视频音频复用到输出（默认开启）")
    parser.add_argument("--no_keep_audio", action="store_false", dest="keep_audio", help="关闭音频复用")
    parser.set_defaults(keep_audio=True)
    return parser.parse_args()


def main():
    args = parse_args()

    RESULT_ROOT = "./results"
    os.makedirs(RESULT_ROOT, exist_ok=True)
    inputs = [args.input]

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    seed, scale, device = args.seed, args.scale, args.device
    sparse_ratio = args.sparse_ratio      # Recommended: 1.5 or 2.0. 1.5 → faster; 2.0 → more stable.
    pipe = init_pipeline()

    for p in inputs:
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        name = os.path.basename(p.rstrip('/'))
        if name.startswith('.'):
            continue
        try:
            LQ, th, tw, F, fps = prepare_input_tensor(
                p, scale=scale, dtype=dtype, device=device, max_frames=args.max_frames
            )
        except Exception as e:
            print(f"[Error] {name}: {e}"); continue

        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), 
            kv_ratio=3.0,
            local_range=11,  # Recommended: 9 or 11. local_range=9 → sharper details; 11 → more stable results.
            color_fix = True,
        )

        video = tensor2video(video)
        save_path = args.output if args.output else os.path.join(RESULT_ROOT, f"FlashVSR_v1.1_Tiny_Long_{name.split('.')[0]}_seed{seed}.mp4")
        save_video(video, save_path, fps=fps, quality=5)

        if args.keep_audio and is_video(p):
            if has_audio_ffprobe(p):
                mux_audio_from_src(p, save_path)
            else:
                print(f"[Audio] 源视频未检测到音频或工具缺失，跳过音频复用：{p}")

    print("Done.")

if __name__ == "__main__":
    main()

