"""
Inference script (inference5.py)
- Sliding-window inference (no random crops)
- Forces ADC/DWI to match T2 grid via ResampleToMatchd
- Inverts transforms back to ORIGINAL T2 geometry (same shape + affine)
- Saves with nibabel (qform/sform set) + optional shape/affine sanity checks

Usage:
  python -m inference5 \
    --t2 /path/T2w.nii.gz \
    --adc /path/ADC.nii.gz \
    --dwi /path/DWI_b1000.nii.gz \
    --out /path/pred_seg.nii.gz \
    --cfg tumor.yaml \
    --ckpt models/tumor.pt
"""

import os
import torch
import numpy as np
import nibabel as nib
from typing import List

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityd,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ConcatItemsd,
    Invertd,
    ResampleToMatchd,
)
from monai.inferers import SlidingWindowInferer

# Support running as a module (package) or as a plain script
try:
    from .model import get_model
    from .utils import load_config
except Exception:
    from model import get_model
    from utils import load_config


def _roi_from_config(config):
    """Best-effort: read roi/patch size from config, else default to 96^3."""
    roi = None
    tfm = getattr(config, "transforms", None)
    if tfm is not None:
        if hasattr(tfm, "rand_spatial_crop_samples") and tfm.rand_spatial_crop_samples:
            roi = tuple(tfm.rand_spatial_crop_samples.get("roi_size", [])) or None
        if (
            roi is None
            and hasattr(tfm, "rand_crop_pos_neg_label")
            and tfm.rand_crop_pos_neg_label
        ):
            roi = tuple(tfm.rand_crop_pos_neg_label.get("spatial_size", [])) or None
    return roi if roi else (96, 96, 96)


def _present_keys(
    t2_path: str, adc_path: str = None, dwi_path: str = None
) -> List[str]:
    keys = []
    if t2_path:
        keys.append("t2")
    if adc_path and os.path.exists(adc_path):
        keys.append("adc")
    if dwi_path and os.path.exists(dwi_path):
        keys.append("dwi")
    return keys


def load_pretrained_model(config, checkpoint_path):
    model = get_model(config).to(config.device)
    print(f"[inference5] loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state if isinstance(state, dict) else state, strict=True)
    model.eval()
    return model


def get_infer_transforms(config, keys: List[str]):
    """
    Inference transforms:
      - Load
      - Channel-first
      - Spacing (if configured)
      - Orientation (if configured)
      - Resample ADC/DWI exactly to T2 grid (FOV/shape)
      - Intensity scale/normalize
      - Concat -> 'image'
    """
    tfms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]
    if getattr(config.transforms, "spacing", None):
        tfms += [
            Spacingd(
                keys=keys,
                pixdim=config.transforms.spacing,
                mode=("bilinear",) * len(keys),
            )
        ]
    if getattr(config.transforms, "orientation", None):
        tfms += [Orientationd(keys=keys, axcodes=config.transforms.orientation)]

    other_keys = [k for k in keys if k != "t2"]
    if other_keys:
        tfms += [
            ResampleToMatchd(
                keys=other_keys, key_dst="t2", mode=("bilinear",) * len(other_keys)
            )
        ]

    tfms += [
        ScaleIntensityd(keys=keys, minv=0, maxv=1),
        NormalizeIntensityd(keys=keys),
        ConcatItemsd(keys=keys, name="image", dim=0),  # (C, D, H, W)
        EnsureTyped(keys=["image", "t2"], device=config.device, track_meta=True),
    ]
    return Compose(tfms)


def inference_pipeline(
    t2_path: str,
    adc_path: str = None,
    dwi_path: str = None,
    output_path: str = "prediction.nii.gz",
    config_path: str = "tumor.yaml",
    checkpoint_path: str = "models/tumor.pt",
    sanity_check: bool = True,
):
    """
    Runs inference and writes a segmentation aligned to the ORIGINAL T2 (same shape + affine).
    Returns:
        saved_path (str): path to the written NIfTI.
    """
    # --- config / device ---
    config = load_config(config_path)
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- capture original T2 metadata ---
    t2_img = nib.load(t2_path)
    original_shape = t2_img.shape
    original_affine = t2_img.affine

    # --- model ---
    model = load_pretrained_model(config, checkpoint_path)

    # --- transforms (no crops) ---
    keys = _present_keys(t2_path, adc_path, dwi_path)
    if "t2" not in keys:
        raise ValueError("t2_path is required and was not found on disk.")
    infer_transforms = get_infer_transforms(config, keys)
    sample = {"t2": t2_path}
    if "adc" in keys:
        sample["adc"] = adc_path
    if "dwi" in keys:
        sample["dwi"] = dwi_path

    data = infer_transforms(sample)
    image = data["image"].unsqueeze(0).to(config.device)  # [1, C, D, H, W]
    print(f"[inference5] preproc image shape: {tuple(image.shape)}")

    # --- sliding window ---
    roi_size = _roi_from_config(config)
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=getattr(config, "sw_batch_size", 1),
        overlap=0.5,
        mode="gaussian",
    )

    with torch.no_grad():
        logits = inferer(inputs=image, network=model)  # [1, out_c, D, H, W]

    # Argmax for multi-class; threshold for single-channel
    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits) > 0.5).float()[0, 0].cpu()
    else:
        pred = torch.argmax(logits, dim=1)[0].cpu().float()  # [D,H,W]

    # Invert back to original T2 geometry
    # Ensure a channel dimension for correct spatial-rank handling during inversion
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)  # [1, D, H, W]

    inv = Compose(
        [
            Invertd(
                keys="pred",
                transform=infer_transforms,
                orig_keys="t2",
                nearest_interp=True,
                to_tensor=True,
                meta_keys="pred_meta_dict",
                orig_meta_keys="t2_meta_dict",
            )
        ]
    )
    inv_out = inv(
        {
            "pred": pred,
            "t2": data["t2"],
            "t2_meta_dict": data.get("t2_meta_dict", {}),
        }
    )
    pred_inv = inv_out["pred"]
    # Remove channel dim if present
    if (
        isinstance(pred_inv, torch.Tensor)
        and pred_inv.ndim == 4
        and pred_inv.shape[0] == 1
    ):
        pred_inv = pred_inv[0]

    # Save with nibabel (set qform/sform so viewers honor the affine)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    seg = pred_inv.detach().cpu().numpy().astype(np.uint8)
    seg = np.squeeze(seg)
    affine = original_affine
    img = nib.Nifti1Image(seg, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, output_path)
    print(f"[inference5] saved: {output_path}")

    # Sanity checks
    if sanity_check:
        pred_img = nib.load(output_path)
        ok_shape = tuple(pred_img.shape) == tuple(original_shape)
        ok_aff = np.allclose(pred_img.affine, original_affine, atol=1e-4)
        print(
            f"[sanity] shape match: {ok_shape} | pred={pred_img.shape}, t2={original_shape}"
        )
        print(f"[sanity] affine match (allclose 1e-4): {ok_aff}")
        if not ok_shape or not ok_aff:
            print(
                "[sanity][WARNING] Saved mask may not perfectly overlay on original T2. "
                "Check transforms & metadata."
            )

    return output_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--t2", required=True, help="Path to T2 NIfTI")
    p.add_argument("--adc", default=None, help="Path to ADC NIfTI (optional)")
    p.add_argument("--dwi", default=None, help="Path to high-b DWI NIfTI (optional)")
    p.add_argument(
        "--out", required=True, help="Output NIfTI path for the segmentation"
    )
    p.add_argument("--cfg", default="tumor.yaml", help="Config YAML")
    p.add_argument("--ckpt", default="models/tumor.pt", help="Checkpoint path")
    p.add_argument(
        "--no_sanity", action="store_true", help="Disable shape/affine sanity checks"
    )
    args = p.parse_args()

    saved = inference_pipeline(
        t2_path=args.t2,
        adc_path=args.adc,
        dwi_path=args.dwi,
        output_path=args.out,
        config_path=args.cfg,
        checkpoint_path=args.ckpt,
        sanity_check=not args.no_sanity,
    )
    print(f"[done] {saved}")
