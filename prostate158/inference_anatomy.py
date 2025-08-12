import os
import torch
import numpy as np
import nibabel as nib

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
    Activationsd,
    AsDiscreted,
    Invertd,
)
from monai.inferers import SlidingWindowInferer

# If used inside a package, keep relative; else fallback to local imports
try:
    from .model import get_model
    from .utils import load_config
except Exception:
    from model import get_model
    from utils import load_config


def _get_roi_size_from_config(config):
    """Best-effort: read roi/patch size from YAML; default to 96^3."""
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


def load_pretrained_model(config, checkpoint_path):
    model = get_model(config).to(config.device)
    print(f"[anatomy-inf] loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state if isinstance(state, dict) else state, strict=True)
    model.eval()
    return model


def get_infer_transforms(config):
    """
    T2-only pipeline:
      Load -> ChannelFirst -> Spacing -> Orientation ->
      Intensity scale/normalize -> Concat(t2->'image') -> EnsureTyped(track_meta)
    """
    keys = ["t2"]
    tfms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]
    if getattr(config.transforms, "spacing", None):
        tfms += [
            Spacingd(keys=keys, pixdim=config.transforms.spacing, mode=("bilinear",))
        ]
    if getattr(config.transforms, "orientation", None):
        tfms += [Orientationd(keys=keys, axcodes=config.transforms.orientation)]

    tfms += [
        ScaleIntensityd(keys=keys, minv=0, maxv=1),
        NormalizeIntensityd(keys=keys),
        ConcatItemsd(keys=keys, name="image", dim=0),  # -> (1, D, H, W)
        EnsureTyped(keys=["image", "t2"], device=config.device, track_meta=True),
    ]
    return Compose(tfms)


def get_post_transforms(infer_transforms):
    """
    Multi-class: softmax -> argmax -> invert back to ORIGINAL T2 geometry.
    """
    return Compose(
        [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            Invertd(
                keys="pred",
                transform=infer_transforms,
                orig_keys="t2",
                nearest_interp=True,  # keep labels crisp
                to_tensor=True,
            ),
        ]
    )


def inference_pipeline(
    t2_path: str,
    output_path: str,
    config_path: str = "anatomy.yaml",
    checkpoint_path: str = "models/anatomy.pt",
    sanity_check: bool = True,
):
    """
    Runs anatomy segmentation (T2-only, out_channels=3) and saves a labelmap
    aligned to the ORIGINAL T2 (same shape + affine).
    """
    # --- config / device ---
    config = load_config(config_path)
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- original T2 metadata ---
    t2_img = nib.load(t2_path)
    original_shape = t2_img.shape
    original_affine = t2_img.affine

    # --- model ---
    model = load_pretrained_model(config, checkpoint_path)

    # --- transforms (NO CROP) ---
    infer_transforms = get_infer_transforms(config)
    data = infer_transforms({"t2": t2_path})
    image = data["image"].unsqueeze(0).to(config.device)  # [1, 1, D, H, W]
    print(f"[anatomy-inf] preproc image shape: {tuple(image.shape)}")

    # --- sliding window inferer ---
    roi_size = _get_roi_size_from_config(config)
    sw_inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=getattr(config, "sw_batch_size", 1),
        overlap=0.5,
        mode="gaussian",
    )

    with torch.no_grad():
        logits = sw_inferer(inputs=image, network=model)  # [1, out_c(=3), D,H,W]

    # simple sanity on channel count
    if logits.shape[1] != 3:
        print(
            f"[WARN] Model produced {logits.shape[1]} channels; expected 3 for anatomy."
        )

    # pack for post (softmax->argmax + invert)
    pred_dict = {"pred": logits[0].cpu(), "t2": data["t2"]}
    post = get_post_transforms(infer_transforms)
    out = post(pred_dict)

    # --- save (nibabel with qform/sform set) ---
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    seg_tensor = out["pred"]  # [D,H,W] on ORIGINAL grid
    seg = seg_tensor.detach().cpu().numpy().astype(np.uint8)
    seg = np.squeeze(seg)
    img = nib.Nifti1Image(seg, original_affine)
    img.set_qform(original_affine, code=1)
    img.set_sform(original_affine, code=1)
    nib.save(img, output_path)
    print(f"[anatomy-inf] saved: {output_path}")

    # --- sanity checks (shape + affine) ---
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
    p.add_argument(
        "--out", required=True, help="Output NIfTI path for the segmentation"
    )
    p.add_argument("--cfg", default="anatomy.yaml", help="Config YAML")
    p.add_argument("--ckpt", default="models/anatomy.pt", help="Checkpoint path")
    p.add_argument(
        "--no_sanity", action="store_true", help="Disable shape/affine sanity checks"
    )
    args = p.parse_args()

    saved = inference_pipeline(
        t2_path=args.t2,
        output_path=args.out,
        config_path=args.cfg,
        checkpoint_path=args.ckpt,
        sanity_check=not args.no_sanity,
    )
    print(f"[done] {saved}")
