import os
import torch
import monai
import numpy as np
import nibabel as nib
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
    Resize,
)
from monai.inferers import SlidingWindowInferer
from .model import get_model
from .utils import load_config


def _get_roi_size_from_config(config):
    # Try to read ROI size from your YAML; default to 96³
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
    print(f"[inference] loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=config.device)
    # handle PyTorch 2.6+ default arg change
    model.load_state_dict(state if isinstance(state, dict) else state, strict=True)
    model.eval()
    return model


def get_infer_transforms(config):
    keys = [
        "t2",
        "adc",
        "dwi",
    ]  # rename here if your YAML uses different image_col names
    tfms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]
    # match training preproc order: spacing -> orientation -> intensity
    if getattr(config.transforms, "spacing", None):
        tfms += [
            Spacingd(
                keys=keys, pixdim=config.transforms.spacing, mode=("bilinear",) * 3
            )
        ]
    if getattr(config.transforms, "orientation", None):
        tfms += [Orientationd(keys=keys, axcodes=config.transforms.orientation)]
    # intensity
    tfms += [
        ScaleIntensityd(keys=keys, minv=0, maxv=1),
        NormalizeIntensityd(keys=keys),
        ConcatItemsd(keys=keys, name="image", dim=0),  # (C,D,H,W)
        EnsureTyped(keys=["image", "t2"], device=config.device, track_meta=True),
    ]
    return Compose(tfms)


def get_post_transforms(config, infer_transforms, save_dir, out_name):
    tfms = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys="pred", argmax=True
        ),  # no hard-coded class index; 2-class -> tumor=1
        # bring prediction back to original T2 space (undo spacing/orientation)
        Invertd(
            keys="pred",
            transform=infer_transforms,
            orig_keys="t2",  # use t2 metadata as reference
            nearest_interp=True,  # keep labels crisp
            to_tensor=True,
        ),
    ]
    return Compose(tfms)


def inference_pipeline(
    t2_path,
    adc_path,
    dwi_path,
    output_path,
    config_path="tumor.yaml",
    checkpoint_path="models/tumor.pt",
):
    # --- config / device ---
    config = load_config(config_path)
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- capture original T2 metadata ---
    try:
        t2_img_ = nib.load(t2_path)
        original_shape = t2_img_.shape
        original_affine = t2_img_.affine
    except Exception:
        original_shape = None
        original_affine = None

    # --- model ---
    model = load_pretrained_model(config, checkpoint_path)

    # --- transforms (NO CROP HERE) ---
    infer_transforms = get_infer_transforms(config)
    data = infer_transforms({"t2": t2_path, "adc": adc_path, "dwi": dwi_path})

    image = data["image"].unsqueeze(0).to(config.device)  # [1, C, D, H, W]
    print(f"[inference] image (after preproc) shape: {tuple(image.shape)}")

    # --- sliding window inferer (same ROI as training) ---
    roi_size = _get_roi_size_from_config(config)
    sw_inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=getattr(config, "sw_batch_size", 1),
        overlap=0.5,
        mode="gaussian",
    )

    with torch.no_grad():
        logits = sw_inferer(inputs=image, network=model)  # [1, out_c, D, H, W]
        pred = {"pred": logits[0].detach().cpu(), "t2": data["t2"]}  # attach t2 meta

    # --- post: softmax->argmax, invert back to original T2, save ---
    save_dir = os.path.dirname(output_path)
    out_name = os.path.basename(output_path)
    post_transforms = get_post_transforms(config, infer_transforms, save_dir, out_name)
    out = post_transforms(pred)

    # --- explicit save to the requested output_path using nibabel ---
    os.makedirs(save_dir or ".", exist_ok=True)
    seg_tensor = out["pred"]

    # Ensure shape matches original T2; if not, resize as a safety fallback
    if original_shape is not None and tuple(seg_tensor.shape) != tuple(original_shape):
        resize_back = Resize(spatial_size=original_shape, mode="nearest")
        seg_tensor = resize_back(seg_tensor)

    seg = seg_tensor.detach().cpu().numpy().astype(np.uint8)
    seg = np.squeeze(seg)

    meta = out.get("pred_meta_dict") or out.get("t2_meta_dict") or {}
    affine = meta.get("affine") or meta.get("original_affine") or original_affine
    if affine is None:
        # fallback to identity if metadata is missing
        affine = np.eye(4, dtype=float)

    img = nib.Nifti1Image(seg, affine)
    nib.save(img, output_path)

    print(f"[inference] saved: {output_path}")


# import os
# import torch
# import monai
# from monai.transforms import (
#     LoadImage,
#     EnsureChannelFirst,
#     ScaleIntensity,
#     NormalizeIntensity,
#     Spacing,
#     Orientation,
#     SaveImage,
#     Compose,
#     ConcatItemsd,
#     LoadImaged,
#     EnsureChannelFirstd,
#     ScaleIntensityd,
#     NormalizeIntensityd,
#     Spacingd,
#     Orientationd,
#     SpatialCropd,
#     CenterSpatialCropd,
# )
# from .model import get_model
# from .utils import load_config


# def load_pretrained_model(config, checkpoint_path):
#     """Load the pretrained model"""
#     model = get_model(config).to(config.device)
#     print(f"Loading checkpoint from {checkpoint_path}")
#     model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
#     model.eval()
#     return model


# def get_transforms(config):
#     """Get basic preprocessing transforms"""
#     # Create dictionary transforms since we have multiple inputs
#     transforms = [
#         LoadImaged(keys=config.data.image_cols),
#         EnsureChannelFirstd(keys=config.data.image_cols),
#     ]

#     if config.transforms.spacing:
#         transforms.append(
#             Spacingd(
#                 keys=config.data.image_cols,
#                 pixdim=config.transforms.spacing,
#                 mode="bilinear",
#             )
#         )

#     if config.transforms.orientation:
#         transforms.append(
#             Orientationd(
#                 keys=config.data.image_cols, axcodes=config.transforms.orientation
#             )
#         )

#     # Add center spatial crop to ensure consistent dimensions
#     transforms.append(
#         CenterSpatialCropd(
#             keys=config.data.image_cols,
#             roi_size=(64, 64, 64),  # Use the same size as in config
#         )
#     )

#     transforms.extend(
#         [
#             ScaleIntensityd(keys=config.data.image_cols, minv=0, maxv=1),
#             NormalizeIntensityd(keys=config.data.image_cols),
#         ]
#     )

#     return monai.transforms.Compose(transforms)


# def inference_pipeline(
#     t2_path,
#     adc_path,
#     dwi_path,
#     output_path,
#     config_path="tumor.yaml",
#     checkpoint_path="models/tumor.pt",
# ):
#     """Run inference on a single case with multiple input channels

#     Args:
#         t2_path: Path to T2W image (NIfTI format)
#         adc_path: Path to ADC image (NIfTI format)
#         dwi_path: Path to DWI image (NIfTI format)
#         output_path: Path where to save the output segmentation
#         config_path: Path to config file
#         checkpoint_path: Path to model checkpoint
#     """
#     # Load config
#     config = load_config(config_path)
#     if torch.cuda.is_available():
#         config.device = "cuda:0"
#     else:
#         config.device = "cpu"

#     # Load model
#     model = load_pretrained_model(config, checkpoint_path)

#     # Setup transforms
#     transforms = get_transforms(config)

#     # Create input dictionary
#     input_dict = {"t2": t2_path, "adc": adc_path, "dwi": dwi_path}

#     # Load and preprocess images
#     print(f"Processing images...")
#     data = transforms(input_dict)

#     # Stack the channels correctly
#     # Each image should be [C, D, H, W]
#     t2 = data["t2"]  # Should already be in correct format from transforms
#     adc = data["adc"]
#     dwi = data["dwi"]

#     # Stack along channel dimension
#     image = torch.cat([t2, adc, dwi], dim=0)  # [3, D, H, W]

#     # Add batch dimension
#     image = image.unsqueeze(0).to(config.device)  # [1, 3, D, H, W]

#     print(f"Input shape: {image.shape}")

#     # Run inference
#     print("Running inference...")
#     with torch.no_grad():
#         output = model(image)
#         # Apply softmax for multi-class segmentation
#         output = torch.softmax(output, dim=1)
#         # Get the tumor class (class 1)
#         tumor_pred = output[:, 1:2]
#         # Threshold at 0.5
#         tumor_pred = (tumor_pred > 0.5).float()

#     # Save output
#     print(f"Saving output to: {output_path}")
#     tumor_pred = tumor_pred.squeeze().cpu()
#     saver = SaveImage(
#         output_dir=os.path.dirname(output_path),
#         output_postfix="",
#         output_ext=".nii.gz",
#         separate_folder=False,
#     )
#     saver(tumor_pred)
