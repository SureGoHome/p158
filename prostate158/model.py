from monai.networks.nets import UNet
from .rrunet import RRUNet3D


def get_model(config):
    t = str(config.model.type).lower().strip()
    print(f"[get_model] model.type = '{t}'")  # ← debug print

    if t == "rrunet3d":
        return RRUNet3D(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            blocks_down=config.model.blocks_down,
            blocks_up=config.model.blocks_up,
            num_init_kernels=config.model.num_init_kernels,
            recurrent=config.model.recurrent,
            residual=config.model.residual,
            attention=config.model.attention,
            se=config.model.se,
        )

    if t == "unet":
        return UNet(
            spatial_dims=config.ndim,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            channels=config.model.channels,
            strides=config.model.strides,
            num_res_units=config.model.num_res_units,
            act=config.model.act,
            norm=config.model.norm,
            dropout=config.model.dropout,
        )

    raise ValueError(
        f"Unrecognized model.type '{config.model.type}'. "
        "Must be one of ['unet', 'rrunet3d']"
    )
