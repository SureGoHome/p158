import os
import yaml
import torch
import monai
import munch


def load_config(fn: str = "config.yaml"):
    "Load config from YAML and return a serialized dictionary object"
    with open(fn, "r") as stream:
        config = yaml.safe_load(stream)
    config = munch.munchify(config)

    if not config.overwrite:
        i = 1
        while os.path.exists(config.run_id + f"_{i}"):
            i += 1
        config.run_id += f"_{i}"

    config.out_dir = os.path.join(config.run_id, config.out_dir)
    config.log_dir = os.path.join(config.run_id, config.log_dir)

    if not isinstance(config.data.image_cols, (tuple, list)):
        config.data.image_cols = [config.data.image_cols]
    if not isinstance(config.data.label_cols, (tuple, list)):
        config.data.label_cols = [config.data.label_cols]

    config.transforms.mode = ("bilinear",) * len(config.data.image_cols) + (
        "nearest",
    ) * len(config.data.label_cols)
    return config


# def num_workers():
#     """Return sensible number of data-loading workers."""
#     try:
#         import resource

#         soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
#         max_by_files = soft_limit // 256
#     except ImportError:
#         # Windows: no resource module
#         import multiprocessing

#         return max(1, multiprocessing.cpu_count() - 2)

#     import multiprocessing

#     cpu_workers = multiprocessing.cpu_count() - 2

#     if max_by_files < cpu_workers:
#         print(
#             "Will not use all available workers: file descriptor limit too low ("
#             f"soft_limit={soft_limit}, allow only={max_by_files * 256})."
#             "Consider increasing system limits if needed."
#         )
#         return max(1, max_by_files)

#     return max(1, cpu_workers)


def num_workers():
    "Get max supported workers -2 for multiprocessing"
    import resource
    import multiprocessing

    # first check for max number of open files allowed on system
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    n_workers=multiprocessing.cpu_count() - 2
    # giving each worker at least 256 open processes should allow them to run smoothly
    max_workers = soft_limit // 256
    if max_workers < n_workers:
        print(
            "Will not use all available workers as number of allowed open files is to small"
            "to ensure smooth multiprocessing. Current limits are:\n"
            f"\t soft_limit: {soft_limit}\n"
            f"\t hard_limit: {hard_limit}\n"
            "try increasing the limits to at least {256*n_workers}."
            "See https://superuser.com/questions/1200539/cannot-increase-open-file-limit-past-4096-ubuntu"
            "for more details"
        )
        return max_workers

    return n_workers

USE_AMP = True if monai.utils.get_torch_version_tuple() >= (1, 6) and torch.cuda.is_available() else False
