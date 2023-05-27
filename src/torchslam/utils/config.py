from abc import ABCMeta
from typing import Any, List, Mapping, Tuple, Type


class config:
    input_path: str | None = None
    input_type: str | None = None
    reverse_sort_filenames: bool = False
    to_hw: int | tuple[int, int] | None = (512, 1024)
    skip_frames: int = 0
    db_dir: str = 'map.db'
    dist_thr: float = 0.1
    feature_dim: int = 128
    topk: int = 8
    ef: int = 128
    radius: float = 0.5
    ransac_ratio: float = 0.6
    ransac_it: int = 8
    ransac_thr: float = 0.75
    device: str = 'cpu'
    num_workers: int = 0
    batch_size: int = 4
    min_match_landmarks: int = 128
    max_n_keyframes_in_local_tracking_queue: int = 8
    camera_model: str = 'equirectangular'
    num_features: int = 256
    patch_size: int = 41
    angle_bins: int = 8
    spatial_bins: int = 4
    scale_n_levels: int = 3
    sigma: float = 1.6
    root_sift: bool = True
    double_image: bool = True
    host: str = '0.0.0.0'
    server_port: int = 7000
    client_port: int = 7001
    map_resolution: float = 0.0001
    no_merge_if_dist_larger_than: float = 0.02
    min_tracking_keyframes: int = 128
    actors: Tuple[str, ...] = (
        'processor',
        'server',
        'optimizer',
    )
    processor_solver_only: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(config, k, v)


def load(**kwargs):
    config(**kwargs)
