from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple


@dataclass(init=True, frozen=True, repr=True, eq=True)
class Configuration(Mapping[str, Any]):
    input_path: str | None = None
    input_type: str | None = None
    reverse_sort_filenames: bool = False
    to_hw: int | tuple[int, int] | None = (128, 256)
    skip_frames: int = 3
    db_dir: str = 'map.db'
    dist_thr: float = 0.1
    feature_dim: int = 512
    topk: int = 8
    ef: int = 128
    radius: float = 0.5
    ransac_ratio: float = 0.6
    ransac_it: int = 8
    ransac_thr: float = 0.65
    device: str = 'cpu'
    num_workers: int = 0
    batch_size: int = 8
    frame_queue_size: int = 8
    min_match_landmarks: int = 128
    camera_model: str = 'equirectangular'

    num_features: int = 256
    patch_size: int = 41
    angle_bins: int = 8
    spatial_bins: int = 8
    scale_n_levels: int = 3
    sigma: float = 1.6
    root_sift: bool = True
    double_image: bool = True

    host: str = '0.0.0.0'
    server_port: int = 7000
    client_port: int = 7001
    actors: Tuple[str, ...] = (
        'processor',
        'server',
        'optimizer',
    )
    processor_solver_only: bool = False

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)
