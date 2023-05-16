from typing import Iterator, Tuple
import torchvision.transforms.functional as TF
from PIL import Image, UnidentifiedImageError
import os
import torchdata.datapipes as dp
from torch import Tensor
import av
from loguru import logger


@dp.functional_datapipe('image_sequence')
class ImageSequenceDataPipe(dp.iter.IterDataPipe):
    def __init__(self, path: str, to_hw: int | Tuple[int, int] | None = None, reverse: bool = False, **kwargs):
        super().__init__()
        self.path = path
        self.to_hw = to_hw
        self.reverse = reverse
        self._size = len(os.listdir(self.path))

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def read_image_sequence(
        path: str, to_hw: int | Tuple[int, int] | None = None, reverse: bool = False, **kwargs
    ) -> Iterator[Tensor]:
        subfiles = os.listdir(path)
        subfiles.sort(reverse=reverse)
        for subfile in subfiles:
            subfile_path = os.path.join(path, subfile)
            try:
                img = Image.open(subfile_path)
                if to_hw is not None:
                    img = TF.resize(img, to_hw, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
                t: Tensor = TF.pil_to_tensor(img)
                yield t
            except UnidentifiedImageError:
                pass

    def __iter__(self) -> Iterator[Tensor]:
        for t in self.read_image_sequence(self.path, self.to_hw, self.reverse):
            yield t


@dp.functional_datapipe('video')
class VideoDataPipe(dp.iter.IterDataPipe):
    def __init__(
        self,
        path: str,
        to_hw: int | Tuple[int, int] | None = None,
        skip_frames: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.to_hw = to_hw
        self.skip_frames = skip_frames
        self._size = int(av.open(path).streams.video[0].frames // (skip_frames + 1))

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def read_video(
        path: str,
        to_hw: int | Tuple[int, int] | None = None,
        skip_frames: int = 0,
        *args,
        **kwargs,
    ) -> Iterator[Tensor]:
        container = av.open(path)
        container.streams.video[0].thread_type = 'AUTO'
        skipped = 0
        for frame in container.decode(video=0):
            if skipped < skip_frames:
                skipped += 1
                continue
            skipped = 0
            img = frame.to_image()
            if to_hw is not None:
                img = TF.resize(img, to_hw, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            img = TF.pil_to_tensor(img)
            yield img

    def __iter__(self) -> Iterator[Tensor]:
        for t in self.read_video(self.path, self.to_hw, self.skip_frames):
            yield t
