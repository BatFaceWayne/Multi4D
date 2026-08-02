import json
import pathlib
from typing import Optional, Text, Union

import numpy as np

PathType = Union[Text, pathlib.PurePosixPath]

from pathlib import PurePosixPath as GPath


class Camera:
  """Class to handle camera geometry."""

  def __init__(self,
               orientation: np.ndarray,
               position: np.ndarray,
               focal_length: Union[np.ndarray, float],
               principal_point: np.ndarray,
               image_size: np.ndarray,
               skew: Union[np.ndarray, float] = 0.0,
               pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
               radial_distortion: Optional[np.ndarray] = None,
               tangential_distortion: Optional[np.ndarray] = None,
               dtype=np.float32):
    """Constructor for camera class."""
    if radial_distortion is None:
      radial_distortion = np.array([0.0, 0.0, 0.0], dtype)
    if tangential_distortion is None:
      tangential_distortion = np.array([0.0, 0.0], dtype)

    self.orientation = np.array(orientation, dtype)
    self.position = np.array(position, dtype)
    self.focal_length = np.array(focal_length, dtype)
    self.principal_point = np.array(principal_point, dtype)
    self.skew = np.array(skew, dtype)
    self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, dtype)
    self.radial_distortion = np.array(radial_distortion, dtype)
    self.tangential_distortion = np.array(tangential_distortion, dtype)
    self.image_size = np.array(image_size, np.uint32)
    self.dtype = dtype

  @classmethod
  def from_json(cls, path: PathType):
    """Loads a JSON camera into memory."""
    path = GPath(path)
    with open(path, 'r') as fp:
      camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
      camera_json['tangential_distortion'] = camera_json['tangential']

    return cls(
        orientation=np.asarray(camera_json['orientation']),
        position=np.asarray(camera_json['position']),
        focal_length=camera_json['focal_length'],
        principal_point=np.asarray(camera_json['principal_point']),
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.asarray(camera_json['radial_distortion']),
        tangential_distortion=np.asarray(camera_json['tangential_distortion']),
        image_size=np.asarray(camera_json['image_size']),
    )

  @property
  def image_size_y(self):
    return self.image_size[1]

  @property
  def image_size_x(self):
    return self.image_size[0]

  @property
  def image_shape(self):
    return self.image_size_y, self.image_size_x
