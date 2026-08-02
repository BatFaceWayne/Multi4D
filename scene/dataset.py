from torch.utils.data import Dataset
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset
    ):
        self.dataset = dataset

    def __getitem__(self, index):
        # Source-image path for the TRASE semantic stage's mask lookup. For dynerf the
        # dataloader holds .../camXX/images/YYYY.png in self.dataset.image_paths; other
        # readers may carry it on caminfo. None when unavailable (mask lookup then skipped).
        try:
            image_path = self.dataset.image_paths[index]
        except Exception:
            image_path = None

        entry = self.dataset[index]
        if hasattr(entry, '_fields'):
            # CameraInfo namedtuple: technicolor (dataset_readers) or nerfies (hyper_loader).
            # Both default cxr/cyr to 0.0; non-zero only for technicolor's off-center
            # principal point (which also selects getProjectionMatrixCV + zfar=1000
            # downstream in Camera).
            image, R, T = entry.image, entry.R, entry.T
            FovX, FovY, time = entry.FovX, entry.FovY, entry.time
            cxr, cyr = entry.cxr, entry.cyr
            image_name = entry.image_name
            if image_path is None:
                image_path = getattr(entry, "image_path", None)
        else:
            # dynerf: plain (image, (R, T), time) tuple; intrinsics live on the dataset.
            image, (R, T), time = entry
            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            cxr = cyr = 0.0
            image_name = f"{index}"
        return Camera(R=R, T=T, FoVx=FovX, FoVy=FovY, image=image,
                      image_name=image_name, time=time,
                      cxr=cxr, cyr=cyr, image_path=image_path)
    def __len__(self):

        return len(self.dataset)
