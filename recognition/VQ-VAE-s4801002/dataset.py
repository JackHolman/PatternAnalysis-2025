import numpy as np
import nibabel as nib
import torch.utils.data
from torch.utils.data import Dataset
import os
from glob import glob
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from normalise import Normalise

class HipMRIStudyDataset(Dataset):
    def __init__(self, images_dir: str, transforms=None):
        self._images_dir = images_dir
        self._transforms = transforms

        # Get paths to all images in the image directory
        self._img_paths = glob(os.path.join(self._images_dir, '*'))

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        if self._transforms:
            data = get_data_2d([self._img_paths[idx]])[0]
            data = self._transforms(data)
        else:
            # just get data normalised
            data = get_data_2d([self._img_paths[idx]], normImage=True)[0]

        return data


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels), ), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

def get_data_2d(imageNames: list[str], normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from names cases list provided into a list for each.

    This function pre-allocates 3D arrays for conv2d to avoid excessive memory usage.

    This function is heavily based on the one provided in the task sheet.

    :param imageNames: Array of image names.
    :param normImage: bool (normalise the image 0.0-1.0)
    :param categorical: bool (whether images have categories)
    :param early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    """

    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0] # sometimes extra dims, remove.
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0] # sometimes extra dims in HipMRI_study data.
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

if __name__ == "__main__":
    # Load a single image and plot, for testing.
    path = "data/case.nii.gz"

    img = get_data_2d([path], normImage=True)[0]
    print(torch.from_numpy(img).shape)
    plt.imshow(img)
    plt.show() # Images are (256, 128)

    # Load the dataset and print the shape of the first batch.
    dataset = HipMRIStudyDataset("data/real_data/train", transforms=transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 128)), transforms.CenterCrop((256, 128)), Normalise()]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    print("Batch size:", next(iter(dataloader)).shape)
    unique = torch.unique(next(iter(dataloader))[0][0])
    print(unique)

