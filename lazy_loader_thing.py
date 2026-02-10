# util sub-routine to lazy-load training data and generate (image, mask) batch pairs following a on-the-fly methodology

import torch
from torch.utils.data import Dataset
import os
from utils import dataset_stuff


class TiffDataset(Dataset):


    def __init__(self):
        self.rasters_dir = "Data/geotiffs/tier1/images"
        self.labels_dir = "Data/geotiffs/tier1/labels"
        self.rasters_paths = os.listdir("Data/geotiffs/tier1/images") # update accordingly gng
        self.labels_paths = os.listdir("Data/geotiffs/tier1/labels") # update accordingly gng


    def __len__(self):

        n_rasters = len(self.rasters_paths)
        n_labels = len(self.labels_paths)

        if  n_rasters == n_labels :
            return n_rasters
        
        else:
            assert f"Number of images and labels are not same.  No. of images : {n_rasters}\n No. of labels: {n_labels}"


    def __getitem__(self, idx):

        raster_path = self.rasters_paths[idx]
        label_path = self.labels_paths[idx]

        assert raster_path.split('.')[0] == label_path.split('.')[0], \
            f"Raster and label file mismatch: {raster_path} vs {label_path}"

        image = dataset_stuff.get_raster_data(rasters_dir=self.rasters_dir, f_path=raster_path)
        mask = dataset_stuff.extract_mask(raster_image=image, labels_dir=self.labels_dir, labels_path=label_path)

        # Image normalization
        image = torch.from_numpy(image).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0  # C,H,W for UNet

        return image, mask


        

