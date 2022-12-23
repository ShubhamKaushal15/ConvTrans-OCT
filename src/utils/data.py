import torch
import numpy as np
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import cv2
import sys
from PIL import Image

class OCTVideos(Dataset):
    
    def __init__(self, transform=ResNet50_Weights.DEFAULT.transforms()):
        """
        ...
        """
        self.data_dir = os.path.join('../../pilot1/patches')

        self.data = os.listdir(self.data_dir)
            
        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        
        curr_dir = os.path.join(self.data_dir, self.data[idx])
        # getting first 50 frames TODO: add padding
        imgs = [self.transform(read_image(os.path.join(curr_dir, f"{img}.png"))) for img in range(50)]

        label = 0 if int(curr_dir[-1]) % 2 == 0 else 1 # 1 is positive glaucoma

        return torch.FloatTensor(np.stack(imgs, axis=0).astype(np.float32)), label



def get_patch(img, x, y, patch_size = 64):
    """
    Given x, y coordinates in img, returns a patch of size = patch_size
    """

    x_min = x - patch_size//2
    x_max = x + patch_size//2

    y_min = y - patch_size//2
    y_max = y + patch_size//2

    return img[y_min : y_max, x_min : x_max, :]

def convert_coordinates(x, y, img_width = 1280, img_height = 720):
    """
    Converts normalized gaze coordinates to 
    unnormalized image coordinates
    """

    return round(x * img_width), round((1 - y) * img_height)

def get_coordinates(df, img_width, img_height):
    """
    Get list of gaze coordinates from dataframe
    """
    grouped_df = df.groupby('world_index')[['x_scaled', 'y_scaled']].mean()
    gaze_coords = list(grouped_df.apply(lambda row: convert_coordinates(row.x_scaled, 
                                                                        row.y_scaled, 
                                                                        img_width, 
                                                                        img_height), axis = 1))

    return gaze_coords

def create_data_from_experiment(fixations_dir, experiment_oct_dir):

    # read csv
    fixations = pd.read_csv(fixations_dir) # 1 and 4 are good

    # get indices when world_timestamp jumps a lot
    change_index = [0] + list(fixations.loc[fixations.world_timestamp.diff() > 2].index) + [fixations.shape[0]]

    # get dataframes based on the breakpoints
    for i in range(0, len(change_index) - 1):

        oct_df = fixations.iloc[change_index[i] : change_index[i + 1], : ]
        
        # only keep fixations on surface
        filtered_df = oct_df.loc[oct_df.on_surf].reset_index(drop = True)

        # read image
        img = cv2.imread(os.path.join(experiment_oct_dir, f"{i + 1}.png"))

        new = Image.new(f"RGBA", (img.shape[1], img.shape[0]))

        # create directory to save patches
        oct_patch_dir = os.path.join(experiment_oct_dir, "patches", f"{i + 1}")
        if not os.path.exists(oct_patch_dir):
            os.makedirs(oct_patch_dir)

        # get coordinates
        gaze_coords = get_coordinates(filtered_df, img.shape[1], img.shape[0])

        # loop through fixation coordinates
        c = 0
        for coord in gaze_coords:

            patch = get_patch(img, coord[0], coord[1], 64)

            if ((patch.shape[0] != 0) and 
                (patch.shape[1] != 0) and # patch should not be on the border
                (patch.mean() <= 250)): # not considering patches which are only white
                
                # save patch
                cv2.imwrite(os.path.join(oct_patch_dir, f"{c}.png"), patch)
                c += 1

                new.paste(Image.fromarray(patch), (coord[0] - 32, coord[1] - 32))
                
        new.save(os.path.join(experiment_oct_dir, f"collage_{i + 1}.png"))

if __name__ == '__main__':
    """
    fixations: name of fixations file
    experiment_oct_dir: name of oct imgs directory for experiment

    Example: python data.py fixations1 pilot1
    """
    fixations = sys.argv[1]
    experiment_oct_dir = sys.argv[2]

    data_dir = os.path.join("..", "..", "..")

    create_data_from_experiment(os.path.join(data_dir, f"{fixations}.csv"), 
                                os.path.join(data_dir, experiment_oct_dir))