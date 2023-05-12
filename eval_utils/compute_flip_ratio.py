import glob
import os
import torch
from tqdm import tqdm

path = "/misc/lmbraid21/faridk/celeb_smile_new"
path = "/misc/lmbraid21/faridk/celeb_age"

correct, counter = 0, 0
for bucket_folder in tqdm(sorted(glob.glob(path + "/bucket*")), leave=False):
    for pth_file in tqdm(sorted(glob.glob(bucket_folder + "/*.pth")), leave=False):
        if not os.path.basename(pth_file)[:-4].isdigit():
            continue
        data = torch.load(pth_file, map_location="cpu")
        counter += 1
        correct += int(data["target"] == data["out_pred"])
print(correct / counter)