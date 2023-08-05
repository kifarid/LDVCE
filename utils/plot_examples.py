import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import glob
import torch
from PIL import Image
from tqdm import tqdm

from utils.fig_utils import get_concat_h

#base_path = "/misc/lmbraid21/faridk/celeb/celeb_age"
#base_path = "/misc/lmbraid21/faridk/celeb/celeb_smile_new"
#base_path = "/misc/lmbraid21/faridk/LDCE_w382_cc23"
#base_path = "/misc/lmbraid21/faridk/LDCE_sd"
#base_path = "/misc/lmbraid21/faridk/ldvce_pets_42_24_correct"
#base_path = "/misc/lmbraid21/faridk/ldvce_flowers_correct_targets"
#base_path = "/misc/lmbraid21/faridk/celeb_age_corrected_8"
base_path = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50"
#base_path = "/misc/lmbraid21/faridk/ldvce_pets_42_24"
base_path = "/misc/lmbraid21/faridk/ldvce_flowers"
base_path = "/misc/lmbraid21/faridk/smileee_1"
save_path = os.path.join(base_path, "examples")

os.makedirs(save_path, exist_ok=True)
os.chmod(save_path, 0o777)
for tmp in ["correct", "incorrect"]:
    os.makedirs(os.path.join(save_path, tmp), exist_ok=True)
    os.chmod(os.path.join(save_path, tmp), 0o777)

bucket_folders = sorted(glob.glob(base_path + "/bucket*"))

counter = 0
for bucket_folder in tqdm(sorted(glob.glob(base_path + "/bucket*")), leave=False):
    for pth_file in tqdm(sorted(glob.glob(bucket_folder + "/*.pth")), leave=False):
        if not os.path.basename(pth_file)[:-4].isdigit():
            continue
        data = torch.load(pth_file, map_location="cpu")
        source, target = data["source"], data["target"]
        source = source.split(",")[0]
        target = target.split(",")[0]
        filename = os.path.basename(pth_file)[:-4]
        original_img = Image.open(os.path.join(bucket_folder, "original", f"{filename}.png"))
        counterfactual_img = Image.open(os.path.join(bucket_folder, "counterfactual", f"{filename}.png"))

        filename = str(counter).zfill(5)
        outfilepath = os.path.join(save_path, "correct" if data["out_pred"] == data["target"] else "incorrect", f"{filename}_{source}_{target}.jpg")
        get_concat_h(original_img, counterfactual_img).save(outfilepath)
        counter += 1