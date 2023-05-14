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
base_path = "/misc/lmbraid21/faridk/LDCE_sd"
save_path = os.path.join(base_path, "examples")

os.makedirs(save_path, exist_ok=True)
os.chmod(save_path, 0o777)

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
        outfilepath = os.path.join(save_path, f"{filename}_{source}_{target}.png")
        get_concat_h(original_img, counterfactual_img).save(outfilepath, dpi=(200, 200))
        counter += 1