import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import glob
import torch
from PIL import Image
from tqdm import tqdm

from utils.fig_utils import get_concat_h, get_concat_v

#base_path = "/misc/lmbraid21/faridk/celeb/celeb_age"
#base_path = "/misc/lmbraid21/faridk/celeb/celeb_smile_new"
#base_path = "/misc/lmbraid21/faridk/LDCE_w382_cc23"

base_path = "/misc/lmbraid21/faridk/LDCE_sd_correct_seed_0/bucket_0_50/"

files_ids = [5, 13, 24, 29, 38, 43, 47, 56]

save_base_path = "/misc/lmbraid21/faridk/evolution_figures"
os.makedirs(save_base_path, exist_ok=True)
os.chmod(save_base_path, 0o777)
save_path = os.path.join(save_base_path, "examples")
os.makedirs(save_path, exist_ok=True)
os.chmod(save_path, 0o777)

counter = 0
file_paths = [os.path.join(base_path, f"{str(file_id).zfill(5)}.pth") for file_id in files_ids]

for pth_file in tqdm(sorted(file_paths), leave=False):

    image_common_name = os.path.basename(pth_file)
    filename =image_common_name[:-4]
    images_across_classifiers = []
    
    data = torch.load(pth_file, map_location="cpu")
    original_img = Image.open(os.path.join(base_path, "original", f"{filename}.png"))
    counterfactual_img = Image.open(os.path.join(base_path, "counterfactual", f"{filename}.png"))

    if 'video' not in  data:
        continue

    video = data['video'].permute(0, 2, 3, 1).cpu().numpy()
    source, target = data["source"], data["target"]
    out_prob = str(round(data["out_tgt_confid"],3)).replace(".", "_")
    source = source.split(",")[0]
    target = target.split(",")[0]
    
    # get 8 frames equally spaced in time from the video
    video_8 = video[::len(video)//8][:8]
    video_8 = [Image.fromarray(frame) for frame in video_8]
    video_8 = [original_img] + video_8 + [counterfactual_img]
    video_10 = get_concat_h(*video_8)
    video_10.save(os.path.join(save_path, f"{filename}_{source}_{target}_{out_prob}.png"))