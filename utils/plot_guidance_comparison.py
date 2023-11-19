import glob
import os
from PIL import Image
import torch
from tqdm import tqdm
from utils.fig_utils import get_concat_h

save_path = "/misc/lmbraid21/faridk/guidance_comparison"

cls_free = "/misc/lmbraid21/faridk/LDCE_sd_free"
cone_projection = "/misc/lmbraid21/faridk/LDCE_sd_default"
consensus = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50"

os.makedirs(save_path, exist_ok=True)
os.chmod(save_path, 0o777)

bucket_folders_cls_free = sorted(glob.glob(cls_free + "/bucket*"))
bucket_folders_cone_projection = sorted(glob.glob(cone_projection + "/bucket*"))
bucket_folders_consensus = sorted(glob.glob(consensus + "/bucket*"))

for idx in tqdm(range(10000), leave=False, total=10000):
    bucket_idx = idx // 1000
    bucket_folder = bucket_folders_consensus[bucket_idx]
    filename = str(idx).zfill(5)
    data = torch.load(os.path.join(bucket_folder, f"{filename}.pth"), map_location="cpu")
    source, target = data["source"], data["target"]
    source = source.split(",")[0]
    target = target.split(",")[0]

    if data["in_pred"] == data["source"] and data["out_pred"] == data["target"]:
        img_original = Image.open(os.path.join(bucket_folder, "original", f"{filename}.png"))
        img_consensus = Image.open(os.path.join(bucket_folder, "counterfactual", f"{filename}.png"))
        img_cls_free = Image.open(os.path.join(bucket_folders_cls_free[bucket_idx], "counterfactual", f"{filename}.png"))
        img_cone_proj = Image.open(os.path.join(bucket_folders_cone_projection[bucket_idx], "counterfactual", f"{filename}.png"))

        outfilepath = os.path.join(save_path, f"{filename}_{source}_{target}.jpg")
        get_concat_h(*[img_original, img_cls_free, img_cone_proj, img_consensus]).save(outfilepath)