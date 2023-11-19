import os
import glob
import torch
from PIL import Image
from tqdm import tqdm

from utils.fig_utils import get_concat_h

base_path = "/misc/lmbraid21/faridk/LDCE_w382_cc23"
base_path_sd = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50"
base_path_no_consensus = "/misc/lmbraid21/faridk/LDCE_w382_cc23_clsg"
base_path_svce = "/misc/lmbraid21/faridk/ImageNetSVCEs_robustOnly"
base_path_dvce = "/misc/lmbraid21/faridk/ImageNetDVCEs_"
#save_path = "/misc/lmbraid21/faridk/imagenet_comparison"
save_path = "/misc/lmbraid21/faridk/imagenet_appendix"
save_path = "/misc/lmbraid21/faridk/imagenet_comparison_new"

indices = [
    574, 
    2429, 
    7768, 
    2812,  
    2951, 
    8574,  
    1951, 
    1852, 
    6926, 
    949, 
    4964, 
    497, 
    262, 
    271, 
    273, 
    2539, 
    279, 
    2016, 
    6926, 
    6926, 
    9022, 
    4023, 
    4964, 
    497, 
    1955, 
    924, 
    1924, 
    1009, 5010, 5012, 829, 15, 22, 7, 1004, 7007, 1023, 267, 2077, 4072, 4075, 7075, 2975, 9306, 9311, 9310, 2306, 2308, 3774, 7770, 9340, 6287, 3289, 5293, 6296, 5296, 7273, 3276, 9290, 1290, 1289, 5290, 6367, 7367, 5013, 1805, 9539,
    8970,
    2414,
    414,
    425,
    2924
]

os.makedirs(save_path, exist_ok=True)
os.chmod(save_path, 0o777)
for tmp in ["correct", "incorrect"]:
    os.makedirs(os.path.join(save_path, tmp), exist_ok=True)
    os.chmod(os.path.join(save_path, tmp), 0o777)

bucket_folders_cls = sorted(glob.glob(base_path + "/bucket*"))
bucket_folders_sd = sorted(glob.glob(base_path_sd + "/bucket*"))
bucket_folders_no_consensus = sorted(glob.glob(base_path_no_consensus + "/bucket*"))
bucket_folders_svce = sorted(glob.glob(base_path_svce + "/bucket*"))
bucket_folders_dvce = sorted(glob.glob(base_path_dvce + "/bucket*"))

indices = range(10000)
for idx in tqdm(indices, leave=False):
    bucket_idx = idx // 1000

    bucket_folder = bucket_folders_sd[bucket_idx]
    filename = str(idx).zfill(5)
    data = torch.load(os.path.join(bucket_folder, f"{filename}.pth"), map_location="cpu")
    source, target = data["source"], data["target"]
    source = source.split(",")[0]
    target = target.split(",")[0]

    original_img = Image.open(os.path.join(bucket_folder, "original", f"{filename}.png"))
    counterfactual_img_sd = Image.open(os.path.join(bucket_folder, "counterfactual", f"{filename}.png"))

    bucket_folder = bucket_folders_cls[bucket_idx]
    counterfactual_img_cls = Image.open(os.path.join(bucket_folder, "counterfactual", f"{filename}.png"))

    bucket_folder = bucket_folders_no_consensus[bucket_idx]
    counterfactual_img_no_consensus = Image.open(os.path.join(bucket_folder, "counterfactual", f"{filename}.png"))

    bucket_folder = bucket_folders_svce[bucket_idx]
    counterfactual_svce = Image.open(os.path.join(bucket_folder, "counterfactual", f"{filename}.png"))

    bucket_folder = bucket_folders_dvce[bucket_idx]
    counterfactual_dvce = Image.open(os.path.join(bucket_folder, "counterfactual", f"{filename}.png"))

    imgs = [original_img, counterfactual_svce, counterfactual_dvce, counterfactual_img_no_consensus, counterfactual_img_cls, counterfactual_img_sd]

    #outfilepath = os.path.join(save_path, f"{filename}_{source}_{target}.png")
    #get_concat_h(original_img, counterfactual_img).save(outfilepath, dpi=(200, 200))
    # outfilepath = os.path.join(save_path, f"{filename}_{source}_{target}.jpg")
    outfilepath = os.path.join(save_path, "correct" if data["out_pred"] == data["target"] else "incorrect", f"{filename}_{source}_{target}.jpg")
    get_concat_h(*imgs).save(outfilepath)