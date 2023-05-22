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

classifiers = ["inception_v3", "vit_b_32", "efficientnet_b7", "convnext_base"]
base_path = "/misc/lmbraid21/faridk/LDCE_sd_"

save_base_path = "/misc/lmbraid21/faridk/multi_classifiers_cat"
os.makedirs(save_base_path, exist_ok=True)
os.chmod(save_base_path, 0o777)
save_path = os.path.join(save_base_path, "examples")
os.makedirs(save_path, exist_ok=True)
os.chmod(save_path, 0o777)

counter = 0

ex_base =  base_path + f"{classifiers[0]}_seed_{0}"
bucket_folder = sorted(glob.glob(ex_base + "/bucket*"))[0]

for pth_file in tqdm(sorted(glob.glob(bucket_folder + "/*.pth")), leave=False):
    if not os.path.basename(pth_file)[:-4].isdigit():
                    continue
    image_common_name = os.path.basename(pth_file)
    filename =image_common_name[:-4]
    images_across_classifiers = []
    for classifier in classifiers:

        image_across_seeds = []
        for seed in range(3):
            current_dir_path = base_path + f"{classifier}_seed_{seed}" + "/bucket_0_50/"
            current_img_pth_path = current_dir_path + image_common_name  
            
            data = torch.load(pth_file, map_location="cpu")
            source, target = data["source"], data["target"]
            source = source.split(",")[0]
            target = target.split(",")[0]
            
            original_img = Image.open(os.path.join(current_dir_path, "original", f"{filename}.png"))
            counterfactual_img = Image.open(os.path.join(current_dir_path, "counterfactual", f"{filename}.png"))

            #filename = str(counter).zfill(5)
            image_across_seeds.append(counterfactual_img)
        image_across_seeds.append(original_img)
        image_across_seeds =get_concat_h(*image_across_seeds)
    
        images_across_classifiers.append(image_across_seeds)
    filename_c = str(counter).zfill(5)
    out_filepath = os.path.join(save_path, f"{filename}_{source}_{target}.jpg")
    get_concat_v(*images_across_classifiers).save(out_filepath)
    counter += 1