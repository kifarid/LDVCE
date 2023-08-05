import os
from PIL import Image
import glob
import torch

from utils.fig_utils import get_concat_h, get_concat_v


for create_for_attr in ["young_old", "old_young", "smile_no smile", "no smile_smile"]:
    if create_for_attr == "young_old" or create_for_attr == "old_young":
        base_path = "/misc/lmbraid21/faridk/celeb/celeb_age"
        if create_for_attr == "young_old":
            indices = [5, 366, 356, 30, 1215, 1628, 2599]
        elif create_for_attr == "old_young":
            indices = [1596, 2, 373, 34, 37, 1168]
        else:
            raise NotImplementedError
    elif create_for_attr == "smile_nosmile" or "nosmile_simle":
        base_path = "/misc/lmbraid21/faridk/celeb/celeb_smile_new"
        if create_for_attr == "smile_no smile":
            # indices = [405, 410, 1378, 1823, 2475]
            indices = [2475, 1823, 405, 410, 1378]
        elif create_for_attr == "no smile_smile":
            indices = [2, 5, 407, 1211, 1354]
        else:
            raise NotImplementedError

    examples_path = os.path.join(base_path, "examples")
    save_path = os.path.join(base_path, "paper_plots")

    buckets = sorted(glob.glob(base_path+"/bucket*"))

    os.makedirs(save_path, exist_ok=True)
    os.chmod(save_path, 0o777)

    examples = []
    for idx in indices[:1]:
        count = str(idx).zfill(5)
        n_samples = 0
        for bucket in buckets:
            if int(count) < n_samples + len(glob.glob(bucket+"/0*.pth")):
                break
            n_samples += len(glob.glob(bucket+"/0*.pth"))

        tmp = str(int(count)-n_samples).zfill(5)
        pth_filename = f"{tmp}.pth"
        data = torch.load(os.path.join(bucket, pth_filename), map_location="cpu")

        print(count, data["target"]==data["out_pred"],data["target"], data["out_pred"])

        filename = f"{count}_{create_for_attr}.png"
        filepath = os.path.join(examples_path, filename)
        example = Image.open(filepath)
        examples.append(example)

    outfilepath = os.path.join(save_path, f"{create_for_attr}.png")
    #get_concat_h(get_concat_v(*examples[:2]), get_concat_v(*examples[2:])).save(outfilepath, dpi=(200, 200))
    #get_concat_v(*examples).save(outfilepath, dpi=(200, 200))

    outfilepath = os.path.join(save_path, f"{create_for_attr}.jpg")
    get_concat_v(*examples).save(outfilepath)