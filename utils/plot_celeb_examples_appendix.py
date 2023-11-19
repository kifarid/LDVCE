import os
from PIL import Image
import glob
import torch

from utils.fig_utils import get_concat_h, get_concat_v


for create_for_attr in ["young_old", "old_young", "smile_no smile", "no smile_smile"]:
    if create_for_attr == "young_old" or create_for_attr == "old_young":
        base_path = "/misc/lmbraid21/faridk/celeb_age_corrected_8"
        if create_for_attr == "young_old":
            indices = [5,6,30,35,51,52,75,91,93,98]
            indices = [5, 30, 35, 51, 52, 91, 98, 739]
        elif create_for_attr == "old_young":
            indices = [21,25,34,39,59,71,74,92,126,161,204,236,267]
            indices = [34, 39, 71, 74, 92, 126, 236, 267]
        else:
            raise NotImplementedError
    elif create_for_attr == "smile_nosmile" or "nosmile_simle":
        base_path = "/misc/lmbraid21/faridk/celeb_smile_corrected_8"
        if create_for_attr == "smile_no smile":
            # indices = [405, 410, 1378, 1823, 2475]
            indices = [109,162,179,216,274,297,441,447,465,470,689,722]
            #indices = [179, 216, 297, 447, 465, 470, 689, 2733]
            indices = [179, 216, 297, 447, 465, 815, 689, 2733]
        elif create_for_attr == "no smile_smile":
            indices = [2,5,16,58,66,144,305,306,401,407,411,426]
            indices = [5, 16, 58, 66, 306, 401, 407, 426]
        else:
            raise NotImplementedError

    examples_path = os.path.join(base_path, "examples", "correct")
    save_path = os.path.join(base_path, "appendix_plots")

    buckets = sorted(glob.glob(base_path+"/bucket*"))

    os.makedirs(save_path, exist_ok=True)
    os.chmod(save_path, 0o777)

    examples = []
    print(len(indices))
    for idx in indices:
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

        filename = f"{count}_{create_for_attr}.jpg"
        filepath = os.path.join(examples_path, filename)
        example = Image.open(filepath)
        examples.append(example)

    #outfilepath = os.path.join(save_path, f"{create_for_attr}.png")
    #get_concat_h(get_concat_v(*examples[:2]), get_concat_v(*examples[2:])).save(outfilepath, dpi=(200, 200))
    #get_concat_v(*examples).save(outfilepath, dpi=(200, 200))

    outfilepath = os.path.join(save_path, f"{create_for_attr}.jpg")
    #get_concat_v(*examples).save(outfilepath)
    get_concat_h(*[get_concat_v(examples[i*2], examples[i*2+1]) for i in range(4)]).save(outfilepath)