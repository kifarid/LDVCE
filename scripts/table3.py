from argparse import Namespace
import numpy as np
from scripts.compute_COUT import compute_cout
from scripts.compute_fid import compute_fid
from scripts.compute_SimSiamSimilarity import compute_s3
import os
import glob
import torch

human_readable = ["cheetah-cougar", "zebra-sorrel", "egyptian-persian cat"]
paths = ["/misc/lmbraid21/faridk/LDCE_zs_ws", "/misc/lmbraid21/faridk/LDCE_cc_ws", "/misc/lmbraid21/faridk/LDCE_ep_ws"]
# cougar, cheetah, zebra, sorrel, egyptian cat, persian cat
class_indices = [(340, 339), (293, 286), (285, 283)]


# S^3

for pair, path, classes in zip(human_readable, paths, class_indices):
    print("#"*5, pair, "#"*5)
    # FID
    args = {
            "output_path": path,
            "sfid": False,
            "sfid_splits": 2,
        }
    fid = compute_fid(Namespace(**args))
    print("FID:", round(fid, 1))
    # sFID 
    args = {
            "output_path": path,
            "sfid": True,
            "sfid_splits": 2,
        }
    fid = compute_fid(Namespace(**args))
    print("sFID:", round(fid, 1))   
    # S^3
    args = {
        "output_path": path,
        "weights_path": "/misc/lmbraid21/schrodi/pretrained_models/simsiam_checkpoint_0099.pth.tar",
        "batch_size": 15,
    }
    s3 = compute_s3(Namespace(**args))
    print("S^3:", round(np.mean(s3).item(), 4))
    # COUT
    couts = []
    for query, target in [classes, reversed(classes)]:
        args = {
            "output_path": path,
            "query_label": query,
            "target_label": target,
            "dataset": "ImageNet",
            "batch_size": 10,
        }
        cout, _ = compute_cout(Namespace(**args))
        couts.append(cout)
    print("COUT:", round(np.mean(couts), 4))

    # FR
    bucket_path = os.path.join(path, "bucket_0_50000")
    n, correct = 0, 0
    for pth_file in sorted(glob.glob(bucket_path + "/*.pth")):
        if not os.path.basename(pth_file)[:-4].isdigit():
            continue
        data = torch.load(pth_file, map_location="cpu")
        correct += int(data["target"] == data["out_pred"])
        n += 1
    fr = correct / n
    print("FR:", round(fr*100, 1))