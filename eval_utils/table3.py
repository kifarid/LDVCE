from argparse import Namespace
import numpy as np
from eval_utils.compute_COUT import compute_cout
from eval_utils.compute_fid import compute_fid
from eval_utils.compute_SimSiamSimilarity import compute_s3
import os
import glob
import torch

human_readable = ["zebra-sorrel", "cheetah-cougar", "egyptian-persian cat"]
base_path = "/misc/lmbraid21/faridk"
folders = [
    "LDCE_zs_ws", "LDCE_zs_ws_l1_sd", "LDCE_zs_ws_l2", "LDCE_zs_ws_l2_sd",
    "LDCE_cc_ws", "LDCE_cc_ws_l1_sd", "LDCE_cc_ws_l2", "LDCE_cc_ws_l2_sd",
    "LDCE_ep_ws", "LDCE_ep_ws_l1_sd", "LDCE_ep_ws_l2", "LDCE_ep_ws_l2_sd",
]
# zebra, sorrel, cougar, cheetah, egyptian cat, persian cat
class_indices = [(340, 339), (293, 286), (285, 283)]

for idx, (pair, classes) in enumerate(zip(human_readable, class_indices)):
    for folder in folders[idx*4:(idx+1)*4]:
        print("#"*5, pair, folder, "#"*5)
        path = os.path.join(base_path, folder)
        # FID
        args = {
                "output_path": path,
                "sfid": False,
                "sfid_splits": 2,
                "class_balanced": True,
            }
        fid = compute_fid(Namespace(**args))
        print("FID:", round(fid, 1))
        # sFID 
        args = {
                "output_path": path,
                "sfid": True,
                "sfid_splits": 2,
                "class_balanced": True,
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