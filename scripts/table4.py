from argparse import Namespace
import numpy as np

from scripts.compute_fid import compute_fid
from scripts.compute_FVA import compute_fva
from scripts.compute_FS import compute_fs
from scripts.compute_MNAC import compute_mnac
from scripts.compute_CD import compute_cd
from scripts.compute_COUT import compute_cout

path = "/misc/lmbraid21/faridk/celeb_proj_0.4_d5_c38"


for query_label in [31, 39]:
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
        "class_balanced": False,
    }
    fid = compute_fid(Namespace(**args))
    print("sFID:", round(fid, 1))   

    # FVA
    args = {
        "output_path": path,
        "weights_path": "/misc/lmbraid21/schrodi/pretrained_models/vggface2_resnet50_ft.pkl",
        "batch_size": 15,
    }
    fva = compute_fva(Namespace(**args))
    print("FVA:", round(np.mean(fva[0])*100, 1))

    # FS
    args = {
        "output_path": path,
        "weights_path": "/misc/lmbraid21/schrodi/pretrained_models/simsiam_checkpoint_0099.pth.tar",
        "batch_size": 15,
    }
    fs = compute_fs(Namespace(**args))
    print("FS:", round(np.mean(fs).item(), 4))

    # MNAC
    args = {
        "output_path": path,
        "oracle_path": "/misc/lmbraid21/schrodi/pretrained_models/celeb_oracle_attribute/celebamaskhq/checkpoint.tar",
        "dataset": "CelebAHQ",
        "batch_size": 15,
    }
    mnac = compute_mnac(Namespace(**args))
    print("MNAC:", round(np.mean(mnac[0]), 2))

    # CD
    args = {
        "output_path": path,
        "oracle_path": "/misc/lmbraid21/schrodi/pretrained_models/celeb_oracle_attribute/celebamaskhq/checkpoint.tar",
        "dataset": "CelebAHQ",
        "celeba_path": "/misc/lmbraid21/schrodi/CelebAMask-HQ",
        "query_label": query_label,
    }
    cd = compute_cd(Namespace(**args))
    print("CD:", round(cd, 2))

    # COUT
    args = {
        "output_path": path,
        "dataset": "CelebAHQ",
        "celeba_path": "/misc/lmbraid21/schrodi/CelebAMask-HQ",
        "query_label": query_label,
        "target_label": -1,
        "batch_size": 10,
        "weights_path": "/misc/lmbraid21/schrodi/pretrained_models/celeba_hq_decision_densenet/celebamaskhq/checkpoint.tar",
    }
    cout = compute_cout(Namespace(**args))
    print("COUT:", round(cout[0], 4))