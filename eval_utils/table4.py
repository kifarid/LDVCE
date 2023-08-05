from argparse import Namespace, ArgumentParser
import numpy as np
import yaml
import glob

from eval_utils.compute_fid import compute_fid
from eval_utils.compute_FVA import compute_fva
from eval_utils.compute_MNAC import compute_mnac
from eval_utils.compute_CD import compute_cd
from eval_utils.compute_COUT import compute_cout


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="/misc/lmbraid21/faridk/celeb_age_corrected_8/")
    args = parser.parse_args()
    path = args.path
    #load config.yaml file from data subdirectories
    

    config = yaml.load(open(glob.glob(args.path + '/**/config.yaml', recursive=True)[0], "r"), Loader=yaml.FullLoader)
    query_label = 39 #config["data"]["query_label"]
    
    for query_label in [query_label]:
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
        fva, fs = compute_fva(Namespace(**args))
        print("FVA:", round(np.mean(fva)*100, 1))

        # FS
        # args = {
        #     "output_path": path,
        #     # "weights_path": "/misc/lmbraid21/schrodi/pretrained_models/simsiam_checkpoint_0099.pth.tar",
        #     "weights_path": "/misc/lmbraid21/schrodi/pretrained_models/vggface2_resnet50_ft.pkl",
        #     "batch_size": 15,
        # }
        # fs = compute_fs(Namespace(**args))
        print("FS:", round(np.mean(fs), 4))

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
