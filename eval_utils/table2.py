from argparse import Namespace

from eval_utils.compute_lpnorms import compute_lp_norms
from eval_utils.compute_validity_metrics import compute_validity_metrics
from eval_utils.compute_fid import compute_fid

#path = "/misc/lmbraid21/faridk/LDCE_w382_cc23" # LDCE (ours)
path = "/misc/lmbraid21/faridk/testing/LDCE_sd"
# path = "/misc/lmbraid21/faridk/ImageNetDVCEs_" # DVCE
# path = "/misc/lmbraid21/faridk/ImageNetSVCEs_robustOnly" # SVCE

# L1 & L2
args = {
    "output_path": path,
    "batch_size": 32,
}
l1, l2 = compute_lp_norms(Namespace(**args))
print("L1:", int(round(l1, 0)))
print("L2:", int(round(l2, 0)))

# Flip ratio & mean conf. -> this is slow, run Karim's evaluation
# args = {
#     "output_path": path,
#     "batch_size": 32,
#     "target_model": "resnet50",
#     "idx_to_tgt": "data/image_idx_to_tgt.yaml",
# }
# fr, mean_conf = compute_validity_metrics(Namespace(**args))

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