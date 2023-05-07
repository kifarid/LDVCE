import argparse
import os
import glob
import shutil
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--output-path', required=True, type=str, default='/path/to/experiment', help='the path of the experiment')
parser.add_argument('--sfid', action="store_true")
parser.add_argument('--sfid_splits', type=int, default=2)
args=parser.parse_args()

if not args.sfid: # FID computation
    real_images_path = os.path.join(args.output_path, "all_originals")
    os.makedirs(real_images_path, mode=777, exist_ok=True)
    os.chmod(real_images_path, 0o777)
    counterfactual_images_path = os.path.join(args.output_path, "all_counterfactuals")
    os.makedirs(counterfactual_images_path, mode=777, exist_ok=True)
    os.chmod(counterfactual_images_path, 0o777)
    
    # create symbolic links
    counter = 0
    for bucket_folder in sorted(glob.glob(args.output_path+ "/bucket*")):
        for original_path, counterfactual_path in zip(sorted(glob.glob(bucket_folder + "/original/*")), sorted(glob.glob(bucket_folder + "/counterfactual/*"))):
            os.symlink(original_path, os.path.join(real_images_path, f"{counter}.png"))
            os.symlink(counterfactual_path, os.path.join(counterfactual_images_path, f"{counter}.png"))
            counter += 1

    import pathlib
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                        'tif', 'tiff', 'webp'}
    path = pathlib.Path(real_images_path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob('*.{}'.format(ext))])

    cmd = "python -m pytorch_fid %s/ %s/ --device cuda" % (real_images_path, counterfactual_images_path)
    os.system(cmd)

    shutil.rmtree(real_images_path)
    shutil.rmtree(counterfactual_images_path)
elif args.sfid: # sFID computation
    def checkNatNum(n):
        if str(n).isdigit() and float(n) == int(n) and int(n) > 0:
            return True
        else:
            return False

    print("Assumes buckets are of equivalent size!")
    buckets_list = sorted(glob.glob(args.output_path+ "/bucket*"))
    
    assert len(buckets_list) // args.sfid_splits * args.sfid_splits == len(buckets_list)

    random.seed(0)
    split1 = random.sample(range(len(buckets_list)), len(buckets_list) // args.sfid_splits)
    split2 = list(set(range(len(buckets_list))) - set(split1))

    for split in [split1, split2]:
        real_images_path = os.path.join(args.output_path, "all_originals")
        os.makedirs(real_images_path, mode=777, exist_ok=True)
        os.chmod(real_images_path, 0o777)
        counterfactual_images_path = os.path.join(args.output_path, "all_counterfactuals")
        os.makedirs(counterfactual_images_path, mode=777, exist_ok=True)
        os.chmod(counterfactual_images_path, 0o777)

        # create symbolic links
        counter = 0
        for bucket_idx in split:
            bucket_folder = buckets_list[bucket_idx]
            for original_path in sorted(glob.glob(bucket_folder + "/original/*")):
                os.symlink(original_path, os.path.join(real_images_path, f"{counter}.png"))
                counter += 1

        other_split = list(set(range(len(buckets_list))) - set(split))
        counter = 0
        for bucket_idx in split:
            bucket_folder = buckets_list[bucket_idx]
            for counterfactual_path in sorted(glob.glob(bucket_folder + "/counterfactual/*")):
                os.symlink(counterfactual_path, os.path.join(counterfactual_images_path, f"{counter}.png"))
                counter += 1

        cmd = "python -m pytorch_fid %s/ %s/ --device cuda" % (real_images_path, counterfactual_images_path)
        os.system(cmd)

        shutil.rmtree(real_images_path)
        shutil.rmtree(counterfactual_images_path)
else:
    raise NotImplementedError