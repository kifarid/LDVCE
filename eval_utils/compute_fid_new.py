import argparse
import glob
import shutil
import os
os.environ['TRANSFORMERS_CACHE'] = '/misc/lmbraid21/faridk/.cache/huggingface/hub'
os.environ["TORCH_HOME"] = '/misc/lmbraid21/faridk/.cache/torch/hub/checkpoints'
import random
import subprocess
import numpy as np
import torch
from tqdm import tqdm


def compute_fid(args):
    if not args.sfid: # FID computation
        real_images_path = os.path.join(args.output_path, "all_originals")
        os.makedirs(real_images_path, mode=777, exist_ok=True)
        os.chmod(real_images_path, 0o777)
        counterfactual_images_path = os.path.join(args.output_path, "all_counterfactuals")
        os.makedirs(counterfactual_images_path, mode=777, exist_ok=True)
        os.chmod(counterfactual_images_path, 0o777)
        
        # create symbolic links
        counter_orig = 0
        counter_gen = 0
        for bucket_folder in sorted(glob.glob(args.output_path+ "/bucket*")):
            for original_path, counterfactual_path in tqdm(zip(sorted(glob.glob(bucket_folder + "/original/*")), sorted(glob.glob(bucket_folder + "/counterfactual/*")))):
                #check if file already exists
                #get new path 
                directory, filename = os.path.split(original_path)
                data_path = os.path.join(os.path.dirname(directory), filename).replace(".png", ".pth")
                
                if args.drop_mismatch:
                    data = torch.load(data_path)
                    if "out_probability" in data:
                        data["out_pred"] = data["out_probability"].argmax()

                else:
                    data = {"out_pred": 0, "target": 0}

                 # or not(args.drop_mismatch):
                if data["out_pred"] == data["target"]:
                    if not os.path.exists(os.path.join(real_images_path, f"{counter_orig}.png")):
                        os.symlink(original_path, os.path.join(real_images_path, f"{counter_orig}.png"))
                    counter_orig+= 1
                
                
                    if not os.path.exists(os.path.join(counterfactual_images_path, f"{counter_gen}.png")):
                        os.symlink(counterfactual_path, os.path.join(counterfactual_images_path, f"{counter_gen}.png"))
                    counter_gen+= 1
                
                if counter_orig >= args.limit_length:
                    break
        
        print(f"counter_orig: {counter_orig}, counter_gen: {counter_gen}")
        import pathlib
        IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                            'tif', 'tiff', 'webp'}
        path = pathlib.Path(real_images_path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in path.glob('*.{}'.format(ext))])

        cmd = ["python", "-m", "dgm_eval", f"{real_images_path}/", 
            f"{counterfactual_images_path}/",
            "--model", "dinov2", "--device", "cuda",
            "--batch_size", "128",
            #"--arch", "vitb16",
            "--metrics", "prdc", "fd",]

        output = subprocess.check_output(cmd, universal_newlines=True)
        #convert output to float
        fid = float(output.split()[-1])
        shutil.rmtree(real_images_path)
        shutil.rmtree(counterfactual_images_path)
        return fid
    
    elif args.sfid: # sFID computation
        sfids = []
        def checkNatNum(n):
            if str(n).isdigit() and float(n) == int(n) and int(n) > 0:
                return True
            else:
                return False

        print("Assumes buckets are of equivalent size!")
        buckets_list = sorted(glob.glob(args.output_path+ "/bucket*"))
        
        if not args.class_balanced:
            random.seed(0)

            # read all
            classes_to_paths = {}
            for pth_file in tqdm(sorted(glob.glob(args.output_path + "/bucket*/*.pth")), leave=False):
                if not os.path.basename(pth_file)[:-4].isdigit():
                    continue
                data = torch.load(pth_file, map_location="cpu")
                if data["source"] in classes_to_paths:
                    classes_to_paths[data["source"]].append(pth_file)
                else:
                    classes_to_paths[data["source"]] = [pth_file]

            # split into 2 sets
            split1, split2 = {
                "original": [],
                "counterfactual": [],
            }, {
                "original": [],
                "counterfactual": [],
            }
            for pth_files in classes_to_paths.values():
                split_idx = len(pth_files) // 2
                for pth_file in pth_files[:split_idx]:
                    dirname = os.path.dirname(pth_file)
                    filename = os.path.basename(pth_file).replace("pth", "png")
                    split1["original"].append(os.path.join(dirname, "original", filename))
                    split2["counterfactual"].append(os.path.join(dirname, "counterfactual", filename))
                for pth_file in pth_files[split_idx:]:
                    dirname = os.path.dirname(pth_file)
                    filename = os.path.basename(pth_file).replace("pth", "png")
                    split1["counterfactual"].append(os.path.join(dirname, "counterfactual", filename))
                    split2["original"].append(os.path.join(dirname, "original", filename))

            for split in [split1, split2]:
                real_images_path = os.path.join(args.output_path, "all_originals")
                os.makedirs(real_images_path, mode=777, exist_ok=True)
                os.chmod(real_images_path, 0o777)
                counterfactual_images_path = os.path.join(args.output_path, "all_counterfactuals")
                os.makedirs(counterfactual_images_path, mode=777, exist_ok=True)
                os.chmod(counterfactual_images_path, 0o777)

                # create symbolic links
                counter = 0
                for original_path in split["original"]:
                    #check if file already exists
                    if not os.path.exists(os.path.join(real_images_path, f"{counter}.png")):
                        os.symlink(original_path, os.path.join(real_images_path, f"{counter}.png"))
                    counter += 1

                counter = 0
                for counterfactual_path in split["counterfactual"]:
                    if not os.path.exists(os.path.join(counterfactual_images_path, f"{counter}.png")):
                        os.symlink(counterfactual_path, os.path.join(counterfactual_images_path, f"{counter}.png"))
                    counter += 1

                cmd = ["python", "-m", "dgm_eval", f"{real_images_path}/", 
                    f"{counterfactual_images_path}/",
                    "--model", "dinov2", "--device", "cuda",
                    "--batch_size", "256",
                    #"--arch", "vitb16",
                    "--metrics", "prdc", "fd",]
                output = subprocess.check_output(cmd, universal_newlines=True)
                #convert output to float
                sfid = float(output.split()[-1])
                sfids.append(sfid)

                shutil.rmtree(real_images_path)
                shutil.rmtree(counterfactual_images_path)

        elif len(buckets_list) == 1:
            random.seed(0)

            examples_per_class = 50

            for split in range(2):
                real_images_path = os.path.join(args.output_path, "all_originals")
                os.makedirs(real_images_path, mode=777, exist_ok=True)
                os.chmod(real_images_path, 0o777)
                counterfactual_images_path = os.path.join(args.output_path, "all_counterfactuals")
                os.makedirs(counterfactual_images_path, mode=777, exist_ok=True)
                os.chmod(counterfactual_images_path, 0o777)

                bucket_folder = buckets_list[0]

                # create symbolic links
                counter = 0
                if split == 0:
                    files = sorted(glob.glob(bucket_folder + "/original/*"))
                    files = files[:25] + files[50:75]
                elif split == 1:
                    files = sorted(glob.glob(bucket_folder + "/original/*"))
                    files = files[25:50] + files[75:]
                for original_path in files:
                    os.symlink(original_path, os.path.join(real_images_path, f"{counter}.png"))
                    counter += 1

                counter = 0
                if split == 0:
                    files = sorted(glob.glob(bucket_folder + "/counterfactual/*"))
                    files = files[25:50] + files[75:]
                elif split == 1:
                    files = sorted(glob.glob(bucket_folder + "/counterfactual/*"))
                    files = files[:25] + files[50:75]
                for counterfactual_path in files:
                    os.symlink(counterfactual_path, os.path.join(counterfactual_images_path, f"{counter}.png"))
                    counter += 1

                cmd = ["python", "-m", "dgm_eval", f"{real_images_path}/", 
                    f"{counterfactual_images_path}/",
                    "--model", "dinov2", "--device", "cuda",
                    "--batch_size", "256",
                    #"--arch", "vitb16",
                    "--metrics", "prdc", "fd",]
                
                output = subprocess.check_output(cmd, universal_newlines=True)
                #convert output to float
                sfid = float(output.split()[-1])
                sfids.append(sfid)

                shutil.rmtree(real_images_path)
                shutil.rmtree(counterfactual_images_path)
        else:
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

                counter = 0
                for bucket_idx in split:
                    bucket_folder = buckets_list[bucket_idx]
                    for counterfactual_path in sorted(glob.glob(bucket_folder + "/counterfactual/*")):
                        os.symlink(counterfactual_path, os.path.join(counterfactual_images_path, f"{counter}.png"))
                        counter += 1

                cmd = ["python", "-m", "dgm_eval", f"{real_images_path}/", 
                    f"{counterfactual_images_path}/",
                    "--model", "dinov2", "--device", "cuda",
                    "--batch_size", "256",
                    #"--arch", "vitb16",
                    "--metrics", "prdc", "fd",]
                output = subprocess.check_output(cmd, universal_newlines=True)
                #convert output to float
                sfid = float(output.split()[-1])
                sfids.append(sfid)

                shutil.rmtree(real_images_path)
                shutil.rmtree(counterfactual_images_path)

        #get the mean
        sfid = np.mean(sfids)
        return sfid

    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default='/misc/lmbraid21/faridk/ImageNetSVCEs_robustOnly/', help='the path of the experiment')
    parser.add_argument('--sfid', action="store_true")
    parser.add_argument('--drop_mismatch', action="store_true",help='drop mismatched pairs')
    parser.add_argument('--sfid_splits', type=int, default=2)
    parser.add_argument('--limit_length', type=int, default=10000)
    args=parser.parse_args()
    #whether we would drop mismatched pairs or not
    print('args.drop_mismatch', args.drop_mismatch)
    fid = compute_fid(args)
    
    print("sFID:" if args.sfid else "FID:", fid)