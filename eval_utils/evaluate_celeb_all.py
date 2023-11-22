from argparse import Namespace, ArgumentParser
import numpy as np
import yaml
import glob
import pandas as pd
import os

from eval_utils.compute_fid import compute_fid
from eval_utils.compute_FVA import compute_fva
from eval_utils.compute_MNAC import compute_mnac
from eval_utils.compute_CD import compute_cd
from eval_utils.compute_COUT import compute_cout
from scripts.evaluate_dvces import create_df, load_yaml_files 
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="/misc/lmbraid21/faridk/celeb_age_corrected_8_sample")
    parser.add_argument("--query_label", type=int, default=39, help="label of the query attribute 39:age 31:smile")
    parser.add_argument("--target_csv", type=str, default="celeb_results_hpo.csv", help="path to target csv file to save and append results to")
    args = parser.parse_args()
    path = args.path
    
    #load config.yaml file from data subdirectories
    columns = ["path", "attribute", "flip_ratio", "num_samples", "FID", "sFID", "FVA", "FS", "MNAC", "CD", "COUT", "strength", "ddim_steps", "classifier_lambda", "dist_lambda"]
    
    args = parser.parse_args()
    target_csv = args.target_csv

    result_list = []
    result_list.append(path.split('/')[-1])

    config = yaml.load(open(glob.glob(args.path + '/**/config.yaml', recursive=True)[0], "r"), Loader=yaml.FullLoader)
    #computing FR:
    query_label = args.query_label 
    assert ((query_label == 31 and "smile" in args.path) or (query_label == 39 and "age" in args.path)), "query label does not match the attribute in the path"
    
    result_list.append( "smile" if query_label == 31 else "age")
    
    df = create_df(args.path)
    flip_ratio = (df['target']==df['out_pred']).mean()
    num_samples = len(df)

    result_list.append(round(flip_ratio,3))
    result_list.append(num_samples)

    for query_label in [query_label]:
        # FID
        args = {
            "output_path": path,
            "sfid": False,
            "sfid_splits": 2,
        }
        fid = compute_fid(Namespace(**args))
        print("FID:", round(fid, 1))
        result_list.append(round(fid, 1))

        # sFID
        args = {
            "output_path": path,
            "sfid": True,
            "sfid_splits": 2,
            "class_balanced": False,
        }
        fid = compute_fid(Namespace(**args))
        print("sFID:", round(fid, 1)) 
        result_list.append(round(fid, 1))

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
        result_list.append(round(np.mean(fva)*100, 1))
        result_list.append(round(np.mean(fs), 3))

        # MNAC
        args = {
            "output_path": path,
            "oracle_path": "/misc/lmbraid21/schrodi/pretrained_models/celeb_oracle_attribute/celebamaskhq/checkpoint.tar",
            "dataset": "CelebAHQ",
            "batch_size": 15,
        }
        mnac = compute_mnac(Namespace(**args))
        print("MNAC:", round(np.mean(mnac[0]), 2))
        result_list.append(round(np.mean(mnac[0]), 2))

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
        result_list.append(round(cd, 2))

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
        result_list.append(round(cout[0], 4))

        # get directory name

        #append the results to the csv file if it exists or create a new one if it does not exist
        #get main configs from the config.yaml file, sampler.classifier_lambda, sampler.dist_lambda, strength, ddim_steps
        # get strength, ddim_steps, sampler.classifier_lambda, sampler.dist_lambda
        result_list.append(config['strength'])
        result_list.append(config['ddim_steps'])
        result_list.append(config['sampler']['classifier_lambda'])
        result_list.append(config['sampler']['dist_lambda'])

        #append the results to the csv file if it exists or create a new one if it does not exist
        if os.path.exists(target_csv):
            df = pd.read_csv(target_csv)
            df = pd.concat([df, pd.DataFrame([result_list], columns=columns)])
            df.to_csv(target_csv, index=False)
            print("results appended to: ",target_csv)
        else:
            print("creating new csv file: ", target_csv)
            print(result_list)
            print(columns)

            df = pd.DataFrame([result_list], columns=columns)
            df.to_csv(target_csv, index=False)
            print("results saved to: ", target_csv)
        
        
            