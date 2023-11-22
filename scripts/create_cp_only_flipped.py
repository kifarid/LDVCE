import os
import torch
import argparse
import yaml
import copy
import pandas as pd
import shutil
from tqdm import tqdm


LMB_USERNAME = 'faridk'


def load_yaml_files(directory_path):
    yaml_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    yaml_files.append(yaml.load(f, Loader=yaml.FullLoader))
    return yaml_files

def copy_to_target(data_dir, target_dir):
    # Initialize wandb run
    
    # Define columns for the wandb table
    columns = None
    data_dict = {}
    # Loop through the files in the data directory
    for bucket in tqdm(sorted(os.listdir(data_dir))):
        if os.path.isdir(os.path.join(data_dir, bucket)):
            #make the target directory
            if 'bucket' not in bucket:
                continue
            os.makedirs(os.path.join(target_dir, bucket), exist_ok=True)
            #create the original and the counterfactual directories
            os.makedirs(os.path.join(target_dir, bucket, 'original'), exist_ok=True)
            os.makedirs(os.path.join(target_dir, bucket, 'counterfactual'), exist_ok=True)
            print("copying the bucket: ", bucket, " to target directory ...")
            count_files_flipped = 0
            

            for i, filename in tqdm(enumerate(sorted(os.listdir(os.path.join(data_dir, bucket))))):
                if filename.endswith(".yaml") or filename.endswith(".yml"):
                    #copy config file to the target directory
                    if not os.path.exists(os.path.join(target_dir, bucket, filename)):
                        shutil.copy(os.path.join(data_dir, bucket, filename), os.path.join(target_dir, bucket, filename))
                    else:
                        os.remove(os.path.join(target_dir, bucket, filename))
                        shutil.copy(os.path.join(data_dir, bucket, filename), os.path.join(target_dir, bucket, filename))
                        
                if filename.endswith(".pth")  and filename.split('.')[0].isdigit():
                    # Load the data from the file
                    data = torch.load(os.path.join(data_dir, bucket, filename))
                    #check if the file is a flipped file
                    if data['out_pred'] == data['target']:
                        #copy the file to the target directory
                        #check if the file is a flipped file
                        #overwrite the file if it already exists
                        if not os.path.exists(os.path.join(target_dir, bucket, filename)):
                            torch.save(data, os.path.join(target_dir, bucket, filename))
                        else:
                            os.remove(os.path.join(target_dir, bucket, filename))
                            torch.save(data, os.path.join(target_dir, bucket, filename))
                        #copy the image to the target directory from the two directories
                        #copy original
                        if not os.path.exists(os.path.join(target_dir, bucket, 'original', filename.split('.')[0] + '.png')):
                            shutil.copy(os.path.join(data_dir, bucket, 'original', filename.split('.')[0] + '.png'), os.path.join(target_dir, bucket, 'original', filename.split('.')[0] + '.png'))
                        else:
                            os.remove(os.path.join(target_dir, bucket, 'original', filename.split('.')[0] + '.png'))
                            shutil.copy(os.path.join(data_dir, bucket, 'original', filename.split('.')[0] + '.png'), os.path.join(target_dir, bucket, 'original', filename.split('.')[0] + '.png'))
                       #copy counterfactual
                        if not os.path.exists(os.path.join(target_dir, bucket, 'counterfactual', filename.split('.')[0] + '.png')):
                            shutil.copy(os.path.join(data_dir, bucket, 'counterfactual', filename.split('.')[0] + '.png'), os.path.join(target_dir, bucket, 'counterfactual', filename.split('.')[0] + '.png'))
                        else:
                            os.remove(os.path.join(target_dir, bucket, 'counterfactual', filename.split('.')[0] + '.png'))
                            shutil.copy(os.path.join(data_dir, bucket, 'counterfactual', filename.split('.')[0] + '.png'), os.path.join(target_dir, bucket, 'counterfactual', filename.split('.')[0] + '.png'))
                        count_files_flipped += 1
                        #clear space in memory
                        

                    else:
                        continue
                    
                    del data
                    

            
            print("number of flipped files in bucket: ", bucket, " is: ", count_files_flipped, "out of: ", i)
        
        #clear memory until the next bucket
        
        
        
    return 


if __name__ == "__main__":
    # Add command line arguments for data directory and table name
    parser = argparse.ArgumentParser(description='Process data directory and table name')
    parser.add_argument('--data-dir', type=str, default='/misc/lmbraid21/faridk/celeb_smile_corrected_8/', help='path to data directory') #np_4_33_40_55

    args = parser.parse_args()
    #change the name of the final directory and add to it _flipped
    target_path = os.path.dirname(args.data_dir) + '_flipped/'

    os.makedirs(target_path, exist_ok=True)
    #load config file from data directory
    config = load_yaml_files(args.data_dir)[0]

    print(config)

    # Call create_table function with data directory and table name arguments
    df = copy_to_target(args.data_dir, target_path)