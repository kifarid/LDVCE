import os
import torch
import argparse
import yaml
import copy
import pandas as pd



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

# Call the function to load all YAML files in the directory and subdirectories

def create_df(data_dir):
    # Initialize wandb run
    
    # Define columns for the wandb table
    columns = None
    data_dict = {}
    # Loop through the files in the data directory
    for bucket in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, bucket)):
            print("logging bucket: ", bucket, " to wandb ...")
            for i, filename in enumerate(sorted(os.listdir(os.path.join(data_dir, bucket)))):
                if filename.endswith(".pth")  and filename.split('.')[0].isdigit():
                    print("logging file: ", filename, " to wandb ...")
                    # Load the data from the file
                    data = torch.load(os.path.join(data_dir, bucket, filename))


                    # Add data to the data dictionary except for the image data
                    for key, value in data.items():
                        #remove columns that have the substring 'video' or 'cgs' in them
                        if 'video' in key or 'cgs' in key or 'image' in key or 'img' in key or 'counterfactual' in key:
                            continue
                        if key not in data_dict:
                            data_dict[key] = []
                        data_dict[key].append(value)
                #break
    # Create dataframe from data dictionary

    df = pd.DataFrame(data_dict)
    return df


if __name__ == "__main__":
    # Add command line arguments for data directory and table name
    parser = argparse.ArgumentParser(description='Process data directory and table name')
    parser.add_argument('--data-dir', type=str, default='/misc/lmbraid21/faridk/LDCE_v8', help='path to data directory')

    args = parser.parse_args()

    #load config file from data directory
    config = load_yaml_files(args.data_dir)[0]

    print(config)

    # Call create_table function with data directory and table name arguments
    df = create_df(args.data_dir)
    df.head()
    # Log wandb table
    #get df without the image columns
    #df_no_img = df.drop(columns=[col for col in df.columns if 'image' in col or 'img' in col or 'counterfactual' in col or 'video' in col or 'cg' in col])

    #get average of each column

