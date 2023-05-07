import os
import wandb
import torch
import argparse
import yaml
import copy


entity = 'kifarid'
enabled = True
project = 'LVCE'

LMB_USERNAME = 'faridk'
#check if directories exist
os.makedirs(f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.cache/wandb", exist_ok=True)
os.makedirs(f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.wandb", exist_ok=True)
os.makedirs(f"/misc/lmbraid21/{LMB_USERNAME}/counterfactuals", exist_ok=True)
os.makedirs(f"/misc/lmbraid21/{LMB_USERNAME}/counterfactuals/checkpoints", exist_ok=True)

os.environ["WANDB_API_KEY"] = 'cff06ca1fa10f98d7fde3bf619ee5ec8550aba11'
os.environ['WANDB_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.wandb"
os.environ['WANDB_DATA_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/counterfactuals"
os.environ['WANDB_CACHE_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.cache/wandb"


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

def create_table(data_dir, table_name, run):
    # Initialize wandb run
    
    # Define columns for the wandb table
    columns = None
    
    # Loop through the files in the data directory
    for bucket in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, bucket)):
            print("logging bucket: ", bucket, " to wandb ...")
            for i, filename in enumerate(sorted(os.listdir(os.path.join(data_dir, bucket)))):
                if filename.endswith(".pth")  and filename.split('.')[0].isdigit():
                    print("logging file: ", filename, " to wandb ...")
                    # Load the data from the file
                    data = torch.load(os.path.join(data_dir, bucket, filename))

                    # Extract column names from the data
                    if columns is None:
                        columns = list(data.keys())
                        #remove columns that have the substring 'video' or 'cgs' in them
                        columns = [column for column in columns if 'video' not in column and 'cgs' not in column]
                        # Create wandb table
                        wandb_table = wandb.Table(columns=columns)

                    # Add data to wandb table
                    row_data= []
                    for column in columns:
                        if "image" in column or "img" in column or "counterfactual" in column:
                            #row_data.append(22)
                            wand_img = wandb.Image(data[column]) # wandb.Image((255.*data[column]).permute(1, 2, 0).to(torch.uint8).numpy())
                            row_data.append(wand_img)
                            #row_data.append(copy.deepcopy(wandb.Image(data[column])))
                        elif "video" in column or "cgs" in column:
                            pass 
                            #row_data.append(wandb.Video(data[column], fps=10, format="gif"))
                        else:
                            row_data.append(data[column])
                
                    #row_data = [data[column] for column in columns]
                    wandb_table.add_data(*row_data)
                #break
                #if i % 10 == 0:
            # wandb.log({"lvces_table"+str(bucket): wandb_table})
            # print("logged bucket: ", bucket, " at index: ", i, " to wandb ...")
    # Log wandb table
    return wandb_table


if __name__ == "__main__":
    # Add command line arguments for data directory and table name
    parser = argparse.ArgumentParser(description='Process data directory and table name')
    parser.add_argument('--data-dir', type=str, default='/misc/lmbraid21/faridk/LDCE_v8', help='path to data directory')
    parser.add_argument('--table-name', type=str, default='lvces_complete', help='name of the wandb table')
    parser.add_argument('--project', type=str, default='LVCE', help='name of the wandb project')
    parser.add_argument('--entity', type=str, default='kifarid', help='name of the wandb entity')
    parser.add_argument('--mode', type=str, choices=['online', 'offline'], default='online', help='wandb mode')


    args = parser.parse_args()

    #load config file from data directory
    config = load_yaml_files(args.data_dir)[0]

    print(config)

    run = wandb.init(entity=args.entity,
     project=args.project, 
     config=config,
    mode=args.mode)
    # Call create_table function with data directory and table name arguments
    table = create_table(args.data_dir, args.table_name, run)
    # Log wandb table
    run.log({args.table_name: table})
    run.finish()
