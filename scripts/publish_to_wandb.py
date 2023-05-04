import os
import wandb
import torch
import argparse


def create_table(data_dir, table_name):
    # Initialize wandb run
    wandb.init(project="your-project-name")
    # Define columns for the wandb table
    columns = None

    # Loop through the files in the data directory
    for bucket in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, bucket)):
            for filename in os.listdir(os.path.join(data_dir, bucket)):
                if filename.endswith(".pt")  and not filename.split('.')[0].isdigit():
                    # Load the data from the file
                    data = torch.load(os.path.join(data_dir, bucket, filename))

                    # Extract column names from the data
                    if columns is None:
                        columns = list(data.keys())

                        # Create wandb table
                        wandb_table = wandb.Table(columns=columns, name=table_name)

                    # Add data to wandb table
                    row_data = [data[column] for column in columns]
                    wandb_table.add_data(*row_data)

    # Log wandb table
    wandb.log({table_name: wandb_table})


if __name__ == "__main__":
    # Add command line arguments for data directory and table name
    parser = argparse.ArgumentParser(description='Process data directory and table name')
    parser.add_argument('--data-dir', type=str, required=True, help='path to data directory')
    parser.add_argument('--table-name', type=str, default='data_table', help='name of the wandb table')
    args = parser.parse_args()

    # Call create_table function with data directory and table name arguments
    create_table(args.data_dir, args.table_name)
