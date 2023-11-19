import torch
import os
import numpy as np
from PIL import Image
import imageio

dir_path = "/misc/lmbraid21/faridk/LDCE_sd_correct_seed_2/bucket_0_50/" 
dir_path = "/misc/lmbraid21/faridk/LDCE_sd_more_clip_seed_0/bucket_0_50/"
# loop over all images

file_list = os.listdir(dir_path)
file_list.sort()
# sort 

for i, filename in enumerate(file_list):
    if filename.endswith(".pth"):
        #loaded = torch.load(dir_path+file)
        print(i, filename)
        file_loaded = torch.load(os.path.join(dir_path, filename))
        if 'video' not in  file_loaded:
            continue
        video = file_loaded['video']
        #create a gif from the video

        video = video.permute(0, 2, 3, 1).cpu().numpy()
        # Save the frames as a GIF
        gif_filename = os.path.splitext(filename)[0] + ".gif"
        gif_path = os.path.join(dir_path, gif_filename)
        imageio.mimsave(gif_path, video, fps=60)  # Adjust the duration as needed
        print(f"Created GIF {i}: {gif_filename}")
        #get
       
