import cv2
import glob
import torch
import numpy as np
import os
import json
import random

#base_path = "/home/simon/Downloads/tmp/user_study_data"

base_path = "/misc/lmbraid21/faridk/misclassifications/user_study_data"

# Lists to store the bounding box coordinates
top_left_corner=[]
bottom_right_corner=[]

# function which will be called on mouse input
def drawRectangle(action, x, y, flags, *userdata):
    image= temp.copy()
    # Referencing global variables 
    global top_left_corner, bottom_right_corner
    # Mark the top left corner when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = [(min(x,image.shape[1]//2-1),y)]
    # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        bottom_right_corner = [(min(x,image.shape[1]//2-1),y)]    
    # Draw the rectangle
    cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0,0,255),2, 8)
    cv2.imshow(title,image)

save_bbox = {}
random.seed(0)

paths = sorted(glob.glob(base_path + "/*.pth"))
for pth_path in paths:
    #random.shuffle(paths) # change order to avoid confounders
    data = torch.load(pth_path)
    assert data["out_pred"] == data["target"]
    in_pred, out_pred = data["in_pred"], data["out_pred"]
    original, counterfactual = (data["image"].permute(1,2,0).numpy()*255).astype(np.uint8), (data["gen_image"].permute(1,2,0).numpy()*255).astype(np.uint8)

    for show_cf in [False, True]: # don't influence users
        if show_cf:
            image = cv2.hconcat([original, counterfactual])
        else:
            image = cv2.hconcat([original, np.zeros_like(original)])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        temp = image.copy()
        title = f"Originally predicted: {in_pred} but should be: {out_pred}"
        cv2.namedWindow(title)
        # highgui function called when mouse events occur
        cv2.setMouseCallback(title, drawRectangle)
        
        k=0
        # Close the window when key q is pressed
        while k!=113 and k!=27:
            # Display the image
            cv2.imshow(title, image)
            k = cv2.waitKey(0)
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:        
                break     
        
        cv2.destroyAllWindows()

        if os.path.basename(pth_path) not in save_bbox:
            save_bbox[os.path.basename(pth_path)] = {}
        save_bbox[os.path.basename(pth_path)][int(show_cf)] = (top_left_corner[0], bottom_right_corner[0])

        top_left_corner=[]
        bottom_right_corner=[]

with open("bbox.json", "w") as f:
    json.dump(save_bbox, f, indent=2)