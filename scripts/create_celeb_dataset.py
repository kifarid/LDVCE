import shutil
import pandas as pd
import numpy as np
import random
import os


celeb_path = '/misc/lmbraid21/schrodi/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt'
df = pd.read_csv(celeb_path, skiprows=[0], delim_whitespace=True)
    # Available columns in the dataframe: ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    #    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    #    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    #    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    #    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    #    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    #    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    #    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    #    'Wearing_Necklace', 'Wearing_Necktie', 'Young']

#load /misc/lmbraid21/schrodi/CelebAMask-HQ/list_eval_partition.csv
df_split = pd.read_csv('/misc/lmbraid21/schrodi/CelebAMask-HQ/list_eval_partition.csv')
#get only the test split
df['image'] = df.index
df.reset_index(drop=True, inplace=True)
df = df[(df_split['split'] == 2).reset_index(drop=True)]

def generate_prompt(row):
    """
    Generates a prompt for a given row of the CelebAMask-HQ dataset.
    each row contains the following columns:
    ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    """

    # Age or Smiling (3/8 for age, 3/8 for smiling, 1/4 for both)
    age_smiling_choice = np.random.choice(np.array(['age', 'smiling', 'both']), p=[3/8, 3/8, 1/4])
    age_prompt = 'Age:18-30' if row['Young'] == 1 else 'Age:old'
    smiling_prompt = 'Face attributes: smiling' if row['Smiling'] == 1 else 'Face attributes: no smile, resting face'
    
    if age_smiling_choice == 'age':
        prompt_form = f"A photograph of a celebrity {age_prompt}, Face, High resolution"
    elif age_smiling_choice == 'smiling':
        prompt_form = f"A photograph of a celebrity, Face, {smiling_prompt}, High resolution"
    else:
        prompt_form = f"A photograph of a celebrity {age_prompt}, Face, {smiling_prompt}, High resolution"
  

    return prompt_form

df['text'] = df.apply(generate_prompt, axis=1)
df = df[['image', 'text']]
#get parent directory of celeb_path
parent_dir = os.path.dirname(celeb_path)

new_dir = '/misc/lmbraid21/faridk/CelebAMask-HQ_lora/'
# make new directory while checking existence
os.makedirs(new_dir, exist_ok=True)
# copy relevant images to the new directory
for i in range(len(df)):
    image_local_path = df.iloc[i]['image']
    image_path = os.path.join(parent_dir, 'CelebA-HQ-img', image_local_path)
    new_image_path = os.path.join(new_dir, image_local_path)
    #copy without permissions
    shutil.copyfile(image_path, new_image_path, follow_symlinks=False)

df.to_csv(new_dir + "/meta_data_new.csv", index=False)
