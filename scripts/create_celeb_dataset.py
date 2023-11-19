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
    
    # Adjectives    
    all_adjectives = ['Attractive', 'Bald', "Male", "Young", "Smiling"]

    # Gender
    #get gender from row
    gender = "male" if row['Male'] == 1 else "female"
    smiling = "smiling" if row['Smiling'] == 1 else "non-smiling"

    #Age
    age = "young" if row['Young'] == 1 else "old"

    # Features
    adjectives = []
    
    # Age or Smiling (3/8 for age, 3/8 for smiling, 1/4 for both)
    age_smiling_choice = np.random.choice(np.array(['age', 'smiling', 'both']), p=[3/8, 3/8, 1/4])
    
    if age_smiling_choice == 'age':
        adjectives.append(age)
    elif age_smiling_choice == 'smiling':
        adjectives.append(smiling)
    else:
        adjectives.extend([age, smiling])
    
    # add gender to adjectives 20% of the time
    adjectives.append(gender) if random.random() > 0.9 else None



    #available_features_row = row[row == 1]
    
    #available_adjectives_row = available_features_row[~available_features_row.index.isin(features)]
    available_features = row[row == 1].index.tolist()
    #filter out the features already used
    available_features = [feature for feature in available_features if feature not in all_adjectives]
    #filter out features with "wearing" in the name
    available_features = [feature for feature in available_features if "Wearing" not in feature]
    #add each to adjective with change 85% of adding each feature
    features = []
    for feature in available_features:
        if random.random() > 0.9:
            features.append(feature.replace("_", " ").lower())

    # Construct the prompt
    prompt = f"A photo of a {' , '.join(adjectives)} celebrity"
    if len(features) > 0:
        prompt += f" with {' and '.join(features)}"


    return prompt

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

df.to_csv(new_dir + "/meta_data.csv", index=False)
