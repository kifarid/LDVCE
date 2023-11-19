from PIL import Image, ImageFilter
import numpy as np
import torch
import cv2
import open_clip
import json
from utils.vision_language_wrapper import VisionLanguageWrapper
from torchvision import transforms
from utils.preprocessor import GenericPreprocessing, CropAndNormalizer
import torchvision
from data.imagenet_classnames import name_map
from utils.dino_linear import LinearClassifier, DINOLinear
import os
import itertools
import math
from copy import deepcopy
from functools import partial

from utils.fig_utils import get_concat_h

device = torch.device('cuda')

def linear_combination(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return img1*mask + img2*(1-mask)

def get_resnet50_imagenet():
    classifier_name = "resnet50"
    classifier_model = getattr(torchvision.models, classifier_name)(pretrained=True)
    classifier_model = CropAndNormalizer(classifier_model)
    classifier_model = classifier_model.eval().to(device)
    return classifier_model

def compute_shapley_values(value_fn, features):
    n = len(features)
    s = n -1
    shapley_values = []
    for i, ith_feature in enumerate(features):
        shapely_value = 0
        tmp_features = [feat for j,feat in enumerate(features) if j!=i]
        for subset in itertools.product([False, True], repeat=s):
            mask = torch.ones((256, 256, 3))
            for on_off, feat in zip(subset, tmp_features):
                mask = feat(mask, on_off)
            mask_with_i = ith_feature(deepcopy(mask), True)
            mask_without_i = ith_feature(deepcopy(mask), False)
            shapely_value += (math.factorial(s)*math.factorial(n-s-1))/math.factorial(n)*(value_fn(mask_with_i).item() - value_fn(mask_without_i).item())
        shapley_values.append(shapely_value)
    return shapley_values

def chin_to_persian():
    filepath = "/misc/lmbraid21/faridk/ldvce_pets_42_24_correct/examples/correct/00245_japanese chin_persian.jpg"
    # img = np.array(Image.open(filepath))
    # original, counterfactual = img[:,:256], img[:,256:]
    
    filepath = "/misc/lmbraid21/faridk/ldvce_pets_42_24_correct/bucket_0_7/01722.pth"
    data = torch.load(filepath)
    assert data["source"] == data["in_pred"] and data["target"] == data["out_pred"]

    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)


    mask = torch.ones((256, 256, 3))
    
    mask[70:170, 70:220] = 1 # all

    # green eyes
    mask[65:95, 155:190] = 0 # left
    mask[90:115, 90:120] = 0 # right

    # snout/mouth
    mask[95:140, 120:180] = 0
    
    # whiskers
    mask[120:170, 70:120] = 0 # left
    mask[100:170, 180:220] = 0 # right

    # copy everything but green eyes
    # mask = np.zeros((256, 256, 3))
    # mask[70:95, 155:190] = 1 # left
    # mask[90:115, 90:120] = 1 # right

    cv2.imwrite("mask.png", (mask*255).numpy())
    cv2.imwrite("masked_cf.png", (counterfactual*mask).numpy())

    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", (linear_combo).numpy())

    tmp = cv2.rectangle(counterfactual.numpy(),(155, 65),(190, 95),(0, 0, 255), 2)
    tmp = cv2.rectangle(tmp,(90, 90),(120, 115),(0, 0, 255), 2)
    tmp = cv2.rectangle(tmp,(120, 95),(180, 140),(255, 0, 0), 2)
    tmp = cv2.rectangle(tmp,(70, 120),(120, 170),(255, 0, 255), 2)
    tmp = cv2.rectangle(tmp,(180, 100),(220, 170),(255, 0, 255), 2)
    cv2.imwrite("rectangles.png", tmp)

    get_concat_h(*[Image.fromarray(im.astype(np.uint8)) for im in [original.numpy(), cv2.UMat.get(tmp), linear_combo.numpy()]]).save("chin_to_persian.jpg")


    # zero-shot OpenClip: https://arxiv.org/pdf/2212.07143.pdf
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    # prompts following https://github.com/openai/CLIP/blob/main/data/prompts.md
    with open("data/pets_idx_to_label.json", "r") as f:
        pets_idx_to_classname = json.load(f)
    prompts = [f"a photo of a {label}, a type of pet." for label in pets_idx_to_classname.values()]
    classifier_model = VisionLanguageWrapper(model, tokenizer, prompts)
    # try running optimization on 224x224 pixel image
    # transforms_list = [preprocess.transforms[0], preprocess.transforms[1], preprocess.transforms[4]]
    transforms_list = [preprocess.transforms[1], preprocess.transforms[4]] # CenterCrop(224, 224), Normalize
    classifier_model = GenericPreprocessing(classifier_model, transforms.Compose(transforms_list))

    with open("data/pets_idx_to_label.json", "r") as f:
        pets_idx_to_classname = json.load(f)
    i2h = {int(k): v for k, v in pets_idx_to_classname.items()}

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[17].item(), 3), round(softmax[23].item(), 3))

    def fn(mask, original, counterfactual, model):
        linear_combo = linear_combination(original, counterfactual, mask)
        input = (linear_combo / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = model(input).cpu()
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        return softmax[23]
    value_fn = partial(fn, original=original, counterfactual=counterfactual, model=classifier_model)
    
    def green_eyes(mask, on_off):
        mask[65:95, 155:190] = int(not on_off) # left
        mask[90:115, 90:120] = int(not on_off) # right
        return mask
    
    def snout(mask, on_off):
        mask[95:140, 120:180] = int(not on_off)
        return mask
    
    def whiskers(mask, on_off):
        mask[120:170, 70:120] = int(not on_off) # left
        mask[100:170, 180:220] = int(not on_off) # right
        return mask

    features = [
        green_eyes,
        snout,
        whiskers
    ]
    shapley_values = compute_shapley_values(value_fn, features)
    print(shapley_values, list(np.array(shapley_values)/ sum(shapley_values)), sum(shapley_values))
    

def soccer_to_golf():
    filepath = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/bucket_1_2/01805.pth"
    data = torch.load(filepath)
    assert data["source"] == data["in_pred"] and data["target"] == data["out_pred"]
    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)
    
    mask = torch.ones((256, 256, 3))
    # ball
    mask[20:130, 85:180] = 0
    #mask[20:80, 100:180] = 0

    # human
    mask[130:250, 110:140] = 0

    cv2.imwrite("masked_cf.png", (counterfactual*mask).numpy())
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", linear_combo.numpy())

    tmp = cv2.rectangle(counterfactual.numpy(),(85, 20),(180, 130),(0, 0, 255), 2)
    tmp = cv2.rectangle(tmp,(110, 130),(140, 250),(255, 0, 0), 2)
    cv2.imwrite("rectangles.png", tmp)

    get_concat_h(*[Image.fromarray(im.astype(np.uint8)) for im in [original.numpy(), cv2.UMat.get(tmp), linear_combo.numpy()]]).save("soccer_to_golf.jpg")

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[805].item(), 3), round(softmax[574].item(), 3))

    def fn(mask, original, counterfactual, model):
        linear_combo = linear_combination(original, counterfactual, mask)
        input = (linear_combo / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = model(input).cpu()
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        return softmax[574]
    value_fn = partial(fn, original=original, counterfactual=counterfactual, model=classifier_model)
    
    def ball(mask, on_off):
        mask[20:130, 85:180] = int(not on_off)
        return mask
    
    def tee(mask, on_off):
        mask[130:250, 110:140] = int(not on_off)
        return mask

    features = [
        ball,
        tee,
    ]
    shapley_values = compute_shapley_values(value_fn, features)
    print(shapley_values, list(np.array(shapley_values)/ sum(shapley_values)), sum(shapley_values))


def sandal_running_shoe():
    filepath = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/bucket_0_1/00022.pth"
    data = torch.load(filepath)
    assert data["source"] == data["in_pred"] and data["target"] == data["out_pred"]

    filepath = "/misc/lmbraid21/faridk/imagenet_appendix/correct/00022_bald eagle_cock.jpg"
    filepath = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/examples/correct/00022_bald eagle_cock.jpg"
    img = np.array(Image.open(filepath))[..., ::-1]
    original, counterfactual = img[:,:256], img[:,-256:]
    
    mask = np.ones((256, 256, 3))
    
    # front of shoe
    mask[20:130, 85:180] = 1

    cv2.imwrite("masked_cf.png", (counterfactual*mask).numpy())
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", (linear_combo).numpy())

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = torch.from_numpy(img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[774].item(), 3), round(softmax[770].item(), 3))

def minibus_to_limousine():
    filepath = "/misc/lmbraid21/faridk/misclassifications/bucket_0_50/00142.pth"
    data = torch.load(filepath)
    assert data["target"] == data["out_pred"]

    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)
    #original = np.flip(np.uint8(original), axis=-1)
    #counterfactual = np.flip(np.uint8(counterfactual), axis=-1)
    
    mask = np.ones((256, 256, 3))
    
    mask[130:, 200:] = 0

    cv2.imwrite("masked_cf.png", (counterfactual*mask).numpy().astype(np.uint8))
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", linear_combo.numpy())

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[627].item(), 3), round(softmax[627].item(), 3))

def maraca_wooden_spon():
    filepath = "/misc/lmbraid21/faridk/misclassifications/bucket_0_50/00227.pth"
    data = torch.load(filepath)
    assert data["target"] == data["out_pred"]

    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)
    #original = np.flip(np.uint8(original), axis=-1)
    #counterfactual = np.flip(np.uint8(counterfactual), axis=-1)
    
    mask = np.ones((256, 256, 3))
    
    mask[25:100, 50:130] = 0
    #mask[100:150, 150:200] = 0

    cv2.imwrite("masked_cf.png", (counterfactual*mask).numpy().astype(np.uint8))
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", linear_combo.numpy())

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[910].item(), 3), round(softmax[910].item(), 3))

def sandal_running_shoe_misclassification():
    filepath = "/misc/lmbraid21/faridk/misclassifications/bucket_0_50/00416.pth"
    data = torch.load(filepath)
    assert data["target"] == data["out_pred"]

    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)
    #original = np.flip(np.uint8(original), axis=-1)
    #counterfactual = np.flip(np.uint8(counterfactual), axis=-1)
    
    mask = np.ones((256, 256, 3))
    
    mask[130:180, 180:220] = 0
    mask[130:180, 120:145] = 0

    cv2.imwrite("masked_cf.png", (counterfactual*mask).numpy().astype(np.uint8))
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", linear_combo.numpy())

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[770].item(), 3), round(softmax[774].item(), 3))


def window_shade_shower_curtain():
    filepath = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/bucket_5_6/05905.pth"
    data = torch.load(filepath)
    assert data["source"] == data["in_pred"] and data["target"] == data["out_pred"]

    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)
    #original = np.flip(np.uint8(original), axis=-1)
    #counterfactual = np.flip(np.uint8(counterfactual), axis=-1)
    
    mask = np.ones((256, 256, 3))
    
    mask[180:, :] = 0

    cv2.imwrite("masked_cf.png", (counterfactual*mask).numpy().astype(np.uint8))
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", linear_combo.numpy())

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[770].item(), 3), round(softmax[774].item(), 3))

def running_shoe_cowboy_boot():
    filepath = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/bucket_2_3/02770.pth"
    data = torch.load(filepath)
    assert data["source"] == data["in_pred"] and data["target"] == data["out_pred"]

    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)
    #original = np.flip(np.uint8(original), axis=-1)
    #counterfactual = np.flip(np.uint8(counterfactual), axis=-1)
    
    mask = np.ones((256, 256, 3))
    
    mask[140:200, 150:210] = 0
    #mask[60:100, 30:60] = 0
    #mask[40:100, 70:140] = 0
    #mask[130:230, 30:220] = 0
    mask[130:230, 110:140] = 0
    mask[:130, 30:220] = 0

    cv2.imwrite("masked_cf.png", cv2.cvtColor((counterfactual*mask).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", cv2.cvtColor(linear_combo.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[770].item(), 3), round(softmax[514].item(), 3))

def running_shoe_cowboy_boot2():
    filepath = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/bucket_3_4/03770.pth"
    data = torch.load(filepath)
    assert data["source"] == data["in_pred"] and data["target"] == data["out_pred"]

    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)
    #original = np.flip(np.uint8(original), axis=-1)
    #counterfactual = np.flip(np.uint8(counterfactual), axis=-1)
    
    mask = np.ones((256, 256, 3))
    
    mask[70:130, 90:200] = 0
    mask[120:170, 10:70] = 0

    cv2.imwrite("masked_cf.png", cv2.cvtColor((counterfactual*mask).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", cv2.cvtColor(linear_combo.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[770].item(), 3), round(softmax[514].item(), 3))

def running_shoe_cowboy_boot2():
    filepath = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/bucket_3_4/03770.pth"
    data = torch.load(filepath)
    assert data["source"] == data["in_pred"] and data["target"] == data["out_pred"]

    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)
    #original = np.flip(np.uint8(original), axis=-1)
    #counterfactual = np.flip(np.uint8(counterfactual), axis=-1)
    
    mask = np.ones((256, 256, 3))
    
    mask[70:130, 90:200] = 0
    mask[120:170, 10:70] = 0

    cv2.imwrite("masked_cf.png", cv2.cvtColor((counterfactual*mask).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", cv2.cvtColor(linear_combo.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[770].item(), 3), round(softmax[514].item(), 3))


def test():
    filepath = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/bucket_3_4/03770.pth"
    data = torch.load(filepath)
    assert data["source"] == data["in_pred"] and data["target"] == data["out_pred"]

    original, counterfactual = (data["image"]*255.).permute(1,2,0), (data["gen_image"]*255.).permute(1,2,0)
    #original = np.flip(np.uint8(original), axis=-1)
    #counterfactual = np.flip(np.uint8(counterfactual), axis=-1)
    
    mask = np.ones((256, 256, 3))
    
    mask[70:130, 90:200] = 0

    linear_combo = torch.clone(original)
    area = linear_combo[70:130, 90:200]
    area = np.array(Image.fromarray(area.numpy().astype(np.uint8)).filter(ImageFilter.BLUR))
    linear_combo[70:130, 90:200] = torch.from_numpy(area)
    cv2.imwrite("combined.png", cv2.cvtColor(linear_combo.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))

    classifier_model = get_resnet50_imagenet()
    
    i2h = name_map

    for img in [original, counterfactual, linear_combo]:
        input = (img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[770].item(), 3), round(softmax[514].item(), 3))

def arum_lily_to_magnolia():
    filepath = "/misc/lmbraid21/faridk/ldvce_flowers_correct_targets/examples/correct/00106_giant white arum lily_magnolia.jpg"
    img = np.array(Image.open(filepath))[..., ::-1]
    original, counterfactual = img[:,:256], img[:,256:]

    mask = np.ones((256, 256, 3))

    mask[20:220, 30:240] = 1
    
    mask[130:180, 180:220] = 0

    mask[170:200, 60:90] = 0

    mask[110:170, 120:160] = 0

    mask[20:130, 60:110] = 0

    mask[40:120, 200:240] = 0

    cv2.imwrite("masked_cf.png", counterfactual*mask)
    linear_combo = linear_combination(original, counterfactual, mask)
    cv2.imwrite("combined.png", linear_combo)

    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device).eval()
    dim = dino.embed_dim
    linear_classifier = LinearClassifier(dim*1, 102)
    linear_classifier.load_state_dict(torch.load("/misc/lmbraid21/schrodi/pretrained_models/dino_flowers_linear.pth", map_location="cpu"), strict=True)
    linear_classifier = linear_classifier.eval().to(device)
    classifier_model = DINOLinear(dino, linear_classifier)
    transforms_list = [transforms.CenterCrop(224), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    classifier_model = GenericPreprocessing(classifier_model, transforms.Compose(transforms_list))

    with open("data/flowers_idx_to_label.json", "r") as f:
        flowers_idx_to_classname = json.load(f)
    flowers_idx_to_classname = {int(k)-1: v for k, v in flowers_idx_to_classname.items()}
    i2h = flowers_idx_to_classname

    for img in [original, counterfactual, linear_combo]:
        input = torch.from_numpy(img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input).cpu()
        pred = logits.argmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)(logits)[0]
        print(pred.item(), i2h[pred.item()], round(softmax[19].item(), 3), round(softmax[86].item(), 3))

def debug():
    classifier_model = get_resnet50_imagenet()
    filepath = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/bucket_0_1/00022.pth"
    i2h = name_map
    data = torch.load(filepath)

    print("Original", data["source"])
    print("CF", data["target"])

    print("### Image tensors ###")
    with torch.no_grad():
        logits = classifier_model(data["image"].to(device)).cpu()
    pred = logits.argmax(dim=1)
    print(pred.item(), data["in_pred"], i2h[pred.item()])

    with torch.no_grad():
        gen_image = data["gen_image"]
        # gen_image = data["gen_image"] + torch.randn(data["gen_image"].size()) * 20/255.
        # gen_image = torchvision.transforms.GaussianBlur(5, sigma=1.0)(data["gen_image"])
        logits = classifier_model(gen_image.to(device)).cpu()
    pred = logits.argmax(dim=1)
    print(pred.item(), data["out_pred"], i2h[pred.item()])

    print("### Saved image by torchvision ###")
    orig_jpg_path = os.path.join(os.path.dirname(filepath), "original", os.path.basename(filepath).replace("pth", "png"))
    img = np.array(Image.open(orig_jpg_path))[..., ::-1]
    with torch.no_grad():
        input = torch.from_numpy(img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input.to(device)).cpu()
    pred = logits.argmax(dim=1)
    print(pred.item(), data["in_pred"], i2h[pred.item()])

    cf_jpg_path = os.path.join(os.path.dirname(filepath), "counterfactual", os.path.basename(filepath).replace("pth", "png"))
    img = np.array(Image.open(cf_jpg_path))[..., ::-1]
    with torch.no_grad():
        input = torch.from_numpy(img / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
        logits = classifier_model(input.to(device)).cpu()
    pred = logits.argmax(dim=1)
    print(pred.item(), data["out_pred"], i2h[pred.item()])

    print("### Concatenated image ###")
    joint_img = "/misc/lmbraid21/faridk/LDCE_sd_correct_3925_50/examples/correct/00022_bald eagle_cock.jpg"
    img = np.array(Image.open(joint_img))[..., ::-1]
    for im in [img[:,:256], img[:,256:]]:
        with torch.no_grad():
            input = torch.from_numpy(im / 255.).to(device).permute(2,0,1).unsqueeze(0).float()
            logits = classifier_model(input.to(device)).cpu()
        pred = logits.argmax(dim=1)
        print(pred.item(), data["out_pred"], i2h[pred.item()])

if __name__ == "__main__":
    #chin_to_persian()
    #soccer_to_golf()
    test()
    #sandal_running_shoe()
    #arum_lily_to_magnolia()
    # debug()