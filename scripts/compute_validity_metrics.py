import os
import torch
import torchvision
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm

from sampling_helpers import disabled_train

def load_img(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return img

def arguments():
    parser = argparse.ArgumentParser(description="Validity metrics evaluation")
    parser.add_argument('--cf_dir', required=True, type=str)
    parser.add_argument('--orig_dir', required=True, type=str)
    parser.add_argument('--model', required=True, type=str, default="resnet50")
    return parser.parse_args()

imagenet_mean = (0.485, 0.456, 0.406),
iamgenet_std = (0.229, 0.224, 0.225),
imagenet_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=imagenet_mean, std=iamgenet_std),
    ]
)

@torch.inference_mode()
def compute_validity_metrics(counterfactual_dir: str, original_dir: str, model: torch.nn.Module, device):
    flipped, confidences = [], []
    softmax = torch.nn.Softmax()
    for cf_example in tqdm(os.listdir(counterfactual_dir)):
        counterfactual = load_img(os.path.join(counterfactual_dir, cf_example))
        original = load_img(os.path.join(original_dir, cf_example)) # TODO

        # TODO cf & orig label
        cf_label = None
        orig_label = None

        inputs = torch.from_numpy(
            np.stack([original, counterfactual], axis=0)
        ).to(device)
        logits = model(inputs)
        softmax_logits = softmax(logits)
        preds = torch.argmax(logits, dim=-1)

        if preds[0] == orig_label and preds[1] == cf_label:
            flipped.append(1)
        elif preds[1] != cf_label:
            flipped.append(0)
        elif preds[0] != orig_label and preds[1] == cf_label: # ignore this case since orig pred is already wrong...
            pass
        else:
            raise NotImplementedError

        confidences.append(softmax_logits[1, cf_label].cpu().item())

    return np.sum(flipped)/len(flipped), np.mean(confidences)

if __name__ == '__main__':
    args = arguments()
    device = torch.device("cuda")

    classifier_model = getattr(torchvision.models, args.model)(pretrained=True).to(device)
    classifier_model = classifier_model.eval()
    classifier_model.train = disabled_train
    
    flip_ratio, confidence = compute_validity_metrics(
        counterfactual_dir=args.cf_dir, 
        original_dir=args.orig_dir, 
        model=classifier_model, 
        device=device
    )

    print(f"Flip ratio: {flip_ratio}")
    print(f"Confidence: {confidence}")
