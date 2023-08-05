import torch
import torchvision

try:
    import shap
except:
    print("WARNING: Run pip install shap")

try:
    import captum
    from captum.attr._utils.visualization import _normalize_attr
except:
    print("WARNING: Run pip install captum")

try:
    from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, AblationCAM, ScoreCAM, EigenCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except:
    print("WARNING: Run pip install grad-cam")

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def compute_gradcam(model, layer, input, cls_idx):
    model.zero_grad()
    #cam = GradCAM(model=model, target_layers=[layer], use_cuda=True)
    #cam = HiResCAM(model=model, target_layers=[layer], use_cuda=True)
    #cam = GradCAMPlusPlus(model=model, target_layers=[layer], use_cuda=True)
    #cam = AblationCAM(model=model, target_layers=[layer], use_cuda=True)
    cam = ScoreCAM(model=model, target_layers=[layer], use_cuda=True)
    #cam = EigenCAM(model=model, target_layers=[layer], use_cuda=True)
    targets = [ClassifierOutputTarget(cls_idx)]
    grayscale_cam = cam(input_tensor=input, targets=targets, aug_smooth=True, eigen_smooth=True)
    return grayscale_cam

def compute_guided_gradcam(model, layer, input, cls_idx):
    model.zero_grad()
    guided_gc = captum.attr.GuidedGradCam(model, layer)
    attribution = guided_gc.attribute(input, target=cls_idx)
    return attribution

def compute_integraded_gradients(model, input, cls_idx):
    model.zero_grad()
    guided_gc = captum.attr.IntegratedGradients(model)
    attribution = guided_gc.attribute(input, target=cls_idx)
    return attribution

def compute_lrp(model, input, cls_idx):
    model.zero_grad()
    lrp = captum.attr.LRP(model)
    attribution = lrp.attribute(input, target=cls_idx)
    return attribution

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(pretrained=True).to(device).eval()

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    imagenet_transform1 = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
        ]
    )
    imagenet_transform2 = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    with open("cat_dog.png", "rb") as f:
        img = Image.open(f)
        img.load()
    img = imagenet_transform1(img)
    input = imagenet_transform2(img).to(device).requires_grad_()
    
    #vals = compute_guided_gradcam(model, model.layer4[-1], input.unsqueeze(0), cls_idx=0)
    #vals = compute_integraded_gradients(model, input.unsqueeze(0), cls_idx=0)
    #vals = compute_lrp(model, input.unsqueeze(0), cls_idx=0)
    #vals = compute_gradcam(model, model.layer4[-1], input.unsqueeze(0), cls_idx=254)
    #vals = compute_gradcam(model, model.layer4[-1], input.unsqueeze(0), cls_idx=281)
    vals = compute_gradcam(model, model.layer4[-1], input.unsqueeze(0), cls_idx=281)

    grayscale_cam = vals[0, :]
    visualization = show_cam_on_image(np.array(img)/255, grayscale_cam, use_rgb=True)
    viz = Image.fromarray(visualization)
    viz.save("tmp.png")


    #attr = np.transpose(vals.squeeze().cpu().detach().numpy(), (1, 2, 0))
    attr = np.transpose(vals, (1, 2, 0))
    # norm_attr = _normalize_attr(attr, "all", 2, reduction_axis=2)
    # cmap = LinearSegmentedColormap.from_list(
    #     "RdWhGn", ["red", "white", "green"]
    # )
    # vmin, vmax = -1, 1

    # plt.imshow(np.mean(img, axis=2), cmap="gray")
    # heat_map = plt.imshow(
    #     norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8
    # )
    # plt.axis("off")
    # plt.savefig("tmp.png")
    # plt.close()

    # heatmap = cv2.applyColorMap(np.uint8(255 * attr[..., 0]), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam = (1 - 0.5) * heatmap + 0.5 * np.array(img)/255
    # cam = cam / np.max(cam)
    # plt.imshow(visualization)
    # plt.axis("off")
    # plt.savefig("tmp.png")
    # plt.close()