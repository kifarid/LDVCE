import time
from itertools import islice

import PIL
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch import distributions as torchd
from torch.nn import functional as F

from ldm.util import instantiate_from_config


# sys.path.append(".")
# sys.path.append('./taming-transformers')


# @title loading utils
def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model#.to(device)  # .cuda()
    model.eval()
    return model


def get_model(cfg_path="configs/latent-diffusion/cin256-v2.yaml", ckpt_path="models/ldm/cin256-v2/model.ckpt"):
    config = OmegaConf.load(cfg_path)
    model = load_model_from_config(config, ckpt_path)
    return model


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((256, 256), resample=PIL.Image.LANCZOS)  # ((w, h), resample=PIL.Image.LANCZOS)
    pil_iamge = image
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image, pil_iamge


def compute_lp_dist(diff, p):
    diff_abs_flat = diff.abs().view(diff.shape[0], -1)
    if p == 1.0:
        lp_dist = torch.sum(diff_abs_flat, dim=1)
    else:
        lp_dist = torch.sum(diff_abs_flat ** p, dim=1)
    return lp_dist


def compute_lp_gradient(diff, p, small_const=1e-12):
    if p < 1:
        grad_temp = (p * (diff.abs() + small_const) ** (

                p - 1)) * diff.sign()
    else:
        grad_temp = (p * diff.abs() ** (p - 1)) * diff.sign()
    return grad_temp


def _renormalize_gradient(grad, eps, small_const=1e-22):
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    grad_norm = torch.where(grad_norm < small_const, grad_norm + small_const, grad_norm)
    grad /= grad_norm
    grad *= eps.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    return grad, grad_norm


def renormalize(a, b, small_const=1e-22):
    # changes(removed detach and restored where)
    a_norm = a.view(a.shape[0], -1).norm(p=2, dim=1).view(b.shape[0], 1, 1, 1)
    a_norm_new = torch.where(a_norm < small_const, a_norm + small_const,
                             a_norm)  # torch.clamp(a_norm, min=small_const) #.detach() #torch.where(a_norm < small_const, a_norm + small_const, a_norm)
    a /= a_norm_new
    a *= b.view(a.shape[0], -1).norm(p=2, dim=1).view(a.shape[0], 1, 1, 1)
    return a, a_norm_new


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):

    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError('need to check')
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


def cone_project(grad_temp_1, grad_temp_2, deg):
    """
    grad_temp_1: gradient of the loss w.r.t. the robust/classifier free
    grad_temp_2: gradient of the loss w.r.t. the non-robust
    projecting the robust/CF onto the non-robust
    """
    angles_before = torch.acos(
        (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))
    ##print('angle before', angles_before)
    grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
    grad_temp_1 = grad_temp_1 - ((grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_2.norm(p=2, dim=1) ** 2)).view(
        grad_temp_1.shape[0], -1) * grad_temp_2
    grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
    # cone_projection = grad_temp_1 + grad_temp_2 45 deg
    radians = torch.tensor([deg], device=grad_temp_1.device).deg2rad()
    ##print('angle after', radians, torch.acos((grad_temp_1*grad_temp_2).sum(1) / (grad_temp_1.norm(p=2,dim=1) * grad_temp_2.norm(p=2,dim=1))))
    cone_projection = grad_temp_1 * torch.tan(radians) + grad_temp_2

    # second classifier is a non-robust one -
    # unless we are less than 45 degrees away - don't cone project
    grad_temp = grad_temp_2.clone()
    loop_projecting = time.time()
    grad_temp[angles_before > radians] = cone_projection[angles_before > radians]

    return grad_temp


def normalize(x):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    x = x - torch.tensor(mean).to(x.device)[None, :, None, None]
    x = x / torch.tensor(std).to(x.device)[None, :, None, None]
    return x


def denormalize(x):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    x = x * torch.tensor(std).to(x.device)[None, :, None, None]
    x = x + torch.tensor(mean).to(x.device)[None, :, None, None]
    return x


def _map_img(x):
    """
    from -1 to 1 to 0 to 1
    """
    return 0.5 * (x + 1)


def _unmap_img(x, from_image_net_dist=False):
    """
    from 0 to 1 to -1 to 1
    """

    return 2. * x - 1


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def generate_samples(model, sampler, classes, n_samples_per_class, ddim_steps, scale, init_image=None, t_enc=None,
                     init_latent=None, ccdddim=False, ddim_eta=0.):
    all_samples = []
    all_probs = []
    all_videos = []

    with torch.no_grad():
        with model.ema_scope():
            tic = time.time()
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)})

            for class_label in classes:
                print(
                    f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class * [class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                if init_latent is not None:
                    y = xc.to(model.device)
                    z_enc = sampler.stochastic_encode(init_latent,
                                                      torch.tensor([t_enc] * (n_samples_per_class)).to(model.device))
                    # decode it
                    if ccdddim:
                        out = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc, y=xc.to(model.device))
                        samples = out["x_dec"]
                        prob = out["prob"]
                        vid = out["video"]

                    else:
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc)

                    x_samples = model.decode_first_stage(samples)
                    x_samples_ddim = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    cat_samples = torch.cat([init_image[:1], x_samples_ddim], dim=0)
                else:

                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                     conditioning=c,
                                                     batch_size=n_samples_per_class,
                                                     shape=[3, 64, 64],
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                                 min=0.0, max=1.0)
                    cat_samples = x_samples_ddim

                all_samples.append(cat_samples)
                all_probs.append(prob) if ccdddim and prob is not None else None
                all_videos.append(vid) if ccdddim and vid is not None else None

    out = {}
    out["samples"] = all_samples
    out["probs"] = all_probs if len(all_probs) > 0 else None
    out["videos"] = all_videos if len(all_videos) > 0 else None
    return out
