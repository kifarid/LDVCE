import numpy as np
import regex as re
import torch
import torchvision
import torchvision.transforms.functional as tf
from torch import distributions as torchd
from torch.nn import functional as F
from tqdm import tqdm
from functools import partial
import time 
import sys

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from sampling_helpers import _map_img, normalize, renormalize, _renormalize_gradient, \
    OneHotDist, compute_lp_dist, cone_project, segment, detect


i2h = dict()
with open('data/imagenet_clsidx_to_label.txt', "r") as f:
    lines = f.read().splitlines()
    assert len(lines) == 1000

    for line in lines:
        key, value = line.split(":")
        i2h[int(key)] = re.sub(r"^'|',?$", "", value.strip())  # value.strip().strip("'").strip(",").strip("\"")

class DinoLoss(torch.nn.Module):
    def __init__(self, dino: torch.nn.Module, loss_identifier: str) -> None:
        super().__init__()
        self.dino = dino
        self.loss_identifier = loss_identifier
        if "cossim" == loss_identifier:
            self.loss = torch.nn.CosineSimilarity()
        elif "1" == loss_identifier:
            self.loss = torch.nn.L1Loss()
        elif "2" == loss_identifier:
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError

    def forward(self, output, target):
        dino_features = normalize(_map_img(tf.center_crop(output, output_size=224)))
        dino_features =  self.dino(output)
        if "cossim" == self.loss_identifier:
            return 1 - self.loss(dino_features, target)
        else:
            return self.loss(dino_features, target)

class CCDDIMSampler(object):
    def __init__(self, model, classifier, model_type="latent", schedule="linear", guidance="free", lp_custom=False,
                 deg_cone_projection=10., denoise_dist_input=True, classifier_lambda=1, dist_lambda=0.15,
                 enforce_same_norms=True, seg_model=None, masked_guidance=False, masked_dist=False,
                 backprop_diffusion=True, mask_alpha = 3., **kwargs):

        super().__init__()
        self.model_type = model_type
        self.lp_custom = lp_custom
        self.images = []
        self.probs = []
        self.classifier_lambda = classifier_lambda
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.classifier = classifier
        self.guidance = guidance
        self.backprop_diffusion = backprop_diffusion
        # self.projected_counterfactuals = projected_counterfactuals
        self.deg_cone_projection = deg_cone_projection
        self.denoise_dist_input = denoise_dist_input
        self.dist_lambda = dist_lambda
        self.enforce_same_norms = enforce_same_norms
        self.seg_model = seg_model
        self.masked_dist = masked_dist
        self.masked_guidance = masked_guidance
        self.mask_alpha = mask_alpha

        self.init_images = None
        self.init_labels = None

    def get_classifier_dist(self, x, t=None):
        """
        Create a distribution over the classifier output space
        Args:
            x: input image for which to create the distribution over the classifier output space range [-1, 1]

        Returns:
            dist: torch distribution over the classifier output space

        """
        x = tf.center_crop(x, 224)
        x = normalize(_map_img(x))
        logit = self.classifier(x)  # (TODO) add option for t here

        dist = torchd.independent.Independent(OneHotDist(logit, validate_args = False), 1)
        return dist

    def get_mask(self):
        """
        this function returns a negative mask given by a segmentation model for the region of interest
        values are higher outside the region of interest
        """
        prompts = []
        for l in self.init_labels:
            prompts.append(re.sub(r'\b(\w)', lambda m: m.group(1).upper(), i2h[l]))

        with torch.no_grad():
            img_to_seg = F.interpolate(normalize(self.init_images), size=(352, 352), mode='bilinear',
                                       align_corners=False).to(self.init_images.device)
            preds = self.seg_model(img_to_seg, prompts)[0]
            preds = F.interpolate(preds, size=self.init_images.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.sigmoid(preds)  # torch.softmax(preds.view(preds.shape[0], -1), dim=1).view(*preds.shape)
            # penalty = 1-preds
            #preds = (preds - preds.min()) / (preds.max() - preds.min())
            preds = (preds - preds.min()) / (preds.max() - preds.min())
            preds = torch.sigmoid(self.mask_alpha*2*(preds-0.5))

        return preds.to(self.init_images.device)

    def get_output(self, x, t, c, index, unconditional_conditioning, use_original_steps=True, quantize_denoised=True,
                   return_decoded=False):
        b, device = x.shape[0], x.device
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([unconditional_conditioning, c])
        with torch.enable_grad() if self.backprop_diffusion else torch.no_grad():
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

        if return_decoded:
            # getting the original denoised image
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            # current prediction for x_0
            # get the original image with range [0, 1] if it is in latent space
            pred_latent_x0 = (x - sqrt_one_minus_at * e_t_uncond) / a_t.sqrt()  # e_t - >  e_t_uncond
            if quantize_denoised:
                pred_latent_x0, _, *_ = self.model.first_stage_model.quantize(pred_latent_x0)

            pred_x0 = self.model.differentiable_decode_first_stage(
                pred_latent_x0)  # if self.model_type == "latent" else pred_latent_x0
            # pred_x0 = torch.clamp((pred_x0 + 1.0) / 2.0, min=0.0, max=1.0)
            return e_t_uncond, e_t, pred_x0

        else:
            return e_t_uncond, e_t

    def conditional_score(self, x, t, c, index, use_original_steps, quantize_denoised, unconditional_guidance_scale=1.,
                          unconditional_conditioning=None, y=None):
        """

        Args:
            x: input image
            t: time step
            c: conditioning
            index: index for the schedule
            use_original_steps: whether to use the original steps
            quantize_denoised: whether to quantize the denoised image
            unconditional_guidance_scale: scale for the unconditional guidance
            unconditional_conditioning: unconditional conditioning
            y: target class


        Returns:
            e_t: score after conditioning

        """
        b, *_, device = *x.shape, x.device
        x = x.detach()  # .requires_grad_()
        # x.requires_grad = True
        prob_best_class = None
        mask_guidance = None
        mask_dist = None
        ## check if gradient tracking is on for x

        if self.masked_guidance or self.masked_dist:
            mask = self.get_mask()

            if self.masked_guidance:
                mask_guidance = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

            if self.masked_dist:
                mask_dist = F.interpolate(1 - mask, size=x.shape[-2:], mode='bilinear', align_corners=False)  # 1 - mask

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            return e_t

        # print("check gradient tracking onf e ", e_t.requires_grad)
        if self.guidance == "free":
            e_t_uncond, e_t, pred_x0 = self.get_output(x, t, c, index, unconditional_conditioning, use_original_steps,
                                                       quantize_denoised, return_decoded=True)

            if self.masked_guidance:
                e_t = e_t_uncond + unconditional_guidance_scale * \
                      renormalize(mask_guidance * (e_t - e_t_uncond), e_t - e_t_uncond)[
                          0]  # mask_guidance * (e_t - e_t_uncond)
            else:
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            return e_t

        # print("check gradient tracking onf e ", e_t.requires_grad)
        score_out = torch.zeros_like(x)

        with torch.enable_grad():
            assert x.requires_grad == False, "x requires grad"

            if self.classifier_lambda != 0:
                x_grad = x.detach().requires_grad_()
                e_t_uncond, e_t, pred_x0 = self.get_output(x_grad, t, c, index, unconditional_conditioning,
                                                           use_original_steps, quantize_denoised=quantize_denoised,
                                                           return_decoded=True)
                classifier_dist = self.get_classifier_dist(pred_x0)
                y_one_hot = F.one_hot(y, classifier_dist.event_shape[-1]).to(device)
                log_probs = classifier_dist.log_prob(y_one_hot)
                prob_best_class = torch.exp(log_probs)
                # print("prob_best_class", prob_best_class, log_probs)

                grad_classifier = torch.autograd.grad(log_probs, x_grad)[
                    0]
        with torch.enable_grad():
            if self.lp_custom:
                assert x.requires_grad == False, "x requires grad"
                x_grad = x.detach().requires_grad_()
                e_t_uncond, e_t, pred_x0 = self.get_output(x_grad, t, c, index, unconditional_conditioning,
                                                           use_original_steps, quantize_denoised=quantize_denoised,
                                                           return_decoded=True)
                # print("computing the lp gradient")
                assert pred_x0.requires_grad == True, "pred_x0 does not require grad"
                pred_x0_0to1 = torch.clamp(_map_img(pred_x0), min=0.0, max=1.0)

                diff = pred_x0_0to1 - self.init_images.to(x.device)
                # print("diff is:", diff.mean())
                lp_dist = compute_lp_dist(diff, self.lp_custom)
                lp_grad = torch.autograd.grad(lp_dist.mean(), x_grad)[0]

        # assert e_t_uncond.requires_grad == True and e_t.requires_grad == True, "e_t_uncond and e_t should require gradients"

        if self.guidance == "projected":
            implicit_classifier_score = (e_t - e_t_uncond)  # .detach()
            # check gradient tracking on implicit_classifier_score
            assert implicit_classifier_score.requires_grad == False, "implicit_classifier_score requires grad"

        if self.lp_custom or self.classifier_lambda != 0:
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)

        if self.classifier_lambda != 0:

            classifier_score = -1 * grad_classifier * (1 - a_t).sqrt()
            assert classifier_score.requires_grad == False, "classifier_score requires grad"
            classifier_score = renormalize(classifier_score * mask_guidance, classifier_score)[
                0] if self.masked_guidance else classifier_score
            implicit_classifier_score = \
                renormalize(implicit_classifier_score * mask_guidance, implicit_classifier_score)[
                    0] if self.masked_guidance else implicit_classifier_score
            # project the gradient of the classifier on the implicit classifier
            classifier_score = cone_project(implicit_classifier_score.view(x.shape[0], -1),
                                            classifier_score.view(x.shape[0], -1),
                                            self.deg_cone_projection).view_as(classifier_score) \
                if self.guidance == "projected" else classifier_score

            if self.enforce_same_norms:
                score_, norm_ = _renormalize_gradient(classifier_score,
                                                      implicit_classifier_score)  # e_t_uncond (AWAREE!!)
                classifier_score = self.classifier_lambda * score_

            else:
                classifier_score *= self.classifier_lambda

            score_out += classifier_score

        # distance gradients
        if self.lp_custom:

            lp_score = -1 * lp_grad * (1 - a_t).sqrt()
            lp_score = mask_dist * lp_score if self.masked_dist else lp_score

            if self.enforce_same_norms:

                score_, norm_ = _renormalize_gradient(lp_score,
                                                      implicit_classifier_score)
                lp_score = self.dist_lambda * score_

            else:

                lp_score *= self.dist_lambda

            score_out -= lp_score

        e_t = e_t_uncond + unconditional_guidance_scale * score_out  # (1 - a_t).sqrt() * grad_out

        # adding images to create a gif
        pred_x0_copy = pred_x0.clone().detach()
        img = torch.clamp(_map_img(pred_x0_copy), min=0.0, max=1.0)
        img = torch.permute(img, (1, 2, 0, 3)).reshape((img.shape[1], img.shape[2], -1))

        self.images.append(img)

        if prob_best_class is not None:
            self.probs.append(prob_best_class)

        return e_t

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            #pass
            # TODO: this is a hack to make it work on CPU
            if attr.device != torch.device("cuda"):
               attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, ):

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, y=None):
        b, *_, device = *x.shape, x.device

        e_t = self.conditional_score(x=x, c=c, t=t, index=index, use_original_steps=use_original_steps,
                                     quantize_denoised=quantize_denoised,
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=unconditional_conditioning, y=y)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas).to(x0.device)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas.to(x0.device)

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, y=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning, y=y)
        out = {}
        out['x_dec'] = x_dec
        out['video'] = torch.stack(self.images, dim=0) if len(self.images) != 0 else None
        # print(f"Video shape: {out['video'].shape}")
        out['prob'] = self.probs[-1].item() if len(self.probs) != 0 else None
        self.images = []
        self.probs = []
        return out


class CCMDDIMSampler(object):
    def __init__(self, model, classifier, model_type="latent", schedule="linear", guidance="free", lp_custom=False,
                 deg_cone_projection=10., denoise_dist_input=True, classifier_lambda=1, dist_lambda=0.15,
                 enforce_same_norms=True, seg_model=None, detect_model=None, masked_guidance=False,
                 backprop_diffusion=True, log_backprop_gradients: bool = False, mask_alpha = 5., **kwargs):

        super().__init__()
        self.model_type = model_type
        self.lp_custom = lp_custom
        self.images = []
        self.probs = []
        self.classifier_lambda = classifier_lambda
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.classifier = classifier
        self.guidance = guidance
        self.backprop_diffusion = backprop_diffusion
        self.log_backprop_gradients = log_backprop_gradients
        # self.projected_counterfactuals = projected_counterfactuals
        self.deg_cone_projection = deg_cone_projection
        self.denoise_dist_input = denoise_dist_input
        self.dist_lambda = dist_lambda
        self.enforce_same_norms = enforce_same_norms
        self.seg_model = seg_model
        self.masked_guidance = masked_guidance
        self.mask_alpha = mask_alpha

        self.init_images = None
        self.init_labels = None            
        self.mask = None
        
        self.detect_model = detect_model
        self.classification_criterion = torch.nn.CrossEntropyLoss()
        
        self.dino_pipeline = False
        if isinstance(self.lp_custom, str) and "dino_" in self.lp_custom:
            # dinov2 requires PyTorch 2.0!
            # dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            # dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            # dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            # dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            self.distance_criterion = DinoLoss(dino=torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').eval(), loss_identifier=self.lp_custom.split("_")[-1])
            self.dino_init_features = None
            self.dino_pipeline = True
        elif isinstance(self.lp_custom, int):
            if self.lp_custom == 1:
                self.distance_criterion = torch.nn.L1Loss()
            elif self.lp_custom == 2:
                self.distance_criterion = torch.nn.MSELoss()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def get_classifier_dist(self, x, t=None):
        """
        Create a distribution over the classifier output space
        Args:
            x: input image for which to create the distribution over the classifier output space range [-1, 1]

        Returns:
            dist: torch distribution over the classifier output space

        """
        x = tf.center_crop(x, 224)
        x = normalize(_map_img(x))
        logit = self.classifier(x)  # (TODO) add option for t here
        dist = torchd.independent.Independent(OneHotDist(logit, validate_args = False), 1)
        return dist

    def get_classifier_logits(self, x, t=None):
        """
        Returns classifier logits
        Args:
            x: input image for which to create the prediction

        Returns:
            logits: logits of output layer of target model

        """
        x = tf.center_crop(x, 224)
        x = normalize(_map_img(x))
        return self.classifier(x)

    def get_dino_features(self, x, device):
        x = normalize(_map_img(tf.center_crop(x, output_size=224)))
        return self.distance_criterion.dino(x.to(device))

    def get_mask_clip_seg(self):
        """
        this function returns a negative mask given by a segmentation model for the region of interest
        values are higher outside the region of interest
        """
        if self.mask is not None:
            return self.mask

        prompts = []

        for l in self.init_labels:
            prompts.append(re.sub(r'\b(\w)', lambda m: m.group(1).upper(), i2h[l]))

        with torch.no_grad():
            img_to_seg = F.interpolate(normalize(self.init_images), size=(352, 352), mode='bilinear',
                                       align_corners=False).to(self.init_images.device)
            preds = self.seg_model(img_to_seg, prompts)[0]
            preds = F.interpolate(preds, size=self.init_images.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.sigmoid(preds)  # torch.softmax(preds.view(preds.shape[0], -1), dim=1).view(*preds.shape)
            # penalty = 1-preds
            preds = (preds - preds.min()) / (preds.max() - preds.min())
            preds = torch.sigmoid(self.mask_alpha*2*(preds-0.5))
        self.mask = preds.to(self.init_images.device)
        return self.mask

    def get_mask(self):
        """
        this function returns a negative mask given by a segmentation model for the region of interest
        values are higher outside the region of interest
        """

        if self.mask is not None:
            return self.mask

        with torch.no_grad():
            print("input range", self.init_images.min(), self.init_images.max())
            image_int8 = (self.init_images[0].permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)
            # detected_boxes = detect(image, text_prompt=i2h[label], model=groundingdino_model, image_source=image_image)
            detected_boxes = detect(normalize(self.init_images[0]).squeeze(),
                                    text_prompt=i2h[self.init_labels[0]].split(',')[0],
                                    model=self.detect_model)  # , image_source=image_int8)
            segmented_frame_masks = segment(image_int8, self.seg_model, boxes=detected_boxes)
            preds = torch.any(segmented_frame_masks, dim=0)
            preds = preds.unsqueeze(0).repeat(self.init_images.shape[0], *(1,) * len(preds.shape))
            # print("preds range after first seg ", preds.min(), preds.max())
        self.mask = preds.to(self.init_images.device)

        return self.mask

    def get_output(self, x, t, c, index, unconditional_conditioning, use_original_steps=True, quantize_denoised=True,
                   return_decoded=False, return_pred_latent_x0=False):
        b, device = x.shape[0], x.device
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([unconditional_conditioning, c])
        with torch.enable_grad() if self.backprop_diffusion else torch.no_grad():
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

        if return_decoded:
            # getting the original denoised image
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            # current prediction for x_0
            # get the original image with range [0, 1] if it is in latent space
            pred_latent_x0 = (x - sqrt_one_minus_at * e_t_uncond) / a_t.sqrt()  # e_t - >  e_t_uncond
            if quantize_denoised:
                pred_latent_x0, _, *_ = self.model.first_stage_model.quantize(pred_latent_x0)

            pred_x0 = self.model.differentiable_decode_first_stage(
                pred_latent_x0)  # if self.model_type == "latent" else pred_latent_x0
            # pred_x0 = torch.clamp((pred_x0 + 1.0) / 2.0, min=0.0, max=1.0)
            
            if return_pred_latent_x0:
                return e_t_uncond, e_t, pred_x0, pred_latent_x0
            else:
                return e_t_uncond, e_t, pred_x0
        else:
            return e_t_uncond, e_t

    def conditional_score(self, x, t, c, index, use_original_steps, quantize_denoised, unconditional_guidance_scale=1.,
                          unconditional_conditioning=None, y=None):
        """

        Args:
            x: input image
            t: time step
            c: conditioning
            index: index for the schedule
            use_original_steps: whether to use the original steps
            quantize_denoised: whether to quantize the denoised image
            unconditional_guidance_scale: scale for the unconditional guidance
            unconditional_conditioning: unconditional conditioning
            y: target class


        Returns:
            e_t: score after conditioning

        """
        b, *_, device = *x.shape, x.device
        x = x.detach()  # .requires_grad_()
        # x.requires_grad = True
        prob_best_class = None
        mask_guidance = None

        ## check if gradient tracking is on for x
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            return e_t

        # print("check gradient tracking onf e ", e_t.requires_grad)
        if self.guidance == "free":
            e_t_uncond, e_t, pred_x0 = self.get_output(x, t, c, index, unconditional_conditioning, use_original_steps,
                                                       quantize_denoised, return_decoded=True)

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            return e_t

        # print("check gradient tracking onf e ", e_t.requires_grad)
        score_out = torch.zeros_like(x)

        with torch.enable_grad():
            x_noise = x.detach().requires_grad_()
            ret_vals = self.get_output(x_noise, t, c, index, unconditional_conditioning,
                                                        use_original_steps, quantize_denoised=quantize_denoised,
                                                        return_decoded=True, return_pred_latent_x0=self.log_backprop_gradients)
            if self.log_backprop_gradients:
                e_t_uncond, e_t, pred_x0, pred_latent_x0 = ret_vals
            else:
                e_t_uncond, e_t, pred_x0 = ret_vals

        with torch.no_grad():
            if self.classifier_lambda != 0:
                with torch.enable_grad():
                    pred_logits = self.get_classifier_logits(pred_x0)
                    log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
                    log_probs = log_probs[:, y.view(-1)]
                    prob_best_class = torch.exp(log_probs).mean().detach()

                    if self.log_backprop_gradients: pred_latent_x0.retain_grad()

                    if self.dino_pipeline:
                        grad_classifier = torch.autograd.grad(log_probs.mean(), x_noise, retain_graph=False)[0]
                    else:
                        grad_classifier = torch.autograd.grad(log_probs.mean(), x_noise, retain_graph=True)[0]

                    if self.log_backprop_gradients:
                        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
                        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
                        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
                        a_t_sqrt = a_t.sqrt()
                        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
                        grad_pred_latent_x0 = pred_latent_x0.grad.data
                        grad_unet_wrt_zt = (grad_classifier*a_t_sqrt/grad_pred_latent_x0 - 1)*(-1/sqrt_one_minus_at)

                        cossim = torch.nn.CosineSimilarity()
                        cossim_wpre = cossim(grad_classifier.view(2, -1), grad_pred_latent_x0.view(2, -1))
                        
                        print(torch.norm(grad_classifier, dim=(2,3)), torch.norm(grad_pred_latent_x0, dim=(2,3)), torch.norm(grad_unet_wrt_zt, dim=(2,3)))
                        print(cossim_wpre)

                    if isinstance(self.lp_custom, str) and "dino_" in self.lp_custom: # retain_graph causes cuda oom issues for dino distance regularizer...
                        x_noise = x.detach().requires_grad_()
                        ret_vals = self.get_output(x_noise, t, c, index, unconditional_conditioning,
                                                                    use_original_steps, quantize_denoised=quantize_denoised,
                                                                    return_decoded=True, return_pred_latent_x0=self.log_backprop_gradients)
                        if self.log_backprop_gradients:
                            e_t_uncond, e_t, pred_x0, pred_latent_x0 = ret_vals
                        else:
                            e_t_uncond, e_t, pred_x0 = ret_vals
                        pred_x0_0to1 = torch.clamp(_map_img(pred_x0), min=0.0, max=1.0)
                        lp_dist = self.distance_criterion(pred_x0_0to1, self.dino_init_features.to(x.device).detach())
                    else:
                        pred_x0_0to1 = torch.clamp(_map_img(pred_x0), min=0.0, max=1.0)
                        lp_dist = self.distance_criterion(pred_x0_0to1, self.init_images.to(x.device))
                
                    lp_grad = torch.autograd.grad(lp_dist.mean(), x_noise, retain_graph=False)[0]

        # assert e_t_uncond.requires_grad == True and e_t.requires_grad == True, "e_t_uncond and e_t should require gradients"

        # if self.guidance == "projected":
        implicit_classifier_score = (e_t - e_t_uncond)  # .detach()
        # check gradient tracking on implicit_classifier_score
        assert implicit_classifier_score.requires_grad == False, "implicit_classifier_score requires grad"

        if self.lp_custom or self.classifier_lambda != 0:
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)

        if self.classifier_lambda != 0:
            classifier_score = -1 * grad_classifier * (1 - a_t).sqrt()
            assert classifier_score.requires_grad == False, "classifier_score requires grad"
            # project the gradient of the classifier on the implicit classifier
            classifier_score = cone_project(implicit_classifier_score.view(x.shape[0], -1),
                                            classifier_score.view(x.shape[0], -1),
                                            self.deg_cone_projection).view_as(classifier_score) \
                if self.guidance == "projected" else classifier_score

            if self.enforce_same_norms:
                score_, norm_ = _renormalize_gradient(classifier_score,
                                                      implicit_classifier_score)  # e_t_uncond (AWAREE!!)
                classifier_score = self.classifier_lambda * score_

            else:
                classifier_score *= self.classifier_lambda

            score_out += classifier_score

        # distance gradients
        if self.lp_custom:

            lp_score = -1 * lp_grad * (1 - a_t).sqrt()

            if self.enforce_same_norms:
                score_, norm_ = _renormalize_gradient(lp_score,
                                                      implicit_classifier_score)
                lp_score = self.dist_lambda * score_

            else:

                lp_score *= self.dist_lambda

            score_out -= lp_score

        e_t = e_t_uncond + unconditional_guidance_scale * score_out  # (1 - a_t).sqrt() * grad_out

        # adding images to create a gif
        pred_x0_copy = pred_x0.clone().detach()
        img = torch.clamp(_map_img(pred_x0_copy), min=0.0, max=1.0)
        img = torch.permute(img, (1, 2, 0, 3)).reshape((img.shape[1], img.shape[2], -1))

        self.images.append(img.cpu())

        if prob_best_class is not None:
            self.probs.append(prob_best_class.cpu())

        return e_t

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            #pass
            # TODO: this is a hack to make it work on CPU
            if attr.device != torch.device("cuda"):
               attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, ):

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, y=None):
        b, *_, device = *x.shape, x.device

        e_t = self.conditional_score(x=x, c=c, t=t, index=index, use_original_steps=use_original_steps,
                                     quantize_denoised=quantize_denoised,
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=unconditional_conditioning, y=y)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas).to(x0.device)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas.to(x0.device)

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, y=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, latent_t_0=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        if self.masked_guidance:
            print("### Getting the mask ###")
            mask = self.get_mask()
            mask = F.interpolate(mask.to(torch.uint8), size=x_latent.shape[-2:])
            # mask = self.get_mask()
            # mask = F.interpolate(mask, size=x_latent.shape[-2:], mode='bilinear', align_corners=True)
            # mask = (mask - mask.min()) / (mask.max() - mask.min())
            # mask[mask < 0.5] = 0.
            # mask[mask >= 0.5] = 1.

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)

        # if latent_t_0:
        #     x_orig = x_latent
        #     x_dec = self.stochastic_encode(x_latent.clone(),
        #                                    torch.tensor([t_start] * (x_latent.shape[0])).to(x_latent.device))
        # else:
        x_dec = x_latent if not latent_t_0 else self.stochastic_encode(x_latent.clone(), torch.tensor([t_start] * (x_latent.shape[0])).to(x_latent.device))
        for i, step in enumerate(iterator):
            tic = time.time()
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)

            if self.masked_guidance and latent_t_0:
                #print("blending with original image")
                img_orig = self.model.q_sample(x_latent.clone(), ts)
                x_dec = img_orig * (1. - mask) + (mask) * x_dec

            
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning, y=y)
            #workaround for long running time
            elapsed_time = time.time() - tic
            if elapsed_time > 6:
                print(f"Iteration time {elapsed_time} exceeded limit 6 secs, terminating program...")
                print("x_dec device: ", x_dec.device)
                sys.exit(1)  # Terminate the program with exit code 1 (indicating an error)                

        out = {}
        out['x_dec'] = x_dec
        out['video'] = torch.stack(self.images, dim=0) if len(self.images) != 0 else None
        out["mask"] = self.mask.to(torch.float32) if self.mask is not None else None
        # print(f"Video shape: {out['video'].shape}")
        out['prob'] = self.probs[-1].item() if len(self.probs) != 0 else None
        self.images = []
        self.probs = []
        self.mask = None
        return out
