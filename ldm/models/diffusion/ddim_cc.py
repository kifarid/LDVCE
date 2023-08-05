"""SAMPLING ONLY."""

import torch
import time
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch import distributions as torchd
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


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
    print(" ratio of dimensions that are cone projected: ", (angles_before > radians).float().mean())
    grad_temp = grad_temp_2.clone()
    loop_projecting = time.time()
    grad_temp[angles_before > radians] = cone_projection[angles_before > radians]

    return grad_temp


def _map_img(x):
    return 0.5 * (x + 1)


class CCDDIMSampler(object):
    def __init__(self, model, classifier, model_type="latent", schedule="linear", guidance="free", lp_custom=False,
                 deg_cone_projection=30., denoise_dist_input=False, classifier_lambda=0.1, dist_lambda=0.15,
                 enforce_same_norms=True, **kwargs):

        super().__init__()
        self.model_type = model_type
        self.lp_custom = lp_custom
        self.classifier_lambda = classifier_lambda
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.classifier = classifier
        self.guidance = guidance
        # self.projected_counterfactuals = projected_counterfactuals
        self.deg_cone_projection = deg_cone_projection
        self.denoise_dist_input = denoise_dist_input
        self.dist_lambda = dist_lambda
        self.enforce_same_norms = enforce_same_norms

        self.init_images = None

    def get_classifier_dist(self, x, t=None):
        """
        Create a distribution over the classifier output space
        Args:
            x: input image for which to create the distribution over the classifier output space

        Returns:
            dist: torch distribution over the classifier output space

        """
        # convert x to range [0,1]
        x = _map_img(x)
        # resize x to the classifier input size
        x = F.interpolate(x, size=self.classifier.input_size, mode='bilinear', align_corners=False)
        logit = self.classifier(x) #(TODO) add option for t here
        dist = torchd.independent.Independent(OneHotDist(logit), 1)
        return dist

    def conditional_score(self, x, t, c, index, use_original_steps, quantize_denoised, unconditional_guidance_scale=1.,
                          unconditional_conditioning=None):
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


        Returns:
            e_t: score after conditioning

        """
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:

            e_t = self.model.apply_model(x, t, c)
            return e_t

        else:

            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

        if self.guidance == "free":
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            return e_t

        if self.guidance == "noisy_classifier":
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            classifier_dist = self.get_classifier_dist(x)
            log_probs = classifier_dist.log_prob(c)
            grad_classifier = torch.autograd.grad(log_probs.sum(), x, retain_graph=True)[
                0]  # TODO using sum instead of mean )

            e_t = e_t - (1 - a_t).sqrt() * grad_classifier * self.classifier_lambda
            return e_t

        grad_out = torch.zeros_like(x)
        with torch.no_grad():
            # getting the original denoised image
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            # current prediction for x_0
            pred_latent_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            if quantize_denoised:
                pred_latent_x0, _, *_ = self.model.first_stage_model.quantize(pred_latent_x0)

            # get the original image with range [0, 1] if it is in latent space
            pred_x0 = self.model.decode_first_stage(pred_latent_x0) if self.model_type == "latent" else pred_latent_x0
            pred_x0 = torch.clamp(_map_img(pred_x0), 0, 1)

            if self.guidance == "projected":
                grad_implicit_classifier = unconditional_guidance_scale * (e_t - e_t_uncond)
                # compute classifier gradient
                keep_denoising_graph = self.denoise_dist_input

            if self.classifier_lambda != 0:

                with torch.enable_grad():

                    # get the gradient of the log probability of the classifier
                    classifier_dist = self.get_classifier_dist(pred_x0)
                    log_probs = classifier_dist.log_prob(c)
                    grad_classifier = torch.autograd.grad(log_probs.sum(), pred_x0, retain_graph=True)[0] #TODO using sum instead of mean )

                    print(f" cone projection: {self.guidance == 'projected'}, angle is {self.deg_cone_projection}}")
                    # project the gradient of the classifier on the implicit classifier
                    grad_class = cone_project(grad_implicit_classifier.view(pred_x0.shape[0], -1),
                                              grad_classifier.view(pred_x0.shape[0], -1),
                                              self.deg_cone_projection).view_as(grad_classifier) \
                        if self.guidance == "projected" else grad_classifier

                if self.enforce_same_norms:
                    grad_, norm_ = _renormalize_gradient(grad_class, e_t)

                    grad_class = self.classifier_lambda * grad_

                else:
                    grad_class *= self.classifier_lambda

                grad_out += grad_class

            # distance gradients
            if self.lp_custom:  # and self.args.range_t < self.tensorboard_counter:
                if not keep_denoising_graph:
                    diff = pred_x0 - self.init_images
                    lp_grad = compute_lp_gradient(diff, self.lp_custom)
                else:
                    with torch.enable_grad():
                        diff = pred_x0 - self.init_images
                        lp_dist = compute_lp_dist(diff, self.lp_custom)
                        lp_grad = torch.autograd.grad(lp_dist.mean(), x)[0]

                if self.enforce_same_norms:

                    grad_, norm_ = _renormalize_gradient(lp_grad, e_t)

                    lp_grad = self.dist_lambda * grad_

                else:
                    lp_grad *= self.dist_lambda

                grad_out -= lp_grad

            e_t = e_t - (1 - a_t).sqrt() * grad_out

            return e_t

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
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
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        e_t = self.conditional_score(x, c, t, index, use_original_steps, quantize_denoised,
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=unconditional_conditioning)
        # if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        #     e_t = self.model.apply_model(x, t, c)
        # else:
        #     x_in = torch.cat([x] * 2)
        #     t_in = torch.cat([t] * 2)
        #     c_in = torch.cat([unconditional_conditioning, c])
        #     e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        #     e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

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
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
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
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
