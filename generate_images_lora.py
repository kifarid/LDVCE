from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
lora_model_path = '/misc/lmbraid21/faridk/lora_finetune_simon'
pipe.unet.load_attn_procs(lora_model_path)
pipe.to("cuda")
# use half the weights from the LoRA finetuned model and half the weights from the base model

# image = pipe(
#     "A pokemon with blue eyes.", num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.}
# ).images[0]

# for i in range(10):
#     prompt = "a celebrity, resting face"
#     image = pipe(prompt).images[0]  
#     image.save(f"{prompt}_{i}.png")


# for i in range(10):
#     prompt = "a non-smiling celebrity"
#     image = pipe(prompt).images[0]  
#     image.save(f"{prompt}_{i}.png")



# #generate 10 images
# for i in range(10):
#     probably_bad_prompt = "a celebrity, resting face, high-quality"
#     image = pipe(probably_bad_prompt).images[0]
#     image.save(f"{probably_bad_prompt}_{i}.png")

# #generate 10 images
# for i in range(10):
#     probably_bad_prompt = "a celebrity, smiling, high-quality"
#     image = pipe(probably_bad_prompt).images[0]
#     image.save(f"{probably_bad_prompt}_{i}.png")

#     #generate 10 images
# for i in range(10):
#     probably_bad_prompt = "a celebrity, smiling"
#     image = pipe(probably_bad_prompt).images[0]
#     image.save(f"{probably_bad_prompt}_{i}.png")


# for i in range(10):
#     probably_bad_prompt = "a person, smiling"
#     image = pipe(probably_bad_prompt).images[0]
#     image.save(f"{probably_bad_prompt}_{i}.png")

# for i in range(5):
#     probably_bad_prompt = "A photo of a young, smiling celebrity with bags under eyes, high-quality, 1024x1024"
#     image = pipe(probably_bad_prompt, guidance_scale=4, cross_attention_kwargs={"scale": 0.}).images[0]
#     image.save(f"{probably_bad_prompt}_{i}_{0.0}.png")


for i in range(5):
    probably_bad_prompt = "A photo of a young, celebrity, resting face, high-quality, 1024x1024"
    image = pipe(probably_bad_prompt, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.0}).images[0]
    image.save(f"{probably_bad_prompt}_{i}.png")
