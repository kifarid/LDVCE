from diffusers import StableDiffusionPipeline
import torch
import os

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
lora_model_path = '/misc/lmbraid21/faridk/lora_finetune_simon_2'
pipe.unet.load_attn_procs(lora_model_path)
pipe.to("cuda")
#fusing lora
pipe.fuse_lora(lora_scale=0.4)
print("fused lora")
# use half the weights from the LoRA finetuned model and half the weights from the base model
for i in range(5):
            #scale range between 0 and 1
            
            #f"A photograph of a celebrity {'Age:18-30 years' if attr == 'young' else 'Age:old'} , face, high resolution"
            probably_bad_prompt = f"A photograph of a celebrity, Face, Face attributes: no smile, resting face, High resolution"
            image = pipe(probably_bad_prompt).images[0]
            image.save(f"lora_examples/fused_{probably_bad_prompt}_{i}.png")

for i in range(5):
            #scale range between 0 and 1
            
            #f"A photograph of a celebrity {'Age:18-30 years' if attr == 'young' else 'Age:old'} , face, high resolution"
            probably_bad_prompt = f"A photograph of a celebrity, Face, Face attributes: smiling, High resolution"
            image = pipe(probably_bad_prompt).images[0]
            image.save(f"lora_examples/fused_{probably_bad_prompt}_{i}.png")
  
pipe.save_pretrained("lora_finetune_simon_2_fused")  
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

scale = 0.0
os.makedirs(f"lora_examples", exist_ok=True)



# for scale in [0.0, 0.25, 0.5, 0.75, 1]:
#         for i in range(5):
#             #scale range between 0 and 1
            
#             #f"A photograph of a celebrity {'Age:18-30 years' if attr == 'young' else 'Age:old'} , face, high resolution"
#             probably_bad_prompt = f"A photograph of a celebrity, Face, Face attributes: no smile, resting face, High resolution"
#             image = pipe(probably_bad_prompt, cross_attention_kwargs={"scale": scale}).images[0]
#             image.save(f"lora_examples/{probably_bad_prompt}_{scale}_{i}.png")
    
