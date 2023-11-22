from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

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

# for i in range(10):
#     probably_bad_prompt = "A photograph of a celebrity, Face, High resolution, Face attributes: smiling"
#     image = pipe(probably_bad_prompt).images[0]
#     image.save(f"{probably_bad_prompt}_{i}.png")

# for i in range(10):
#     probably_bad_prompt = "A photograph of a celebrity, Face, High resolution, Face attributes: no smile, resting face"
#     image = pipe(probably_bad_prompt).images[0]
#     image.save(f"{probably_bad_prompt}_{i}.png")

for i in range(10):
    probably_bad_prompt = "A photograph of a celebrity, Face Attributes:smiling, Face, High resolution"
    image = pipe(probably_bad_prompt).images[0]
    image.save(f"{probably_bad_prompt}_{i}.png")

