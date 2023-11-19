import json
import os
import numpy as np

seed = 0
for axiom in ["composition", "reversibility"]:
    base_path = f"/misc/lmbraid21/faridk/axiomatic/{axiom}/{seed}/bucket_0_50"
    json_files = [os.path.join(base_path, file) for file in os.listdir(base_path) if ".json" in file]
    assert len(json_files) == 100
    pixel = {k:[] for k in range(10)}
    latents = {k:[] for k in range(10)}
    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
        for k, d in enumerate(data):
            pixel[k].append(d["pixel"])
            latents[k].append(d["latent"])
    print("#"*5, axiom,"#"*5)
    for k in range(10):
        print(k, "pixel", np.mean(pixel[k]))
        print(k, "latent", np.mean(latents[k]))