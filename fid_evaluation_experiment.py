import fid_metric
import os
import json
import uuid

def run_fid_evaluation(pipe, file):
    
    generated_dir = "" # specify directory where generated images will be saved

    with open(file) as f:
        all_prompts_dict = json.load(f)
        total_imgs = list(all_prompts_dict.keys())
        prompts = [all_prompts_dict[k][0] for k in total_imgs]
            
    for i, prompt in enumerate(prompts):
        image = pipe(prompt, num_inference_steps=8, guidance_scale=0).images[0]
        file_name = f"./generated_images/idx{i}_{uuid.uuid4().hex}.png"
        image.save(file_name)
    
    train_dir = "" # specify directory where training images stats are stored

    fid_score = fid_metric.calculate_fid_given_paths(paths=[train_dir, generated_dir], batch_size=100, device="cuda", dims=2048, num_workers=0)
    print(fid_score)