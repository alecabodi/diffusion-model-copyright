import random 

def read_prompts(size):
    n_samples = 10
    n_data = 100
    filepath = 'raw_data/pool.txt'
        
    # Open the file and read the prompts
    if filepath != '':
        with open(filepath, 'r') as file:
            prompt = file.readlines()
    else:
        raise ValueError('Prompt file not found')

    prompt_list = [p.strip() for p in prompt]
    len_prompts = len(prompt_list)
    
    sampled_indices = random.sample(range(n_data), n_samples)

    for i in sampled_indices:
        prompt_list.append(prompt_list[i % len_prompts])
        
    return prompt_list
                

