"""The code builds on implementation at https://github.com/somepago/DCR"""

import math
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import uuid
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import seaborn as sns

import dino_models
from einops import rearrange

from PIL import PngImagePlugin

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

class ImagePathDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, i
    
    @staticmethod
    def process_and_save_image(img_path, transform=None):

        Path('').mkdir(parents=True, exist_ok=True) # Adjust the path as needed

        img = Image.open(img_path)
        
        if transform is not None:
            img = transform(img)
        
        img = transforms.ToPILImage()(img)
        
        path = '' # Adjust the path as needed
        img.save(f"{path}{uuid.uuid4().hex}.png")


def run_semantic_similarity(query_dir, values_dir, similarity_metric='splitloss', scale='large'):

    dinomapping = {
        'vit_base' : 'dino_vitb16',
        'vit_base8' : 'dino_vitb8',
        'vit_small' : 'dino_vits16',
        'resnet50': 'dino_resnet50',
        'vit_base_cifar10' : 'dino_vitb_cifar10'
    }
        
    model = dino_models.__dict__[dinomapping['vit_base']](pretrained=True, global_pool = '')

    for name, module in model.named_modules():
        print(f"Module Name: {name}, Module Class: {module.__class__.__name__}")

    if scale == 'large':
        large_scale = True
        small_scale = False
    else:
        large_scale = False
        small_scale = True
    
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print('using CPU, this will be slow')
    
    cudnn.benchmark = True

    # Data loading code
    query_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]), # pixel values are in range [-1, 1]
                    ])
    
    value_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]), # pixel values are in range [-1, 1]
                    ])

    query_files = list(glob.glob(query_dir + '/**/*.JPEG', recursive=True)) + list(glob.glob(query_dir + '/**/*.png', recursive=True)) + list(glob.glob(query_dir + '/**/*.jpg', recursive=True))
    query_files = sorted([pathlib.Path(f) for f in query_files ])

    values_files = list(glob.glob(values_dir + '/**/*.JPEG', recursive=True)) + list(glob.glob(values_dir + '/**/*.png', recursive=True)) + list(glob.glob(values_dir + '/**/*.jpg', recursive=True))
    values_files = sorted([pathlib.Path(f) for f in values_files])
    
    MAX_SAMPLES = 5000 # Maximum number of samples to use for testing
    
    if large_scale:
        query_files = query_files[:MAX_SAMPLES]
        values_files = values_files[:MAX_SAMPLES]
    
    dataset_query = ImagePathDataset(query_files, query_transform) # the query data
    dataset_values = ImagePathDataset(values_files, value_transform) # the reference data

    data_loader_values = torch.utils.data.DataLoader(
        dataset_values,
        batch_size=10,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=10,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    
    print(f"train: {len(dataset_values)} imgs / query: {len(dataset_query)} imgs")
    model.eval()

    print("--------Features extraction--------")
    
    values_features, values_numpatches = extract_features(model, data_loader_values, similarity_metric)
    query_features, query_numpatches = extract_features(model, data_loader_query, similarity_metric)
    
    values_features = nn.functional.normalize(values_features, dim=1, p=2)
    query_features = nn.functional.normalize(query_features, dim=1, p=2)

    assert values_numpatches == query_numpatches

    num_chunks = values_numpatches
    
    print("--------Semantic similarity--------")
    
    if similarity_metric == 'splitloss':

        v = rearrange(values_features, 'n (c p) -> n c p ', c=num_chunks)
        q = rearrange(query_features, 'm (c p) -> m c p ', c=num_chunks)
        print(f"Shape of v: {v.shape}, Shape of q: {q.shape}")
        chunk_dp = torch.einsum('ncp, mcp -> nmc', v, q)
        # sim = reduce(chunk_dp, 'bc -> b', 'max')
        print(chunk_dp)
        print(chunk_dp.shape)
        # Normalize the inner products by the norms of the chunks
        v_norms = torch.norm(v, dim=2).unsqueeze(0)
        q_norms = torch.norm(q, dim=2).unsqueeze(1)
        norm_product = v_norms * q_norms
        print(norm_product.shape)
        norm_product = norm_product.permute(1, 0, 2)  # Permute to match chunk_dp shape
        print(norm_product.shape)  

        normalized_chunk_dp = chunk_dp / norm_product
        print(f"Normalized chunk inner products (chunk_dp): {normalized_chunk_dp}")

        # Aggregate the maximum inner product across chunks
        sim_max = torch.max(normalized_chunk_dp, dim=2).values 
        print(f"Similarity (split-product) max: {sim_max}")

        sim_mean = torch.mean(normalized_chunk_dp, dim=2)
        print(f"Similarity (split-product) mean: {sim_mean}")

        # torch.save(sim_max, '') # Adjust the path as needed

        if small_scale:

            dirname = "" # Adjust the path as needed
            os.makedirs(dirname, exist_ok=True)

            sim_mean_np = sim_mean.cpu().numpy()

            plt.figure(figsize=(10, 8))
            heatmap = sns.heatmap(sim_mean_np, cmap='viridis', cbar_kws={'label': 'Similarity Score'})
            # plt.colorbar(heatmap.collections[0], label='Similarity Score')
            plt.xlabel('Query Index', fontsize=14)
            plt.ylabel('Value Index', fontsize=14)
            plt.title('Heatmap of Mean Similarity Scores', fontsize=16)
            heatmap.tick_params(axis='both', which='major', labelsize=14)
            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.yaxis.label.set_size(14)
            colorbar.ax.tick_params(labelsize=14)
            plt.savefig(f'{dirname}/mean_similarity_heatmap.png')

            plt.figure(figsize=(10, 8))
            plt.hist(sim_mean_np.flatten(), bins=50, color='blue', alpha=0.7)
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.title('Histogram of Mean Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/mean_similarity_histogram.png')

            plt.figure(figsize=(12, 6))
            sns.kdeplot(sim_mean_np.flatten(), fill=True, color='blue')
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.title('Density Plot of Mean Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14) 
            plt.savefig(f'{dirname}/mean_similarity_density.png')

            # Sort the similarity scores
            sorted_similarities = np.sort(sim_mean_np.flatten())
            cumulative_probs = np.linspace(0, 1, len(sorted_similarities))

            # Compute the complementary cumulative probabilities
            ccdf_probs = 1 - cumulative_probs

            # Plot the CCDF
            plt.figure(figsize=(10, 8))
            plt.plot(sorted_similarities, ccdf_probs, color='blue')
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Complementary Cumulative Probability', fontsize=14)
            plt.title('CCDF of Mean Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14) 
            plt.savefig(f'{dirname}/mean_similarity_ccdf.png')

            plt.figure(figsize=(12, 6))
            sns.boxplot(data=[sim_mean_np[i, :] for i in range(sim_mean_np.shape[0])])
            plt.xlabel('Group of Value Images', fontsize=14)
            plt.ylabel('Similarity Score', fontsize=14)
            plt.title('Box Plot of Mean Similarity Scores by Groups', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14) 
            plt.savefig(f'{dirname}/mean_similarity_boxplot.png')
            

            sim_max_np = sim_max.cpu().numpy()

            # Create the heatmap of the downsampled similarity scores
            plt.figure(figsize=(10, 8))
            heatmap = sns.heatmap(sim_max_np, cmap='viridis', cbar_kws={'label': 'Similarity Score'})
            # plt.colorbar(heatmap.collections[0], label='Similarity Score')
            plt.xlabel('Query Index', fontsize=14)
            plt.ylabel('Value Index', fontsize=14)
            plt.title('Heatmap of Max Similarity Scores', fontsize=16)
            heatmap.tick_params(axis='both', which='major', labelsize=14)
            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.yaxis.label.set_size(14)
            colorbar.ax.tick_params(labelsize=14)
            plt.savefig(f'{dirname}/max_similarity_heatmap.png')

            plt.figure(figsize=(10, 8))
            plt.hist(sim_max_np.flatten(), bins=50, color='blue', alpha=0.7)
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.title('Histogram of Max Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/max_similarity_histogram.png')

            plt.figure(figsize=(12, 6))
            sns.kdeplot(sim_max_np.flatten(), fill=True, color='blue')
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.title('Density Plot of Max Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/max_similarity_density.png')

            # Sort the similarity scores
            sorted_similarities = np.sort(sim_max_np.flatten())
            cumulative_probs = np.linspace(0, 1, len(sorted_similarities))

            # Compute the complementary cumulative probabilities
            ccdf_probs = 1 - cumulative_probs

            # Plot the CCDF
            plt.figure(figsize=(10, 8))
            plt.plot(sorted_similarities, ccdf_probs, color='blue')
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Complementary Cumulative Probability', fontsize=14)
            plt.title('CCDF of Max Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/max_similarity_ccdf.png')

            plt.figure(figsize=(12, 6))
            sns.boxplot(data=[sim_max_np[i, :] for i in range(sim_max_np.shape[0])])
            plt.xlabel('Group of Value Images', fontsize=14)
            plt.ylabel('Similarity Score', fontsize=14)
            plt.title('Box Plot of Max Similarity Scores by Groups', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/max_similarity_boxplot.png')

            # Set the similarity threshold
            mean_threshold = 0.4

            # Set the similarity threshold
            max_threshold = 0.7

            indices_above_threshold_mean = np.where(sim_mean_np > mean_threshold)
            indices_above_threshold_max = np.where(sim_max_np > max_threshold)

            # Indices
            rows_above_threshold, cols_above_threshold = indices_above_threshold_mean

            print(f"Indices where mean similarity is above {mean_threshold}:")
            for row, col in zip(rows_above_threshold, cols_above_threshold):
                print(f"Value Index: {row}, Query Index: {col}")
                print(query_files[col])
                print(values_files[row])

            rows_above_threshold, cols_above_threshold = indices_above_threshold_max

            print(f"Indices where max similarity is above {max_threshold}:")
            for row, col in zip(rows_above_threshold, cols_above_threshold):
                print(f"Value Index: {row}, Query Index: {col}")
                print(query_files[col])
                print(values_files[row])           
                        
        elif large_scale:
            
            dirname = "" # Adjust the path as needed   
            os.makedirs(dirname, exist_ok=True)
            
            sim_mean_np = sim_mean.cpu().numpy()

            # Downsample factor for large scale plotting
            downsample_factor = 10 

            # Calculate the padding size needed
            pad_height = (downsample_factor - sim_mean_np.shape[0] % downsample_factor) % downsample_factor
            pad_width = (downsample_factor - sim_mean_np.shape[1] % downsample_factor) % downsample_factor

            # Pad the array with zeros
            padded_sim_mean_np = np.pad(sim_mean_np, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

            # Downsample the similarity matrix by taking mean over downsample_factor x downsample_factor blocks
            downsampled_sim_mean = padded_sim_mean_np.reshape(
                (padded_sim_mean_np.shape[0] // downsample_factor, downsample_factor, 
                padded_sim_mean_np.shape[1] // downsample_factor, downsample_factor)
            ).mean(axis=(1, 3))

            print(f"Original shape: {sim_mean_np.shape}")
            print(f"Padded shape: {padded_sim_mean_np.shape}")
            print(f"Downsampled shape: {downsampled_sim_mean.shape}")

            # Create the heatmap of the downsampled similarity scores
            plt.figure(figsize=(10, 8))
            heatmap = sns.heatmap(downsampled_sim_mean, cmap='viridis', cbar_kws={'label': 'Similarity Score'})
            # plt.colorbar(heatmap.collections[0], label='Similarity Score')
            plt.xlabel('Query Index (downsampled)', fontsize=14)
            plt.ylabel('Value Index (downsampled)', fontsize=14)
            plt.title('Heatmap of Mean Similarity Scores', fontsize=16)
            heatmap.tick_params(axis='both', which='major', labelsize=14)
            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.yaxis.label.set_size(14)
            colorbar.ax.tick_params(labelsize=14)
            plt.savefig(f'{dirname}/mean_similarity_heatmap.png')

            plt.figure(figsize=(12, 6))
            plt.hist(sim_mean_np.flatten(), bins=100, color='blue', alpha=0.7)
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.title('Histogram of Mean Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/mean_similarity_histogram.png')

            plt.figure(figsize=(12, 6))
            sns.kdeplot(sim_mean_np.flatten(), fill=True, color='blue')
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.title('Density Plot of Mean Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/mean_similarity_density.png')

            sorted_similarities = np.sort(sim_mean_np.flatten())
            cumulative_probs = np.linspace(0, 1, len(sorted_similarities))

            # Compute the complementary cumulative probabilities
            ccdf_probs = 1 - cumulative_probs

            plt.figure(figsize=(10, 8))
            plt.plot(sorted_similarities, ccdf_probs, color='blue')
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Complementary Cumulative Probability', fontsize=14)
            plt.title('CCDF of Mean Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/mean_similarity_ccdf.png')

            # plt.figure(figsize=(12, 6))
            # sns.boxplot(data=[sim_mean_np[i, :] for i in range(0, sim_mean_np.shape[0], sim_mean_np.shape[0] // 10)])
            # plt.xlabel('Group of Value Images', fontsize=14)
            # plt.ylabel('Similarity Score', fontsize=14)
            # plt.title('Box Plot of Similarity Scores by Groups', fontsize=16)
            # plt.tick_params(axis='both', which='major', labelsize=14)
            # plt.savefig('results_path/similarity_boxplot.png')
            
            sim_max_np = sim_max.cpu().numpy()

            # Calculate the padding size needed
            pad_height = (downsample_factor - sim_max_np.shape[0] % downsample_factor) % downsample_factor
            pad_width = (downsample_factor - sim_max_np.shape[1] % downsample_factor) % downsample_factor

            # Pad the array with zeros
            padded_sim_max_np = np.pad(sim_max_np, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

            # Downsample the similarity matrix by taking mean over downsample_factor x downsample_factor blocks
            downsampled_sim_max = padded_sim_max_np.reshape(
                (padded_sim_max_np.shape[0] // downsample_factor, downsample_factor, 
                padded_sim_max_np.shape[1] // downsample_factor, downsample_factor)
            ).max(axis=(1, 3))

            print(f"Original shape: {sim_max_np.shape}")
            print(f"Padded shape: {padded_sim_max_np.shape}")
            print(f"Downsampled shape: {downsampled_sim_max.shape}")

            # Create the heatmap of the downsampled similarity scores
            plt.figure(figsize=(10, 8))
            heatmap = sns.heatmap(downsampled_sim_max, cmap='viridis', cbar_kws={'label': 'Similarity Score'})
            # plt.colorbar(heatmap.collections[0], label='Similarity Score')
            plt.xlabel('Query Index (downsampled)', fontsize=14)
            plt.ylabel('Value Index (downsampled)', fontsize=14)
            plt.title('Heatmap of Max Similarity Scores', fontsize=16)
            heatmap.tick_params(axis='both', which='major', labelsize=14)
            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.yaxis.label.set_size(14)
            colorbar.ax.tick_params(labelsize=14)
            plt.savefig(f'{dirname}/max_similarity_heatmap.png')

            plt.figure(figsize=(12, 6))
            plt.hist(sim_max_np.flatten(), bins=100, color='blue', alpha=0.7)
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.title('Histogram of Max Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/max_similarity_histogram.png')

            plt.figure(figsize=(12, 6))
            sns.kdeplot(sim_max_np.flatten(), fill=True, color='blue')
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.title('Density Plot of Max Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/max_similarity_density.png')

            sorted_similarities = np.sort(sim_max_np.flatten())
            cumulative_probs = np.linspace(0, 1, len(sorted_similarities))

            # Compute the complementary cumulative probabilities
            ccdf_probs = 1 - cumulative_probs

            plt.figure(figsize=(10, 8))
            plt.plot(sorted_similarities, ccdf_probs, color='blue')
            plt.xlabel('Similarity Score', fontsize=14)
            plt.ylabel('Complementary Cumulative Probability', fontsize=14)
            plt.title('CCDF of Max Similarity Scores', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.savefig(f'{dirname}/max_similarity_ccdf.png')

            # plt.figure(figsize=(12, 6))
            # sns.boxplot(data=[sim_max_np[i, :] for i in range(0, sim_max_np.shape[0], sim_max_np.shape[0] // 10)])
            # plt.xlabel('Group of Value Images', fontsize=14)
            # plt.ylabel('Similarity Score', fontsize=14)
            # plt.title('Box Plot of Similarity Scores by Groups', fontsize=16)
            # plt.tick_params(axis='both', which='major', labelsize=14)
            # plt.savefig(f'{dirname}/max_similarity_boxplot.png')

            # Set the similarity threshold
            mean_threshold = 0.4

            # Set the similarity threshold
            max_threshold = 0.7

                        
            # indices_above_threshold_mean = np.where(sim_mean_np > mean_threshold)
            indices_above_threshold_max = np.where(sim_max_np > max_threshold)

            # Indices
            rows_above_threshold, cols_above_threshold = indices_above_threshold_mean

            # Print the indices
            print(f"Indices where mean similarity is above {mean_threshold}:")
            for row, col in zip(rows_above_threshold, cols_above_threshold):
                print(f"Value Index: {row}, Query Index: {col}")
                print(query_files[col])
                print(values_files[row])

            # Indices
            rows_above_threshold, cols_above_threshold = indices_above_threshold_max

            # Print the indices
            print(f"Indices where max similarity is above {max_threshold}:")
            for row, col in zip(rows_above_threshold, cols_above_threshold):
                print(f"Value Index: {row}, Query Index: {col}")
                print(query_files[col])
                print(values_files[row])

        else:
            raise NotImplementedError
            
            # "This is legacy code from repo https://github.com/somepago/DCR
       
            # sim = torch.max(chunk_dp, dim=1).values
            # print(sim)
        
            # # Debugging: Check shapes
            # print(f"Shape of v: {v.shape}, Shape of q: {q.shape}")
            
            # # Compute the Euclidean distance for corresponding chunks
            # chunk_distances = torch.cdist(v, q, p=1)  # Shape: (batch_size, num_chunks, num_chunks)
            
            # # Debugging: Print intermediate values
            # print(f"Chunk distances: {chunk_distances}")
            # print(f"Chunk distances shape: {chunk_distances.shape}")
            
            # # Aggregate the maximum distance across chunks
            # sim = reduce(chunk_distances, 'b c p -> b c', 'min')
            # sim_min = sim.min(dim=1).values  # Shape: (batch_size,)


            # # Compute the Euclidean distance for corresponding chunks
            # chunk_distances2 = torch.cdist(v, v, p=1)  # Shape: (batch_size, num_chunks, num_chunks)
            
            # # Debugging: Print intermediate values
            # print(f"Chunk distances: {chunk_distances2}")
            
            # # Aggregate the maximum distance across chunks
            # sim2 = reduce(chunk_distances2, 'b c p -> b c', 'min')
            # sim2_min = sim2.min(dim=1).values  # Shape: (batch_size,)
            
            # # Debugging: Print final distance values
            # print(f"Distance (split-product): {sim2_min}")
    
    else:
        sim = torch.mm(values_features, query_features.T)
        sim_ref = torch.mm(values_features, values_features.T)

        ranks = torch.argsort(-sim, dim=0).cpu().numpy()

        ######################
        ret_savepath = f'test_path/'
        os.makedirs(ret_savepath,exist_ok=True)

        simscores = sim.T 
        bg_simscores = sim_ref.T

        torch.save(simscores.cpu(), os.path.join(ret_savepath, "similarity.pth"))
        torch.save(bg_simscores.cpu(), os.path.join(ret_savepath, "similarity_wtrain.pth"))

        main_v,main_l = simscores.topk(1,axis=1,largest=True)
        bg_v,bg_l = bg_simscores.topk(1,axis=1,largest=True)
        bg_v = bg_v[:,-1] #remove the first one since it is to self.

        plt.figure(figsize=(6,4))

        x0 =  main_v.cpu().numpy()
        x1 = bg_v.cpu().numpy()
        bin_width= 0.005
        
        nbins = math.ceil(1 / bin_width)
        bins = np.linspace(0,1, nbins)

        fig = plt.hist(x0, bins, alpha=0.4, label='sim(gen,train)',density=True)
        fig = plt.hist(x1, bins, alpha=0.6, label='sim(train,train)',density=True)
        plt.legend(loc='upper right')

        plt.savefig(f"{ret_savepath}/histogram.png")
        plt.figure()

        #### Computing mean similarity and FID scores of generations ######

        sim_mean = np.mean(x0)
        sim_std = np.std(x0)
        sim_75pc = np.percentile(x0, 75)
        sim_90pc = np.percentile(x0, 90)
        sim_95pc = np.percentile(x0, 95)

        bg_mean = np.mean(x1)
        bg_std = np.std(x1)
        bg_75pc = np.percentile(x1, 75)
        bg_90pc = np.percentile(x1, 90)
        bg_95pc = np.percentile(x1, 95)
        
        sim_gt_05pc = np.sum(x0 > 0.5)/(x0.shape[0])

        print('Simscores @x% part done')
        print({
            'sim_mean':sim_mean,
            'sim_std':sim_std,
            'sim_75pc' : sim_75pc,
            'sim_90pc' : sim_90pc,
            'sim_95pc' : sim_95pc,
            'sim_gt_05pc' : sim_gt_05pc,
            'bg_mean':bg_mean,
            'bg_std' : bg_std,
            'bg_75pc' : bg_75pc,
            'bg_90pc' : bg_90pc,
            'bg_95pc' : bg_95pc
        })


@torch.no_grad()
def extract_features(model: nn.Module, 
                     data_loader: torch.utils.data.DataLoader, 
                     similarity_metric='splitloss',
                     layer=-1):
    features = None

    count = 0
    for samples, index in data_loader:
        print("iteration: ", count)
        count += 1  
        samples = samples.cuda(non_blocking=True) if torch.cuda.is_available() else samples
        index = index.cuda(non_blocking=True) if torch.cuda.is_available() else index

        if similarity_metric == 'splitloss':
            
            if  layer > 1:
                feats = model.module.get_intermediate_layers(samples,layer)[0].clone()
            
            else:
                feats = model(samples).clone()

        else:
            
            if  layer > 1:
                feats = model.module.get_intermediate_layers(samples,layer)[0][:,0,:].clone()
        
            else:
                feats = model(samples).clone()
        
        numpatches = feats.shape[1]
        feats = rearrange(feats, 'b h w -> b (h w)')
        
        # init storage feature matrix
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if torch.cuda.is_available():
                features = features.cuda(non_blocking=True)
            
            print(f"Storing features into tensor of shape {features.shape}")

        index_all = index
        feats_all = feats

        if torch.cuda.is_available():
            features.index_copy_(0, index_all, feats_all)
        else:
            features.index_copy_(0, index_all.cpu(), feats_all.cpu())
    
    print(features)
    return features, numpatches

if __name__ == '__main__':
    query_dir = '' # adjust as needed
    val_dir = '' # adjust as needed
    run_semantic_similarity(query_dir, val_dir, similarity_metric='splitloss', scale='small')