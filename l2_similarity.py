import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.backends.cudnn as cudnn
import os
import pathlib
import numpy as np
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
import glob


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
        return img
    
    def get_stacked_images(self):
        images = [self.__getitem__(i) for i in range(len(self))]
        return torch.stack(images)

def run_l2_similarity(query_dir, values_dir, scale='large'):

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    if scale == 'large':
        large_scale = True
        small_scale = False
    else:
        large_scale = False
        small_scale = True

    cudnn.benchmark = True

    # Data loading code
    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]), # pixel values are in range [-1, 1]
                    ])

    query_files = list(glob.glob(query_dir + '/**/*.JPEG', recursive=True)) + list(glob.glob(query_dir + '/**/*.png', recursive=True)) + list(glob.glob(query_dir + '/**/*.jpg', recursive=True))
    query_files = sorted([pathlib.Path(f) for f in query_files])

    values_files = list(glob.glob(values_dir + '/**/*.JPEG', recursive=True)) + list(glob.glob(values_dir + '/**/*.png', recursive=True)) + list(glob.glob(values_dir + '/**/*.jpg', recursive=True))
    values_files = sorted([pathlib.Path(f) for f in values_files])

    dataset_query = ImagePathDataset(query_files, transform) # the query data
    dataset_values = ImagePathDataset(values_files, transform) # the reference data

    query_images_stacked = dataset_query.get_stacked_images()
    values_images_stacked = dataset_values.get_stacked_images()

    print('Query images shape: ', query_images_stacked.shape)
    print('Values images shape: ', values_images_stacked.shape)

    # Add a batch dimension to each tensor
    query_images_stacked = query_images_stacked.unsqueeze(1)  # Shape: (num_query_images, 1, C, H, W)
    values_images_stacked = values_images_stacked.unsqueeze(0)  # Shape: (1, num_value_images, C, H, W)
    print('Query images shape: ', query_images_stacked.shape)
    print('Values images shape: ', values_images_stacked.shape)

    # Broadcast the tensors to create all pairs
    query_images_broadcasted = query_images_stacked.expand(-1, values_images_stacked.size(1), -1, -1, -1)
    values_images_broadcasted = values_images_stacked.expand(query_images_stacked.size(0), -1, -1, -1, -1)
    print('Query images broadcasted shape: ', query_images_broadcasted.shape)
    print('Values images broadcasted shape: ', values_images_broadcasted.shape)

    query_images_flattened = query_images_broadcasted.reshape(-1, 3, 224, 224)
    values_images_flattened = values_images_broadcasted.reshape(-1, 3, 224, 224)

    distances_flattened = compute_l2_distances_in_batches(values_images_flattened, query_images_flattened, batch_size=25)
    # Reshape the distances back to a 2D tensor
    num_query_images = query_images_broadcasted.size(0)
    num_value_images = values_images_broadcasted.size(1)
    distances = distances_flattened.view(num_query_images, num_value_images).t() # Shape: (num_value_images, num_query_images)

    distances = distances / (distances + 1) # Normalize the distances to [0, 1]
    sim = (1 - distances)*100
    # save
    # torch.save(sim, '') # add path 

    # Print the distances
    print('Similarity shape: ', sim.shape)
    print('Similarity: ', sim)

    if small_scale:
        
        sim = sim.cpu().numpy()

        dirname = '' # add path
        os.makedirs(dirname, exist_ok=True)

        # Create the heatmap of the downsampled similarity scores
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(sim, cmap='viridis', cbar_kws={'label': 'Similarity Score'})
        plt.xlabel('Query Index', fontsize=14)
        plt.ylabel('Value Index', fontsize=14)
        plt.title('Heatmap of L2 Similarity Scores', fontsize=16)
        heatmap.tick_params(axis='both', which='major', labelsize=14)
        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.yaxis.label.set_size(14)
        colorbar.ax.tick_params(labelsize=14)
        plt.savefig(f'{dirname}/l2_similarity_heatmap.png')

        plt.figure(figsize=(10, 8))
        plt.hist(sim.flatten(), bins=50, color='blue', alpha=0.7)
        plt.xlabel('Similarity Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Histogram of L2 Similarity Scores', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14) 
        plt.savefig(f'{dirname}/l2_similarity_histogram.png')

        plt.figure(figsize=(12, 6))
        sns.kdeplot(sim.flatten(), fill=True, color='blue')
        plt.xlabel('Similarity Score', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Density Plot of L2 Similarity Scores', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)  # Set font size for the tick labels
        plt.savefig(f'{dirname}/l2_similarity_density.png')

        # Sort the similarity scores
        sorted_similarities = np.sort(sim.flatten())

        # Compute the cumulative probabilities
        cumulative_probs = np.linspace(0, 1, len(sorted_similarities))
        ccdf = 1 - cumulative_probs

        # Plot the CDF
        plt.figure(figsize=(10, 8))
        plt.plot(sorted_similarities, ccdf, color='blue')
        plt.xlabel('Similarity Score', fontsize=14)
        plt.ylabel('Complementary Cumulative Probability', fontsize=14)
        plt.title('CCDF of L2 Similarity Scores', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(f'{dirname}/l2_similarity_ccdf.png')

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=[sim[i, :] for i in range(sim.shape[0])])
        plt.xlabel('Group of Value Images', fontsize=14)
        plt.ylabel('Similarity Score', fontsize=14)
        plt.title('Box Plot of L2 Similarity Scores by Groups', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(f'{dirname}/l2_similarity_boxplot.png')

        # Set the similarity threshold
        threshold = 0.5

        # Compute the number of queries that have at least one similarity score above the threshold
        num_queries_above_threshold = np.sum(np.any(sim > threshold, axis=0))
        print(f"Number of query images with L2 similarity lower than {threshold}: {num_queries_above_threshold}")

        indices_above_threshold = np.where(sim > threshold)

        # Indices
        rows_above_threshold, cols_above_threshold = indices_above_threshold

        # Print the indices
        print(f"Indices where L2 similarity is above {threshold}:")
        for row, col in zip(rows_above_threshold, cols_above_threshold):
            print(f"Value Index: {row}, Query Index: {col}")
            print(query_files[col])
            print(values_files[row])

    if large_scale:
        sim = sim.cpu().numpy()

        painting_name = query_dir.split('/')[-2]
        dirname = '' # add path
        os.makedirs(dirname, exist_ok=True)

        # Downsample factor
        downsample_factor = 10

        # Calculate the padding size needed
        pad_height = (downsample_factor - sim.shape[0] % downsample_factor) % downsample_factor
        pad_width = (downsample_factor - sim.shape[1] % downsample_factor) % downsample_factor

        # Pad the array with zeros
        padded_sim_np = np.pad(sim, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

        # Downsample the similarity matrix by taking mean over downsample_factor x downsample_factor blocks
        downsampled_sim = padded_sim_np.reshape(
            (padded_sim_np.shape[0] // downsample_factor, downsample_factor,
             padded_sim_np.shape[1] // downsample_factor, downsample_factor)
        ).mean(axis=(1, 3))

        print(f"Original shape: {sim.shape}")
        print(f"Padded shape: {padded_sim_np.shape}")
        print(f"Downsampled shape: {downsampled_sim.shape}")

        # Create the heatmap of the downsampled similarity scores
        plt.figure(figsize=(14, 12))
        heatmap = sns.heatmap(downsampled_sim, cmap='viridis', cbar_kws={'label': 'Similarity Score'})
        plt.xlabel('Query Index (downsampled)', fontsize=14)
        plt.ylabel('Value Index (downsampled)', fontsize=14)
        plt.title('Heatmap of L2 Similarity Scores', fontsize=16)
        heatmap.tick_params(axis='both', which='major', labelsize=14)
        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.yaxis.label.set_size(14)
        colorbar.ax.tick_params(labelsize=14)
        plt.savefig(f'{dirname}/l2_similarity_heatmap.png')

        plt.figure(figsize=(12, 6))
        plt.hist(downsampled_sim.flatten(), bins=100, color='blue', alpha=0.7)
        plt.xlabel('Similarity Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Histogram of L2 Similarity Scores', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(f'{dirname}/l2_similarity_histogram.png')

        plt.figure(figsize=(12, 6))
        sns.kdeplot(downsampled_sim.flatten(), fill=True, color='blue')
        plt.xlabel('Similarity Score', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Density Plot of L2 Similarity Scores', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)

        # Sort the similarity scores
        sorted_similarities = np.sort(downsampled_sim.flatten())

        # Compute the cumulative probabilities
        cumulative_probs = np.linspace(0, 1, len(sorted_similarities))

        # Plot the CDF
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_similarities, cumulative_probs, color='blue')
        plt.xlabel('Similairity Score', fontsize=14)
        plt.ylabel('Complementary Cumulative Probability', fontsize=14)
        plt.title('CDF of L2 Similarity Scores', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(f'{dirname}/l2_similarity_ccdf.png')

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=[downsampled_sim[i, :] for i in range(downsampled_sim.shape[0])])
        plt.xlabel('Group of Value Images', fontsize=14)
        plt.ylabel('Similarity Score', fontsize=14)
        plt.title('Box Plot of L2 Similarity Scores by Groups', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(f'{dirname}/l2_similarity_boxplot.png')

        # Set the similarity threshold
        threshold = 0.5

        # Compute the number of queries that have at least one similarity score above the threshold
        num_queries_above_threshold = np.sum(np.any(downsampled_sim > threshold, axis=0))
        print(f"Number of query images with L2 similarity lower than {threshold}: {num_queries_above_threshold}")

        indices_above_threshold = np.where(downsampled_sim < threshold)

        # Indices
        rows_above_threshold, cols_above_threshold = indices_above_threshold

        # Print the indices
        print(f"Indices where L2 similarity is below {threshold}:")
        for row, col in zip(rows_above_threshold, cols_above_threshold):
            print(f"Value Index: {row}, Query Index: {col}")
            print(query_files[col])
            print(values_files[row])


def compute_l2_distances_in_batches(values_images_flattened, query_images_flattened, batch_size=25):
    num_batches = (values_images_flattened.size(0) + batch_size - 1) // batch_size
    distances_list = []
    print('num_batches: ', num_batches)
    for i in range(num_batches):
        print(i)
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, values_images_flattened.size(0))

        batch_values = values_images_flattened[start_idx:end_idx]
        batch_queries = query_images_flattened[start_idx:end_idx]

        if torch.cuda.is_available():
            batch_values = batch_values.cuda()
            batch_queries = batch_queries.cuda()

        with torch.no_grad():
            distances_batch = torch.norm(batch_values - batch_queries, p=2, dim=(1, 2, 3))

        distances_list.append(distances_batch)

    distances_flattened = torch.cat(distances_list)
    return distances_flattened


if __name__ == '__main__':
    query_dir = '' # add path
    val_dir = '' # add path
    run_l2_similarity(query_dir, val_dir, scale='small')