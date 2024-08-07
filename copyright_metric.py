import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load sem_sim and perc_sim from files 
sem_sim = torch.load('').cuda()  # Adjust the path as needed
perc_sim = torch.load('').cuda() # Adjust the path as needed

assert sem_sim.shape == perc_sim.shape, "The sem_sim and perc_sim must have the same shape"

copyright_loss = 0.5 * sem_sim + 0.5 * perc_sim
print(copyright_loss)

dirname = "" # Adjust the path as needed

copyright_loss_np = copyright_loss.cpu().numpy()

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(copyright_loss_np-0.1, cmap='viridis', cbar_kws={'label': 'Copyright Score'}, vmax=0.77)
# plt.colorbar(heatmap.collections[0], label='Similarity Score')
plt.xlabel('Query Index', fontsize=14)
plt.ylabel('Value Index', fontsize=14)
plt.title('Heatmap of Copyright Scores', fontsize=16)
heatmap.tick_params(axis='both', which='major', labelsize=0)
colorbar = heatmap.collections[0].colorbar
colorbar.ax.yaxis.label.set_size(14)
colorbar.ax.tick_params(labelsize=14)
plt.savefig(f'{dirname}/similarity_heatmap.png')

plt.figure(figsize=(10, 8))
plt.hist(copyright_loss_np.flatten(), bins=100, color='blue', alpha=0.7)
# plt.xlim(right=0.77)
plt.xlabel('Copyright Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Copyright Scores', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(f'{dirname}/similarity_histogram.png')

plt.figure(figsize=(12, 6))
sns.kdeplot(copyright_loss_np.flatten(), fill=True, color='blue')
plt.xlim(right=0.87)
plt.xlabel('Copyright Score', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Density Plot of Copyright Scores', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.savefig(f'{dirname}/similarity_density.png')

# Sort the similarity scores
sorted_copyright_losses = np.sort(copyright_loss_np.flatten())
cumulative_probs = np.linspace(0, 1, len(sorted_copyright_losses))

# Compute the complementary cumulative probabilities
ccdf_probs = 1 - cumulative_probs

# Plot the CCDF
plt.figure(figsize=(10, 8))
plt.plot(sorted_copyright_losses, ccdf_probs, color='blue')
plt.xlabel('Copyright Score', fontsize=14)
plt.ylabel('Complementary Cumulative Probability', fontsize=14)
plt.title('CCDF of Copyright Scores', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.savefig(f'{dirname}/similarity_ccdf.png')

plt.figure(figsize=(12, 6))
sns.boxplot(data=[copyright_loss_np[i, :] for i in range(copyright_loss_np.shape[0])])
plt.ylim(top=0.77)
plt.xlabel('Group of Value Images', fontsize=14)
plt.ylabel('Copyright Score', fontsize=14)
plt.title('Box Plot of Copyright Scores by Groups', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.savefig(f'{dirname}/similarity_boxplot.png')