import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import random
from tqdm import tqdm

# Load the training data
train_in_df = pd.read_csv('data/train_in.csv', header=None)
train_out_df = pd.read_csv('data/train_out.csv', header=None)

# Load the test data
test_in_df = pd.read_csv('data/test_in.csv', header=None)
test_out_df = pd.read_csv('data/test_out.csv', header=None)

# Convert the dataframes to NumPy arrays
train_data = train_in_df.to_numpy()
train_label = train_out_df.to_numpy().flatten()
test_data = test_in_df.to_numpy()
test_label = test_out_df.to_numpy().flatten()

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(train_data)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1500)
tsne_result = tsne.fit_transform(train_data)

# UMAP
umap_result = umap.UMAP(n_neighbors=5, min_dist=0.3).fit_transform(train_data)

# Plot the results

class_labels = np.unique(train_label)

result_list = [pca_result, tsne_result, umap_result]
result_names = ['PCA', 't-SNE', 'UMAP']

# Create a figure with 3 subplots
plt.figure(figsize=(15, 6))

# Loop through the results and plot each one
for i, result in enumerate(result_list):
    plt.subplot(1, 3, i+1)
    for label in class_labels:
        plt.scatter(result[train_label == label, 0], result[train_label == label, 1], label=label, s=10)
    plt.title(result_names[i])

plt.subplot(1,3,2)
plt.legend(class_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=5)

plt.tight_layout()
plt.show()

# Function to classify a point using Nearest Mean Classifier
def nearest_mean_classifier(points, centers):
    distances = [np.linalg.norm(points - center) for center in centers]
    return np.argmin(distances)

# Function to calculate the accuracy of classification
def calculate_accuracy(data, labels, centers):
    correct = 0
    for i in range(len(data)):
        predicted_label = nearest_mean_classifier(data[i], centers)
        if predicted_label == labels[i]:
            correct += 1
    accuracy = (correct / len(data)) * 100
    return accuracy