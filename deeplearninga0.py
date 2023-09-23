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