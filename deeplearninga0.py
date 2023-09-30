import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import random
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

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

# Calculate Means from the Training Set
means = []
for label in class_labels:
    mean = np.mean(train_data[train_label == label], axis=0)
    means.append(mean)

# Apply the Nearest Mean Classifier on the Training Set
nmc_train_pred = []
for point in train_data:
    nmc_train_pred.append(nearest_mean_classifier(point, means))
nmc_train_pred = np.array(nmc_train_pred)
nmc_train_accuracy = np.sum(nmc_train_pred == train_label) / len(train_label) * 100
print('Training Accuracy: {:.2f}%'.format(nmc_train_accuracy))

# Apply the Nearest Mean Classifier on the Test Set
nmc_test_pred = []
for point in test_data:
    nmc_test_pred.append(nearest_mean_classifier(point, means))
nmc_test_pred = np.array(nmc_test_pred)
nmc_test_accuracy = np.sum(nmc_test_pred == test_label) / len(test_label) * 100
print('Test Accuracy: {:.2f}%'.format(nmc_test_accuracy))

# Visualize means of each digit
# Can be used for Task 1 Q1
plt.figure(figsize=(20, 8))
for i, center in enumerate(means):
    plt.subplot(2, 5, i+1)
    plt.imshow(center.reshape(16, 16), cmap='gray')
    plt.title('Digit {}'.format(i))
plt.tight_layout()
plt.show()

# Train the KNN classifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_data, train_label)

knn_train_pred = knn.predict(train_data)
knn_test_pred = knn.predict(test_data)

# Evaluate
knn_train_accuracy = np.sum(knn_train_pred == train_label) / len(train_label) * 100
knn_test_accuracy = np.sum(knn_test_pred == test_label) / len(test_label) * 100

print('KNN Training Accuracy: {:.2f}%'.format(knn_train_accuracy))
print('KNN Test Accuracy: {:.2f}%'.format(knn_test_accuracy))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix for the Nearest Mean Classifier
nmc_confusion_matrix = confusion_matrix(train_label, nmc_train_pred)

# Calculate the confusion matrix for the KNN Classifier
knn_confusion_matrix = confusion_matrix(train_label, knn_train_pred)

# Plot the confusion matrices for Nearest Mean Classifier and KNN Classifier
precent_confusion_matrix = [nmc_confusion_matrix / np.sum(nmc_confusion_matrix, axis=1) * 100,
                            knn_confusion_matrix / np.sum(knn_confusion_matrix, axis=1) * 100]
matrix_names = ['Nearest Mean Classifier', 'KNN Classifier']

plt.figure(figsize=(15, 6))
for i, matrix in enumerate(precent_confusion_matrix):
    plt.subplot(1, 2, i+1)
    plt.imshow(matrix, cmap='Blues')
    plt.title(matrix_names[i])
    plt.colorbar(format='%d%%')

    # Add labels to each cell
    for j in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            plt.text(k, j, '{:.2f}%'.format(matrix[j, k]), ha='center', va='center', size=8)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))

plt.suptitle('Confusion Matrices of Nearest Mean Classifier and KNN Classifier', fontsize=16)
plt.show()

# Define multi class perceptron
# Arguments:
# num_classes: Number of classes
# num_features: Number of features

# Methods:
# train: Train the model
# Arguments:
# X_train: Input data
# y_train: Output data
# lr: Learning rate
# epochs: Number of epochs

# predict: Predict the class of the input data
# Arguments:
# X_test: Input data
# Returns:
# y_pred: Predicted class of the input data

# Input data is 16x16 images of digits 0-9
# Output data is the digit itself
# Number of classes = 10
# Number of features = 257 (16x16 + 1 bias term)

class MultiClassPerceptron():
    def __init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.num_features = num_features
        # Add bias term
        # Initialize weights to random values
        self.weights = np.random.rand(num_classes, num_features + 1)

    def train(self, X_train, y_train, lr=0.01, epochs=100):
        for epoch in tqdm(range(epochs)):
            for i in range(len(X_train)):
                # Add bias term
                x = np.append(X_train[i], 1)
                # Predict the class
                y_pred = np.dot(self.weights, x)

                # Label the true class
                y_true = np.zeros(self.num_classes)
                y_true[y_train[i]] = 1

                # Update weights
                # Formula: w_true = w_true + lr * x
                #          w_pred = w_pred - lr * x
                if y_pred.argmax() != y_true.argmax():
                    self.weights[y_true.argmax()] += lr * x
                    self.weights[y_pred.argmax()] -= lr * x

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            # Add bias term
            x = np.append(x, 1)
            # Predict the class
            y_pred.append(np.argmax(np.dot(self.weights, x)))
        return y_pred