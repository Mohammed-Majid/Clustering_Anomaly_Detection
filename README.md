# Unsupervised Clustering and Anomaly Detection on Network Traffic Data

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview

This project performs clustering and anomaly detection on network traffic data. We use dimensionality reduction techniques such as PCA (Principal Component Analysis) and t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualize high-dimensional data and apply K-Means clustering to identify patterns and detect anomalies (Silhouette score of 0.81).

Key components of the project include:
- **Data Preprocessing**: Standardizing and encoding features from the dataset.
- **Dimensionality Reduction**: Using PCA and t-SNE to reduce the data to two dimensions for easier computation & visualization.
- **Clustering**: Applying K-Means clustering to group similar data points.
- **Anomaly Detection**: Labeling anomalies based on cluster sizes (we assume the minority class is the anomaly as it is normally the case in the real world).


<img width="889" alt="Screen Shot 2024-08-08 at 1 22 42 AM" src="https://github.com/user-attachments/assets/38252b2c-89d9-46ef-aff8-b75f89f46898">



## Features

- **Clustering Visualization**: Visualize clustering results using PCA and t-SNE.
- **Anomaly Detection**: Identify and label anomalies based on clustering results.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/mohammed-majid/clustering-anomaly-detection.git
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the model Script**: Execute the script to perform clustering and anomaly detection.
    ```bash
    python model.ipynb
    ```

2. **Review Results**: Check the output in the console or view the generated plots for clustering visualizations.

3. **Explore Data**: The dataset will be updated with cluster labels and anomaly indicators.

## Acknowledgements

This project was developed using the following libraries and tools:
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
