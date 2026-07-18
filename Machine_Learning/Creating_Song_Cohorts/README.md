# Creating Song Cohorts

## Project Overview

This project groups Spotify songs into meaningful cohorts using unsupervised machine learning. Audio features such as danceability, energy, tempo, loudness, and valence are used to cluster songs with similar musical characteristics. A simple recommendation system suggests similar songs from the same cluster.

---

## Problem Statement

Analyze Spotify audio features to identify natural song groups and recommend similar songs based on clustering.

---

## Dataset

- **Source:** Kaggle - Spotify Tracks Dataset
- **Records:** ~114,000 songs
- **Features:** 21

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Selection
5. Feature Scaling
6. K-Means Clustering
7. DBSCAN Clustering
8. PCA Visualization
9. Song Recommendation System
10. Export Results

---

## Machine Learning Techniques

- K-Means Clustering
- DBSCAN
- StandardScaler
- PCA

---

## Recommendation System

Given a song title, the system:

- Finds its cluster
- Identifies songs with similar audio characteristics
- Recommends songs from the same cluster

---

## Key Insights

- Songs naturally form groups based on audio features.
- K-Means effectively identifies musical cohorts.
- DBSCAN detects dense groups and outliers.
- PCA provides an intuitive visualization of song clusters.
- Cluster-based recommendations offer a simple content-based recommendation system.

---

## Project Structure

```text
Creating_Song_Cohorts
│
├── data
├── models
├── notebooks
├── outputs
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Future Improvements

- Cosine Similarity Recommendations
- Nearest Neighbors Recommendation Engine
- Streamlit Web App
- Spotify API Integration

---

## Author

**Veda Vaishnavi Penumatcha**