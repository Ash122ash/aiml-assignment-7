# ==========================================
# AI-ML Assignment 7
# Customer Segmentation using
# K-Means Clustering and PCA
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------------------------
# Task 1 : Data Understanding
# ----------------------------------------

df = pd.read_csv("Mall_Customers.csv")

print("First Five Records\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nNumerical Features:")
print(df.select_dtypes(include=["number"]).columns.tolist())

print("\nCategorical Features:")
print(df.select_dtypes(include=["object", "string"]).columns.tolist())

# ----------------------------------------
# Task 2 : Data Preprocessing
# ----------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Remove CustomerID
df.drop("CustomerID", axis=1, inplace=True)

# Encode Gender
df["Gender"] = df["Gender"].map({
    "Male": 0,
    "Female": 1
})

# Standardize Features
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

# ----------------------------------------
# Task 3 : Elbow Method
# ----------------------------------------

wcss = []

for i in range(1, 11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_data)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker="o")

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.show()

# ----------------------------------------
# Train Final Model
# ----------------------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)

df["Cluster"] = clusters

# ----------------------------------------
# Scatter Plot
# ----------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Clusters")

plt.show()

# ----------------------------------------
# PCA
# ----------------------------------------

pca = PCA(n_components=2)

pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))

plt.scatter(
    pca_data[:,0],
    pca_data[:,1],
    c=df["Cluster"],
    cmap="viridis"
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("PCA Visualization")

plt.show()