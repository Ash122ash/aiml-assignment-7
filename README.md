# Customer Segmentation using K-Means Clustering and Principal Component Analysis (PCA)

## Objective

The objective of this project is to segment shopping mall customers into different groups using K-Means Clustering based on customer characteristics and spending behaviour. Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset and visualize the clusters effectively.

---

## Dataset Link

Mall Customer Segmentation Dataset

https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Methodology

1. Load the dataset using Pandas.
2. Display the first five records, dataset information, and summary statistics.
3. Check for missing values.
4. Remove the `CustomerID` column.
5. Encode the `Gender` column.
6. Standardize the numerical features using `StandardScaler`.
7. Apply the Elbow Method to determine the optimal number of clusters.
8. Train a K-Means Clustering model.
9. Assign cluster labels to each customer.
10. Apply Principal Component Analysis (PCA) to reduce the dataset to two dimensions.
11. Visualize the Elbow Curve, customer clusters, and PCA projection.

---

## Results

The Elbow Method identified five clusters as an appropriate choice for customer segmentation. K-Means successfully grouped customers with similar spending behaviour, and PCA effectively visualized these clusters in two dimensions, making them easier to interpret.

---

## Conclusion

This project demonstrates the use of K-Means Clustering and Principal Component Analysis for customer segmentation. The resulting customer groups can help businesses improve marketing strategies, customer targeting, and promotional campaigns. PCA provides a simple and effective way to visualize high-dimensional customer data while preserving important information.
