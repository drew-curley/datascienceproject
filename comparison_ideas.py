import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example feature vectors
x_optimal = np.array([0.8, 1.5, 2.0, 1.2, 0.9])  # Optimal project
x_new = np.array([0.7, 1.4, 1.8, 1.3, 1.0])       # New project
weights = np.array([0.2, 0.3, 0.1, 0.25, 0.15])   # Feature importance

# 1. Euclidean Distance
distance = np.linalg.norm(x_new - x_optimal)
print(f"Euclidean Distance: {distance:.4f}")

# 2. Weighted Distance
weighted_distance = np.sqrt(np.sum(weights * (x_new - x_optimal) ** 2))
print(f"Weighted Distance: {weighted_distance:.4f}")

# 3. Cosine Similarity
cos_sim = cosine_similarity([x_new], [x_optimal])[0, 0]
print(f"Cosine Similarity: {cos_sim:.4f}")

# 4. Relative Evaluation Score
# Assume model predictions for evaluation scores
y_optimal = 4.8
y_new = 4.2
delta_evaluation = abs(y_optimal - y_new)
print(f"Relative Evaluation Score Difference: {delta_evaluation:.4f}")
