import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Separate target and features
    y = data.iloc[:, 0].values  # Evaluation (target)
    X = data.iloc[:, 1:].values  # Features (factors)
    return X, y

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Find the optimal project (highest predicted evaluation in the training set)
    y_pred_train = model.predict(X_train)
    optimal_idx = np.argmax(y_pred_train)
    optimal_project = X_train[optimal_idx]
    optimal_evaluation = y_pred_train[optimal_idx]
    
    print(f"Optimal Project Evaluation: {optimal_evaluation:.2f}")
    return model, optimal_project, optimal_evaluation

# Compare new project to optimal project
def compare_to_optimal(new_project, optimal_project, feature_importances):
    # Euclidean Distance
    euclidean_distance = np.linalg.norm(new_project - optimal_project)
    
    # Weighted Distance
    weighted_distance = np.sqrt(np.sum(feature_importances * (new_project - optimal_project) ** 2))
    
    # Cosine Similarity
    cos_sim = cosine_similarity([new_project], [optimal_project])[0, 0]
    
    return euclidean_distance, weighted_distance, cos_sim

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    file_path = "data.csv"  # Replace with your file path
    data = load_data(file_path)
    X, y = preprocess_data(data)

    # Normalize features (optional but recommended for distance calculations)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train the model and get the optimal project
    model, optimal_project, optimal_evaluation = train_model(X, y)

    # Feature importances (normalized for weighting distances)
    feature_importances = model.feature_importances_
    feature_importances /= feature_importances.sum()

    # Define a new project for comparison
    new_project_raw = np.random.rand(1, X.shape[1])  # Replace with real new project data
    new_project = scaler.transform(new_project_raw)  # Scale the new project using the same scaler

    # Compare new project to the optimal project
    euclidean_dist, weighted_dist, cos_sim = compare_to_optimal(new_project[0], optimal_project, feature_importances)

    print(f"Euclidean Distance: {euclidean_dist:.4f}")
    print(f"Weighted Distance: {weighted_dist:.4f}")
    print(f"Cosine Similarity: {cos_sim:.4f}")
