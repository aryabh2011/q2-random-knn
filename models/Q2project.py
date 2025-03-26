import random
import math
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def generate_datasets():
    datasets = []
    configurations = [
        {"n_samples": 100, "n_features": 5, "n_clusters": 3, "cluster_std": 1.0},
        {"n_samples": 200, "n_features": 5, "n_clusters": 4, "cluster_std": 1.5},
        {"n_samples": 300, "n_features": 6, "n_clusters": 5, "cluster_std": 0.8},
        {"n_samples": 150, "n_features": 4, "n_clusters": 2, "cluster_std": 2.0},
        {"n_samples": 250, "n_features": 3, "n_clusters": 3, "cluster_std": 0.5},
        {"n_samples": 400, "n_features": 7, "n_clusters": 6, "cluster_std": 1.2},
        {"n_samples": 180, "n_features": 4, "n_clusters": 2, "cluster_std": 1.8},
        {"n_samples": 220, "n_features": 5, "n_clusters": 4, "cluster_std": 0.7},
        {"n_samples": 350, "n_features": 6, "n_clusters": 5, "cluster_std": 1.1},
        {"n_samples": 100, "n_features": 3, "n_clusters": 2, "cluster_std": 1.5},
    ]
    for config in configurations:
        X, y = make_blobs(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            centers=config["n_clusters"],
            cluster_std=config["cluster_std"],
            random_state=42,
        )
        datasets.append((X, y, config))
    return datasets

def random_knn(data, n_models=3):
    feature_names = [f"feature_{i}" for i in range(len(data[0]) - 1)] + ["target"]
    models = []
    for _ in range(n_models):
        chosen_features = random.sample(feature_names[:-1], min(3, len(feature_names) - 1))
        chosen_features.append("target")
        indices = [feature_names.index(f) for f in chosen_features]
        n_instances = len(data)
        k = int(math.sqrt(n_instances))
        sampled_data = random.choices(data, k=k)
        new_data = [[row[i] for i in indices] for row in sampled_data]
        models.append((new_data, indices[:-1]))
    return models

def knn_predict(models, test_point):
    predictions = []
    for model_data, feature_indices in models:
        reduced_test_point = [test_point[i] for i in feature_indices]
        distances = []
        for row in model_data:
            dist = sum((x - y) ** 2 for x, y in zip(reduced_test_point, row[:-1]))
            distances.append((row[-1], dist))
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = [dist[0] for dist in distances[:3]]
        votes = {}
        for n in nearest_neighbors:
            if n not in votes:
                votes[n] = 0
            votes[n] += 1
        predictions.append(max(votes, key=votes.get))
    final_votes = {}
    for p in predictions:
        if p not in final_votes:
            final_votes[p] = 0
        final_votes[p] += 1
    return max(final_votes, key=final_votes.get)

def random_forest(data, feature_names, n):
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    X = [row[:-1] for row in data]  
    y = [row[-1] for row in data]   
    rf.fit(X, y) 
    return rf

def rf_predict(model, test_point):
    return model.predict([test_point[:-1]])[0]

def compare_algorithms():
    datasets = generate_datasets()
    for X, y, config in datasets:
        data = [list(row) + [target] for row, target in zip(X, y)]
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
        
        knn_models = random_knn(train_data, n_models=5)
        knn_predictions = [knn_predict(knn_models, test) for test in test_data]
        knn_accuracy = sum(1 for pred, true in zip(knn_predictions, [row[-1] for row in test_data]) if pred == true) / len(test_data)
        
        rf_model = random_forest(train_data, feature_names=[f"feature_{i}" for i in range(len(train_data[0]) - 1)] + ["target"], n=5)
        rf_predictions = [rf_predict(rf_model, test) for test in test_data]
        rf_accuracy = sum(1 for pred, true in zip(rf_predictions, [row[-1] for row in test_data]) if pred == true) / len(test_data)
        
        print(f"Dataset config: {config}")
        print(f"Random kNN Accuracy: {knn_accuracy:.2f}")
        print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
        print("-" * 40)

if __name__ == "__main__":
    compare_algorithms()
