import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import sys
import json
import wandb
import requests
import os
import optuna
from optuna.samplers import TPESampler

def parse_args():
    parser = argparse.ArgumentParser(description='XGBoost Classifier for Graph Embeddings')
    parser.add_argument("--dataset", type=str, default="MulDiGraph", choices=["B4E", "MulDiGraph", "TXNT"],
                        help="Dataset to use (default: B4E)")
    parser.add_argument("--embedding_file", type=str, default=None,
                        help="Path to embedding file (default: ../data/dataset/Data_after_FE/graph_emb.txt)")
    parser.add_argument("--tag_file", type=str, default=None,
                        help="Path to tag file (default: ../data/processed_data/account_tags.pkl)")
    parser.add_argument("--feature_select", type=str, default="Selected Features",
                        help="Feature selection method (default: Selected Features)")
    parser.add_argument("--ratio", type=str, default="5:5",
                            help="Ratio of normal accounts to phisher accounts (default: 5:5)")
    parser.add_argument("--project", type=str, default="SGDiT SGCR",
                        help="WandB project name (default: Feature Engineer GMM)")
    parser.add_argument("--n_estimators", type=int, default=400,
                        help="Number of gradient boosted trees (default: 100)")
    parser.add_argument("--max_depth", type=int, default=6,
                        help="Maximum tree depth (default: 6)")
    parser.add_argument("--learning_rate", type=float, default=0.3,
                        help="Boosting learning rate (default: 0.3)")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Subsample ratio of training instances (default: 1.0)")
    parser.add_argument("--colsample_bytree", type=float, default=1.0,
                        help="Subsample ratio of columns when constructing each tree (default: 1.0)")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable hyperparameter optimization with Optuna")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of optimization trials (default: 50)")
    args = parser.parse_args()

    # Set default file paths based on dataset if not provided
    if args.embedding_file is None:
        args.embedding_file = f"../data/dataset/Data_after_FE/graph_emb.txt"
    if args.tag_file is None:
        args.tag_file = f"../data/processed_data/account_tags.pkl"

    return args

def save_misclassified_indices(misclassified_indices, y_true, y_pred, original_indices, dataset_name, split_name, threshold):
    """
    Lưu các indices bị dự đoán sai vào file JSON
    """
    filename = f"misclassified_{dataset_name}_{split_name}_threshold_{threshold:.1f}_xgboost.json"
    
    # Chuyển đổi boolean mask thành indices nếu cần
    if misclassified_indices.dtype == bool:
        misclassified_pos = np.where(misclassified_indices)[0]
    else:
        misclassified_pos = misclassified_indices
    
    result = {
        "dataset": dataset_name,
        "split": split_name, 
        "threshold": float(threshold),
        "total_misclassified": int(len(misclassified_pos)),
        "misclassified_indices": [int(x) for x in misclassified_pos.tolist()],
        "original_indices": [int(original_indices[i]) for i in misclassified_pos],
        "true_labels": [int(y_true[i]) for i in misclassified_pos],
        "predicted_labels": [int(y_pred[i]) for i in misclassified_pos]
    }
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved {len(misclassified_pos)} misclassified indices to {filename}")

def get_max_index(tag_file):
    with open(tag_file, 'rb') as f:
        tags = pickle.load(f)
    return len(tags) - 1

def get_embedding_dim(embedding_file):
    with open(embedding_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                return len(parts) - 1
    return 0

def load_embeddings(file_path, max_index, embedding_dim):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            index = int(parts[0])
            embedding = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if len(embedding) == embedding_dim:
                embeddings[index] = embedding
    print(f"Number of embeddings loaded: {len(embeddings)}", end='\r')
    return embeddings

def load_tags(file_path, max_index):
    with open(file_path, 'rb') as f:
        tags = pickle.load(f)
    print(f"Number of tags loaded: {len(tags)}", end='\r')
    filtered_tags = {k: v for k, v in tags.items()}
    return filtered_tags

def prepare_data(embeddings, tags):
    valid_indices = sorted(set(embeddings.keys()) & set(tags.keys()))

    X = np.zeros((len(valid_indices), len(list(embeddings.values())[0])), dtype=np.float32)
    y = np.zeros(len(valid_indices), dtype=np.int32)

    for i, idx in enumerate(valid_indices):
        X[i] = embeddings[idx]
        y[i] = tags[idx]

    return X, y, valid_indices

def optimize_hyperparameters(X_train, y_train, n_trials=50):
    """
    Tối ưu hóa hyperparameters bằng Optuna
    """
    def objective(trial):
        # Định nghĩa search space cho các hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        # Tạo model với hyperparameters được suggest
        model = XGBClassifier(**params)
        
        # Sử dụng cross-validation để đánh giá
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        return scores.mean()
    
    # Tạo study và optimize
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nOptimization completed!")
    print(f"Best F1 score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return study.best_params

def train_and_evaluate(X, y, valid_indices, dataset_name, args):
    # Chia dữ liệu: 80% train, 20% test
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, valid_indices, test_size=0.2, random_state=42)

    print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Tối ưu hóa hyperparameters nếu được yêu cầu
    if args.optimize:
        best_params = optimize_hyperparameters(X_train, y_train, args.n_trials)
        
        # Khởi tạo model với best parameters
        model = XGBClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    else:
        # Sử dụng parameters mặc định hoặc từ command line
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
    # Training
    model.fit(X_train, y_train)

    def evaluate_split(X_split, y_split, split_indices, split_name):
        y_prob = model.predict_proba(X_split)[:, 1]
        thresholds = np.arange(0.1, 1.0, 0.1)
        target_threshold = 0.5
        precision_03 = 0.0
        recall_03 = 0.0
        f1_03 = 0.0

        print(f"\n=== {split_name} Split Results ===")
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            print(f"\nThreshold: {threshold:.2f}")
            print(classification_report(y_split, y_pred, zero_division=0, digits=4))

            if abs(threshold - target_threshold) < 1e-5:
                precision_03 = precision_score(y_split, y_pred, labels=[1], average=None, zero_division=0)[0] if 1 in y_split else 0.0
                recall_03 = recall_score(y_split, y_pred, labels=[1], average=None, zero_division=0)[0] if 1 in y_split else 0.0
                f1_03 = f1_score(y_split, y_pred, labels=[1], average=None, zero_division=0)[0] if 1 in y_split else 0.0
                
                # Tìm các indices bị dự đoán sai
                misclassified_mask = (y_split != y_pred)
                misclassified_local_indices = np.where(misclassified_mask)[0]
                misclassified_original_indices = np.array(split_indices)[misclassified_local_indices]
                
                # Lưu các indices bị dự đoán sai
                save_misclassified_indices(
                    misclassified_mask, y_split, y_pred, 
                    split_indices, dataset_name, split_name, target_threshold
                )
                
                # Confusion Matrix - numpy array
                cm = confusion_matrix(y_split, y_pred)
                cm_array = np.array(cm)
                print(f"\n=== Confusion Matrix (Threshold 0.5) - {split_name} ===")
                print(f"Numpy Array:\n{cm_array}")
                
                if cm_array.shape == (2, 2):
                    tn, fp, fn, tp = cm_array.ravel()
                    print(f"True Negatives (TN): {tn}")
                    print(f"False Positives (FP): {fp}")
                    print(f"False Negatives (FN): {fn}")
                    print(f"True Positives (TP): {tp}")
                    print(f"Total samples: {tn + fp + fn + tp}")
                
                print(f"\nThreshold 0.5 - Label 1 Results:")
                print(f"Label 1 - Precision: {precision_03:.4f}")
                print(f"Label 1 - Recall: {recall_03:.4f}")
                print(f"Label 1 - F1 Score: {f1_03:.4f}")
        
        return precision_03, recall_03, f1_03

    # Đánh giá trên test set  
    test_precision, test_recall, test_f1 = evaluate_split(X_test, y_test, indices_test, "Test")

    # Feature importance
    feature_importance = model.feature_importances_
    print(f"\nTop 10 Most Important Features:")
    important_features = np.argsort(feature_importance)[-10:][::-1]
    for i, feat_idx in enumerate(important_features):
        print(f"  {i+1}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")

    return test_precision, test_recall, test_f1

def main():
    args = parse_args()
    embedding_file = args.embedding_file
    tag_file = args.tag_file

    max_index = get_max_index(tag_file)
    print(f"Max index from tags: {max_index}")
    embedding_dim = get_embedding_dim(embedding_file)
    print(f"Embedding dimension: {embedding_dim}")

    embeddings = load_embeddings(embedding_file, max_index, embedding_dim)
    tags = load_tags(tag_file, max_index)
    print(f"Number of tags loaded: {len(tags)}")

    X, y, valid_indices = prepare_data(embeddings, tags)

    if args.optimize:
        print(f"\nOptuna Optimization Settings:")
        print(f"  n_trials: {args.n_trials}")
        print(f"  Optimization enabled: True")
    else:
        print(f"\nXGBoost Parameters (Manual):")
        print(f"  n_estimators: {args.n_estimators}")
        print(f"  max_depth: {args.max_depth}")
        print(f"  learning_rate: {args.learning_rate}")
        print(f"  subsample: {args.subsample}")
        print(f"  colsample_bytree: {args.colsample_bytree}")

    test_precision, test_recall, test_f1 = train_and_evaluate(X, y, valid_indices, args.dataset, args)

    print(f"\n=== Final Metrics for Label 1 (Threshold 0.5) ===")
    print(f"Test Set:")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")

if __name__ == '__main__':
    main()