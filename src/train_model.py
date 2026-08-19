import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from src.preprocessing import load_and_clean_data, split_data, get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    trained_pipelines = {}

    print("=" * 60)
    print("Training and Evaluating Models...")
    print("=" * 60)

    for name, model in candidate_models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

    comparison_df = pd.DataFrame(results).sort_values(
        by='F1-Score',
        ascending=False
    ).reset_index(drop=True)

    return comparison_df, trained_pipelines


def tune_best_model(X_train, y_train, preprocessor):
    print("\nRunning Hyperparameter Tuning for Gradient Boosting...")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5],
        'classifier__subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_


def extract_feature_importance(best_pipeline):
    preprocessor = best_pipeline.named_steps['preprocessor']
    classifier = best_pipeline.named_steps['classifier']

    cat_cols = preprocessor.named_transformers_['cat'].named_steps[
        'onehot'
    ].get_feature_names_out(CATEGORICAL_FEATURES)

    all_feature_names = list(NUMERIC_FEATURES) + list(cat_cols)

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_

        fi_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        })

        fi_df = fi_df.sort_values(
            by='Importance',
            ascending=False
        ).reset_index(drop=True)

        print("\nTop 10 Important Features:")
        print(fi_df.head(10))

        return fi_df

    return None


def main():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )

    data_path = os.path.join(
        project_root,
        'data',
        'customer_churn.csv'
    )

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(
            f"Error: Dataset not found at {data_path}. "
            "Please place your CSV file there."
        )
        return

    df = load_and_clean_data(data_path)

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = get_preprocessor()

    comparison_df, _ = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    print("\n--- Model Performance Comparison ---")
    print(comparison_df.to_string(index=False))

    best_model = tune_best_model(
        X_train,
        y_train,
        preprocessor
    )

    y_pred = best_model.predict(X_test)

    print("\n--- Final Model Evaluation on Test Set ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=['No Churn', 'Churn']
        )
    )

    extract_feature_importance(best_model)

    model_file = os.path.join(
        model_dir,
        'customer_churn_model.pkl'
    )

    joblib.dump(best_model, model_file)

    print(f"\nModel pipeline successfully saved to {model_file}")


if __name__ == '__main__':
    main()