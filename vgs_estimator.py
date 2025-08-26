import logging
import os
import sys

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

df = pd.read_csv("data/video_game_sales.csv")

df_cleaned = df.dropna()
logger.info(f"Rows after removing nulls: {len(df_cleaned)}")

# Convert Year back to integer after removing nulls
df_cleaned["Year"] = df_cleaned["Year"].astype(int)

features = ["Platform", "Year", "Genre", "Publisher"]
target = "NA_Sales"

# Rename rarely occuring publisher below threshold to 'Other'
publisher_threshold = 20
publisher_counts = df_cleaned["Publisher"].value_counts()
df_cleaned["Publisher"] = df_cleaned["Publisher"].apply(
    lambda x: x if publisher_counts[x] >= publisher_threshold else "Other"
)

categorical_features = ["Platform", "Genre", "Publisher"]
numerical_features = ["Year"]

X = df_cleaned[features]
y = df_cleaned[target]

logger.info(f"Training features: {features}")
logger.info(f"Target variable: {target}")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop=None, handle_unknown="ignore"),
            categorical_features,
        ),
        ("num", "passthrough", numerical_features),
    ]
)

pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])

param_grid = {"regressor__alpha": [0.01, 0.1, 1, 10, 100]}

# Set up k-fold cross-validation
cv_settings = {"n_splits": 5, "shuffle": True, "random_state": 42}
cv = KFold(
    n_splits=cv_settings["n_splits"],
    shuffle=cv_settings["shuffle"],
    random_state=cv_settings["random_state"],
)

grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring="neg_mean_squared_error"
)

# Set up MLflow tracking and experiment
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
mlflow.set_experiment("MLflow Quickstart")

logger.info("Starting MLflow run...")

with mlflow.start_run():

    logger.info("Starting grid search cross-validation...")
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    logger.info(f"Best hyperparameters: {best_params}")
    mlflow.log_params(best_params)

    mlflow.log_param("cv_n_splits", cv_settings["n_splits"])
    mlflow.log_param("cv_shuffle", cv_settings["shuffle"])
    mlflow.log_param("cv_random_state", cv_settings["random_state"])

    logger.info("Evaluating best estimator with cross-validation...")
    mse_scores = -cross_val_score(
        grid_search.best_estimator_, X, y, cv=cv, scoring="neg_mean_squared_error"
    )

    mean_mse = mse_scores.mean()
    std_mse = mse_scores.std()

    logger.info(f"Cross-validation MSE scores: {mse_scores}")
    logger.info(f"Mean CV MSE: {mean_mse:.6f}")
    logger.info(f"Std CV MSE: {std_mse:.6f}")

    mlflow.log_metric("mean_cv_mse", mean_mse)
    mlflow.log_metric("std_mse", std_mse)

    # Optional, log each fold's MSE
    for i, score in enumerate(mse_scores):
        mlflow.log_metric(f"fold_{i+1}_mse", score)

    logger.info("Saving grid search results...")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df["mean_mse"] = -results_df["mean_test_score"]
    results_df["std_mse"] = results_df["std_test_score"]
    results_path = "mini_mlops_project/data/grid_search_results.csv"
    results_df.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path)

    logger.info("Logging model to MLflow...")
    signature = infer_signature(X, grid_search.best_estimator_.predict(X))
    input_example = X.iloc[[0]].to_dict(orient="records")[0]

    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        "ridge_pipeline_model",
        signature=signature,
        input_example=input_example,
        registered_model_name="vgs_ridge_pipeline_model",
    )

    # Log feature information for reference
    mlflow.log_text(str(categorical_features), "categorical_features.txt")
    mlflow.log_text(str(numerical_features), "numerical_features.txt")

    logger.info("MLflow run completed successfully!")
