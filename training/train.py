import logging
import os

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"Using MLflow tracking URI: {tracking_uri}")

    try:
        mlflow.set_experiment("Video Game Sales Prediction")
        logger.info("Successfully connected to MLflow server")
    except Exception as e:
        logger.error(f"Failed to connect to MLflow: {e}")
        return

    try:
        df = pd.read_csv("data/video_game_sales.csv")
        logger.info(f"Loaded {len(df)} rows of data")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    df_cleaned = df.dropna()
    logger.info(f"Rows after removing nulls: {len(df_cleaned)}")

    df_cleaned["Year"] = df_cleaned["Year"].astype(int)

    features = ["Platform", "Year", "Genre", "Publisher"]
    target = "NA_Sales"

    # Map rarely occuring publishers below threshold to "Other"
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
    logger.info(f"Training data shape: {X.shape}")

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

    cv_settings = {"n_splits": 5, "shuffle": True, "random_state": 42}
    cv = KFold(
        n_splits=cv_settings["n_splits"],
        shuffle=cv_settings["shuffle"],
        random_state=cv_settings["random_state"],
    )

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="neg_mean_squared_error"
    )

    logger.info("Starting MLflow run...")
    with mlflow.start_run(run_name="video_game_sales_training"):

        mlflow.log_params(cv_settings)
        mlflow.log_param("publisher_threshold", publisher_threshold)

        logger.info("Starting grid search cross-validation...")
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        logger.info(f"Best hyperparameters: {best_params}")
        mlflow.log_params(best_params)

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
        mlflow.log_metric("std_cv_mse", std_mse)
        mlflow.log_metric("best_cv_score", -grid_search.best_score_)

        for i, score in enumerate(mse_scores):
            mlflow.log_metric(f"fold_{i+1}_mse", score)

        signature = infer_signature(X, grid_search.best_estimator_.predict(X))
        input_example = X.iloc[:5]  # Use first 5 rows as example

        logger.info("Logging model to MLflow...")
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name="video_game_sales_model",
            )

            logger.info(f"Model registered successfully!")
            logger.info(f"Model URI: {model_info.model_uri}")

            mlflow.log_dict(
                {"categorical_features": categorical_features},
                "categorical_features.json",
            )
            mlflow.log_dict(
                {"numerical_features": numerical_features}, "numerical_features.json"
            )

            performance_summary = {
                "mean_cv_mse": float(mean_mse),
                "std_cv_mse": float(std_mse),
                "best_alpha": float(best_params["regressor__alpha"]),
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
            mlflow.log_dict(performance_summary, "model_performance.json")

        except Exception as e:
            logger.error(f"Failed to log/register model: {e}")
            raise

        logger.info("Training completed successfully!")

        run = mlflow.active_run()
        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Experiment ID: {run.info.experiment_id}")


if __name__ == "__main__":
    main()
