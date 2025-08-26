# mlflow-fastapi-playground

Mini mlops sandbox to test and play around with mlflow for logging models, fastapi for serving models, and prometheus for scraping metrics.

## Roadmap

- [x] Take model and add MLflow logging
- [x] Save model in MLflow registry
- [x] Expose simple `/predict` endpoing via FastAPI
- [x] Format code (remove unnecessary comments, replace `print` with `logger`)
- [ ] Package app in Docker
- [ ] Expose metrics (latency, request count) via prometheus
- [ ] Visualize metrics in Grafana
