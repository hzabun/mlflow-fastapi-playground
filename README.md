# MLflow FastAPI Playground

> A mini MLOps sandbox to experiment with MLflow (for model logging), FastAPI (for model serving), and Prometheus (for metrics scraping).

---

## Table of Contents

- [MLflow FastAPI Playground](#mlflow-fastapi-playground)
  - [Table of Contents](#table-of-contents)
  - [Features \& Roadmap](#features--roadmap)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Setup \& Usage](#setup--usage)
  - [Known issues](#known-issues)
  - [References](#references)

---

## Features & Roadmap

- [x] Train model and add MLflow logging
- [x] Save model in MLflow registry
- [x] Expose `/predict` endpoint via FastAPI
- [x] Format code (remove unnecessary comments, use logger)
- [ ] Package app in Docker
- [ ] Expose metrics (latency, request count) via Prometheus
- [ ] Visualize metrics in Grafana

---

## Project Structure

```
├── data/
│   └── video_game_sales.csv
├── fastapi-app/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── mlflow-server/
│   └── Dockerfile
├── mlflow_data/
│   └── artifacts/
├── training/
│   ├── train.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yaml
└── README.md
```

---

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/)
- Download the [Video Game Sales dataset](https://www.kaggle.com/datasets/gregorut/videogamesales) from Kaggle

---

## Setup & Usage

1. **Download the dataset**

   - Download `vgsales.csv` from Kaggle and place it in the `./data` directory as `video_game_sales.csv`.

2. **Build and run the stack**

   ```sh
   docker compose up --build
   ```

3. **Access the services**
   - FastAPI: [http://localhost:8000](http://localhost:8000)
   - MLflow UI: [http://localhost:5000](http://localhost:5000)
   - Prometheus: [http://localhost:9090](http://localhost:9090) _(if enabled)_
   - Grafana: [http://localhost:3000](http://localhost:3000) _(if enabled)_

---

## Known issues

- There's currently a bug when the FastAPI server starts, as it cannot find the registered model
- This will be fixed in the near future

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
