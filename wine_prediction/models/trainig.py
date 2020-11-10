import json
import pickle

import click
from loguru import logger
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split


def load_data():
    wine_dataset = load_wine()
    logger.info("Load datasets...")
    X_train, X_test, y_train, y_test = train_test_split(
        wine_dataset["data"], wine_dataset["target"], test_size=0.2, random_state=42
    )
    logger.info("Split datasets...")
    return X_train, y_train, X_test, y_test


@click.command()
@click.argument("model_path", type=click.File("wb", lazy=True))
@click.argument("metrics_path", type=click.File("w", lazy=True))
def train(model_path, metrics_path):
    X_train, y_train, X_test, y_test, = load_data()
    model = RandomForestClassifier(n_estimators=400)
    logger.info("Fitting model..")
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics = {
        "test_precision": precision_score(y_test, y_test_pred, average="macro"),
        "train_precision": precision_score(y_train, y_train_pred, average="macro"),
    }
    logger.info("Test precision {}", metrics["test_precision"])
    pickle.dump(model, model_path)
    json.dump(metrics, metrics_path)


if __name__ == "__main__":
    train()
