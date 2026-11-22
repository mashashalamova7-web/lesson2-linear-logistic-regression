"""
Модуль модификации моделей: линейная и многоклассовая логистическая регрессия.
Реализует регуляризацию, early stopping, расширенные метрики и визуализацию.
"""

import logging
import unittest
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ЛОГИРОВАНИЕ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ModelModification")



# МОДУЛЬНОСТЬ: Классы


class EarlyStopping:
    """Механизм ранней остановки обучения при отсутствии улучшений на валидации."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, path: str = "models/checkpoint.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model: nn.Module) -> None:
        torch.save(model.state_dict(), self.path)


class LinearRegressionMod(nn.Module):
    """Линейная регрессия с поддержкой L1 и L2 регуляризации."""

    def __init__(self, in_features: int, l1_lambda: float = 0.0, l2_lambda: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Возвращаем 1D тензор (batch_size,) для удобства сравнения с y_batch
        return self.linear(x).squeeze(-1)

    def compute_regularization_loss(self) -> torch.Tensor:
        """Вычисление штрафов L1 (Lasso) и L2 (Ridge) по весам модели."""
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return self.l1_lambda * l1_norm + self.l2_lambda * l2_norm


class MulticlassLogisticRegression(nn.Module):
    """Многоклассовая (и бинарная) логистическая регрессия."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)



# Функции обучения и оценки

def plot_history(history: Dict[str, List[float]], save_path: str):
    """3. ВИЗУАЛИЗАЦИЯ: График динамики потерь."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Val Loss", marker='s')
    plt.title("Динамика обучения (Linear Regression)")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"График обучения сохранен: {save_path}")


def train_linear_regression(
    model: LinearRegressionMod,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.01,
    patience: int = 7,
    save_path: str = "models/best_linreg.pt",
) -> Dict[str, List[float]]:
    """Цикл обучения линейной регрессии с Early Stopping и регуляризацией."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, path=save_path)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch) # .squeeze(-1) уже внутри forward
            loss = criterion(preds, y_batch) + model.compute_regularization_loss()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch) # Без регуляризации на валидации
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping сработал на эпохе {epoch+1}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return history


def train_logistic_regression(
    model: MulticlassLogisticRegression,
    train_loader: DataLoader,
    val_loader: DataLoader, # Добавлен val_loader для консистентности
    epochs: int = 50,
    lr: float = 0.01,
) -> Dict[str, List[float]]:
    """Цикл обучения многоклассовой логистической регрессии."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        history["train_loss"].append(train_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch.long())
                val_loss += loss.item() * X_batch.size(0)
        history["val_loss"].append(val_loss / len(val_loader.dataset))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs} | Train CE: {history['train_loss'][-1]:.4f} | Val CE: {history['val_loss'][-1]:.4f}")

    return history


def evaluate_classification(
    model: MulticlassLogisticRegression,
    data_loader: DataLoader,
    class_names: Optional[List[str]] = None,
    plot_path: Optional[str] = "plots/confusion_matrix.png",
) -> Dict[str, float]:
    """Расчет Precision, Recall, F1, ROC-AUC и сохранение Confusion Matrix."""
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.numpy())
            all_preds.append(preds.numpy())
            all_targets.append(y_batch.numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    avg_mode = "binary" if model.num_classes == 2 else "weighted"

    metrics = {
        "precision": precision_score(y_true, y_pred, average=avg_mode, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg_mode, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=avg_mode, zero_division=0),
    }

    try:
        if model.num_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except ValueError as e:
        logger.warning(f"Не удалось рассчитать ROC-AUC: {e}")
        metrics["roc_auc"] = 0.0

    logger.info(f"Метрики классификации: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
                f"F1={metrics['f1_score']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}")

    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names or "auto", yticklabels=class_names or "auto")
        plt.xlabel("Предсказанный класс")
        plt.ylabel("Истинный класс")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix сохранена в {plot_path}")

    return metrics



# ТЕСТИРОВАНИЕ (Unit Tests)
class TestModelModification(unittest.TestCase):
    def test_linear_regression_regularization(self):
        model = LinearRegressionMod(in_features=4, l1_lambda=0.1, l2_lambda=0.05)
        reg_loss = model.compute_regularization_loss()
        self.assertGreater(reg_loss.item(), 0.0)

    def test_early_stopping(self):
        es = EarlyStopping(patience=2, path="models/test_es.pt")
        model = LinearRegressionMod(2)
        es(1.0, model)
        self.assertFalse(es.early_stop)
        es(1.1, model) # Ухудшение
        self.assertFalse(es.early_stop)
        es(1.2, model) # Ухудшение 2-й раз -> срабатывание
        self.assertTrue(es.early_stop)

    def test_multiclass_logistic_pipeline(self):
        X = torch.randn(50, 5)
        y = torch.randint(0, 3, (50,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)
        
        model = MulticlassLogisticRegression(in_features=5, num_classes=3)
        history = train_logistic_regression(model, loader, loader, epochs=5)
        
        self.assertEqual(len(history["train_loss"]), 5)
        metrics = evaluate_classification(model, loader, plot_path=None)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)



# Для проверки работы всего пайплайна

def run_demo():
    """Демонстрация работы модифицированных моделей на синтетических данных."""
    logger.info("ЗАПУСК ДЕМО-ПИПАЙПЛАЙНА")

    # 1. Демонстрация Линейной Регрессии
    logger.info("\n--- 1. Линейная регрессия с L1/L2 и Early Stopping ---")
    X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                            torch.tensor(y_train, dtype=torch.float32)), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                          torch.tensor(y_val, dtype=torch.float32)), batch_size=16)

    lin_model = LinearRegressionMod(in_features=5, l1_lambda=0.01, l2_lambda=0.01)
    history_reg = train_linear_regression(lin_model, train_loader, val_loader, epochs=30, patience=5)
    plot_history(history_reg, "plots/linreg_history.png")

    # 2. Демонстрация Многоклассовой Логистической Регрессии
    logger.info("\n 2. Многоклассовая логистическая регрессия ")
    X_clf, y_clf = make_classification(n_samples=300, n_features=6, n_classes=3, n_informative=4, random_state=42)
    X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    train_loader_c = DataLoader(TensorDataset(torch.tensor(X_train_c, dtype=torch.float32), 
                                              torch.tensor(y_train_c, dtype=torch.long)), batch_size=16, shuffle=True)
    val_loader_c = DataLoader(TensorDataset(torch.tensor(X_val_c, dtype=torch.float32), 
                                            torch.tensor(y_val_c, dtype=torch.long)), batch_size=16)

    log_model = MulticlassLogisticRegression(in_features=6, num_classes=3)
    train_logistic_regression(log_model, train_loader_c, val_loader_c, epochs=30)
    
    evaluate_classification(
        log_model, val_loader_c, 
        class_names=["Class 0", "Class 1", "Class 2"], 
        plot_path="plots/multiclass_cm.png"
    )
    logger.info("Демо-пайплайн успешно завершен. Проверьте папку 'plots/'.\n")


if __name__ == "__main__":
    # Сначала запускаем демонстрацию, чтобы показать, что код работает
    run_demo()
    
    # Затем запускаем unit-тесты для проверки корректности логики
    logger.info("Запуск unit-тестов")
    unittest.main(argv=["first-arg-is-ignored"], exit=False, verbosity=2)