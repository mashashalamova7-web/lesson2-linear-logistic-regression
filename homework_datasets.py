"""
Модуль для обучения и оценки моделей на табличных данных.
Реализует кастомный Dataset, циклы обучения, визуализацию и unit-тесты.
"""

import logging
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

# 5. ЛОГИРОВАНИЕ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TabularMLPipeline")

# 1. МОДУЛЬНОСТЬ: Классы и функции

class TabularCSVDataset(Dataset):
    """
    Кастомный PyTorch Dataset для табличных данных из CSV.
    
    Поддерживает числовые, категориальные и бинарные признаки.
    Позволяет сохранить состояние препроцессоров (scaler, encoder) 
    для корректной обработки валидационных/тестовых данных (режим fit=False).
    
    Attributes:
        X (torch.Tensor): Тензор признаков формы (N, n_features).
        y (torch.Tensor): Тензор целевой переменной формы (N,).
        scaler (StandardScaler): Обученный скейлер для числовых признаков.
        encoder (OneHotEncoder): Обученный энкодер для категориальных признаков.
    """

    def __init__(
        self,
        csv_path: str,
        target_col: str,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        bin_cols: Optional[List[str]] = None,
        task: str = "regression",
        scaler: Optional[StandardScaler] = None,
        encoder: Optional[OneHotEncoder] = None,
        fit: bool = True,
    ):
        logger.debug(f"Инициализация Dataset. Путь: {csv_path}, fit={fit}")
        self.df = pd.read_csv(csv_path)
        self.target_col = target_col
        self.task = task

        # Автоопределение типов колонок, если они не переданы явно
        if not (num_cols or cat_cols or bin_cols):
            self.num_cols, self.cat_cols, self.bin_cols = [], [], []
            feature_cols = [c for c in self.df.columns if c != target_col]
            for col in feature_cols:
                unique_vals = set(self.df[col].dropna().unique())
                if self.df[col].dtype == "object" or str(self.df[col].dtype) == "category":
                    self.cat_cols.append(col)
                elif unique_vals.issubset({0, 1, True, False}):
                    self.bin_cols.append(col)
                else:
                    self.num_cols.append(col)
        else:
            self.num_cols, self.cat_cols, self.bin_cols = num_cols or [], cat_cols or [], bin_cols or []

        features_list = []

        # 1. Числовые признаки
        if self.num_cols:
            num_data = self.df[self.num_cols].fillna(self.df[self.num_cols].median())
            if fit:
                self.scaler = StandardScaler()
                num_scaled = self.scaler.fit_transform(num_data)
            else:
                self.scaler = scaler
                num_scaled = self.scaler.transform(num_data)
            features_list.append(num_scaled)
        else:
            self.scaler = None

        # 2. Категориальные признаки
        if self.cat_cols:
            cat_data = self.df[self.cat_cols].fillna("MISSING").astype(str)
            if fit:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                cat_encoded = self.encoder.fit_transform(cat_data)
            else:
                self.encoder = encoder
                cat_encoded = self.encoder.transform(cat_data)
            features_list.append(cat_encoded)
        else:
            self.encoder = None

        # 3. Бинарные признаки
        if self.bin_cols:
            bin_data = self.df[self.bin_cols].fillna(0).astype(np.float32).values
            features_list.append(bin_data)

        self.X = torch.tensor(np.hstack(features_list).astype(np.float32), dtype=torch.float32)
        
        y_raw = self.df[target_col].values
        self.y = torch.tensor(y_raw, dtype=torch.float32 if task == "regression" else torch.long)
        
        logger.debug(f"Dataset создан. Форма X: {self.X.shape}, Форма y: {self.y.shape}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LinearRegressionMod(nn.Module):
    """Простая линейная регрессия с поддержкой L1 и L2 регуляризации."""
    def __init__(self, in_features: int, l1_lambda: float = 0.0, l2_lambda: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def regularization_loss(self) -> torch.Tensor:
        l1 = self.l1_lambda * torch.sum(torch.abs(self.linear.weight))
        l2 = self.l2_lambda * torch.sum(self.linear.weight ** 2)
        return l1 + l2


class BinaryLogisticRegression(nn.Module):
    """Логистическая регрессия для бинарной классификации."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1) # BCEWithLogitsLoss применяет sigmoid внутри


# 3. ВИЗУАЛИЗАЦИЯ

def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """
    Строит и сохраняет график динамики потерь (Loss) при обучении.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Validation Loss", marker='s')
    plt.title("Динамика обучения модели")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"График истории обучения сохранен: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: str):
    """
    Строит и сохраняет матрицу ошибок (Confusion Matrix).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Матрица ошибок")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Матрица ошибок сохранена: {save_path}")


# ОСНОВНОЙ ПАЙПЛАЙН

def prepare_datasets_on_disk() -> Tuple[str, str]:
    """Генерирует и сохраняет CSV-файлы в папку data/ для воспроизводимости."""
    Path("data").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # 1. Регрессия
    housing = fetch_california_housing(as_frame=True).frame
    housing["OceanProximity"] = np.where(housing["Longitude"] < -121, "NEAR_OCEAN", "INLAND")
    housing["IsNewHouse"] = (housing["HouseAge"] < 15).astype(int)
    reg_path = "data/housing_regression.csv"
    housing.to_csv(reg_path, index=False)

    # 2. Классификация
    cancer = load_breast_cancer(as_frame=True).frame
    clf_path = "data/cancer_classification.csv"
    cancer.to_csv(clf_path, index=False)

    logger.info(f"Датасеты успешно сохранены: {reg_path}, {clf_path}")
    return reg_path, clf_path


def run_experiments():
    """Запуск полного цикла экспериментов: подготовка, обучение, оценка."""
    logger.info("Запуск пайплайна машинного обучения...")
    reg_csv, clf_csv = prepare_datasets_on_disk()

    # --- Эксперимент 1: Регрессия ---
    logger.info("=== Начало эксперимента 1: Линейная регрессия ===")
    df_reg = pd.read_csv(reg_csv)
    train_df, val_df = train_test_split(df_reg, test_size=0.2, random_state=42)
    
    train_df.to_csv("data/reg_train.csv", index=False)
    val_df.to_csv("data/reg_val.csv", index=False)

    train_ds = TabularCSVDataset("data/reg_train.csv", target_col="MedHouseVal", task="regression", fit=True)
    val_ds = TabularCSVDataset(
        "data/reg_val.csv", target_col="MedHouseVal", task="regression",
        num_cols=train_ds.num_cols, cat_cols=train_ds.cat_cols, bin_cols=train_ds.bin_cols,
        scaler=train_ds.scaler, encoder=train_ds.encoder, fit=False,
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model_reg = LinearRegressionMod(in_features=train_ds.X.shape[1], l1_lambda=0.001, l2_lambda=0.01)
    optimizer = optim.Adam(model_reg.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    epochs = 30
    
    for epoch in range(epochs):
        model_reg.train()
        train_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model_reg(bx)
            loss = criterion(preds, by) + model_reg.regularization_loss()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model_reg.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                preds = model_reg(bx)
                val_loss += criterion(preds, by).item()
                
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        if (epoch + 1) % 10 == 0:
            logger.info(f"Эпоха {epoch+1}/{epochs} | Train MSE: {history['train_loss'][-1]:.4f} | Val MSE: {history['val_loss'][-1]:.4f}")

    plot_training_history(history, "plots/regression_history.png")
    logger.info(f"Эксперимент 1 завершен. Финальный Val MSE: {history['val_loss'][-1]:.4f}")

    # Эксперимент 2: Классификация
    logger.info("=== Начало эксперимента 2: Логистическая регрессия ===")
    df_clf = pd.read_csv(clf_csv)
    train_c, test_c = train_test_split(df_clf, test_size=0.2, random_state=42)
    
    train_c.to_csv("data/clf_train.csv", index=False)
    test_c.to_csv("data/clf_test.csv", index=False)

    train_clf_ds = TabularCSVDataset("data/clf_train.csv", target_col="target", task="classification", fit=True)
    test_clf_ds = TabularCSVDataset(
        "data/clf_test.csv", target_col="target", task="classification",
        num_cols=train_clf_ds.num_cols, cat_cols=train_clf_ds.cat_cols, bin_cols=train_clf_ds.bin_cols,
        scaler=train_clf_ds.scaler, encoder=train_clf_ds.encoder, fit=False,
    )

    train_clf_loader = DataLoader(train_clf_ds, batch_size=32, shuffle=True)
    test_clf_loader = DataLoader(test_clf_ds, batch_size=32)

    model_clf = BinaryLogisticRegression(in_features=train_clf_ds.X.shape[1])
    optimizer_clf = optim.Adam(model_clf.parameters(), lr=0.01)
    criterion_clf = nn.BCEWithLogitsLoss()

    model_clf.train()
    for epoch in range(50):
        for bx, by in train_clf_loader:
            optimizer_clf.zero_grad()
            loss = criterion_clf(model_clf(bx), by.float())
            loss.backward()
            optimizer_clf.step()
            
    logger.info("Обучение классификатора завершено. Оценка на тесте...")
    
    model_clf.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for bx, by in test_clf_loader:
            logits = model_clf(bx)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            all_preds.extend(preds.numpy())
            all_true.extend(by.numpy())
            
    plot_confusion_matrix(all_true, all_preds, class_names=["Malignant (0)", "Benign (1)"], save_path="plots/cancer_cm.png")
    accuracy = np.mean(np.array(all_true) == np.array(all_preds))
    logger.info(f"Эксперимент 2 завершен. Точность на тесте: {accuracy:.4f}")


# 4. ТЕСТИРОВАНИЕ (Unit Tests)

class TestTabularMLPipeline(unittest.TestCase):
    """Unit-тесты для критических компонентов пайплайна."""

    @classmethod
    def setUpClass(cls):
        """Создание тестовых данных перед запуском тестов."""
        Path("data").mkdir(exist_ok=True)
        cls.sample_path = "data/test_sample.csv"
        df = pd.DataFrame({
            "num_col": [10.0, 20.0, np.nan, 40.0],
            "cat_col": ["A", "B", "A", "C"],
            "bin_col": [0, 1, 0, 1],
            "target": [1.5, 2.5, 3.0, 4.5],
        })
        df.to_csv(cls.sample_path, index=False)

    def test_dataset_shape_and_nans(self):
        """Проверка корректности формы выходного тензора и отсутствия NaN."""
        ds = TabularCSVDataset(self.sample_path, target_col="target", task="regression", fit=True)
        # 1 (num) + 3 (one-hot cat) + 1 (bin) = 5 признаков
        self.assertEqual(ds.X.shape, (4, 5))
        self.assertFalse(torch.isnan(ds.X).any(), "В тензоре признаков обнаружены NaN!")

    def test_preprocessor_reuse(self):
        """Проверка, что scaler и encoder корректно передаются и работают в режиме fit=False."""
        ds_train = TabularCSVDataset(self.sample_path, target_col="target", fit=True)
        # Создаем второй датасет на тех же данных, но с переданными препроцессорами
        ds_val = TabularCSVDataset(
            self.sample_path, target_col="target", fit=False,
            scaler=ds_train.scaler, encoder=ds_train.encoder,
            num_cols=ds_train.num_cols, cat_cols=ds_train.cat_cols, bin_cols=ds_train.bin_cols
        )
        self.assertIsNotNone(ds_val.scaler)
        self.assertIsNotNone(ds_val.encoder)
        self.assertEqual(ds_train.X.shape, ds_val.X.shape)

    def test_model_forward_pass(self):
        """Проверка прохождения тензора через модели без ошибок."""
        batch_size, n_features = 16, 10
        x_dummy = torch.randn(batch_size, n_features)
        
        lin_model = LinearRegressionMod(in_features=n_features)
        out_lin = lin_model(x_dummy)
        self.assertEqual(out_lin.shape, (batch_size,))
        
        log_model = BinaryLogisticRegression(in_features=n_features)
        out_log = log_model(x_dummy)
        self.assertEqual(out_log.shape, (batch_size,))


if __name__ == "__main__":
    # Запуск основного пайплайна
    run_experiments()
    
    # Запуск unit-тестов (argv=[""] нужен, чтобы unittest не пытался парсить аргументы скрипта)
    print("ЗАПУСК UNIT-ТЕСТОВ")
    unittest.main(argv=[""], exit=False, verbosity=2)