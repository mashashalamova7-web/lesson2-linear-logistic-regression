## Задание 3: Эксперименты и анализ (20 баллов)

### 3.1 Исследование гиперпараметров (10 баллов)
# Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
# Визуализируйте результаты в виде графиков или таблиц

### 3.2 Feature Engineering (10 баллов)
# Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# Устанавливаем стиль графиков
plt.style.use('seaborn-v0_8-whitegrid')

## 3.1 Исследование гиперпараметров (10 баллов)

def create_dataset(n_samples=1000):
    """Создаем синтетический датасет для экспериментов."""
    X = torch.randn(n_samples, 10)
    # Создаем целевую переменную с нелинейной зависимостью
    y = 2 * X[:, 0] + 0.5 * X[:, 1]**2 - X[:, 2] * X[:, 3] + torch.randn(n_samples) * 0.1
    return X, y

def train_model(X_train, y_train, X_val, y_val, learning_rate=0.01, batch_size=32, 
                optimizer_name='Adam', epochs=50):
    """Обучает простую нейросеть с заданными гиперпараметрами."""
    
    # Создаем DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Простая модель
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    criterion = nn.MSELoss()
    
    # Выбираем оптимизатор
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")
    
    # Обучение
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Считаем среднюю потерю за эпоху
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Валидация
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val).squeeze()
            val_loss = criterion(val_predictions, y_val).item()
            val_losses.append(val_loss)
    
    return train_losses, val_losses

def experiment_learning_rate():
    """Эксперимент с разными learning rate."""
    print("Эксперимент 1: Скорость обучения")
    
    X, y = create_dataset()
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    results = {}
    
    for lr in learning_rates:
        train_losses, val_losses = train_model(
            X_train, y_train, X_val, y_val,
            learning_rate=lr, batch_size=32, optimizer_name='Adam'
        )
        results[lr] = val_losses
        print(f"LR={lr}: финальная val_loss={val_losses[-1]:.4f}")
    
    # График
    plt.figure(figsize=(10, 6))
    for lr, losses in results.items():
        plt.plot(losses, label=f'LR={lr}')
    plt.xlabel('Эпоха')
    plt.ylabel('Validation Loss')
    plt.title('Влияние learning rate на обучение')
    plt.legend()
    plt.savefig('experiment_lr.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("График сохранен: experiment_lr.png\n")

def experiment_batch_size():
    """Эксперимент с разными размерами батча."""
    print(" Эксперимент 2: Размер батча")
    
    X, y = create_dataset()
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]
    
    batch_sizes = [8, 32, 128, 512]
    results = {}
    
    for bs in batch_sizes:
        train_losses, val_losses = train_model(
            X_train, y_train, X_val, y_val,
            learning_rate=0.01, batch_size=bs, optimizer_name='Adam'
        )
        results[bs] = val_losses
        print(f"Batch size={bs}: финальная val_loss={val_losses[-1]:.4f}")
    
    # График
    plt.figure(figsize=(10, 6))
    for bs, losses in results.items():
        plt.plot(losses, label=f'Batch={bs}')
    plt.xlabel('Эпоха')
    plt.ylabel('Validation Loss')
    plt.title('Влияние размера батча на обучение')
    plt.legend()
    plt.savefig('experiment_batch.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("График сохранен: experiment_batch.png\n")

def experiment_optimizer():
    """Эксперимент с разными оптимизаторами."""
    print("Эксперимент 3: Оптимизаторы")
    
    X, y = create_dataset()
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]
    
    optimizers = ['SGD', 'Adam', 'RMSprop']
    results = {}
    
    for opt_name in optimizers:
        train_losses, val_losses = train_model(
            X_train, y_train, X_val, y_val,
            learning_rate=0.01, batch_size=32, optimizer_name=opt_name
        )
        results[opt_name] = val_losses
        print(f"Optimizer={opt_name}: финальная val_loss={val_losses[-1]:.4f}")
    
    # График
    plt.figure(figsize=(10, 6))
    for opt_name, losses in results.items():
        plt.plot(losses, label=opt_name)
    plt.xlabel('Эпоха')
    plt.ylabel('Validation Loss')
    plt.title('Сравнение оптимизаторов')
    plt.legend()
    plt.savefig('experiment_optimizer.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("График сохранен: experiment_optimizer.png\n")


## 3.2 Feature Engineering (10 баллов)

def create_features_basic(X):
    """Базовые признаки (как есть)."""
    return X

def create_features_polynomial(X):
    """Добавляем полиномиальные признаки (квадраты)."""
    X_poly = torch.cat([X, X**2], dim=1)
    return X_poly

def create_features_interactions(X):
    """Добавляем взаимодействия между признаками (попарные произведения)."""
    interactions = []
    n_features = X.shape[1]
    for i in range(n_features):
        for j in range(i+1, n_features):
            interactions.append((X[:, i] * X[:, j]).unsqueeze(1))
    X_interact = torch.cat([X] + interactions, dim=1)
    return X_interact

def create_features_statistical(X):
    """Добавляем статистические признаки (среднее, дисперсия по строкам)."""
    row_mean = X.mean(dim=1, keepdim=True)
    row_std = X.std(dim=1, keepdim=True)
    X_stats = torch.cat([X, row_mean, row_std], dim=1)
    return X_stats

def create_features_all(X):
    """Комбинируем все виды признаков."""
    X_poly = X**2
    interactions = []
    n_features = X.shape[1]
    for i in range(n_features):
        for j in range(i+1, n_features):
            interactions.append((X[:, i] * X[:, j]).unsqueeze(1))
    X_interact = torch.cat(interactions, dim=1)
    row_mean = X.mean(dim=1, keepdim=True)
    row_std = X.std(dim=1, keepdim=True)
    
    X_all = torch.cat([X, X_poly, X_interact, row_mean, row_std], dim=1)
    return X_all

def evaluate_features():
    """Сравниваем качество модели с разными наборами признаков."""
    print("3.2 Feature Engineering\n")
    
    X, y = create_dataset(n_samples=1000)
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]
    
    feature_sets = {
        'Базовые': create_features_basic,
        'Полиномиальные': create_features_polynomial,
        'Взаимодействия': create_features_interactions,
        'Статистические': create_features_statistical,
        'Все признаки': create_features_all
    }
    
    results = {}
    
    for name, feature_fn in feature_sets.items():
        # Создаем признаки
        X_train_feat = feature_fn(X_train)
        X_val_feat = feature_fn(X_val)
        
        print(f"Набор '{name}': {X_train_feat.shape[1]} признаков")
        
        # Обучаем модель
        train_losses, val_losses = train_model(
            X_train_feat, y_train, X_val_feat, y_val,
            learning_rate=0.01, batch_size=32, optimizer_name='Adam', epochs=100
        )
        
        results[name] = val_losses[-1]
        print(f"  Финальная val_loss: {val_losses[-1]:.4f}\n")
    
    # График сравнения
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.xlabel('Набор признаков')
    plt.ylabel('Validation Loss (меньше = лучше)')
    plt.title('Сравнение качества моделей с разными признаками')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('experiment_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("График сохранен: experiment_features.png\n")
    
    # Выводим лучший результат
    best_feature_set = min(results, key=results.get)
    print(f"Лучший набор признаков: '{best_feature_set}' с val_loss={results[best_feature_set]:.4f}")


if __name__ == "__main__":
    print("ЗАДАНИЕ 3: ЭКСПЕРИМЕНТЫ И АНАЛИЗ")
    print()
    
    # 3.1 Гиперпараметры
    experiment_learning_rate()
    experiment_batch_size()
    experiment_optimizer()
    
    # 3.2 Feature Engineering
    evaluate_features()
    