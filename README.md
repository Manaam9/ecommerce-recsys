Готово — вот полный, аккуратно структурированный README с добавленным блоком трансляции задачи 👇
# 🛒 Рекомендации товаров в электронной коммерции

## Описание проекта

Цель проекта — построить рекомендательную систему для интернет-магазина, которая предсказывает, какие товары следует рекомендовать пользователю.

Проект охватывает полный ML-цикл:

* исследование данных;
* подготовку признаков;
* построение и сравнение моделей;
* логирование экспериментов;
* развёртывание модели в виде API-сервиса;
* контейнеризацию;
* пайплайн переобучения;
* мониторинг.

---

## Данные

Используются три таблицы:

### 1. `category_tree.csv`

Содержит иерархию категорий:

* `parentid` — родительская категория;
* `categoryid` — дочерняя категория.

### 2. `events.csv`

Лог пользовательских событий:

* `timestamp` — время события;
* `visitorid` — идентификатор пользователя;
* `event` — тип события (`view`, `addtocart`, `transaction`);
* `itemid` — идентификатор товара;
* `transactionid` — идентификатор транзакции.

### 3. `item_properties.csv`

История свойств товаров:

* `timestamp` — время изменения свойства;
* `itemid` — идентификатор товара;
* `property` — название свойства;
* `value` — значение свойства.

---

## Постановка задачи

Необходимо рекомендовать пользователю товары, которые с высокой вероятностью приведут к добавлению в корзину.

### Целевое действие

Основной таргет:

* `addtocart`

Дополнительные сигналы:

* `transaction` как сильный позитивный сигнал;
* `view` как слабый сигнал интереса.

Также купленные товары исключаются из рекомендаций.

---

## Метрики

### Оффлайн-метрики

Для оценки качества рекомендаций используются:

* Recall@K (основная)
* Precision@K
* MAP@K
* NDCG@K
* HitRate@K

Основная метрика:

* Recall@10

Дополнительные:

* MAP@10
* NDCG@10

---

### Бизнес-метрики

* CTR рекомендаций
* Add-to-cart rate
* Conversion to purchase
* GMV uplift

---

## Трансляция задачи

### Формулировка

Задача решается как **learning-to-rank**:

> необходимо упорядочить товары так, чтобы наиболее релевантные находились в топе рекомендаций.

---

### Обоснование метрик

С учётом EDA:

* высокая разреженность
* дисбаланс классов
* implicit feedback
* важность top-K

классические метрики (accuracy, F1) не применимы.

Используются:

* Recall@K
* NDCG@K
* MAP@K
* HitRate@K

---

### Связь с бизнесом

* Recall → покрытие рекомендаций
* NDCG → UX
* MAP → качество ранжирования

Финальная цель:

* рост Conversion Rate
* рост Revenue

---

## Подход к решению

Используется двухэтапная архитектура:

```text
Candidate Generation → Ranking
```

---

### 1. Candidate Generation

* Top Popular
* ALS
* item-to-item
* recently popular
* category-based

---

### 2. Ranking

Модель:

* LightGBM / CatBoost / XGBoost

Признаки:

* user behavior
* item popularity
* interaction history
* time features

---

## Исследование данных (EDA)

Ключевые выводы:

* сильная разреженность
* выраженный cold-start
* long-tail пользователей и товаров
* addtocart — сильный сигнал
* вечер — пик конверсий
* item properties — богатый источник признаков

Результат:

* `notebooks/01_eda.ipynb`

---

## Эксперименты

Сравниваются:

* Top Popular
* ALS
* Hybrid модель

Результат:

* `notebooks/02_modeling.ipynb`
* модели в `models/`

---

## MLflow

### Запуск

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

---

### Использование

Логируются:

* параметры моделей
* метрики (Recall@K, MAP@K, NDCG@K)
* модели
* feature importance

---

### Хранение артефактов

⚠️ Не добавляются в git:

```bash
mlflow.db
mlruns/
mlartifacts/
```

---

## Структура проекта

```text
ecommerce-recsys/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── data/
├── models/
├── scripts/
├── docker/
├── airflow/
└── src/
```

---

## Запуск

```bash
git clone <repo_url>
cd ecommerce-recsys

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

## Docker

```bash
docker build -t ecommerce-recsys .
docker run -p 8000:8000 ecommerce-recsys
```

---

## Airflow

* DAG: `airflow/dags/retrain_recsys.py`

---

## Мониторинг

* latency
* error rate
* coverage
* drift
* offline-метрики

---

## Итог

Данные:

* sparse
* long-tail
* cold-start

Решение:
👉 гибридная рекомендательная система (ALS + ranking)

