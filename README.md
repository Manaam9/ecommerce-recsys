# Рекомендации товаров в электронной коммерции

## Описание проекта

Цель проекта — построить рекомендательную систему для интернет-магазина, которая предсказывает, какие товары следует рекомендовать пользователю.

Проект охватывает полный ML-цикл:
- исследование данных;
- подготовку признаков;
- построение и сравнение моделей;
- логирование экспериментов;
- развёртывание модели в виде API-сервиса;
- контейнеризацию;
- пайплайн переобучения;
- мониторинг.

## Данные

Используются три таблицы:

### 1. `category_tree.csv`
Содержит иерархию категорий:
- `parentid` — родительская категория;
- `categoryid` — дочерняя категория.

### 2. `events.csv`
Лог пользовательских событий:
- `timestamp` — время события;
- `visitorid` — идентификатор пользователя;
- `event` — тип события (`view`, `addtocart`, `transaction`);
- `itemid` — идентификатор товара;
- `transactionid` — идентификатор транзакции.

### 3. `item_properties.csv`
История свойств товаров:
- `timestamp` — время изменения свойства;
- `itemid` — идентификатор товара;
- `property` — название свойства;
- `value` — значение свойства.

## Постановка задачи

Необходимо рекомендовать пользователю товары, которые с высокой вероятностью приведут к добавлению в корзину.

### Целевое действие
Основной таргет:
- `addtocart`

Дополнительные сигналы:
- `transaction` как сильный позитивный сигнал;
- `view` как слабый сигнал интереса.

Также купленные товары можно исключать из рекомендаций.

## Метрики

### Оффлайн-метрики
Для оценки качества рекомендаций используются:
- Recall@K
- Precision@K
- MAP@K
- NDCG@K
- HitRate@K

Основная метрика:
- Recall@10

Дополнительные:
- MAP@10
- NDCG@10

### Бизнес-метрики
Потенциально можно отслеживать:
- CTR рекомендаций;
- Add-to-cart rate;
- Conversion to purchase;
- GMV uplift.

## Структура проекта

```text
ecommerce-recsys/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── mlruns/
├── scripts/
│   ├── setup_vm.sh
│   ├── run_mlflow.sh
│   └── train_model.py
├── docker/
│   └── Dockerfile
├── airflow/
│   └── dags/
│       └── retrain_recsys.py
└── src/
    ├── api/
    │   └── app.py
    ├── data/
    │   ├── load_data.py
    │   └── preprocess.py
    ├── features/
    │   └── build_features.py
    ├── models/
    │   ├── baseline.py
    │   ├── als.py
    │   ├── ranker.py
    │   └── inference.py
    ├── monitoring/
    │   └── metrics.py
    └── utils/
        └── config.py
````

## Исследование данных

В рамках EDA анализируются:

* число пользователей, товаров и событий;
* распределение событий по типам;
* временной диапазон данных;
* популярные товары;
* активность пользователей;
* разреженность user-item матрицы;
* cold-start по пользователям и товарам.

Результат:

* ноутбук `notebooks/01_eda.ipynb`.

## Подход к решению

Решение строится в два этапа:

### 1. Candidate Generation

Генерация кандидатов несколькими способами:

* Top Popular;
* ALS;
* item-to-item/co-visitation;
* recently popular;
* category-based candidates.

### 2. Ranking

Для кандидатов рассчитываются признаки:

* число просмотров товара пользователем;
* число добавлений в корзину;
* recency событий;
* популярность товара;
* конверсия товара;
* категориальные признаки;
* similarity-признаки.

Финальный ранжировщик:

* LightGBM / CatBoost / XGBoost.

## Эксперименты

В проекте планируется сравнение:

* baseline Top Popular;
* ALS;
* гибридной модели с ранжированием.

Логирование экспериментов выполняется через MLflow.

Результат:

* ноутбук `notebooks/02_modeling.ipynb`;
* сохранённый артефакт модели в `models/`.

## Запуск окружения

### 1. Клонирование репозитория

```bash
git clone <repo_url>
cd ecommerce-recsys
```

### 2. Подготовка окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Подготовка директорий

```bash
mkdir -p data/raw data/processed models mlruns
```

## Запуск Jupyter

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

Для подключения с локальной машины:

```bash
ssh -L 8888:localhost:8888 -i <ssh_key> <user>@<server_ip>
```

## Запуск MLflow

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

Для подключения с локальной машины:

```bash
ssh -L 5000:localhost:5000 -i <ssh_key> <user>@<server_ip>
```

## Обучение модели

Пример запуска обучения:

```bash
python scripts/train_model.py
```

В результате сохраняются:

* обученная модель;
* метрики;
* артефакты эксперимента в MLflow.

## API-сервис

Сервис предоставляет рекомендации по `visitorid`.

### Основные endpoints

* `GET /health` — проверка работоспособности;
* `POST /recommend` — получение рекомендаций.

Пример запуска:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## Docker

Для удобства деплоя сервис упакован в Docker.

### Сборка образа

```bash
docker build -t ecommerce-recsys -f docker/Dockerfile .
```

### Запуск контейнера

```bash
docker run -p 8000:8000 ecommerce-recsys
```

## Переобучение модели

Переобучение организовано через Airflow DAG:

* загрузка данных;
* предобработка;
* генерация признаков;
* обучение модели;
* сохранение артефактов.

Файл DAG:

* `airflow/dags/retrain_recsys.py`

## Мониторинг

Мониторинг включает:

### Технические метрики

* latency;
* error rate;
* CPU/RAM usage;
* requests per second.

### ML-метрики

* доля пустых рекомендаций;
* средняя длина списка рекомендаций;
* coverage;
* popularity bias;
* drift данных;
* деградация оффлайн-метрик.

## Воспроизводимость

Для воспроизводимости:

* фиксируются версии зависимостей в `requirements.txt`;
* фиксируются random seeds;
* логируются параметры и артефакты в MLflow.

## Результаты

В ходе проекта должны быть получены:

* EDA ноутбук;
* ноутбук с экспериментами;
* обученная модель;
* API-сервис;
* Dockerfile;
* DAG переобучения;
* описание мониторинга;
* оформленный `README.md`.

