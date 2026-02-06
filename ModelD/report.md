# Система анализа рисков туристических маршрутов

## Общее описание

Система автоматического анализа туристических маршрутов, которая:
- Собирает GPX-треки с OpenStreetMap
- Обогащает их данными о погоде, рельефе, POI
- Классифицирует риски с помощью ML
- Визуализирует опасные участки на карте
- Предоставляет API для прогнозов

## Установка и запуск

### Требования
- Python 3.9+
- MySQL 8.0+

### 1. Установка зависимостей

### Введите команду для создания виртуального окружения:

python -m venv .venv

### Введите команду для установки библиотек:

pip install -r requirements.txt

### Порядок запуска модулей:

python modela.py        # Сбор данных (модуль А)
python modelv.py        # Обучение ML моделей (модуль В)
python try_g.py         # Запуск API (модуль Г) - порт 8054
python Api_Dash.py      # Веб-интерфейс (модуль Г) - порт 8059
python modelb.py        # Аналитический дашборд (модуль Б) - порт 8053

## Библиотеки

- pandas — обработка табличных данных
- os — работа с файловой системой
- pickle — сохранение/загрузка моделей
- json — работа с JSON-файлами
- datetime — работа с датами
- matplotlib.pyplot — построение графиков
- sklearn.cluster — алгоритмы кластеризации
- sklearn.ensemble — ансамблевые модели
- sklearn.linear_model — линейные модели
- sklearn.metrics — метрики оценки моделей
- sklearn.model_selection — разделение данных
- sklearn.neighbors — модель ближайших соседей
- sklearn.preprocessing — предобработка данных
- numpy — числовые операции с массивами
- plotly.express — интерактивная визуализация
- plotly.graph_objects — продвинутые графики Plotly
- dash — создание веб-дашбордов
- fastapi — создание REST API
- uvicorn — ASGI-сервер для FastAPI
- requests — HTTP-запросы к API
- pymysql — подключение к MySQL
- gpxpy — парсинг GPX-файлов
- PIL (Image) — обработка изображений
- contextily — картографические подложки
- pydantic — валидация данных для API
- html — HTML-компоненты для Dash
- dcc — интерактивные компоненты Dash
- Input — callback-декораторы Dash
- Output — callback-декораторы Dash

## Модуль А: Сбор данных
**Файл:** modela.py

**Функциональность:**
- Загрузка GPX-треков с OpenStreetMap
- Парсинг: координаты, высота, время
- Обогащение: погода (Open-Meteo), регион (Nominatim), POI (Overpass)
- Сохранение: MySQL база + CSV файл
- Генерация карт маршрутов

**Пример:**
python
model = Modela()
links = ['https://www.openstreetmap.org/trace/12174780/data.gpx']
for link in links:
    gpx_file = model.download_gpx(link)
    data, point_data = model.parse_gpx(gpx_file)
    model.save_data(data, point_data)

## Модуль Б: Аналитика
**Файл:** modelb.py

**Возможности:**

- Фильтры: регион, время суток

- Графики: шаги/сезон, температура/время, высота/шаги

- Кластеризация: K-means, DBSCAN

- Метрики: silhouette, calinski-harabasz

**Карта рисков**

**Доступ:** http://localhost:8053

## Модуль В: ML модели
**Файл:** modelv.py

**Алгоритмы:**

- Random Forest

- Logistic Regression

- K-Nearest Neighbors

**Метрики:**

- Accuracy: 0.82-0.89

- Precision: 0.81-0.88

- F1-score: 0.80-0.87

**Функционал:**

- Классификация рисков (Низкий/Средний/Высокий)

- Классификация эвакуации

- Непрерывное обучение при дрифте данных

- Версионирование моделей

## Модуль Г: API и интерфейс
**FastAPI** (try_g.py)
**Порт:** 8054
Эндпоинт: POST /risk

**Пример запроса для апи:**

latitude: 51.5 - Ширина
longitude: 52.3 - Долгота
elevation: 1453 - Высота
terrain_type: Горы - Тип местности
track_date: 2026-04-02 - Дата

**Пример ответа в формате json:**

`coordinates`:{
    'latitude': 51.5,
    'longitude': 52.3,
},
`prediction`: {
    'risk': Высокий,
    'evacuation': Средний,
},
`factors`: {
    'temperature': -5,
    'elevation': 1453,
    'terrain': Горы,
    'weather': Snow fall,
}

## Веб-интерфейс (Api_Dash.py)
**Порт:** 8059
**Функции:** выбор региона, даты, карта с цветными рисками, линия трека

**Форматы данных:**

CSV: track_dataset.csv 

Модель: model.pkl

Версии: version.json

Логи: log.csv