ml_project
==============================

Predicting heart disease

Настройка окружения
----------
```
make environment
conda activate ml_project
make requirements
```

Генерация отчёта EDA
--------
`make eda_report`

Обучение
-----------
Для дефолтных параметров
`python -m heart_disease.models.train_model`

Для первого конфига/эксперимента
`python -m heart_disease.models.train_model --config-path=${PWD}/config/experiment_1`

Для второго конфига/эксперимента
`python -m heart_disease.models.train_model --config-path=${PWD}/config/experiment_2`

Предсказание
-----------
Для дефолтных параметров
`python -m heart_disease.models.predict_model`

Для первого конфига/эксперимента
`python -m heart_disease.models.predict_model --config-path=${PWD}/config/experiment_1`

Для второго конфига/эксперимента
`python -m heart_disease.models.predict_model --config-path=${PWD}/config/experiment_2`

Пути к файлам можно указывать через файлы конфигов, либо через 
параметры командной строки (e.g. `python -m heart_disease.models.predict_model data_path="data/new_data.csv"`)


Сборка образа
-----------------
```shell
python -m heart_disease.models.train_model
docker build . -t ml_project:latest
```

Публикация образа
-----------------
```shell
docker tag ml_project:latest polikutinevgeny/ml_project:latest
docker push polikutinevgeny/ml_project:latest
```

Запуск образа
-------------
```shell
docker pull polikutinevgeny/ml_project:latest
docker run -p 8000:80 polikutinevgeny/ml_project:latest
```
