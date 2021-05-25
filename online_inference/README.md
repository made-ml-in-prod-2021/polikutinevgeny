Сборка образа
-----------------
Сначала обучить и положить артефакты рядом, затем:
```shell
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

Протыкать скриптом:
`python -m make_request`
