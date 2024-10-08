# Проект распознавания русского жестового языка

**Цель проекта**:  Разработка модели глубокого обучения для распознавания жестов рук на основе видеопоследовательностей.

## Пайплайн

1. **Загрузка и организация данных**:
    - Данные организованы в директории, где каждый поддиректорий представляет отдельный класс жеста.
    - Внутри каждого класса находятся кадры видео, именованные по шаблону "{video_id}_frame_{frame_number}.jpg".
    - Класс `VideoSequenceDataset` загружает эти данные, группируя кадры по видео и выбирая 32 кадра для каждой последовательности.
    - Для обучающего набора применяется стратифицированное разделение на обучающую и валидационную выборки с сохранением соотношения классов.

2. **Аугментация изображений**:
    - Используется библиотека Albumentations для аугментации обучающих данных.
    - Применяются следующие преобразования:
        - Горизонтальное отражение (вероятность 100%)
        - Оптическое искажение или сеточное искажение (вероятность 50%)
        - Случайное изменение яркости/контраста или цветовой баланс (вероятность 80%)
        - Добавление шума (Гауссов или ISO) (вероятность 60%)
        - Размытие в движении или Гауссово размытие (вероятность 60%)
    - Все изображения изменяют размер до 640x640 пикселей.
    - Для валидационного набора применяется только изменение размера без аугментаций.

3. **Извлечение признаков (эмбеддингов)**:
    - Используются две предобученные модели:
        - ResNext50: извлекает общие визуальные признаки.
        - ViT (Vision Transformer): специализированная модель для распознавания жестов рук, предобученная на датасете "dima806/hand_gestures_image_detection".
    - Процесс извлечения эмбеддингов:
        - Каждое изображение последовательности пропускается через модели ResNext50 и ViT.
        - Выходы ResNext50 и ViT объединяются для создания богатого представления каждого кадра.
        - Выход ResNext50 разделяется на три части по 768 элементов, с дополнением нулями при необходимости.
        - Для ViT используется среднее значение выходов последнего слоя.
        - Изображения разделяются на левую и правую половины, для каждой из которых извлекаются эмбеддинги с помощью ViT и ResNext50.

4. **Организация эмбеддингов**:
    - Эмбеддинги организуются в структуру из 12 каналов для каждого кадра последовательности:
        - [левая часть изображения (ViT), правая часть изображения (ViT),
        - левая часть ResNext (3 куска), правая часть ResNext (3 куска),
        - все изображение (ViT), все изображение ResNext (3 куска)]

5. **Сохранение предобработанных данных**:
    - Извлеченные эмбеддинги и метки классов сохраняются в HDF5 файлы:
        - 'preprocessed_train_dataset.h5' для обучающего набора.
        - 'preprocessed_val_dataset.h5' для валидационного набора.
    - HDF5 формат выбран для эффективного хранения и быстрой загрузки больших объемов данных.
    - Данные сжимаются с помощью алгоритма "gzip".

6. **Загрузка данных для обучения**:
    - Класс `HDF5Dataset` используется для эффективной загрузки предобработанных данных во время обучения.
    - Это позволяет избежать повторной обработки изображений при каждой эпохе, значительно ускоряя процесс обучения.

7. **Модель**:
    - Используется модель `CNNBiLSTMWithAttention`, которая состоит из:
        - Двунаправленного LSTM слоя с остаточным соединением (`ResidualLSTM`).
        - Механизма внимания к последовательности (`SequenceAttentionLayer`).
        - Механизма внимания к каналу (`ChannelAttentionLayer`).
        - Полносвязного слоя для классификации.

8. **Обучение**:
    - Модель обучается с помощью оптимизатора RAdam и функции потерь CrossEntropyLoss.
    - Используется OneCycleLR для планирования скорости обучения.
    - Во время обучения отслеживаются метрики потерь и точности на обучающей и валидационной выборках.
    - Сохраняется модель с наилучшей точностью на валидационной выборке.
    - Вычисляется и выводится статистика по механизмам внимания (средние значения, индексы максимальных и минимальных значений).