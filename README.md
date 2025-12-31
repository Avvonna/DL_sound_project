# ASR Project (DeepSpeech2)

Репозиторий содержит реализацию системы автоматического распознавания речи (ASR) на архитектуре **DeepSpeech2** с использованием CTC Loss.  
Проект выполнен в рамках ДЗ№2 Sound DL

## Особенности реализации

* **Архитектура:** DeepSpeech2 (CNN feature extractor + GRU layers + FC).
* **Лосс:** CTC Loss.
* **Декодинг:**
   * Greedy Decoding (Argmax).
   * **Custom Beam Search** (реализован вручную на Python/PyTorch без внешних библиотек).
* **Аугментации:** Реализован широкий спектр аудио и спектральных аугментаций.
* **Логирование:** WandB.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Avvonna/DL_sound_project.git
   cd DL_sound_project
   ```

2. Установите зависимости

   - [requirements_inference.txt](./requirements_inference.txt) для инференса
   - [requirements_all.txt](./requirements_all.txt) для разработки

   ```bash
   pip install -r requirements_inference.txt
   ```

3. **Загрузка весов модели:**
   Для запуска инференса необходимо скачать обученные веса (`best_model.pth`).  
   Веса модели доступны по [**ссылке**](https://drive.google.com/file/d/12k5HUXaGLweeOGXSZqiqrhUG9CMjKujr/view?usp=drive_link) - можно скачать вручную
   
   Веса модели можно также загрузить с помощью команды:

   ```bash
   # Создаем папку и скачиваем веса
   mkdir -p saved
   gdown --id 12k5HUXaGLweeOGXSZqiqrhUG9CMjKujr -O saved/best_model.pth
   ```

## Инференс

Для инференса используется скрипт [**inference.py**](./inference.py).

### На своем датасете

#### Подготовка данных

Данные должны быть организованы в структуру `CustomDirDataset`:

```text
my_dataset/
├── audio/
│   ├── file1.wav
│   ├── file2.mp3
├── ├── file2.flac
│   └── ...
└── transcriptions/  (опционально, для подсчета метрик)
    ├── file1.txt
    └── ...
```

Для запуска инференса можно воспользоваться следующей командой

```bash
python inference.py \
   inferencer.save_path="predictions" \
   inferencer.from_pretrained="saved/best_model.pth" \
   inferencer.data_dir="path/to/my_dataset" \
   inferencer.device="cuda" \
   decoding.decode_type="beam" \
   decoding.beam_size=20
```
* Результаты (текстовые файлы) будут сохранены в папку `data/saved/predictions/inference/`.
* Если в датасете есть папка `transcriptions`, метрики (CER/WER) будут выведены в консоль.

### На готовых датасетах Librispeech

Для проверки метрик можно использовать следующую команду

```bash
python inference.py \
   datasets=librispeech_test \
   inferencer.from_pretrained="saved/model_best.pth" \
   inferencer.save_path="data/saved/test_results"
```

### Подсчет метрик (отдельно)
Если у вас уже есть предсказания и ground truth, можно пересчитать метрики скриптом [**calc_metrics.py**](./calc_metrics.py):

```bash
python calc_metrics.py \
   --gt_dir path/to/my_dataset/transcriptions \
   --pred_dir data/saved/predictions/inference \
   --out_json metrics_details.json
```

## Демонстрация
Для быстрой проверки работы модели доступен **Jupyter Notebook**:
`demo_colab.ipynb`

Он позволяет:
1. Загрузить отдельно выбранные аудиофайлы для получения транскрипций
2. Скачать датасет с Google Drive для транскрипций или оценки метрик (если `transcriptions` уже есть)
3. Записать голос с микрофона (в Colab) и получить транскрипцию.

[**Открыть в Google Colab**](https://colab.research.google.com/github/Avvonna/DL_sound_project/blob/main/demo_colab.ipynb)

## Отчет о проделанной работе

### 1. Воспроизведение обучения
Обучение проводилось в несколько этапов для достижения стабильности и качества:

1. **Overfit sanity check:** Обучение на одном батче с помощью [baseline_onebatch.yaml](./src/configs/experiment/baseline_onebatch.yaml), чтобы убедиться, что пайплайн работает (Loss падает до 0)
2. **Pre-training:** Обучение на `Librispeech train-clean-100` с помощью [deepspeech_librispeech.yaml](./src/configs/experiment/deepspeech_librispeech.yaml).
3. **(in process) Fine-tuning:** Дообучение на смеси `Librispeech` + `Common Voice` для улучшения устойчивости к разным условиям записи: [deepspeech_finetune_librispeech_cv.yaml](./src/configs/experiment/deepspeech_finetune_librispeech_cv.yaml).

Команда для запуска обучения:
```bash
python train.py experiment=deepspeech_librispeech
```

### 2. Логи обучения
Полные логи обучения, графики Loss, градиентов доступны в WandB: [**LINK TO W&B REPORT**](https://wandb.ai/vaalkaev-hse/DL_sound_HW2/reports/ASR-Project-DeepSpeech2---VmlldzoxNTUwNjkwOA?accessToken=gvbv10puhzyq4mje381gaou4alham2r4bncvyklg0l21uukg8duymz3qfa8rn818)

**Динамика обучения:**
* Сеть начала выдавать хороший скор примерно после 90 эпохи.
* CTC Loss стабильно снижался, но требовал аккуратного подбора Learning Rate (использовался OneCycleLR).

### 3. Результаты и Метрики
Метрики на тестовых данных (`Librispeech test-clean` / `test-other`):

| Dataset | Decoding | WER (%) | CER (%) |
| :--- | :--- | :--- | :--- |
| Test Clean | Greedy | [26.6] | [8.6] |
| **Test Clean** | **Beam Search** | **[26.2]** | **[8.5]** |
| Test Other | Greedy | [26.6] | [8.6] |
| Test Other | Beam Search | [26.2] | [8.5] |

*Beam Search (размер луча 20) показал прирост качества примерно **на 0.4%** WER по сравнению с Greedy.*

### 4. Эксперименты и Аугментации
Для выполнения требований и улучшения качества были реализованы и проверены следующие аугментации (см. [**wav_augs**](./src/transforms/wav_augs/) и [**spec_augs**](./src/transforms/spec_augs/)):

1. **Gain:** Случайное изменение громкости.
2. **PitchShift:** Сдвиг тональности.
3. **SpeedPerturb:** Изменение скорости (через ресемплинг).
4. **ColoredNoise:** Добавление цветного шума (с настраиваемым SNR и "цветом" шума - alpha).
5. **SpecAugment:** Frequency Masking и Time Masking (на спектрограммах).

**Наблюдения:**
* Спектральные аугментации (`Masking`) наиболее эффективны для борьбы с переобучением.
* Добавление шума (`ColoredNoise`) значительно улучшило качество на `test-other`.
* Важно не переборщить с вероятностью аугментаций (`p`), иначе модель перестает сходиться.

### 5. Реализованные Бонусы / Требования
* **DeepSpeech2 Architecture**: Реализована сверточная часть + GRU слои.
* **Custom Beam Search**: Реализован на чистом Python (см. [**ctc_text_encoder**](src/text_encoder/ctc_text_encoder.py)). Поддерживает `beam_size`, `beam_threshold` и оптимизацию через `topk_per_timestep`.
   * В таблице метрик выше видно, что Beam Search дает WER ниже, чем Argmax.
* **Аугментации**: Реализовано более 4-х типов (см. раздел выше).
* **Inference Pipeline**: Полностью рабочий скрипт [**inference.py**](./inference.py) и [**calc_metrics.py**](./calc_metrics.py).
* **Mixed Datasets**: Реализован [**LibriSpeechCommonVoiceMixedDataset**](./src/datasets/mixed_librispeech_commonvoice.py) для смешивания датасетов с балансировкой долей.

## Структура проекта

```
.
├── data/                   # Данные и сохраненные модели
├── src/
│   ├── configs/            # Конфиги Hydra (model, train, datasets...)
│   ├── datasets/           # Классы датасетов (Librispeech, CommonVoice, CustomDir)
│   ├── logger/             # WandB и визуализация
│   ├── loss/               # CTC Loss wrapper
│   ├── metrics/            # WER, CER (Argmax & Beam)
│   ├── model/              # DeepSpeech2
│   ├── text_encoder/       # Работа с текстом и CTC Beam Search
│   ├── trainer/            # Логика обучения и инференса
│   ├── transforms/         # Аугментации (Wav & Spec)
│   └── utils/              # Утилиты
├── train.py                # Скрипт обучения
├── inference.py            # Скрипт инференса
├── calc_metrics.py         # Подсчет метрик
├── demo_colab.ipynb        # Демонстрационный ноутбук
├── requirements_colab.txt  # Зависимости для GoogleColab
└── requirements_all.txt    # Все используемые локально зависимости
```
