# KaoAnime — перенос стиля «селфи → аниме»

KaoAnime переносит реальное фото лица в аниме-стиль с помощью нейросетевого
image-to-image перевода между двумя неспаренными доменами. Основная модель —
**Neural Optimal Transport (NOT)** (Korotin et al., ICLR 2023); как альтернатива
доступен **CycleGAN**, разделяющий с NOT общие строительные блоки.

## Смысловое содержание проекта

### Задача

Есть два набора изображений лиц без парных соответствий: домен **A** — реальные
фото (CelebA), домен **B** — аниме-лица (safebooru). Цель — обучить отображение
`T: A → B`, переносящее фото в аниме-стиль с сохранением геометрии и идентичности
лица (unpaired image-to-image translation).

### Данные

- **CelebA** — публичный датасет лиц (~200k изображений). Используется выровненная
  версия `img_align_celeba`; домен A дополнительно фильтруется по полу (women-only)
  по `list_attr_celeba.csv`.
- **AlignedAnimeFaces (safebooru)** — ~500k предварительно выровненных аниме-лиц.
- Оба домена приводятся к квадрату `128×128`: реальные лица — выравниванием по
  ключевым точкам (MediaPipe), аниме — центральным кропом. В каждом домене выделен
  held-out тест.

Данные версионируются через **DVC** и в git не хранятся. Для быстрой проверки есть
**демо-выборка** (~800 МБ, 3000/3000 train + 500/500 test), которая скачивается
автоматически; полный датасет (~100 ГБ) загружается из открытых источников по
запросу (см. [Данные](#данные-1)).

### Метод

NOT (strong OT) обучает пару сетей: транспорт `T` (UNet) переносит A в B, потенциал
Канторовича `f` (ResNet без BatchNorm) их различает.

```
T_loss = MSE(X, T(X)) − f(T(X))      # на каждый шаг f делается t_iters шагов T
f_loss = f(T(X))      − f(Y)
```

Слагаемое `MSE(X, T(X))` штрафует сильные изменения пикселей, поэтому транспорт
получается «дешёвым» и cycle-consistency не нужна. Качество переноса отслеживается
метрикой **FID** во время обучения.

### Используемые библиотеки

[PyTorch](https://pytorch.org/) + [PyTorch Lightning](https://lightning.ai/) —
модели и цикл обучения; [Hydra](https://hydra.cc/) — конфиги;
[MLflow](https://mlflow.org/) — трекинг экспериментов;
[DVC](https://dvc.org/) — версионирование данных и моделей;
[torchmetrics](https://lightning.ai/docs/torchmetrics/) — FID;
[MediaPipe](https://developers.google.com/mediapipe) — выравнивание лиц.

## Технические детали

### Setup

Проект использует менеджер пакетов [uv](https://docs.astral.sh/uv/); PyTorch
ставится из индекса CUDA 12.8 (см. `pyproject.toml`).

```bash
uv sync                    # окружение + все зависимости (включая PyTorch)
uv sync --group dev        # dev-инструменты (pytest, jupyter, pre-commit)
uv run pre-commit install  # git-хуки
```

Python всегда вызывается через `uv run python ...`.

### Данные

Датасет скачивается автоматически: `train.py`/`infer.py` вызывают `ensure_data()`,
который при отсутствии данных загружает нужную выборку.

- **demo** (по умолчанию, `data.variant=demo`) — публичный `demo.zip` качается с
  Google Drive через `gdown` без аутентификации и распаковывается в `data/demo/`.
- **full** (`data.variant=full`) — CelebA (`gdown`) + AlignedAnimeFaces (Kaggle API,
  нужен `~/.kaggle/kaggle.json`), затем раскладка `scripts/prepare_dataset.py`.
  Чтобы не качать 100 ГБ повторно, можно указать уже существующие пути:

  ```bash
  uv run python train.py data.variant=full data.root_a=<celeba_dir> data.root_b=<anime_dir>
  ```

### Train

Единая точка входа — `train.py`; модель выбирается параметром `model_type`.

```bash
uv run python train.py                      # NOT (по умолчанию)
uv run python train.py model_type=cyclegan  # CycleGAN
```

Любой гиперпараметр переопределяется через Hydra-CLI:

```bash
uv run python train.py not_.t_iters=5 not_.t_lr=5e-5 data.batch_size=32
```

Метрики, гиперпараметры и графики обучения логируются в MLflow (адрес — в конфиге,
по умолчанию `http://127.0.0.1:8080`; локальный сервер: `uv run mlflow server
--host 127.0.0.1 --port 8080`).

### Infer

Точка входа — `infer.py`: принимает чекпойнт и изображение (или директорию),
сохраняет переносы в выходную папку.

```bash
uv run python infer.py \
    eval.checkpoint=<path/to/last.ckpt> \
    eval.input=data/demo/testA \
    eval.output_dir=outputs/eval
```

Вход — изображения `.jpg/.png/.webp/.bmp`; флаг `eval.align=true` включает
предварительное выравнивание лица.

### Overall — структура проекта

```
kaoanime-selfie2anime/
├── kaoanime/                # основной Python-пакет
│   ├── models/              # строительные блоки сетей (ResNet, UNet, PatchGAN, NOT-потенциал)
│   ├── losses/              # лоссы CycleGAN
│   ├── data/                # скачивание датасета (demo/full) и пути
│   ├── utils/               # датасеты, трансформы, выравнивание, FID
│   ├── inference/           # пайплайн инференса
│   ├── callbacks.py         # MLflow-чекпойнт-callback
│   ├── config*.py           # Hydra-конфиги (общий + NOT + CycleGAN)
│   ├── model_not.py         # LightningModule NOT
│   └── model_cyclegan.py    # LightningModule CycleGAN
├── scripts/                 # prepare_dataset, export_models, align_dataset
├── notebooks/               # исследовательские ноутбуки
├── tests/                   # pytest-тесты
├── train.py · infer.py · eval.py   # CLI точки входа
├── pyproject.toml · uv.lock        # зависимости
└── .pre-commit-config.yaml         # хуки качества кода
```

### Инференс и поставка

Обученная модель сохраняется как Lightning-чекпойнт (`.ckpt`) и логируется в MLflow.
Лучшие чекпойнты публикуются в DVC-remote `models` командой
`scripts/export_models.py`. Экспорт в ONNX/TensorRT и inference-сервер — в планах.
