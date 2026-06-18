# KaoAnime — перенос стиля «селфи → аниме»

KaoAnime превращает реальное фото лица в аниме-портрет с помощью нейросетевого
image-to-image переноса между двумя доменами без парных данных. Основная модель —
**Neural Optimal Transport (NOT)** (Korotin et al., ICLR 2023); в репозитории также
есть альтернативная реализация **CycleGAN**, разделяющая с NOT общие строительные блоки.

## Смысловое содержание проекта

### Задача

Дано два неспаренных набора изображений лиц:

- **домен A** — реальные фотографии лиц (CelebA);
- **домен B** — аниме-лица (safebooru).

Парных соответствий «фото ↔ его аниме-версия» нет, поэтому задача решается методами
unpaired image-to-image translation. Цель — обучить отображение `T: A → B`, которое
переносит фото в аниме-стиль, сохраняя идентичность и геометрию лица.

### Данные

- **CelebA** — крупный публичный датасет лиц знаменитостей (~200k изображений
  `178×218`). В проекте используется выровненная версия `img_align_celeba` с
  дополнительной фильтрацией по полу (women-only) и нормализацией кропа.
- **Anime faces (safebooru)** — набор предварительно выровненных аниме-лиц.
- Оба домена приводятся к каноническому квадратному кропу `128×128`:
  - реальные лица — выравниванием по лицевым ключевым точкам (MediaPipe Face Landmarker);
  - аниме-лица — фиксированным центральным кропом (они уже выровнены).
- Под отложенный тест в каждом домене вынесено по held-out поднабору изображений.

Датасет (~411 МБ, 7000 изображений) версионируется через **DVC** (`data.dvc`).

### Метод

**Neural Optimal Transport (strong OT)** обучает пару сетей:

- `T` — транспортное отображение (UNet-генератор), переносящее A в B;
- `f` — потенциал Канторовича (ResNet-классификатор без BatchNorm/spectral norm).

Шаг оптимизации:

```
# внутренний цикл по T (t_iters раз):
T_loss = MSE(X, T(X)) − f(T(X)).mean()      # эффективный транспорт + «обмануть» потенциал
# обновление f (один раз):
f_loss = f(T(X)).mean() − f(Y).mean()        # максимизировать разделение по Канторовичу
```

Слагаемое `MSE(X, T(X))` штрафует большие изменения пикселей, поэтому транспорт
получается «дешёвым» — cycle-consistency, как в CycleGAN, не требуется.

Качество переноса контролируется метрикой **FID** (Fréchet Inception Distance),
которая считается по скользящему окну во время обучения.

### Используемые библиотеки

- [PyTorch](https://pytorch.org/) — определение и обучение моделей.
- [PyTorch Lightning](https://lightning.ai/) — цикл обучения и инфраструктура.
- [Hydra](https://hydra.cc/) — конфигурирование гиперпараметров.
- [MLflow](https://mlflow.org/) — трекинг экспериментов.
- [DVC](https://dvc.org/) — версионирование данных и моделей.
- [torchmetrics](https://lightning.ai/docs/torchmetrics/) — расчёт FID.
- [MediaPipe](https://developers.google.com/mediapipe) — выравнивание лиц по ключевым точкам.

---

## Технические детали

### Setup

Проект использует пакетный менеджер [uv](https://docs.astral.sh/uv/). Зависимости
описаны в `pyproject.toml`, версии зафиксированы в `uv.lock`. PyTorch ставится из
индекса CUDA 12.8 (см. `[[tool.uv.index]]` в `pyproject.toml`).

```bash
# 1. установить uv (если ещё не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. создать виртуальное окружение и поставить все зависимости (включая PyTorch)
uv sync

# 3. поставить dev-инструменты (pytest, jupyter, pre-commit)
uv sync --group dev

# 4. настроить git-хуки
uv run pre-commit install

# 5. (опционально) проверить, что всё в порядке
uv run pre-commit run -a
uv run pytest
```

Все вызовы Python выполняются через `uv run python ...`, чтобы использовать
окружение проекта.

### Данные (DVC)

Данные и модели версионируются через DVC на двух Google Drive remote
(`data` — демо-датасет, `models` — обученные модели), в git не хранятся.

Демо-датасет (по умолчанию, `data.variant=demo`) скачивается автоматически:
`train.py`/`infer.py` вызывают `ensure_data()`, который при отсутствии данных
делает `dvc pull`. Вручную:

```bash
uv run dvc pull -r data data/demo
```

Полный датасет (`data.variant=full`) скачивается `download_data()`: CelebA через
`gdown`, AlignedAnimeFaces через Kaggle API (нужен `~/.kaggle/kaggle.json`).
Раскладка по директориям — `scripts/prepare_dataset.py`. Чтобы не качать 100 ГБ
заново, можно указать существующие пути:

```bash
uv run python train.py data.variant=full \
    data.root_a=<celeba_dir> data.root_b=<anime_dir>
```

Экспорт лучших моделей в remote `models` (сохраняет `.pt` и пушит в DVC):

```bash
uv run python scripts/export_models.py --checkpoints checkpoints/not_ep10.ckpt
```

### Train

Единая точка входа — `train.py`. Модель выбирается параметром `model_type`
(`not` — по умолчанию, либо `cyclegan`).

```bash
# обучить NOT-модель (по умолчанию)
uv run python train.py

# обучить CycleGAN
uv run python train.py model_type=cyclegan
```

Любой гиперпараметр переопределяется через Hydra-CLI:

```bash
uv run python train.py model_type=not not_.t_iters=5 not_.t_lr=5e-5 data.batch_size=32
```

Метрики и графики обучения логируются в MLflow (адрес задаётся в конфиге,
по умолчанию `http://127.0.0.1:8080`). Поднять локальный сервер для тестов:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

#### Препроцессинг (опционально)

Для офлайн-выравнивания датасета используется вспомогательный скрипт:

```bash
# реальные лица — выравнивание по ключевым точкам MediaPipe
uv run python scripts/align_dataset.py --input  <src> --output data/celeba_aligned \
    --mode real --size 128 --workers 8

# аниме-лица — центральный кроп
uv run python scripts/align_dataset.py --input  <src> --output data/anime_aligned \
    --mode center-crop --size 128 --workers 8
```

### Infer

Точка входа в инференс — `infer.py` (публичный API). Принимает чекпойнт и путь к
изображению или директории, сохраняет переносы в выходную директорию.

```bash
uv run python infer.py \
    eval.checkpoint=outputs/<run>/last.ckpt \
    eval.input=data/selfie2anime/testA \
    eval.output_dir=outputs/eval
```

Формат входных данных: изображения `.jpg/.jpeg/.png/.webp/.bmp`. На вход подаётся
одиночный файл или директория с файлами; на выходе — аниме-версии под теми же именами.
Флаг `eval.align=true` включает предварительное выравнивание лица.

### Overall — структура проекта

```
kaoanime-selfie2anime/
├── kaoanime/                  # основной Python-пакет
│   ├── models/                # строительные блоки сетей
│   │   ├── resnet.py          #   ResNet-генератор и дискриминатор
│   │   ├── unet.py            #   UNet-генератор (транспорт T для NOT)
│   │   ├── patch_discriminator.py
│   │   ├── not_potential.py   #   потенциал Канторовича f
│   │   └── weights_init.py    #   инициализация весов
│   ├── losses/                # лоссы CycleGAN
│   ├── utils/                 # датасеты, трансформы, выравнивание, FID-расписание
│   ├── inference/             # пайплайн инференса
│   ├── callbacks.py           # MLflow-чекпойнт-callback
│   ├── config.py              # единая точка входа в конфиги (Hydra)
│   ├── config_cyclegan.py     # гиперпараметры CycleGAN
│   ├── config_not.py          # гиперпараметры NOT
│   ├── model_cyclegan.py      # LightningModule CycleGAN
│   └── model_not.py           # LightningModule NOT
├── scripts/
│   └── align_dataset.py       # офлайн-выравнивание датасета
├── notebooks/                 # исследовательские ноутбуки (EDA, отладка пайплайна)
├── tests/                     # pytest-тесты
├── train.py                   # CLI: обучение
├── infer.py                   # CLI: инференс (публичный API)
├── eval.py                    # CLI: оценка на тесте
├── data.dvc                   # DVC-указатель на датасет
├── pyproject.toml             # зависимости и настройки инструментов
├── uv.lock                    # зафиксированные версии зависимостей
└── .pre-commit-config.yaml    # хуки качества кода
```

### Inference / внедрение

Натренированная модель сохраняется как Lightning-чекпойнт (`.ckpt`) и логируется
как артефакт в MLflow. Для продакшена планируется экспорт в ONNX и TensorRT, а также
поднятие inference-сервера (см. разделы Production preparation / Inference server —
_в разработке_).
