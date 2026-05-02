import os
from pathlib import Path

# === Пути ===
BASE_DIR = Path(__file__).resolve().parent
SIGNS_IMAGES_DIR = BASE_DIR / "signs_images"
MODEL_PATTERN = "**/best.pt"
RUNS_DIR = BASE_DIR / "runs" / "detect"
DATASET_SAVE_PATH = "./dataset_roboflow"

# === Roboflow ===
ROBOFLOW_API_KEY = "7gmhvML7rFgZluA9mKwh"
ROBOFLOW_WORKSPACE = "s-workspace-imvbb"
ROBOFLOW_PROJECT = "find-signs-2-ku4tw"
ROBOFLOW_VERSION = 13

# === UI ===
WINDOW_GEOMETRY = "1400x900"
WINDOW_TITLE = "Детектор дорожных знаков"
THEME = "clam"
OVERLAY_HIDE_DELAY = 3000
PLAY_SPEEDS = ["0.25×", "0.5×", "1×", "2×", "4×"]
DEFAULT_PLAY_SPEED = 1.0
DEFAULT_CONFIDENCE = 0.25
DEFAULT_FPS = 30

STYLES = {
    'slider': {'troughcolor': '#e0e0e0', 'background': '#4a90e2', 'thumbcolor': '#4a90e2'},
    'button_active': '#4a90e2',
    'button_pressed': '#357abd',
    'overlay_bg': '#000000',
    'overlay_gradient_steps': 60
}

# === 🔑 МАППИНГ ИМЁН МОДЕЛИ → КЛЮЧИ БАЗЫ ===
# Сюда добавляйте точные названия классов, которые выводит ваша модель
CLASS_NAME_MAPPING = {
    'sign Give Way': 'sign_give_way',
    'Give Way': 'sign_give_way',
    'signNo parking': 'no_parking',
    'sign No parking pls': 'no_parking',
    'No parking': 'no_parking',
    'priority road': 'priority_road',
    'Main road': 'priority_road',
    'sign main road': 'priority_road',
    'no stopping': 'no_stopping',
    'No stopping': 'no_stopping',
    'pedestrian crossing': 'pedestrian_crossing',
    'speed limit': 'speed_limit',
    'sign Maximum speed limit 20': 'speed_limit',
    'sign Maximum speed limit 30': 'speed_limit',
    'sign Maximum speed limit 40': 'speed_limit',
    'sign Maximum speed limit 50': 'speed_limit',
    'sign Maximum speed limit 80': 'speed_limit',
    'no entry': 'no_entry',
    'sign No entry': 'no_entry',
    'sign Movement Prohibition': 'no_entry',
    'parking': 'parking',
    'sign One Way': 'sign',
    'sign move straight': 'sign',
    'sign overtaking is prohibited': 'warning',
    'sign right turn is prohibited': 'warning',
    'sign two-way traffic': 'warning',
    'sign Dangerous bend': 'warning',
    'traffic light': 'traffic_light',
    'warning': 'warning',
    'sign': 'sign'
}

# === База знаков (ключи должны совпадать со значениями в CLASS_NAME_MAPPING) ===
SIGNS_DATABASE = {
    'sign_give_way': {
        'name': 'Уступи дорогу',
        'description': 'Дорожный знак приоритета, предписывающий уступить дорогу ТС на пересекаемой дороге.',
        'image_path': str(SIGNS_IMAGES_DIR / 'sign_give_way.png'),
        'color': '#DC143C'
    },
    'no_parking': {
        'name': 'Остоновка запрещена',
        'description': 'Запрещается остановка более 5 минут транспортных средств.',
        'image_path': str(SIGNS_IMAGES_DIR / 'no_parking.png'),
        'color': '#1E90FF'
    },
    'priority_road': {
        'name': 'Главная дорога',
        'description': 'Дорога, на которой водитель имеет право преимущественного проезда нерегулируемых перекрёстков.',
        'image_path': str(SIGNS_IMAGES_DIR / 'sign_main_road.png'),
        'color': '#FFD700'
    },
    'no_stopping': {
        'name': 'Остановка запрещена',
        'description': 'Запрещается остановка и стоянка транспортных средств.',
        'image_path': str(SIGNS_IMAGES_DIR / 'no_stopping.png'),
        'color': '#1E90FF'
    },
    'pedestrian_crossing': {
        'name': 'Пешеходный переход',
        'description': 'Знак обозначает зону пешеходного перехода. Водители должны уступить дорогу пешеходам.',
        'image_path': str(SIGNS_IMAGES_DIR / 'pedestrian_crossing.png'),
        'color': '#4169E1'
    },
    'speed_limit': {
        'name': 'Ограничение скорости',
        'description': 'Запрещается движение со скоростью, превышающей указанную на знаке.',
        'image_path': str(SIGNS_IMAGES_DIR / 'speed_limit.png'),
        'color': '#DC143C'
    },
    'no_entry': {
        'name': 'Въезд запрещён',
        'description': 'Запрещается въезд всех транспортных средств в данном направлении.',
        'image_path': str(SIGNS_IMAGES_DIR / 'no_entry.png'),
        'color': '#DC143C'
    },
    'parking': {
        'name': 'Парковка',
        'description': 'Знак обозначает место, где разрешена стоянка транспортных средств.',
        'image_path': str(SIGNS_IMAGES_DIR / 'parking.png'),
        'color': '#4169E1'
    },
    'traffic_light': {
        'name': 'Светофорное регулирование',
        'description': 'Перекрёсток, движение на котором регулируется светофором.',
        'image_path': str(SIGNS_IMAGES_DIR / 'traffic_light.png'),
        'color': '#32CD32'
    },
    'warning': {
        'name': 'Прочие опасности',
        'description': 'Участок дороги, на котором имеются опасности.',
        'image_path': str(SIGNS_IMAGES_DIR / 'warning.png'),
        'color': '#FFD700'
    },
    'sign': {
        'name': 'Дорожный знак',
        'description': 'Общий дорожный знак',
        'image_path': str(SIGNS_IMAGES_DIR / 'default.png'),
        'color': '#808080'
    }
}

def ensure_dirs():
    SIGNS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)