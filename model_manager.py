import os
import glob
import yaml
import threading
from collections import deque
from ultralytics import YOLO

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

import config


class ModelManager:
    def __init__(self, log_callback=None, progress_callback=None):
        self.model = None
        self.trained_model_path = None
        self.is_training = False
        self.dataset_path = ""
        self.dataset_ready = False
        self.log_callback = log_callback or (lambda msg: print(msg))
        self.progress_callback = progress_callback or (lambda v: None)
        self.training_log = deque(maxlen=100)
        self.loss_history = {'box': [], 'cls': [], 'dfl': []}

    def load_model(self, model_path=None):
        if model_path is None:
            matches = glob.glob(str(config.BASE_DIR / config.MODEL_PATTERN), recursive=True)
            if not matches:
                return False, "Модель best.pt не найдена"
            model_path = os.path.abspath(matches[0])
        try:
            self.model = YOLO(model_path)
            self.trained_model_path = model_path
            self.log_callback(f"✅ Модель загружена: {model_path}")
            return True, model_path
        except Exception as e:
            self.log_callback(f"❌ Ошибка загрузки модели: {e}")
            return False, str(e)

    def infer_frame(self, frame, conf=config.DEFAULT_CONFIDENCE):
        if self.model is None:
            return None
        return self.model(frame, conf=conf, verbose=False)[0]

    def get_sign_info(self, class_name):
        """Возвращает информацию о знаке, применяя маппинг имён"""
        mapped_name = config.CLASS_NAME_MAPPING.get(class_name, class_name)
        return config.SIGNS_DATABASE.get(mapped_name, {
            'name': class_name,
            'description': 'Дорожный знак',
            'image_path': str(config.SIGNS_IMAGES_DIR / 'default.png'),
            'color': '#808080'
        })

    def download_dataset(self, on_complete=None, on_error=None):
        if not ROBOFLOW_AVAILABLE:
            error = "Установите: pip install roboflow"
            self.log_callback(f"❌ {error}")
            if on_error: on_error(error)
            return
        if not config.ROBOFLOW_API_KEY:
            error = "API ключ Roboflow не настроен"
            self.log_callback(f"❌ {error}")
            if on_error: on_error(error)
            return

        def _download_thread():
            try:
                yaml_path = os.path.join(config.DATASET_SAVE_PATH, 'data.yaml')
                if os.path.exists(yaml_path):
                    self.dataset_path = config.DATASET_SAVE_PATH
                    self.dataset_ready = True
                    self.log_callback("✅ Датасет уже загружен")
                    if on_complete: on_complete(self.dataset_path)
                    return
                self.log_callback("📥 Скачивание датасета Roboflow...")
                rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
                workspace = rf.workspace(config.ROBOFLOW_WORKSPACE)
                project = workspace.project(config.ROBOFLOW_PROJECT)
                version = project.version(config.ROBOFLOW_VERSION)
                dataset = version.download("yolov8", config.DATASET_SAVE_PATH)
                self.dataset_path = config.DATASET_SAVE_PATH
                self.dataset_ready = True
                self.log_callback(f"✅ Датасет загружен: {self.dataset_path}")
                if on_complete: on_complete(self.dataset_path)
            except Exception as ex:
                error_msg = str(ex)
                self.log_callback(f"❌ Ошибка: {error_msg}")
                if on_error: on_error(error_msg)

        threading.Thread(target=_download_thread, daemon=True).start()

    def start_training(self, params, data_yaml_path, on_progress=None, on_complete=None, on_error=None):
        if not self.dataset_ready:
            error = "Датасет не загружен"
            self.log_callback(f"❌ {error}")
            if on_error: on_error(error)
            return
        self.is_training = True
        self.loss_history = {'box': [], 'cls': [], 'dfl': []}

        def _training_thread():
            try:
                model = YOLO("yolo11n.pt")
                with open(data_yaml_path, 'r', encoding='utf-8') as f:
                    yaml_content = yaml.safe_load(f)
                train_path = val_path = None
                for train_sub, val_sub in [('images/train', 'images/val'), ('train', 'valid'), ('Dataset/train', 'Dataset/val')]:
                    t_full = os.path.join(self.dataset_path, train_sub)
                    v_full = os.path.join(self.dataset_path, val_sub)
                    if os.path.exists(t_full) and os.path.exists(v_full):
                        train_path = os.path.abspath(t_full)
                        val_path = os.path.abspath(v_full)
                        break
                if not train_path:
                    raise Exception("Не найдены папки train/val в датасете")
                fixed_yaml = {
                    'train': train_path.replace('\\', '/'),
                    'val': val_path.replace('\\', '/'),
                    'nc': yaml_content.get('nc', len(yaml_content.get('names', []))),
                    'names': yaml_content.get('names', [])
                }
                fixed_yaml_path = os.path.join(self.dataset_path, 'data_fixed.yaml')
                with open(fixed_yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(fixed_yaml, f, default_flow_style=False, allow_unicode=True)
                self.log_callback(f"✅ Обучение: epochs={params['epochs']}, batch={params['batch']}, imgsz={params['imgsz']}")
                results = model.train(
                    data=fixed_yaml_path, epochs=params['epochs'], batch=params['batch'],
                    imgsz=params['imgsz'], device=params['device'], project=str(config.RUNS_DIR),
                    name='traffic_signs', exist_ok=True, verbose=True,
                )
                matches = glob.glob(str(config.RUNS_DIR / 'traffic_signs' / 'weights' / 'best.pt'))
                if matches:
                    self.trained_model_path = os.path.abspath(matches[0])
                    self.log_callback(f"✅ Обучение завершено: {self.trained_model_path}")
                    if on_complete: on_complete(self.trained_model_path)
                else:
                    raise Exception("Файл best.pt не найден после обучения")
            except Exception as ex:
                error_msg = str(ex)
                self.log_callback(f"❌ Ошибка обучения: {error_msg}")
                if on_error: on_error(error_msg)
            finally:
                self.is_training = False

        threading.Thread(target=_training_thread, daemon=True).start()

    def stop_training(self):
        self.is_training = False
        self.log_callback("⏹ Обучение остановлено")