import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import glob
from pathlib import Path
import config
from model_manager import ModelManager
from video_handler import VideoHandler
from ui_components import UIComponents


class TrafficSignDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(config.WINDOW_GEOMETRY)
        config.ensure_dirs()

        self.model_mgr = ModelManager(log_callback=self.log_message)
        self.video_hdl = VideoHandler()

        self.overlay_visible = True
        self.hide_timer = None
        self.sign_images_cache = {}
        self.current_signs_widgets = []
        self.obj_count = 0
        self.avg_accuracy = 0.0
        self.last_signs_signature = ""

        self.setup_ui()
        self.load_sign_images()
        self.bind_hotkeys()

    def setup_ui(self):
        style = ttk.Style()
        UIComponents.apply_styles(style)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.det_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.det_frame, text="🔍 Детекция")
        self.create_detection_tab()

        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="🎓 Обучение")
        self.create_training_tab()

        self.status_var = tk.StringVar(value="Загрузите модель для начала работы")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.FLAT,
                  anchor=tk.W, font=('Arial', 9)).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    def create_detection_tab(self):
        top_bar = ttk.Frame(self.det_frame)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.status_label = ttk.Label(top_bar, text="⚠️ Модель не загружена", font=('Arial', 10, 'bold'),
                                      foreground='red')
        self.status_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(top_bar, text="📂 Загрузить модель", command=self.load_trained_model).pack(side=tk.LEFT, padx=5)

        main_layout = ttk.PanedWindow(self.det_frame, orient=tk.HORIZONTAL)
        main_layout.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_panel = ttk.Frame(main_layout)
        main_layout.add(left_panel, weight=3)
        self.video_container = ttk.Frame(left_panel, relief="sunken", borderwidth=1)
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.display_label = tk.Label(self.video_container, text="Загрузите модель и видео", font=('Arial', 12),
                                      background='#1a1a1a', foreground='#888888', anchor=tk.CENTER)
        self.display_label.pack(fill=tk.BOTH, expand=True)

        self.overlay_cfg = UIComponents.create_overlay_controls(self.video_container, callbacks={})
        cfg = self.overlay_cfg

        # 🔥 ПРИВЯЗЫВАЕМ ТОЛЬКО ТЕ КНОПКИ, КОТОРЫЕ ОСТАЛИСЬ
        if 'load_btn' in cfg:
            cfg['load_btn'].config(command=self.load_file)
        if 'play_pause' in cfg:
            cfg['play_pause'].config(command=self.toggle_play_pause)


        cfg['video_slider'].bind("<ButtonPress-1>", self.on_slider_press)
        cfg['video_slider'].bind("<ButtonRelease-1>", self.on_slider_release)




        side_panel = ttk.Frame(main_layout)
        main_layout.add(side_panel, weight=1)
        ttk.Label(side_panel, text="🚦 Обнаруженные знаки:", font=('Arial', 11, 'bold')).pack(anchor=tk.W, pady=5)
        signs_cfg = UIComponents.create_signs_panel(side_panel, config.SIGNS_DATABASE)
        self.signs_canvas = signs_cfg['canvas']
        self.signs_scrollable = signs_cfg['frame']

        self.stats_frame = ttk.LabelFrame(side_panel, text="📊 Статистика", padding=5)
        self.stats_frame.pack(fill=tk.X, pady=5)
        self.info_text = tk.Text(self.stats_frame, height=8, width=35, state=tk.DISABLED, font=('Arial', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # 🔥 ПАНЕЛЬ: Детальная информация о знаке
        self.detail_frame = ttk.LabelFrame(side_panel, text="📋 Информация о знаке", padding=10)
        self.detail_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        detail_content = tk.Frame(self.detail_frame, bg='#ffffff')
        detail_content.pack(fill=tk.BOTH, expand=True)

        self.detail_image_label = tk.Label(detail_content, text="Нет данных",
                                           font=('Arial', 10), bg='#f5f5f5',
                                           width=150, height=150, relief='sunken')
        self.detail_image_label.pack(pady=10)

        self.detail_desc_text = tk.Text(detail_content, height=6, width=35,
                                        wrap=tk.WORD, font=('Arial', 9),
                                        bg='#f9f9f9', relief='flat')
        self.detail_desc_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.detail_desc_text.config(state=tk.DISABLED)

        self.detail_name_label = tk.Label(detail_content, text="",
                                          font=('Arial', 11, 'bold'),
                                          bg='#ffffff', fg='#DC143C')
        self.detail_name_label.pack(pady=(5, 0))

    def create_training_tab(self):
        left = ttk.Frame(self.train_frame)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        rf_frame = ttk.LabelFrame(left, text="☁️ Roboflow Dataset", padding=10)
        rf_frame.pack(fill=tk.X, pady=5)
        ttk.Label(rf_frame, text=f"Project: {config.ROBOFLOW_PROJECT}", font=('Arial', 10, 'bold')).pack(anchor=tk.W,
                                                                                                         pady=5)
        self.rf_download_btn = ttk.Button(rf_frame, text="📥 Скачать датасет", command=self.download_roboflow_dataset)
        self.rf_download_btn.pack(fill=tk.X, pady=5)

        self.train_cfg = UIComponents.create_training_panel(left)
        self.train_cfg['train_btn'].config(command=self.start_training)
        self.train_cfg['stop_btn'].config(command=self.stop_training)

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            right = ttk.Frame(self.train_frame)
            right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            graph_frame = ttk.LabelFrame(right, text=" Потери", padding=10)
            graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            self.figure, self.ax = plt.subplots(figsize=(5, 3))
            self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except ImportError:
            pass

    def load_file(self):
        if self.model_mgr.model is None:
            messagebox.showerror("Ошибка", "Сначала загрузите модель")
            return
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if not path: return
        success, res = self.video_hdl.load_video(path)
        if success:
            info = self.video_hdl.get_frame_info()
            self.overlay_cfg['video_slider'].config(from_=0, to=info['total'], value=0)
            self.overlay_cfg['time_total'].config(text=VideoHandler.frames_to_time(info['total'], info['fps']))
            self.overlay_cfg['frame_info'].config(text=f"Кадр: 0 / {info['total']}")
            self.status_var.set(f"📹 {os.path.basename(path)} | {info['fps']:.1f} FPS")

            # 🔥 ВКЛЮЧАЕМ ТОЛЬКО КНОПКУ PLAY
            if 'play_pause' in self.overlay_cfg:
                self.overlay_cfg['play_pause'].config(state=tk.NORMAL)
        else:
            messagebox.showerror("Ошибка", res)

    def render_frame(self, frame):
        if frame is None: return
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            w = self.display_label.winfo_width() or 1280
            h = self.display_label.winfo_height() or 720
            if w > 10 and h > 10:
                img.thumbnail((w, h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            self.display_label.config(image=photo)
            self.display_label.image = photo
        except Exception as e:
            print(f"⚠️ Ошибка рендера: {e}")

    def toggle_play_pause(self):
        if not self.video_hdl.capture or not self.video_hdl.capture.isOpened(): return
        if self.model_mgr.model is None:
            self.show_temp_message("⚠️ Загрузите модель!")
            return
        self.video_hdl.is_playing = not self.video_hdl.is_playing
        is_playing = self.video_hdl.is_playing

        if 'play_pause' in self.overlay_cfg:
            self.overlay_cfg['play_pause'].config(text="⏸" if is_playing else "▶",
                                                  background='#e74c3c' if is_playing else '#333')

        # ❌ Удалены строки, отключающие step_back/forward

        if is_playing and not self.video_hdl.is_processing:
            self.video_hdl.is_processing = True
            threading.Thread(target=self._video_loop, daemon=True).start()

    def _video_loop(self):
        while self.video_hdl.is_processing and self.video_hdl.capture and self.video_hdl.capture.isOpened():
            if not self.video_hdl.is_playing:
                time.sleep(0.05)
                continue

            start_t = time.time()
            if self.video_hdl.seek_frame is not None:
                self.video_hdl.capture.set(cv2.CAP_PROP_POS_FRAMES, self.video_hdl.seek_frame)
                self.video_hdl.seek_frame = None
                continue

            ret, frame = self.video_hdl.capture.read()
            if not ret:
                self.root.after(0, self.stop_processing)
                break

            current_pos = int(self.video_hdl.capture.get(cv2.CAP_PROP_POS_FRAMES))
            if self.model_mgr.model:
                results = self.model_mgr.infer_frame(frame)
                if results:
                    plotted = results.plot()
                    self.root.after(0, self.render_frame, plotted)
                    self.root.after(0, self.update_statistics, results)

            if not self.video_hdl.is_slider_dragging:
                self.root.after(0, self._update_slider, current_pos)

            if self.video_hdl.is_playing:
                elapsed = time.time() - start_t
                target = (1 / self.video_hdl.fps_video) / self.video_hdl.play_speed
                if target > elapsed:
                    time.sleep(target - elapsed)

    def _update_slider(self, pos):
        try:
            self.overlay_cfg['video_slider'].set(pos)
            self.overlay_cfg['frame_info'].config(text=f"Кадр: {pos} / {self.video_hdl.total_frames}")
            self.overlay_cfg['time_current'].config(text=VideoHandler.frames_to_time(pos, self.video_hdl.fps_video))
        except Exception:
            pass

    def step_frame(self, direction):
        # Этот метод нужен для горячих клавиш (Стрелки), даже если кнопок нет
        self.video_hdl.pause()
        self.video_hdl.step_frame(direction)
        self.update_frame_display()

    def update_frame_display(self):
        if self.video_hdl.capture and self.video_hdl.capture.isOpened() and self.model_mgr.model:
            ret, frame = self.video_hdl.capture.read()
            if ret:
                self.video_hdl.capture.set(cv2.CAP_PROP_POS_FRAMES,
                                           max(0, int(self.video_hdl.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1))
                ret, frame = self.video_hdl.capture.read()
                if ret:
                    results = self.model_mgr.infer_frame(frame)
                    if results:
                        self.render_frame(results.plot())
                        self.update_statistics(results)

    def stop_processing(self):
        self.video_hdl.stop()
        if 'play_pause' in self.overlay_cfg:
            self.overlay_cfg['play_pause'].config(text="▶", background='#333')
            self.overlay_cfg['play_pause'].config(state=tk.NORMAL)




    def on_slider_press(self, event):
        self.video_hdl.is_slider_dragging = True
        self.video_hdl.pause()

    def on_slider_release(self, event):
        if self.video_hdl.capture and self.video_hdl.capture.isOpened():
            target = int(float(self.overlay_cfg['video_slider'].get()))
            self.video_hdl.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.video_hdl.seek_frame = target
            self.update_frame_display()
        self.video_hdl.is_slider_dragging = False
        if self.video_hdl.is_playing:
            self.video_hdl.is_playing = True
            if 'play_pause' in self.overlay_cfg:
                self.overlay_cfg['play_pause'].config(text="", background='#e74c3c')






    def cancel_hide_timer(self):
        if self.hide_timer:
            self.root.after_cancel(self.hide_timer)
            self.hide_timer = None



    def toggle_fullscreen(self):
        is_fs = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not is_fs)
        # 🔥 Защита: обновляем иконку только если кнопка существует
        if 'fullscreen_btn' in self.overlay_cfg:
            self.overlay_cfg['fullscreen_btn'].config(text="❐" if not is_fs else "⛶")

    def show_temp_message(self, text, duration=1500):
        msg = tk.Label(self.video_container, text=text, font=('Arial', 10, 'bold'), background='#222',
                       foreground='white', relief=tk.FLAT, padx=15, pady=8)
        msg.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
        self.root.after(duration, msg.destroy)

    def update_statistics(self, results):
        boxes = results.boxes
        self.obj_count = len(boxes)
        class_counts = {}
        class_confs = {}

        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.model_mgr.model.names[cls_id]
            class_counts.setdefault(label, 0)
            class_confs.setdefault(label, [])
            class_counts[label] += 1
            class_confs[label].append(conf)

        current_signature = str(sorted(class_counts.items()))

        if current_signature == self.last_signs_signature:
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            total_conf = sum(np.mean(c) for c in class_confs.values()) if class_confs else 0
            self.avg_accuracy = total_conf / len(class_confs) * 100 if class_confs else 0
            for lbl, confs in class_confs.items():
                self.info_text.insert(tk.END, f"✓ {lbl}: {len(confs)} шт. | {sum(confs) / len(confs):.1%}\n")
            self.info_text.config(state=tk.DISABLED)
            self.status_var.set(f"Всего: {self.obj_count} | Точность: {self.avg_accuracy:.1f}%")

            if not class_counts:
                self.detail_name_label.config(text="Нет обнаруженных знаков", fg='#808080')
                self.detail_desc_text.config(state=tk.NORMAL)
                self.detail_desc_text.delete(1.0, tk.END)
                self.detail_desc_text.insert(tk.END, "Ожидаю обнаружение дорожных знаков...")
                self.detail_desc_text.config(state=tk.DISABLED)
                self.detail_image_label.config(image="", text="Нет данных")
            return

        self.last_signs_signature = current_signature

        for widget in self.current_signs_widgets:
            widget.destroy()
        self.current_signs_widgets = []

        for label, count in class_counts.items():
            avg_conf = sum(class_confs[label]) / len(class_confs[label])
            mapped_label = config.CLASS_NAME_MAPPING.get(label, label)

            if mapped_label not in self.sign_images_cache:
                sign_info = config.SIGNS_DATABASE.get(mapped_label)
                if sign_info:
                    img_path = sign_info['image_path']
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).resize((80, 80), Image.Resampling.LANCZOS)
                            self.sign_images_cache[mapped_label] = ImageTk.PhotoImage(img)
                        except Exception as e:
                            print(f"️ Ошибка загрузки {img_path}: {e}")
                    else:
                        default_path = str(config.SIGNS_IMAGES_DIR / 'default.png')
                        if os.path.exists(default_path) and 'default' not in self.sign_images_cache:
                            try:
                                img = Image.open(default_path).resize((80, 80), Image.Resampling.LANCZOS)
                                self.sign_images_cache['default'] = ImageTk.PhotoImage(img)
                                self.sign_images_cache[mapped_label] = self.sign_images_cache['default']
                            except:
                                pass
                else:
                    print(f"⚠️ Нет записи в базе для класса: {mapped_label}")

            sign_info = self.model_mgr.get_sign_info(label)
            card = UIComponents.create_sign_card(
                self.signs_scrollable, mapped_label, sign_info, count, avg_conf, self.sign_images_cache
            )
            self.current_signs_widgets.append(card)

        if class_counts:
            most_detected_label = max(class_counts.items(), key=lambda x: x[1])[0]
            sign_info = self.model_mgr.get_sign_info(most_detected_label)
            mapped_name = config.CLASS_NAME_MAPPING.get(most_detected_label, most_detected_label)

            if mapped_name not in self.sign_images_cache:
                self._load_sign_image_dynamic(mapped_name)

            self.update_sign_detail(most_detected_label, sign_info, self.sign_images_cache)
        else:
            self.detail_name_label.config(text="Нет обнаруженных знаков", fg='#808080')
            self.detail_desc_text.config(state=tk.NORMAL)
            self.detail_desc_text.delete(1.0, tk.END)
            self.detail_desc_text.insert(tk.END, "Ожидаю обнаружение дорожных знаков...")
            self.detail_desc_text.config(state=tk.DISABLED)
            self.detail_image_label.config(image="", text="Нет данных")

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        total_conf = sum(np.mean(c) for c in class_confs.values()) if class_confs else 0
        self.avg_accuracy = total_conf / len(class_confs) * 100 if class_confs else 0
        for lbl, confs in class_confs.items():
            self.info_text.insert(tk.END, f"✓ {lbl}: {len(confs)} шт. | {sum(confs) / len(confs):.1%}\n")
        self.info_text.config(state=tk.DISABLED)
        self.status_var.set(f"Всего: {self.obj_count} | Точность: {self.avg_accuracy:.1f}%")

        self.signs_canvas.update_idletasks()
        self.signs_canvas.configure(scrollregion=self.signs_canvas.bbox("all"))

    def load_trained_model(self):
        success, res = self.model_mgr.load_model()
        if success:
            print("📋 Имена классов в модели:")
            for i, name in enumerate(self.model_mgr.model.names.values()):
                print(f"  {i}: '{name}'")
            self.status_label.config(text="✅ Модель загружена", foreground='green')
            self.status_var.set(f"Модель: {os.path.basename(res)}")
            if 'load_btn' in self.overlay_cfg:
                self.overlay_cfg['load_btn'].config(state=tk.NORMAL)
        else:
            messagebox.showerror("Ошибка", res)

    def download_roboflow_dataset(self):
        self.rf_download_btn.config(state=tk.DISABLED)
        self.model_mgr.download_dataset(
            on_complete=lambda p: self.root.after(0, lambda: self.dataset_loaded(p)),
            on_error=lambda e: self.root.after(0, lambda: self.rf_download_btn.config(state=tk.NORMAL))
        )

    def dataset_loaded(self, path):
        self.train_cfg['train_btn'].config(state=tk.NORMAL)
        self.log_message(f"✅ Датасет: {path}")

    def start_training(self):
        params = {
            'epochs': int(self.train_cfg['entries']['Эпохи:'].get()),
            'batch': int(self.train_cfg['entries']['Batch size:'].get()),
            'imgsz': int(self.train_cfg['entries']['Размер:'].get()),
            'device': self.train_cfg['entries']['device'].get()
        }
        yaml_path = os.path.join(self.model_mgr.dataset_path, 'data.yaml')
        self.model_mgr.start_training(params, yaml_path, on_progress=self.update_progress,
                                      on_complete=self.on_training_complete, on_error=self.on_training_error)
        self.train_cfg['train_btn'].config(state=tk.DISABLED)
        self.train_cfg['stop_btn'].config(state=tk.NORMAL)

    def stop_training(self):
        self.model_mgr.stop_training()
        self.train_cfg['stop_btn'].config(state=tk.DISABLED)

    def on_training_complete(self, model_path):
        messagebox.showinfo("Успех", f"✅ Обучение завершено!\nМодель: {model_path}")
        self.train_cfg['train_btn'].config(state=tk.NORMAL)

    def on_training_error(self, error):
        messagebox.showerror("Ошибка", error)
        self.train_cfg['train_btn'].config(state=tk.NORMAL)
        self.train_cfg['stop_btn'].config(state=tk.DISABLED)

    def update_progress(self, value):
        self.train_cfg['progress_bar']['value'] = value

    def log_message(self, msg):
        self.train_cfg['log_text'].config(state=tk.NORMAL)
        self.train_cfg['log_text'].insert(tk.END, msg + "\n")
        self.train_cfg['log_text'].see(tk.END)
        self.train_cfg['log_text'].config(state=tk.DISABLED)

    def bind_hotkeys(self):
        self.root.bind('<space>', lambda e: self.toggle_play_pause())
        self.root.bind('<Left>', lambda e: self.step_frame(-1))
        self.root.bind('<Right>', lambda e: self.step_frame(1))
        self.root.bind('<f>', lambda e: self.toggle_fullscreen())
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False)
        if self.root.attributes('-fullscreen') else None)

    def load_sign_images(self):
        print("📂 Загрузка изображений знаков...")
        for class_name, info in config.SIGNS_DATABASE.items():
            img_path = info['image_path']
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).resize((80, 80), Image.Resampling.LANCZOS)
                    self.sign_images_cache[class_name] = ImageTk.PhotoImage(img)
                except Exception as e:
                    print(f"❌ Ошибка: {e}")

    def _load_sign_image_dynamic(self, db_key):
        if db_key in self.sign_images_cache:
            return
        sign_info = config.SIGNS_DATABASE.get(db_key)
        img_path = sign_info['image_path'] if sign_info else None
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).resize((80, 80), Image.Resampling.LANCZOS)
                self.sign_images_cache[db_key] = ImageTk.PhotoImage(img)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {img_path}: {e}")
        else:
            default_path = str(config.SIGNS_IMAGES_DIR / 'default.png')
            if os.path.exists(default_path):
                try:
                    img = Image.open(default_path).resize((80, 80), Image.Resampling.LANCZOS)
                    self.sign_images_cache[db_key] = ImageTk.PhotoImage(img)
                except:
                    pass

    def update_sign_detail(self, class_name, sign_info, image_cache):
        if not sign_info: return
        self.detail_name_label.config(text=sign_info['name'], fg=sign_info['color'])
        self.detail_desc_text.config(state=tk.NORMAL)
        self.detail_desc_text.delete(1.0, tk.END)
        self.detail_desc_text.insert(tk.END, sign_info['description'])
        self.detail_desc_text.config(state=tk.DISABLED)

        mapped_name = config.CLASS_NAME_MAPPING.get(class_name, class_name)
        if mapped_name in image_cache:
            img = image_cache[mapped_name]
            self.detail_image_label.config(image=img, text="", width=150, height=150)
            self.detail_image_label.image = img
        else:
            img_path = sign_info.get('image_path')
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).resize((120, 120), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.detail_image_label.config(image=photo, text="", width=150, height=150)
                    self.detail_image_label.image = photo
                    image_cache[mapped_name] = photo
                except Exception as e:
                    self.detail_image_label.config(image="", text="❌ Нет изображения")
            else:
                self.detail_image_label.config(image="", text="❌ Нет изображения")

    def on_closing(self):
        self.video_hdl.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()