import tkinter as tk
from tkinter import ttk
import config


class UIComponents:
    @staticmethod
    def apply_styles(style: ttk.Style):
        style.theme_use(config.THEME)
        style.configure('Horizontal.TScale',
                        troughcolor=config.STYLES['slider']['troughcolor'],
                        background=config.STYLES['slider']['background'])
        style.map('Modern.TButton',
                  background=[('active', config.STYLES['button_active']),
                              ('pressed', config.STYLES['button_pressed'])],
                  foreground=[('active', 'white'), ('pressed', 'white')])

    @staticmethod
    def create_overlay_controls(parent, callbacks):
        overlay = tk.Frame(parent, background=config.STYLES['overlay_bg'])
        overlay.place(relx=0, rely=1, relwidth=1, height=60, anchor='sw')
        canvas = tk.Canvas(overlay, height=60, highlightthickness=0, background='#000000')
        canvas.pack(fill=tk.X, side=tk.BOTTOM)
        for i in range(config.STYLES['overlay_gradient_steps']):
            alpha = int(120 * (i / config.STYLES['overlay_gradient_steps']))
            color = f'#{alpha:02x}{alpha:02x}{alpha:02x}'
            canvas.create_line(0, i, 3000, i, fill=color)
        controls = tk.Frame(canvas, background=config.STYLES['overlay_bg'])
        controls.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.95)

        slider_frame = tk.Frame(controls, background=config.STYLES['overlay_bg'])
        slider_frame.pack(fill=tk.X, pady=(8, 4))
        time_current = tk.Label(slider_frame, text="00:00", font=('Arial', 9),
                                foreground='white', background=config.STYLES['overlay_bg'])
        time_current.pack(side=tk.LEFT)
        video_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        video_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        time_total = tk.Label(slider_frame, text="00:00", font=('Arial', 9),
                              foreground='white', background=config.STYLES['overlay_bg'])
        time_total.pack(side=tk.RIGHT)
        frame_info = tk.Label(slider_frame, text="Кадр: 0 / 0", font=('Arial', 9),
                              foreground='white', background=config.STYLES['overlay_bg'])
        frame_info.pack(side=tk.RIGHT, padx=10)

        btns = tk.Frame(controls, background=config.STYLES['overlay_bg'])
        btns.pack(fill=tk.X)
        left_btns = tk.Frame(btns, background=config.STYLES['overlay_bg'])
        left_btns.pack(side=tk.LEFT)
        load_btn = tk.Button(left_btns, text="📁", font=('Arial', 11),
                             background='#333', foreground='white', relief=tk.FLAT,
                             padx=12, pady=4, cursor='hand2')
        load_btn.pack(side=tk.LEFT, padx=2)
        tk.Frame(left_btns, width=1, background='#666').pack(side=tk.LEFT, padx=8, fill=tk.Y)


        center_btns = tk.Frame(btns, background=config.STYLES['overlay_bg'])
        center_btns.pack(side=tk.LEFT, expand=True)

        play_pause = tk.Button(center_btns, text="▶", font=('Arial', 14, 'bold'),
                               background='#333', foreground='white', relief=tk.FLAT,
                               padx=16, pady=4, cursor='hand2')
        play_pause.pack(side=tk.LEFT, padx=8)


        right_btns = tk.Frame(btns, background=config.STYLES['overlay_bg'])
        right_btns.pack(side=tk.RIGHT)
        stop_btn = tk.Button(right_btns, text="⏹", font=('Arial', 11),
                             background='#333', foreground='white', relief=tk.FLAT,
                             padx=12, pady=4, cursor='hand2')
        stop_btn.pack(side=tk.LEFT, padx=2)


        return {
            'overlay': overlay, 'canvas': canvas, 'controls': controls,
            'time_current': time_current, 'time_total': time_total,
            'video_slider': video_slider, 'frame_info': frame_info,
            'load_btn': load_btn,
            'play_pause': play_pause, 'stop_btn': stop_btn
        }

    @staticmethod
    def create_signs_panel(parent, signs_db):
        container = ttk.Frame(parent)
        canvas = tk.Canvas(container, highlightthickness=0, bg='#f0f0f0')  # Явный фон
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        # Создаем окно внутри Canvas
        window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")

        # Привязываем изменение размера Canvas к ширине внутренней рамки
        def _configure_canvas(event):
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind('<Configure>', _configure_canvas)

        # Обновление scrollregion при изменении размера внутренней рамки
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return {'container': container, 'canvas': canvas, 'frame': scrollable}

    @staticmethod
    def create_sign_card(parent, class_name, sign_info, count, avg_conf, sign_images_cache):
        # ✅ tk.Frame вместо ttk.Frame устраняет конфликты отрисовки
        card = tk.Frame(parent, bg='#ffffff', relief='solid', bd=1)
        card.pack(fill=tk.X, padx=5, pady=5, ipady=5)

        header = tk.Frame(card, bg='#ffffff')
        header.pack(fill=tk.X, padx=5, pady=5)

        if class_name in sign_images_cache:
            # ✅ Явный фон + защита от GC
            img_lbl = tk.Label(header, image=sign_images_cache[class_name], bg='#ffffff')
            img_lbl.image = sign_images_cache[class_name]
            img_lbl.pack(side=tk.LEFT, padx=10, pady=2)
        else:
            # Заглушка для отладки
            debug_lbl = tk.Label(header, text=f"❓ {class_name}", font=('Arial', 10),
                                 bg='#ffcccc', width=10, height=2)
            debug_lbl.pack(side=tk.LEFT, padx=10, pady=5)

        text_frame = tk.Frame(card, bg='#ffffff')
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(text_frame, text=sign_info['name'], font=('Arial', 11, 'bold'),
                 fg=sign_info['color'], bg='#ffffff', anchor='w').pack(fill=tk.X)
        tk.Label(text_frame, text=f"Обнаружено: {count} шт.", font=('Arial', 9),
                 bg='#ffffff', anchor='w').pack(fill=tk.X)

        desc = tk.Text(card, height=3, width=40, wrap=tk.WORD, font=('Arial', 9),
                       relief=tk.FLAT, bg='#F5F5F5')
        desc.insert(tk.END, sign_info['description'])
        desc.config(state=tk.DISABLED)
        desc.pack(padx=10, pady=5, fill=tk.X)

        conf_frame = tk.Frame(card, bg='#ffffff')
        conf_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(conf_frame, text="Точность:", bg='#ffffff').pack(side=tk.LEFT)
        conf_bar = ttk.Progressbar(conf_frame, length=150, mode='determinate', maximum=100)
        conf_bar.pack(side=tk.LEFT, padx=5)
        conf_bar['value'] = avg_conf * 100
        conf_text = tk.Label(conf_frame, text=f"{avg_conf:.1%}", font=('Arial', 9, 'bold'),
                             fg='green' if avg_conf > 0.7 else 'orange', bg='#ffffff')
        conf_text.pack(side=tk.LEFT)

        return card

    @staticmethod
    def create_training_panel(parent):
        main = ttk.Frame(parent)
        params = ttk.LabelFrame(main, text="⚙️ Параметры", padding=10)
        params.pack(fill=tk.X, pady=5)
        entries = {}
        for i, (label, default) in enumerate([("Эпохи:", "50"), ("Batch size:", "16"), ("Размер:", "640")]):
            var = tk.StringVar(value=default)
            ttk.Label(params, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(params, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=5)
            entries[label] = var
        ttk.Label(params, text="Устройство:").grid(row=3, column=0, sticky=tk.W, pady=5)
        device_var = tk.StringVar(value="cpu")
        device_combo = ttk.Combobox(params, textvariable=device_var, values=["cpu", "0"], width=10, state="readonly")
        device_combo.grid(row=3, column=1, padx=5)
        entries['device'] = device_var

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=20)
        train_btn = ttk.Button(btn_frame, text="🚀 Обучить модель")
        train_btn.pack(fill=tk.X, pady=5)
        stop_btn = ttk.Button(btn_frame, text="⏹ Остановить", state=tk.DISABLED)
        stop_btn.pack(fill=tk.X, pady=5)

        progress = ttk.LabelFrame(main, text="📊 Прогресс", padding=10)
        progress.pack(fill=tk.X, pady=5)
        progress_bar = ttk.Progressbar(progress, maximum=100)
        progress_bar.pack(fill=tk.X, pady=5)
        progress_lbl = ttk.Label(progress, text="Ожидание...")
        progress_lbl.pack()

        log_frame = ttk.LabelFrame(main, text="📝 Лог", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        log_text = tk.Text(log_frame, height=15, width=60, state=tk.DISABLED)
        log_text.pack(fill=tk.BOTH, expand=True)

        return {'main': main, 'entries': entries, 'train_btn': train_btn, 'stop_btn': stop_btn,
                'progress_bar': progress_bar, 'progress_lbl': progress_lbl, 'log_text': log_text}