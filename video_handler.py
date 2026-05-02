import cv2
import time
from PIL import Image, ImageTk
import config

class VideoHandler:
    def __init__(self):
        self.capture = None
        self.is_processing = False
        self.is_playing = False
        self.play_speed = config.DEFAULT_PLAY_SPEED
        self.total_frames = 0
        self.fps_video = config.DEFAULT_FPS
        self.current_fps = 0
        self.seek_frame = None
        self.is_slider_dragging = False

    def load_video(self, path):
        if self.capture:
            self.capture.release()
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            return False, "Не удалось открыть видео"
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_video = self.capture.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
        ret, frame = self.capture.read()
        if ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True, path
        return False, "Не удалось прочитать кадр"

    def get_frame_info(self):
        if not self.capture:
            return {'current': 0, 'total': 0, 'fps': 0}
        return {
            'current': int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)),
            'total': self.total_frames,
            'fps': self.fps_video
        }

    @staticmethod
    def frames_to_time(frame_num, fps):
        if fps <= 0: return "00:00"
        total_sec = int(frame_num / fps)
        h, rem = divmod(total_sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def seek_to_frame(self, frame_num):
        if not self.capture: return False
        target = max(0, min(self.total_frames, frame_num))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.seek_frame = target
        return True

    def step_frame(self, direction):
        if not self.capture: return False
        current = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        new_pos = max(0, min(self.total_frames, current + direction))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        return True

    def pause(self):
        """Ставит воспроизведение на паузу без закрытия потока"""
        self.is_playing = False

    def stop(self):
        """Безопасная остановка с очисткой ресурсов"""
        self.is_processing = False
        self.is_playing = False
        self.seek_frame = None
        if self.capture is not None:
            time.sleep(0.05)
            try:
                self.capture.release()
            except Exception:
                pass
            finally:
                self.capture = None

    def release(self):
        self.stop()

    def render_frame(self, frame):
        if frame is None: return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(image=img)