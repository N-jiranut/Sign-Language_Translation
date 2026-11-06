"""
TSL Assist: Real-time Sign Language Translator UI (English Version)
Graphical User Interface for Sign Language Translation & Communication Hub.
Designed for Assistive Technology, Deaf & Hard of Hearing (DHH) accessibility.

Includes:
  - Tab 1: Live Translation (Hero Card, TTS Audio, Conversation Log, Two-Way Staff Reply)
  - Tab 2: Gesture Studio & AI Trainer (In-app Recording & 1-Click Model Retraining)
"""
import os
import sys
import time
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Project imports
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent

for p in (ROOT_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from dataset import load_label_mapping, load_raw_dataset, split_dataset, save_label_mapping
from features import extract_frame_features, resample_sequence
from landmarkers import Camera, Tracker, hands_by_side, draw_hand, draw_pose
from models import SignGRU

# Try loading ML engines
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
except ImportError:
    joblib = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    torch = None


# --------------------------------------------------------------------------
# Text-to-Speech (TTS) Helper using Windows SAPI (Zero extra dependencies)
# --------------------------------------------------------------------------
def speak_text_async(text):
    """Speaks English text asynchronously using Windows native SAPI."""
    if not text or text.startswith("(") or text == "-":
        return

    def _speak():
        try:
            clean = text.replace('"', '').replace("'", "").strip()
            if not clean:
                return
            cmd = f'(New-Object -ComObject SAPI.SpVoice).Speak("{clean}")'
            subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", cmd],
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    threading.Thread(target=_speak, daemon=True).start()


# --------------------------------------------------------------------------
# Main GUI Application
# --------------------------------------------------------------------------
class SignLanguageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSL Assist - Sign Language Translation & Gesture Studio")
        self.geometry("1320x840")
        self.minsize(1120, 720)
        self.configure(bg="#121212")

        # Styling
        self._setup_styles()

        # Model & State Variables
        self.model_type = "gru"  # 'gru' or 'rf'
        self.classifier = None
        self.id_to_label = {}
        self.threshold = 0.65
        self.auto_speak = tk.BooleanVar(value=True)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.trigger_mode = tk.StringVar(value="auto")  # 'auto' or 'manual'
        self.selected_scenario = tk.StringVar(value="Hospital & Clinic")
        self.active_model_name = tk.StringVar(value="PyTorch GRU")

        # Studio variables (Tab 2)
        self.studio_target_word = tk.StringVar(value="hello")
        self.studio_person = tk.StringVar(value=config.DEFAULT_PERSON)
        self.studio_recording = False
        self.studio_countdown = 0

        # Camera & State Machine
        self.cam = None
        self.tracker = None
        self.is_running = False
        self.thread = None
        self._video_canvas_item = None

        self.state = "idle"  # 'idle', 'arming', 'recording', 'result'
        self.arm_start_time = None
        self.rec_start_time = None
        self.current_recording = []
        self.last_sign = "-"
        self.last_conf = 0.0
        self.fps = 0.0
        self.latency_ms = 0.0

        # Build UI layout
        self._build_header()
        self._build_main_content()
        self._build_footer()

        # Keyboard shortcuts
        self.bind("<space>", lambda e: self.trigger_manual_record())

        # Load AI Model in background
        self.after(200, self._load_model_safe)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Colors
        self.C_BG = "#121212"
        self.C_CARD = "#1E1E1E"
        self.C_CARD_LIGHT = "#282828"
        self.C_BORDER = "#3A3A3A"
        self.C_ACCENT_GREEN = "#00E676"
        self.C_ACCENT_YELLOW = "#FFD600"
        self.C_ACCENT_RED = "#FF1744"
        self.C_ACCENT_CYAN = "#00E5FF"
        self.C_TEXT = "#FFFFFF"
        self.C_MUTED = "#A0A0A0"

        style.configure("TFrame", background=self.C_BG)
        style.configure("Card.TFrame", background=self.C_CARD, relief="flat")
        style.configure("TLabel", background=self.C_CARD, foreground=self.C_TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.C_CARD, foreground=self.C_MUTED, font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=self.C_BG, foreground=self.C_TEXT, font=("Segoe UI", 14, "bold"))
        style.configure("TCheckbutton", background=self.C_CARD, foreground=self.C_TEXT, font=("Segoe UI", 10))
        style.configure("TCombobox", fieldbackground=self.C_CARD_LIGHT, background=self.C_BORDER, foreground=self.C_TEXT)

        # Notebook tabs styling
        style.configure("TNotebook", background=self.C_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=self.C_CARD_LIGHT, foreground=self.C_TEXT,
                        padding=[14, 6], font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", self.C_CARD), ("active", self.C_BORDER)],
                  foreground=[("selected", self.C_ACCENT_CYAN), ("active", self.C_TEXT)])

    # ----------------------------------------------------------------------
    # 1. Header Bar
    # ----------------------------------------------------------------------
    def _build_header(self):
        header_frame = tk.Frame(self, bg="#181818", height=54, highlightbackground=self.C_BORDER, highlightthickness=1)
        header_frame.pack(side="top", fill="x")

        # Title + Badge
        title_box = tk.Frame(header_frame, bg="#181818")
        title_box.pack(side="left", padx=16, pady=10)

        lbl_logo = tk.Label(title_box, text="🤟 TSL ASSIST", font=("Segoe UI", 14, "bold"), fg=self.C_ACCENT_CYAN, bg="#181818")
        lbl_logo.pack(side="left")

        lbl_sub = tk.Label(title_box, text=" |  Sign Language Translation Hub", font=("Segoe UI", 11), fg=self.C_MUTED, bg="#181818")
        lbl_sub.pack(side="left")

        # Scenario dropdown
        scenario_box = tk.Frame(header_frame, bg="#181818")
        scenario_box.pack(side="left", padx=20)
        tk.Label(scenario_box, text="Scenario: ", font=("Segoe UI", 9), fg=self.C_MUTED, bg="#181818").pack(side="left")
        combo = ttk.Combobox(scenario_box, textvariable=self.selected_scenario, values=[
            "Hospital & Clinic", "General Services", "Police & Emergency", "University & Classroom"
        ], state="readonly", width=18)
        combo.pack(side="left")

        # Model Selector Dropdown
        model_box = tk.Frame(header_frame, bg="#181818")
        model_box.pack(side="left", padx=15)
        tk.Label(model_box, text="Engine: ", font=("Segoe UI", 9), fg=self.C_MUTED, bg="#181818").pack(side="left")
        combo_model = ttk.Combobox(model_box, textvariable=self.active_model_name, values=[
            "PyTorch GRU", "Random Forest"
        ], state="readonly", width=15)
        combo_model.pack(side="left")
        combo_model.bind("<<ComboboxSelected>>", self._on_model_switch)

        # Right-side Status Badges
        right_box = tk.Frame(header_frame, bg="#181818")
        right_box.pack(side="right", padx=16)

        self.lbl_camera_status = tk.Label(right_box, text="● Camera: Standby", font=("Segoe UI", 9, "bold"), fg="#FF9100", bg="#181818")
        self.lbl_camera_status.pack(side="left", padx=8)

        self.lbl_model_status = tk.Label(right_box, text="● Model: Loading...", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg="#181818")
        self.lbl_model_status.pack(side="left", padx=8)

    # ----------------------------------------------------------------------
    # 2. Main Content (Dual-Pane: Left 58% / Right 42%)
    # ----------------------------------------------------------------------
    def _build_main_content(self):
        content_frame = tk.Frame(self, bg=self.C_BG)
        content_frame.pack(side="top", fill="both", expand=True, padx=14, pady=12)

        content_frame.columnconfigure(0, weight=6, minsize=580)
        content_frame.columnconfigure(1, weight=4, minsize=440)
        content_frame.rowconfigure(0, weight=1)

        # Left Panel (Viewfinder)
        self._build_left_viewfinder(content_frame)

        # Right Panel (Tabbed Hub: Translation & Gesture Studio)
        self._build_right_tabbed_hub(content_frame)

    # ----------------------------------------------------------------------
    # Left Viewfinder Panel (Camera + Signer Feedback)
    # ----------------------------------------------------------------------
    def _build_left_viewfinder(self, parent):
        left_card = tk.Frame(parent, bg=self.C_CARD, highlightbackground=self.C_BORDER, highlightthickness=1)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # Title bar of viewfinder
        v_head = tk.Frame(left_card, bg=self.C_CARD)
        v_head.pack(fill="x", padx=14, pady=(10, 6))

        tk.Label(v_head, text="Camera Viewfinder & Motion Tracker", font=("Segoe UI", 11, "bold"),
                 fg=self.C_TEXT, bg=self.C_CARD).pack(side="left")

        # Mode Badge (READY / RECORDING / RESULT)
        self.badge_state = tk.Label(v_head, text="IDLE", font=("Segoe UI", 9, "bold"),
                                    fg="#121212", bg=self.C_MUTED, padx=10, pady=2)
        self.badge_state.pack(side="right")

        # Video Canvas
        self.canvas_video = tk.Canvas(left_card, bg="#0A0A0A", highlightthickness=0)
        self.canvas_video.pack(fill="both", expand=True, padx=12, pady=4)

        # Environmental Diagnostics Bar
        diag_frame = tk.Frame(left_card, bg=self.C_CARD_LIGHT, height=32)
        diag_frame.pack(fill="x", padx=12, pady=6)

        self.lbl_distance = tk.Label(diag_frame, text="📏 Distance: --", font=("Segoe UI", 9), fg=self.C_MUTED, bg=self.C_CARD_LIGHT)
        self.lbl_distance.pack(side="left", padx=10, pady=4)

        self.lbl_hands_found = tk.Label(diag_frame, text="✋ Hands: L[-] R[-]", font=("Segoe UI", 9), fg=self.C_MUTED, bg=self.C_CARD_LIGHT)
        self.lbl_hands_found.pack(side="left", padx=15, pady=4)

        self.lbl_fps = tk.Label(diag_frame, text="⚡ FPS: 0.0", font=("Segoe UI", 9), fg=self.C_MUTED, bg=self.C_CARD_LIGHT)
        self.lbl_fps.pack(side="right", padx=10, pady=4)

        # Viewfinder Control Bar
        ctrl_frame = tk.Frame(left_card, bg=self.C_CARD)
        ctrl_frame.pack(fill="x", padx=12, pady=(4, 12))

        # Start/Stop Button
        self.btn_camera = tk.Button(ctrl_frame, text="▶ Start Camera", font=("Segoe UI", 10, "bold"),
                                    bg=self.C_ACCENT_GREEN, fg="#121212", activebackground="#00C853",
                                    relief="flat", padx=14, pady=5, command=self.toggle_camera)
        self.btn_camera.pack(side="left", padx=(0, 8))

        # Manual Trigger Button
        self.btn_trigger = tk.Button(ctrl_frame, text="🔘 Trigger (Space)", font=("Segoe UI", 9),
                                     bg=self.C_CARD_LIGHT, fg=self.C_TEXT, activebackground=self.C_BORDER,
                                     relief="flat", padx=10, pady=5, command=self.trigger_manual_record)
        self.btn_trigger.pack(side="left", padx=4)

        # Trigger Mode Radio
        tk.Label(ctrl_frame, text="Trigger:", font=("Segoe UI", 9), fg=self.C_MUTED, bg=self.C_CARD).pack(side="left", padx=(10, 4))
        tk.Radiobutton(ctrl_frame, text="Auto (Chest Line)", variable=self.trigger_mode, value="auto",
                       bg=self.C_CARD, fg=self.C_TEXT, selectcolor=self.C_CARD_LIGHT, activebackground=self.C_CARD,
                       font=("Segoe UI", 9)).pack(side="left")
        tk.Radiobutton(ctrl_frame, text="Manual", variable=self.trigger_mode, value="manual",
                       bg=self.C_CARD, fg=self.C_TEXT, selectcolor=self.C_CARD_LIGHT, activebackground=self.C_CARD,
                       font=("Segoe UI", 9)).pack(side="left", padx=(4, 0))

        # Skeleton Checkbox
        chk_skel = tk.Checkbutton(ctrl_frame, text="Skeleton", variable=self.show_skeleton,
                                  bg=self.C_CARD, fg=self.C_TEXT, selectcolor=self.C_CARD_LIGHT,
                                  activebackground=self.C_CARD, font=("Segoe UI", 9))
        chk_skel.pack(side="right")

    # ----------------------------------------------------------------------
    # Right Tabbed Hub (Tab 1: Live Translation | Tab 2: Gesture Studio)
    # ----------------------------------------------------------------------
    def _build_right_tabbed_hub(self, parent):
        right_container = tk.Frame(parent, bg=self.C_BG)
        right_container.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.notebook = ttk.Notebook(right_container)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Live Translation
        tab_translation = tk.Frame(self.notebook, bg=self.C_CARD, highlightbackground=self.C_BORDER, highlightthickness=1)
        self.notebook.add(tab_translation, text=" 🤟 Live Translation ")
        self._build_tab_translation(tab_translation)

        # Tab 2: Gesture Studio & Trainer
        tab_studio = tk.Frame(self.notebook, bg=self.C_CARD, highlightbackground=self.C_BORDER, highlightthickness=1)
        self.notebook.add(tab_studio, text=" ⚙️ Gesture Studio & Trainer ")
        self._build_tab_studio(tab_studio)

    # ----------------------------------------------------------------------
    # Tab 1: Translation Hub View
    # ----------------------------------------------------------------------
    def _build_tab_translation(self, parent):
        # 1. HERO RESULT CARD
        hero_frame = tk.Frame(parent, bg="#16222F", highlightbackground=self.C_ACCENT_CYAN, highlightthickness=1)
        hero_frame.pack(fill="x", padx=12, pady=(12, 8))

        h_title_bar = tk.Frame(hero_frame, bg="#16222F")
        h_title_bar.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(h_title_bar, text="TRANSLATED SIGN OUTPUT", font=("Segoe UI", 8, "bold"), fg=self.C_ACCENT_CYAN, bg="#16222F").pack(side="left")
        
        self.lbl_conf_tag = tk.Label(h_title_bar, text="Confidence: --%", font=("Segoe UI", 8, "bold"), fg=self.C_MUTED, bg="#16222F")
        self.lbl_conf_tag.pack(side="right")

        # Large English Word Display
        self.lbl_hero_word = tk.Label(hero_frame, text="Awaiting Sign...", font=("Segoe UI", 26, "bold"),
                                      fg=self.C_TEXT, bg="#16222F", pady=10)
        self.lbl_hero_word.pack()

        # Confidence Bar (Canvas)
        self.canvas_conf_bar = tk.Canvas(hero_frame, height=6, bg="#0D161F", highlightthickness=0)
        self.canvas_conf_bar.pack(fill="x", padx=14, pady=(0, 10))

        # Hero Action Bar (Speak button, Auto-TTS toggle)
        h_actions = tk.Frame(hero_frame, bg="#16222F")
        h_actions.pack(fill="x", padx=12, pady=(0, 10))

        self.btn_speak = tk.Button(h_actions, text="🔊 Speak (Audio)", font=("Segoe UI", 9, "bold"),
                                   bg=self.C_ACCENT_CYAN, fg="#121212", activebackground="#00B0FF",
                                   relief="flat", padx=10, pady=3, command=self.speak_current_sign)
        self.btn_speak.pack(side="left")

        chk_tts = tk.Checkbutton(h_actions, text="Auto-Speak on detect", variable=self.auto_speak,
                                 bg="#16222F", fg=self.C_TEXT, selectcolor="#0D161F",
                                 activebackground="#16222F", font=("Segoe UI", 9))
        chk_tts.pack(side="left", padx=10)

        # 2. CONVERSATION TIMELINE / LOG
        tk.Label(parent, text="Conversation Timeline", font=("Segoe UI", 10, "bold"),
                 fg=self.C_TEXT, bg=self.C_CARD).pack(anchor="w", padx=14, pady=(8, 4))

        log_container = tk.Frame(parent, bg=self.C_CARD)
        log_container.pack(fill="both", expand=True, padx=12, pady=2)

        self.txt_timeline = tk.Text(log_container, bg="#161616", fg=self.C_TEXT, font=("Segoe UI", 9),
                                    wrap="word", relief="flat", padx=8, pady=8, state="disabled")
        self.txt_timeline.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_container, orient="vertical", command=self.txt_timeline.yview)
        scrollbar.pack(side="right", fill="y")
        self.txt_timeline.config(yscrollcommand=scrollbar.set)

        # Timeline helper buttons
        t_btn_box = tk.Frame(parent, bg=self.C_CARD)
        t_btn_box.pack(fill="x", padx=12, pady=(4, 8))
        tk.Button(t_btn_box, text="Clear Log", font=("Segoe UI", 8), bg=self.C_CARD_LIGHT, fg=self.C_MUTED,
                  relief="flat", padx=6, command=self.clear_log).pack(side="right")

        # 3. TWO-WAY STAFF RESPONSE (For hearing person to reply to deaf user)
        two_way_frame = tk.Frame(parent, bg=self.C_CARD_LIGHT, highlightbackground=self.C_BORDER, highlightthickness=1)
        two_way_frame.pack(fill="x", padx=12, pady=(0, 12))

        tw_head = tk.Frame(two_way_frame, bg=self.C_CARD_LIGHT)
        tw_head.pack(fill="x", padx=10, pady=(6, 2))
        tk.Label(tw_head, text="💬 Staff Two-Way Reply (Display to Signer)", font=("Segoe UI", 9, "bold"),
                 fg=self.C_ACCENT_GREEN, bg=self.C_CARD_LIGHT).pack(side="left")

        # Quick preset buttons
        quick_frame = tk.Frame(two_way_frame, bg=self.C_CARD_LIGHT)
        quick_frame.pack(fill="x", padx=10, pady=2)
        presets = ["Yes", "No", "Please wait a moment", "Follow me please"]
        for p in presets:
            btn = tk.Button(quick_frame, text=p, font=("Segoe UI", 8), bg=self.C_BORDER, fg=self.C_TEXT,
                            relief="flat", padx=5, pady=1,
                            command=lambda msg=p: self.send_staff_reply(msg))
            btn.pack(side="left", padx=2)

        # Input & Send
        input_box = tk.Frame(two_way_frame, bg=self.C_CARD_LIGHT)
        input_box.pack(fill="x", padx=10, pady=(4, 8))

        self.entry_reply = tk.Entry(input_box, font=("Segoe UI", 10), bg="#141414", fg=self.C_TEXT,
                                    insertbackground=self.C_TEXT, relief="flat", highlightbackground=self.C_BORDER, highlightthickness=1)
        self.entry_reply.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=3)
        self.entry_reply.bind("<Return>", lambda e: self.send_staff_reply(self.entry_reply.get()))

        btn_send = tk.Button(input_box, text="Send ↵", font=("Segoe UI", 9, "bold"), bg=self.C_ACCENT_GREEN,
                             fg="#121212", relief="flat", padx=10, pady=3,
                             command=lambda: self.send_staff_reply(self.entry_reply.get()))
        btn_send.pack(side="right")

    # ----------------------------------------------------------------------
    # Tab 2: Gesture Studio & 1-Click AI Retrainer
    # ----------------------------------------------------------------------
    def _build_tab_studio(self, parent):
        studio_box = tk.Frame(parent, bg=self.C_CARD)
        studio_box.pack(fill="both", expand=True, padx=14, pady=12)

        # 1. Section: Collect & Record Gesture Samples
        sec1 = tk.LabelFrame(studio_box, text=" 1. Record Gesture Samples Directly from Camera ",
                             font=("Segoe UI", 10, "bold"), fg=self.C_ACCENT_CYAN, bg=self.C_CARD,
                             highlightbackground=self.C_BORDER, highlightthickness=1, padx=12, pady=10)
        sec1.pack(fill="x", pady=(0, 10))

        row1 = tk.Frame(sec1, bg=self.C_CARD)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Target Gesture Name:", font=("Segoe UI", 9), fg=self.C_TEXT, bg=self.C_CARD).pack(side="left")
        
        self.entry_word = tk.Entry(row1, textvariable=self.studio_target_word, font=("Segoe UI", 10),
                                   bg=self.C_CARD_LIGHT, fg=self.C_TEXT, insertbackground=self.C_TEXT,
                                   width=18, relief="flat", highlightbackground=self.C_BORDER, highlightthickness=1)
        self.entry_word.pack(side="left", padx=8)

        # Quick preset words
        presets_box = tk.Frame(sec1, bg=self.C_CARD)
        presets_box.pack(fill="x", pady=4)
        tk.Label(presets_box, text="Quick Presets:", font=("Segoe UI", 8), fg=self.C_MUTED, bg=self.C_CARD).pack(side="left")
        for w in ["hello", "thank_you", "hospital", "yes", "no", "sorry", "help"]:
            tk.Button(presets_box, text=w, font=("Segoe UI", 8), bg=self.C_BORDER, fg=self.C_TEXT, relief="flat",
                      command=lambda val=w: self.studio_target_word.set(val)).pack(side="left", padx=2)

        # Action: Record 1 Sample
        row2 = tk.Frame(sec1, bg=self.C_CARD)
        row2.pack(fill="x", pady=(10, 4))
        self.btn_studio_rec = tk.Button(row2, text="🔴 Record Sample (3.. 2.. 1..)", font=("Segoe UI", 10, "bold"),
                                        bg=self.C_ACCENT_RED, fg="#FFFFFF", relief="flat", padx=14, pady=5,
                                        command=self.start_studio_recording)
        self.btn_studio_rec.pack(side="left")

        self.lbl_studio_status = tk.Label(row2, text="Ready to record sample", font=("Segoe UI", 9),
                                          fg=self.C_MUTED, bg=self.C_CARD)
        self.lbl_studio_status.pack(side="left", padx=12)

        # 2. Section: Current Dataset Inventory
        sec2 = tk.LabelFrame(studio_box, text=" 2. Current Dataset Classes & Sample Counts ",
                             font=("Segoe UI", 10, "bold"), fg=self.C_TEXT, bg=self.C_CARD,
                             highlightbackground=self.C_BORDER, highlightthickness=1, padx=12, pady=10)
        sec2.pack(fill="both", expand=True, pady=6)

        self.txt_dataset_info = tk.Text(sec2, height=6, bg="#161616", fg=self.C_TEXT, font=("Consolas", 9),
                                        relief="flat", padx=8, pady=6)
        self.txt_dataset_info.pack(fill="both", expand=True)

        tk.Button(sec2, text="🔄 Refresh Counts", font=("Segoe UI", 8), bg=self.C_CARD_LIGHT, fg=self.C_TEXT,
                  relief="flat", command=self._refresh_dataset_summary).pack(anchor="e", pady=(4, 0))

        # 3. Section: 1-Click Retrain Model
        sec3 = tk.LabelFrame(studio_box, text=" 3. Train AI Models (1-Click Training) ",
                             font=("Segoe UI", 10, "bold"), fg=self.C_ACCENT_GREEN, bg=self.C_CARD,
                             highlightbackground=self.C_BORDER, highlightthickness=1, padx=12, pady=10)
        sec3.pack(fill="x", pady=(8, 0))

        row_train = tk.Frame(sec3, bg=self.C_CARD)
        row_train.pack(fill="x")

        self.btn_train_models = tk.Button(row_train, text="⚡ Retrain AI Models Now", font=("Segoe UI", 10, "bold"),
                                          bg=self.C_ACCENT_GREEN, fg="#121212", relief="flat", padx=16, pady=6,
                                          command=self.retrain_models_async)
        self.btn_train_models.pack(side="left")

        self.lbl_train_status = tk.Label(row_train, text="Click to train Random Forest & GRU on all current samples",
                                         font=("Segoe UI", 9), fg=self.C_MUTED, bg=self.C_CARD)
        self.lbl_train_status.pack(side="left", padx=14)

        # Initial refresh
        self.after(500, self._refresh_dataset_summary)

    # ----------------------------------------------------------------------
    # 3. Footer Bar
    # ----------------------------------------------------------------------
    def _build_footer(self):
        footer_frame = tk.Frame(self, bg="#101010", height=28)
        footer_frame.pack(side="bottom", fill="x")

        self.lbl_footer_info = tk.Label(footer_frame,
                                        text="KMUTT CPE Senior Project  |  TSL Isolated Sign Translator",
                                        font=("Segoe UI", 8), fg=self.C_MUTED, bg="#101010")
        self.lbl_footer_info.pack(side="left", padx=16, pady=4)

        self.lbl_footer_tech = tk.Label(footer_frame,
                                        text="Engine: MediaPipe Tasks + Linear Interpolation  |  Latency: -- ms",
                                        font=("Segoe UI", 8), fg=self.C_MUTED, bg="#101010")
        self.lbl_footer_tech.pack(side="right", padx=16, pady=4)

    # ----------------------------------------------------------------------
    # AI Model Loading & Inference
    # ----------------------------------------------------------------------
    def _on_model_switch(self, event=None):
        name = self.active_model_name.get()
        if "GRU" in name:
            self.model_type = "gru"
        else:
            self.model_type = "rf"
        self._load_model_safe()

    def _load_model_safe(self):
        """Loads trained models (PyTorch GRU or Scikit-learn RF)."""
        try:
            if not config.SAVED_LABELS.exists():
                self.lbl_model_status.config(text="● No labels found", fg=self.C_ACCENT_RED)
                return

            label_map = load_label_mapping(config.SAVED_LABELS)
            self.id_to_label = {v: k for k, v in label_map.items()}

            if self.model_type == "gru" and config.SAVED_TORCH_MODEL.exists() and torch is not None:
                ckpt = torch.load(config.SAVED_TORCH_MODEL, map_location="cpu")
                model = SignGRU(
                    input_size=ckpt["input_size"],
                    hidden_size=ckpt["hidden_size"],
                    num_layers=ckpt["num_layers"],
                    num_classes=ckpt["num_classes"],
                )
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self.classifier = model
                self.lbl_model_status.config(text=f"● Model: GRU ({len(self.id_to_label)} gestures)", fg=self.C_ACCENT_GREEN)

            elif config.SAVED_SKLEARN_MODEL.exists() and joblib is not None:
                self.classifier = joblib.load(config.SAVED_SKLEARN_MODEL)
                self.lbl_model_status.config(text=f"● Model: RF ({len(self.id_to_label)} gestures)", fg=self.C_ACCENT_GREEN)

            else:
                self.lbl_model_status.config(text="● Untrained (Run train script)", fg="#FF9100")

        except Exception as e:
            self.lbl_model_status.config(text=f"● Load error: {e}", fg=self.C_ACCENT_RED)

    def _predict(self, seq_24x112):
        """Predicts gesture class and confidence from (24, 112) matrix."""
        if self.classifier is None:
            return "Untrained Model", 0.0

        if self.model_type == "rf":
            flat = seq_24x112.reshape(1, -1)
            probs = self.classifier.predict_proba(flat)[0]
            idx = int(np.argmax(probs))
            label = str(self.classifier.classes_[idx])
            return label, float(probs[idx])

        elif self.model_type == "gru":
            with torch.no_grad():
                tensor_in = torch.tensor(seq_24x112[None, ...], dtype=torch.float32)
                logits = self.classifier(tensor_in)
                probs = torch.softmax(logits, dim=-1)[0].numpy()
                idx = int(np.argmax(probs))
                label = self.id_to_label.get(idx, f"Class {idx}")
                return label, float(probs[idx])

        return "Unknown", 0.0

    # ----------------------------------------------------------------------
    # Camera Loop & Real-time State Machine
    # ----------------------------------------------------------------------
    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        try:
            self.cam = Camera()
            self.tracker = Tracker()
            self.is_running = True
            self.btn_camera.config(text="⏹ Stop Camera", bg=self.C_ACCENT_RED)
            self.lbl_camera_status.config(text="● Camera: Live", fg=self.C_ACCENT_GREEN)

            self.thread = threading.Thread(target=self._camera_worker, daemon=True)
            self.thread.start()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Unable to start camera:\n{e}\n\nPlease check camera connection.")

    def stop_camera(self):
        self.is_running = False
        if self.cam:
            self.cam.release()
            self.cam = None
        if self.tracker:
            self.tracker.close()
            self.tracker = None

        self.btn_camera.config(text="▶ Start Camera", bg=self.C_ACCENT_GREEN)
        self.lbl_camera_status.config(text="● Camera: Stopped", fg=self.C_MUTED)
        self.canvas_video.delete("all")
        self._video_canvas_item = None

    def trigger_manual_record(self):
        """Triggers recording manually (via button or Spacebar)."""
        if not self.is_running:
            self.start_camera()
            time.sleep(0.5)
        self.state = "recording"
        self.rec_start_time = time.perf_counter()
        self.current_recording = []
        self._update_badge("RECORDING", self.C_ACCENT_RED)

    def _camera_worker(self):
        last_id = 0
        prev_t = time.perf_counter()

        while self.is_running and self.cam is not None:
            frame, last_id = self.cam.read(last_id)
            if frame is None:
                time.sleep(0.01)
                continue

            if config.FLIP_IMAGE:
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            hand_res, pose_res, _ = self.tracker.process(frame)
            t_infer = (time.perf_counter() - t0) * 1000

            hands = hands_by_side(hand_res)
            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
            frame_vec = extract_frame_features(pose_lms, hands)

            # Measure FPS
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            cur_fps = (1.0 / dt) if dt > 0 else 30.0
            self.fps = 0.9 * self.fps + 0.1 * cur_fps
            self.latency_ms = 0.8 * self.latency_ms + 0.2 * t_infer

            # State Machine Calculations
            h, w = frame.shape[:2]
            ready_y = int(0.65 * h)
            sh_w = 0.35
            if pose_lms is not None:
                try:
                    l_sh = pose_lms[config.POSE_KEEP["l_shoulder"]]
                    r_sh = pose_lms[config.POSE_KEEP["r_shoulder"]]
                    sh_w = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
                    ready_y = int(np.clip(((l_sh.y + r_sh.y) / 2.0 + 0.65 * max(sh_w, 0.15)), 0.35, 0.85) * h)
                except Exception:
                    pass

            # Check if hand is in ready zone
            hand_in_zone = False
            for side in ("Left", "Right"):
                lms = hands.get(side)
                if lms is not None and len(lms) > 0:
                    if int(lms[0].y * h) < ready_y:
                        hand_in_zone = True
                        break

            # State Transitions
            mode = self.trigger_mode.get()
            if self.state == "idle":
                if mode == "auto" and hand_in_zone:
                    if self.arm_start_time is None:
                        self.arm_start_time = now
                        self._update_badge("ARMING...", self.C_ACCENT_YELLOW)
                    elif (now - self.arm_start_time) >= 0.35:
                        self.state = "recording"
                        self.rec_start_time = now
                        self.current_recording = [frame_vec]
                        self._update_badge("RECORDING", self.C_ACCENT_RED)
                else:
                    self.arm_start_time = None
                    self._update_badge("READY", self.C_ACCENT_GREEN)

            elif self.state == "recording":
                self.current_recording.append(frame_vec)
                elapsed = now - self.rec_start_time
                if elapsed >= 1.8:
                    # Recording finished
                    self.state = "result"
                    self._update_badge("PROCESSING", self.C_ACCENT_CYAN)

                    # Check if this was a Studio sample recording
                    if self.studio_recording:
                        self._save_studio_sample(self.current_recording)
                        self.studio_recording = False
                    else:
                        if len(self.current_recording) >= 10:
                            seq_24 = resample_sequence(np.array(self.current_recording), target_len=config.TARGET_FRAMES)
                            label, conf = self._predict(seq_24)
                            self.after(0, self._on_prediction_ready, label, conf)
                        else:
                            self.after(0, self._on_prediction_ready, "(No motion)", 0.0)

            elif self.state == "result":
                # Cooldown
                if not hand_in_zone or (now - self.rec_start_time) > 3.2:
                    self.state = "idle"
                    self.arm_start_time = None

            # Render Overlays
            view = frame.copy()
            if self.show_skeleton.get():
                if pose_lms is not None:
                    draw_pose(view, pose_lms)
                for side, lms in hands.items():
                    if lms is not None:
                        draw_hand(view, lms, side)

            # Draw Ready Line Guide
            line_color = (0, 220, 255) if hand_in_zone else (100, 100, 100)
            cv2.line(view, (0, ready_y), (w, ready_y), line_color, 2 if hand_in_zone else 1)
            cv2.putText(view, "READY ZONE (Chest)", (w - 180, ready_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)

            # Border glow when recording
            if self.state == "recording":
                cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 0, 255), 5)
                cv2.circle(view, (w - 30, 30), 12, (0, 0, 255), -1)
                cv2.putText(view, "REC", (w - 75, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            # Update Diagnostics
            l_found = hands.get("Left") is not None
            r_found = hands.get("Right") is not None
            sh_dist_str = f"{sh_w:.2f}" if pose_lms is not None else "--"

            self.after(0, self._update_diagnostics, sh_dist_str, l_found, r_found, self.fps, self.latency_ms)

            # Render to Tkinter Canvas efficiently
            rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            self.after(0, self._render_canvas, rgb)

    def _render_canvas(self, rgb_array):
        """Draws RGB frame onto canvas with persistent image item (zero memory leak)."""
        c_w = self.canvas_video.winfo_width()
        c_h = self.canvas_video.winfo_height()
        if c_w < 10 or c_h < 10:
            return

        img = Image.fromarray(rgb_array)
        img.thumbnail((c_w, c_h), Image.Resampling.BILINEAR)
        img_tk = ImageTk.PhotoImage(image=img)

        if self._video_canvas_item is None:
            self._video_canvas_item = self.canvas_video.create_image(c_w // 2, c_h // 2, image=img_tk, anchor="center")
        else:
            self.canvas_video.coords(self._video_canvas_item, c_w // 2, c_h // 2)
            self.canvas_video.itemconfig(self._video_canvas_item, image=img_tk)

        self.canvas_video.image = img_tk  # Keep reference

    def _update_badge(self, text, bg_color):
        self.after(0, lambda: self.badge_state.config(text=text, bg=bg_color))

    def _update_diagnostics(self, dist_str, l_found, r_found, fps, lat):
        dist_color = self.C_ACCENT_GREEN if dist_str != "--" else self.C_MUTED
        self.lbl_distance.config(text=f"📏 Shoulder Dist: {dist_str}", fg=dist_color)

        l_str = "L[✓]" if l_found else "L[-]"
        r_str = "R[✓]" if r_found else "R[-]"
        hands_color = self.C_ACCENT_GREEN if (l_found or r_found) else self.C_MUTED
        self.lbl_hands_found.config(text=f"✋ Hands: {l_str} {r_str}", fg=hands_color)
        self.lbl_fps.config(text=f"⚡ {fps:4.1f} FPS")
        self.lbl_footer_tech.config(text=f"Latency: {lat:4.1f} ms  |  Engine: {self.active_model_name.get()}")

    # ----------------------------------------------------------------------
    # Prediction Handling & Conversation Timeline
    # ----------------------------------------------------------------------
    def _on_prediction_ready(self, label, conf):
        self.last_sign = label
        self.last_conf = conf

        clean_label = label.replace("_", " ").title() if label else "Unknown"
        self.lbl_hero_word.config(text=clean_label)
        self.lbl_conf_tag.config(text=f"Confidence: {int(conf * 100)}%")

        # Draw Confidence Bar
        self.canvas_conf_bar.delete("all")
        bar_w = self.canvas_conf_bar.winfo_width()
        fill_w = int(bar_w * np.clip(conf, 0.0, 1.0))
        bar_color = self.C_ACCENT_GREEN if conf >= self.threshold else "#FF9100"
        self.canvas_conf_bar.create_rectangle(0, 0, fill_w, 6, fill=bar_color, width=0)

        # Append to Conversation Timeline
        if conf >= 0.50 and label not in ("(No motion)", "Unknown", "Untrained Model"):
            ts = time.strftime("%H:%M:%S")
            self._append_to_timeline(f"🤟 [Signer] {clean_label} ({int(conf * 100)}%)", "#00E5FF")

            # Auto-speak if enabled
            if self.auto_speak.get():
                speak_text_async(clean_label)

    def speak_current_sign(self):
        """Manual click to speak the current hero sign."""
        if self.last_sign and self.last_sign != "-":
            clean = self.last_sign.replace("_", " ").title()
            speak_text_async(clean)

    def send_staff_reply(self, message):
        """Sends reply from hearing staff/companion to display on screen."""
        msg = message.strip()
        if not msg:
            return
        ts = time.strftime("%H:%M:%S")
        self._append_to_timeline(f"👨‍💼 [Staff] {msg}", self.C_ACCENT_GREEN)
        self.entry_reply.delete(0, tk.END)

    def _append_to_timeline(self, text, color):
        ts = time.strftime("%H:%M:%S")
        self.txt_timeline.config(state="normal")
        self.txt_timeline.insert(tk.END, f"[{ts}] {text}\n")
        self.txt_timeline.see(tk.END)
        self.txt_timeline.config(state="disabled")

    def clear_log(self):
        self.txt_timeline.config(state="normal")
        self.txt_timeline.delete("1.0", tk.END)
        self.txt_timeline.config(state="disabled")

    # ----------------------------------------------------------------------
    # Tab 2: Gesture Studio Methods (Record Samples & Retrain AI)
    # ----------------------------------------------------------------------
    def start_studio_recording(self):
        """Starts countdown then records 1 sample for the target gesture."""
        word = self.studio_target_word.get().strip().lower().replace(" ", "_")
        if not word:
            messagebox.showwarning("Warning", "Please specify a target gesture word first.")
            return

        if not self.is_running:
            self.start_camera()
            time.sleep(0.5)

        self.btn_studio_rec.config(state="disabled")
        self._do_countdown(3, word)

    def _do_countdown(self, seconds_left, word):
        if seconds_left > 0:
            self.lbl_studio_status.config(text=f"Get ready: Recording in {seconds_left}...", fg=self.C_ACCENT_YELLOW)
            self._update_badge(f"START IN {seconds_left}", self.C_ACCENT_YELLOW)
            self.after(1000, lambda: self._do_countdown(seconds_left - 1, word))
        else:
            self.lbl_studio_status.config(text=f"RECORDING NOW for [{word.upper()}]!", fg=self.C_ACCENT_RED)
            self.studio_recording = True
            self.state = "recording"
            self.rec_start_time = time.perf_counter()
            self.current_recording = []
            self._update_badge("STUDIO REC", self.C_ACCENT_RED)

    def _save_studio_sample(self, frames):
        """Saves recorded frames directly into data/raw/<word>/."""
        word = self.studio_target_word.get().strip().lower().replace(" ", "_")
        person = self.studio_person.get()

        if len(frames) >= 10:
            folder = config.DATA_RAW_DIR / word
            folder.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            file_path = folder / f"{person}_{ts}.npy"
            np.save(str(file_path), np.array(frames, dtype=np.float32))
            msg = f"✓ Saved {file_path.name} to '{word}' ({len(frames)} frames)"
            self.lbl_studio_status.config(text=msg, fg=self.C_ACCENT_GREEN)
        else:
            self.lbl_studio_status.config(text="⚠ Too few frames detected. Discarded.", fg="#FF9100")

        self.btn_studio_rec.config(state="normal")
        self._refresh_dataset_summary()

    def _refresh_dataset_summary(self):
        """Refreshes the sample count summary in Tab 2."""
        raw_dir = config.DATA_RAW_DIR
        self.txt_dataset_info.delete("1.0", tk.END)

        if not raw_dir.exists():
            self.txt_dataset_info.insert(tk.END, "No 'data/raw' folder found.\n")
            return

        class_folders = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
        if not class_folders:
            self.txt_dataset_info.insert(tk.END, "No gesture classes found yet in data/raw/.\n")
            return

        total_samples = 0
        self.txt_dataset_info.insert(tk.END, f"{'GESTURE CLASS':<22} | {'SAMPLES':<10} | {'STATUS'}\n")
        self.txt_dataset_info.insert(tk.END, "-" * 50 + "\n")

        for f in class_folders:
            files = list(f.glob("*.npy"))
            cnt = len(files)
            total_samples += cnt
            status = "Ready" if cnt >= 5 else "Need more (≥5)"
            self.txt_dataset_info.insert(tk.END, f"{f.name:<22} | {cnt:<10} | {status}\n")

        self.txt_dataset_info.insert(tk.END, "-" * 50 + "\n")
        self.txt_dataset_info.insert(tk.END, f"Total Classes: {len(class_folders)} | Total Samples: {total_samples}\n")

    def retrain_models_async(self):
        """Retrains Random Forest and PyTorch GRU in background thread."""
        self.btn_train_models.config(state="disabled")
        self.lbl_train_status.config(text="Training in progress... Please wait 3-5 seconds", fg=self.C_ACCENT_YELLOW)

        def _worker():
            try:
                # Load dataset
                X, y, groups, label_to_id = load_raw_dataset()
                if len(label_to_id) < 2:
                    self.after(0, lambda: messagebox.showwarning("Training Error", "Need at least 2 distinct gesture classes to train!"))
                    self.after(0, self._on_train_finished, False, "Need at least 2 classes")
                    return

                # 1. Train Random Forest
                X_train, X_test, y_train, y_test = split_dataset(X, y, groups, test_size=0.2)
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)

                clf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42)
                clf.fit(X_train_flat, y_train)
                preds = clf.predict(X_test_flat)
                rf_acc = accuracy_score(y_test, preds)

                config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(clf, config.SAVED_SKLEARN_MODEL)
                save_label_mapping(label_to_id, config.SAVED_LABELS)

                # 2. Train PyTorch GRU
                y_encoded = np.array([label_to_id[name] for name in y], dtype=np.int64)
                X_train_g, X_test_g, y_train_g, y_test_g = split_dataset(X, y_encoded, groups, test_size=0.2)

                train_ds = TensorDataset(torch.tensor(X_train_g, dtype=torch.float32), torch.tensor(y_train_g, dtype=torch.long))
                loader = DataLoader(train_ds, batch_size=min(16, max(2, len(train_ds))), shuffle=True)

                device = torch.device("cpu")
                model = SignGRU(input_size=config.NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=len(label_to_id)).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()

                for epoch in range(35):
                    model.train()
                    for bx, by in loader:
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()

                # Save Torch model
                torch.save({
                    "model_state": model.state_dict(),
                    "label_to_id": label_to_id,
                    "input_size": config.NUM_FEATURES,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "num_classes": len(label_to_id),
                }, config.SAVED_TORCH_MODEL)

                self.after(0, self._on_train_finished, True, f"Success! RF Acc: {rf_acc*100:.0f}%, {len(label_to_id)} classes ready")

            except Exception as e:
                self.after(0, self._on_train_finished, False, f"Train failed: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_train_finished(self, success, message):
        self.btn_train_models.config(state="normal")
        if success:
            self.lbl_train_status.config(text=f"✓ {message}", fg=self.C_ACCENT_GREEN)
            self._load_model_safe()
            self._refresh_dataset_summary()
            messagebox.showinfo("Training Complete", f"AI Models retrained successfully!\n\n{message}\n\nYou can now test your new gestures in Tab 1!")
        else:
            self.lbl_train_status.config(text=f"✗ {message}", fg=self.C_ACCENT_RED)

    def on_close(self):
        self.stop_camera()
        self.destroy()


# --------------------------------------------------------------------------
# Launcher
# --------------------------------------------------------------------------
if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()
