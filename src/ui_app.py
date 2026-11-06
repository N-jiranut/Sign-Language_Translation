"""
Modern Desktop Assistive System for Sign Language Translation (TSL Assist)
KMUTT Senior Project - Thai Sign Language Real-Time Translation & Gesture Studio

Key Components:
  - 50/50 Dual-Pane Interface: Camera Viewfinder (Left) & Translation Hub / Gesture Studio (Right)
  - Aspect-ratio Dynamic Camera Scaling: Scales up to fill window when maximized, scales down smoothly when resized
  - Adjustable Font Size & High-Legibility Accessibility for Staff Two-Way Communicator & Conversation Log
  - Thread-Safe Model Switching & Asynchronous PyTorch GRU / Random Forest Classifier
  - Two-Way Real-Time Communication with customizable scenario presets
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# Safe AI Model Imports
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
except ImportError:
    joblib = None
    RandomForestClassifier = None
    accuracy_score = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

# Local module imports
_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from dataset import augment_dataset, load_label_mapping, load_raw_dataset, save_label_mapping, split_dataset
from features import extract_frame_features, resample_sequence
from landmarkers import Camera, Tracker, draw_hand, draw_pose, hands_by_side
from models import SignGRU

# --------------------------------------------------------------------------
# Font Scaling Presets for High Legibility & Accessibility
# --------------------------------------------------------------------------
FONT_LEVELS = [
    {
        "name": "Normal (100%)",
        "log_body": 11,
        "log_signer": 12,
        "log_staff": 11,
        "log_time": 9,
        "presets": 9,
        "entry": 11,
        "btn_send": 9,
        "header_sub": 9,
        "padding_y": 3,
        "padding_x": 8,
    },
    {
        "name": "Large (120%)",
        "log_body": 13,
        "log_signer": 15,
        "log_staff": 13,
        "log_time": 10,
        "presets": 11,
        "entry": 13,
        "btn_send": 10,
        "header_sub": 10,
        "padding_y": 4,
        "padding_x": 10,
    },
    {
        "name": "Extra Large (140%)",
        "log_body": 16,
        "log_signer": 18,
        "log_staff": 16,
        "log_time": 11,
        "presets": 13,
        "entry": 15,
        "btn_send": 12,
        "header_sub": 12,
        "padding_y": 6,
        "padding_x": 12,
    },
    {
        "name": "Huge (160%)",
        "log_body": 19,
        "log_signer": 22,
        "log_staff": 19,
        "log_time": 12,
        "presets": 15,
        "entry": 17,
        "btn_send": 14,
        "header_sub": 14,
        "padding_y": 7,
        "padding_x": 14,
    },
]

# --------------------------------------------------------------------------
# Scenario Presets Loader (reads data/scenario_presets.json)
# --------------------------------------------------------------------------
def load_scenario_data():
    """Loads scenario presets from data/scenario_presets.json with resilient fallback."""
    cfg_file = getattr(config, "PRESETS_CONFIG_PATH", config.DATA_DIR / "scenario_presets.json")
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            staff_map = {}
            gesture_map = {}
            hints_map = {}
            for k, v in data.items():
                staff_map[k] = v.get("staff_replies", [])
                gesture_map[k] = v.get("recommended_signs", [])
                hints_map[k] = (
                    v.get("category", "General"),
                    v.get("accent_color", "#00E5FF"),
                    v.get("hint_text", "")
                )
            return staff_map, gesture_map, hints_map, list(data.keys())
        except Exception as e:
            print(f"[Warning] Failed to parse {cfg_file}: {e}")

    # Built-in Default Scenarios
    data = {
        "Hospital & Clinic": {
            "category": "Healthcare",
            "accent_color": "#00E5FF",
            "hint_text": "Context Signs: hospital, doctor, medicine, pain, help, yes, no, thank_you",
            "staff_replies": [
                "Wait for doctor",
                "Take medication",
                "Follow nurse",
                "Please sit here",
                "Where does it hurt?",
                "Are you feeling better?"
            ],
            "recommended_signs": ["hospital", "doctor", "medicine", "pain", "help", "yes", "no", "thank_you"]
        },
        "General Services": {
            "category": "Customer Service",
            "accent_color": "#00E676",
            "hint_text": "Context Signs: hello, thank_you, sorry, please, yes, no",
            "staff_replies": [
                "How can I help you?",
                "Please wait a moment",
                "Have a great day",
                "Thank you very much"
            ],
            "recommended_signs": ["hello", "thank_you", "sorry", "yes", "no"]
        },
        "Police & Emergency": {
            "category": "Public Safety",
            "accent_color": "#FF1744",
            "hint_text": "Context Signs: emergency, help, accident, police, danger, safe",
            "staff_replies": [
                "You are safe here",
                "Officer is on the way",
                "Do you need ambulance?",
                "Please stay calm"
            ],
            "recommended_signs": ["emergency", "help", "accident", "danger", "safe", "yes", "no"]
        },
        "University & Classroom": {
            "category": "Education",
            "accent_color": "#FFD600",
            "hint_text": "Context Signs: hello, thank_you, question, finish, book, yes, no",
            "staff_replies": [
                "Class starts soon",
                "Submit assignment",
                "Room 304",
                "Please see instructor"
            ],
            "recommended_signs": ["hello", "thank_you", "question", "finish", "book", "yes", "no"]
        }
    }
    staff_map = {k: v["staff_replies"] for k, v in data.items()}
    gesture_map = {k: v["recommended_signs"] for k, v in data.items()}
    hints_map = {k: (v["category"], v["accent_color"], v["hint_text"]) for k, v in data.items()}
    return staff_map, gesture_map, hints_map, list(data.keys())


# --------------------------------------------------------------------------
# Text-to-Speech (TTS) Helper using Windows SAPI (Zero extra dependencies)
# --------------------------------------------------------------------------
_tts_lock = threading.Lock()

def speak_text_async(text):
    """Speaks English text asynchronously using Windows native SAPI with no popup console window."""
    if not text or text.startswith("(") or text == "-":
        return

    def _speak():
        # Sanitize to prevent command injection, PowerShell string bugs, or console flash
        clean = re.sub(r"[^a-zA-Z0-9\s\.,!\?]", "", text).strip()
        if not clean:
            return

        with _tts_lock:
            try:
                cmd = f'(New-Object -ComObject SAPI.SpVoice).Speak("{clean}")'
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                    timeout=5,
                    creationflags=creationflags,
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

        # Threading lock for model switching and inference
        self.model_lock = threading.Lock()

        # UI Styling & Font Scaling
        self.font_scale_idx = 1  # Default to Large (120%) for superior readability
        self._setup_styles()

        # Model & State Variables
        self.scenario_staff_replies, self.scenario_gesture_presets, self.scenario_hints, self.scenario_list = load_scenario_data()
        self.model_type = "gru"  # 'gru' or 'rf'
        self.classifier = None
        self.id_to_label = {}
        self.threshold = 0.60    # Unified recognition threshold
        self.auto_speak = tk.BooleanVar(value=True)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.trigger_mode = tk.StringVar(value="auto")  # 'auto' or 'manual'
        self.selected_scenario = tk.StringVar(value=self.scenario_list[0] if self.scenario_list else "Hospital & Clinic")
        self.active_model_name = tk.StringVar(value="PyTorch GRU")

        # Thread-safe copies of control variables for camera worker
        self.trigger_mode_val = "auto"
        self.show_skeleton_val = True

        # Track active notebook tab: 0 = Translation, 1 = Studio
        self.active_tab_idx = 0

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
        self._render_pending = False

        self.state = "idle"  # 'idle', 'countdown', 'recording', 'result'
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

        # Initialize Scenario Presets & Font Sizing
        self._render_scenario_buttons()
        self._apply_font_scale()

        # Keyboard shortcuts (Safe focus checking)
        self.bind("<space>", self._on_space_pressed)

        # Tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Sync variable traces
        self.trigger_mode.trace_add("write", lambda *args: self._sync_trigger_mode())
        self.show_skeleton.trace_add("write", lambda *args: self._sync_skeleton())

        # Load AI Model in background
        self.after(200, self._load_model_safe)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _sync_trigger_mode(self):
        self.trigger_mode_val = self.trigger_mode.get()

    def _sync_skeleton(self):
        self.show_skeleton_val = self.show_skeleton.get()

    def _on_tab_changed(self, event=None):
        try:
            self.active_tab_idx = self.notebook.index(self.notebook.select())
        except Exception:
            self.active_tab_idx = 0

    def _on_space_pressed(self, event):
        """Triggers manual record ONLY if not currently focused in a text entry box."""
        if event and isinstance(event.widget, (tk.Entry, tk.Text, ttk.Combobox, ttk.Entry)):
            return  # Allow normal typing of space characters in text fields
        self.trigger_manual_record()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Color Palette
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

        # Brand Title
        brand_frame = tk.Frame(header_frame, bg="#181818")
        brand_frame.pack(side="left", padx=16, pady=10)

        tk.Label(brand_frame, text="🤟 TSL ASSIST", font=("Segoe UI", 14, "bold"), fg=self.C_ACCENT_CYAN, bg="#181818").pack(side="left")
        tk.Label(brand_frame, text=" | Sign Language Translation Hub", font=("Segoe UI", 10), fg=self.C_MUTED, bg="#181818").pack(side="left", padx=(4, 0))

        # Scenario Context Switcher
        scenario_frame = tk.Frame(header_frame, bg="#181818")
        scenario_frame.pack(side="left", padx=24, pady=10)

        tk.Label(scenario_frame, text="Scenario:", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg="#181818").pack(side="left", padx=(0, 6))

        self.combo_scenario = ttk.Combobox(
            scenario_frame,
            textvariable=self.selected_scenario,
            values=self.scenario_list,
            state="readonly",
            width=22,
            font=("Segoe UI", 9, "bold")
        )
        self.combo_scenario.pack(side="left")
        self.combo_scenario.bind("<<ComboboxSelected>>", self._on_scenario_switch)

        # Presets Manager Button
        btn_presets = tk.Button(
            scenario_frame,
            text="⚙️ Presets",
            font=("Segoe UI", 8, "bold"),
            bg="#263238",
            fg="#80D8FF",
            activebackground="#37474F",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2",
            command=self.open_presets_settings
        )
        btn_presets.pack(side="left", padx=(8, 0))

        # AI Engine Switcher (GRU vs RF)
        engine_frame = tk.Frame(header_frame, bg="#181818")
        engine_frame.pack(side="left", padx=14, pady=10)

        tk.Label(engine_frame, text="Engine:", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg="#181818").pack(side="left", padx=(0, 6))
        self.combo_model = ttk.Combobox(
            engine_frame,
            textvariable=self.active_model_name,
            values=["PyTorch GRU", "Random Forest"],
            state="readonly",
            width=14,
            font=("Segoe UI", 9)
        )
        self.combo_model.pack(side="left")
        self.combo_model.bind("<<ComboboxSelected>>", self._on_model_switch)

        # Status Indicators (Right side of header)
        status_frame = tk.Frame(header_frame, bg="#181818")
        status_frame.pack(side="right", padx=16, pady=10)

        self.lbl_camera_status = tk.Label(status_frame, text="● Camera: Ready", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg="#181818")
        self.lbl_camera_status.pack(side="left", padx=10)

        self.lbl_model_status = tk.Label(status_frame, text="● Model: Loading...", font=("Segoe UI", 9, "bold"), fg=self.C_ACCENT_YELLOW, bg="#181818")
        self.lbl_model_status.pack(side="left", padx=10)

    # ----------------------------------------------------------------------
    # 2. Main Dual-Pane Content (Viewfinder Left / Hub Right)
    # ----------------------------------------------------------------------
    def _build_main_content(self):
        content_frame = tk.Frame(self, bg=self.C_BG)
        content_frame.pack(side="top", fill="both", expand=True, padx=12, pady=10)

        # Equal 50/50 split between Camera Viewfinder and Translation Hub
        content_frame.columnconfigure(0, weight=1, uniform="panels")
        content_frame.columnconfigure(1, weight=1, uniform="panels")
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
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Title bar of viewfinder
        v_head = tk.Frame(left_card, bg=self.C_CARD)
        v_head.pack(fill="x", padx=14, pady=(8, 4))

        tk.Label(v_head, text="Camera Viewfinder & Motion Tracker", font=("Segoe UI", 11, "bold"),
                 fg=self.C_TEXT, bg=self.C_CARD).pack(side="left")

        # Mode Badge (READY / RECORDING / RESULT)
        self.badge_state = tk.Label(v_head, text="IDLE", font=("Segoe UI", 9, "bold"),
                                    fg="#121212", bg=self.C_MUTED, padx=10, pady=2)
        self.badge_state.pack(side="right")

        # Video Canvas (Scales cleanly in all window sizes)
        self.canvas_video = tk.Canvas(left_card, bg="#0A0A0A", highlightthickness=0)
        self.canvas_video.pack(fill="both", expand=True, padx=12, pady=4)

        # Environmental Diagnostics Bar
        diag_frame = tk.Frame(left_card, bg=self.C_CARD_LIGHT, height=30)
        diag_frame.pack(fill="x", padx=12, pady=4)

        self.lbl_distance = tk.Label(diag_frame, text="📏 Distance: --", font=("Segoe UI", 9), fg=self.C_MUTED, bg=self.C_CARD_LIGHT)
        self.lbl_distance.pack(side="left", padx=10, pady=3)

        self.lbl_hands_found = tk.Label(diag_frame, text="✋ Hands: L[-] R[-]", font=("Segoe UI", 9), fg=self.C_MUTED, bg=self.C_CARD_LIGHT)
        self.lbl_hands_found.pack(side="left", padx=12, pady=3)

        self.lbl_fps = tk.Label(diag_frame, text="⚡ FPS: 0.0", font=("Segoe UI", 9), fg=self.C_MUTED, bg=self.C_CARD_LIGHT)
        self.lbl_fps.pack(side="right", padx=10, pady=3)

        # Viewfinder Control Bar (Clean 2-Row Layout)
        ctrl_frame = tk.Frame(left_card, bg=self.C_CARD)
        ctrl_frame.pack(fill="x", padx=12, pady=(4, 10))

        # Row 1: Camera Actions & Skeleton Toggle
        row1 = tk.Frame(ctrl_frame, bg=self.C_CARD)
        row1.pack(fill="x", pady=(0, 4))

        self.btn_camera = tk.Button(row1, text="▶ Start Camera", font=("Segoe UI", 9, "bold"),
                                    bg=self.C_ACCENT_GREEN, fg="#121212", activebackground="#00C853",
                                    relief="flat", padx=12, pady=4, cursor="hand2", command=self.toggle_camera)
        self.btn_camera.pack(side="left", padx=(0, 6))

        self.btn_trigger = tk.Button(row1, text="🔘 Trigger (Space)", font=("Segoe UI", 9),
                                     bg=self.C_CARD_LIGHT, fg=self.C_TEXT, activebackground=self.C_BORDER,
                                     relief="flat", padx=10, pady=4, cursor="hand2", command=self.trigger_manual_record)
        self.btn_trigger.pack(side="left", padx=4)

        chk_skel = tk.Checkbutton(row1, text="Skeleton Overlay", variable=self.show_skeleton,
                                  bg=self.C_CARD, fg=self.C_TEXT, selectcolor=self.C_CARD_LIGHT,
                                  activebackground=self.C_CARD, font=("Segoe UI", 9))
        chk_skel.pack(side="right")

        # Row 2: Detection Mode Selector
        row2 = tk.Frame(ctrl_frame, bg=self.C_CARD)
        row2.pack(fill="x", pady=(2, 0))

        tk.Label(row2, text="Trigger Mode:", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg=self.C_CARD).pack(side="left", padx=(0, 6))
        
        tk.Radiobutton(row2, text="Auto (Chest Line)", variable=self.trigger_mode, value="auto",
                       bg=self.C_CARD, fg=self.C_TEXT, selectcolor=self.C_CARD_LIGHT, activebackground=self.C_CARD,
                       font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))
        tk.Radiobutton(row2, text="Manual (Spacebar)", variable=self.trigger_mode, value="manual",
                       bg=self.C_CARD, fg=self.C_TEXT, selectcolor=self.C_CARD_LIGHT, activebackground=self.C_CARD,
                       font=("Segoe UI", 9)).pack(side="left")

    # ----------------------------------------------------------------------
    # Right Tabbed Hub (Tab 1: Live Translation | Tab 2: Gesture Studio)
    # ----------------------------------------------------------------------
    def _build_right_tabbed_hub(self, parent):
        right_container = tk.Frame(parent, bg=self.C_BG)
        right_container.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

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
        # 1. HERO RESULT CARD (Packed at TOP)
        hero_frame = tk.Frame(parent, bg="#15212E", highlightbackground=self.C_ACCENT_CYAN, highlightthickness=1)
        hero_frame.pack(side="top", fill="x", padx=12, pady=(8, 4))

        h_title_bar = tk.Frame(hero_frame, bg="#15212E")
        h_title_bar.pack(fill="x", padx=12, pady=(6, 2))
        tk.Label(h_title_bar, text="TRANSLATED SIGN OUTPUT", font=("Segoe UI", 8, "bold"), fg=self.C_ACCENT_CYAN, bg="#15212E").pack(side="left")

        self.lbl_scenario_tag = tk.Label(h_title_bar, text=f"[{self.selected_scenario.get()}]", font=("Segoe UI", 8, "bold"), fg=self.C_ACCENT_CYAN, bg="#15212E")
        self.lbl_scenario_tag.pack(side="left", padx=8)

        self.lbl_conf_tag = tk.Label(h_title_bar, text="Confidence: --%", font=("Segoe UI", 8, "bold"), fg=self.C_MUTED, bg="#15212E")
        self.lbl_conf_tag.pack(side="right")

        # Large English Word Display
        self.lbl_hero_word = tk.Label(hero_frame, text="Awaiting Sign...", font=("Segoe UI", 24, "bold"),
                                      fg=self.C_TEXT, bg="#15212E", pady=4)
        self.lbl_hero_word.pack()

        # Context Guidance / Hint
        init_hint = self.scenario_hints.get(self.selected_scenario.get(), ("", "", ""))[2]
        self.lbl_scenario_hint = tk.Label(
            hero_frame,
            text=init_hint,
            font=("Segoe UI", 8),
            fg="#8BA3BA",
            bg="#15212E"
        )
        self.lbl_scenario_hint.pack(pady=(0, 4))

        # Confidence Bar (Canvas)
        self.canvas_conf_bar = tk.Canvas(hero_frame, height=5, bg="#0D161F", highlightthickness=0)
        self.canvas_conf_bar.pack(fill="x", padx=14, pady=(0, 8))

        # Hero Action Bar (Speak button, Auto-TTS toggle)
        h_actions = tk.Frame(hero_frame, bg="#15212E")
        h_actions.pack(fill="x", padx=12, pady=(0, 8))

        self.btn_speak = tk.Button(h_actions, text="🔊 Speak (Audio)", font=("Segoe UI", 9, "bold"),
                                   bg=self.C_ACCENT_CYAN, fg="#121212", activebackground="#00B0FF",
                                   relief="flat", padx=10, pady=2, command=self.speak_current_sign)
        self.btn_speak.pack(side="left")

        chk_tts = tk.Checkbutton(h_actions, text="Auto-Speak on detect", variable=self.auto_speak,
                                 bg="#15212E", fg=self.C_TEXT, selectcolor="#0D161F",
                                 activebackground="#15212E", font=("Segoe UI", 9))
        chk_tts.pack(side="left", padx=10)

        # 2. TWO-WAY STAFF RESPONSE (Packed at BOTTOM so it is ALWAYS visible)
        self.two_way_frame = tk.LabelFrame(
            parent,
            text=" 💬 Staff Two-Way Communicator (Display to Signer) ",
            font=("Segoe UI", 10, "bold"),
            fg=self.C_ACCENT_GREEN,
            bg="#16202C",
            highlightbackground="#27394D",
            highlightthickness=1,
            padx=10,
            pady=6
        )
        self.two_way_frame.pack(side="bottom", fill="x", padx=12, pady=(4, 10))

        # Instructions / Context Bar + Edit Presets Button
        tw_bar = tk.Frame(self.two_way_frame, bg="#16202C")
        tw_bar.pack(fill="x", padx=2, pady=(0, 2))

        self.lbl_tw_hint = tk.Label(
            tw_bar,
            text="Quick response presets (Click to send instantly):",
            font=("Segoe UI", 9),
            fg="#8FA3B8",
            bg="#16202C"
        )
        self.lbl_tw_hint.pack(side="left")

        btn_edit_words = tk.Button(
            tw_bar,
            text="⚙️ Edit Words",
            font=("Segoe UI", 8),
            bg="#203040",
            fg="#80D8FF",
            activebackground="#2E445B",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=6,
            pady=1,
            cursor="hand2",
            command=self.open_presets_settings
        )
        btn_edit_words.pack(side="right")

        # Quick preset buttons (dynamically populated based on Scenario)
        self.quick_frame = tk.Frame(self.two_way_frame, bg="#16202C")
        self.quick_frame.pack(fill="x", padx=0, pady=(2, 4))

        # Custom Reply Input Row
        input_box = tk.Frame(self.two_way_frame, bg="#16202C")
        input_box.pack(fill="x", padx=0, pady=(3, 2))

        self.entry_reply = tk.Entry(
            input_box,
            font=("Segoe UI", 12),
            bg="#0D151E",
            fg="#FFFFFF",
            insertbackground="#00E5FF",
            relief="flat",
            highlightbackground="#2F445B",
            highlightcolor=self.C_ACCENT_GREEN,
            highlightthickness=1
        )
        self.entry_reply.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=6)
        self.entry_reply.bind("<Return>", lambda e: self.send_staff_reply(self.entry_reply.get()))

        self.btn_send_reply = tk.Button(
            input_box,
            text=" Send Reply ↵ ",
            font=("Segoe UI", 10, "bold"),
            bg=self.C_ACCENT_GREEN,
            fg="#0A1E11",
            activebackground="#00C853",
            activeforeground="#000000",
            relief="flat",
            padx=16,
            pady=5,
            cursor="hand2",
            command=lambda: self.send_staff_reply(self.entry_reply.get())
        )
        self.btn_send_reply.pack(side="right")

        # 3. CONVERSATION TIMELINE HEADER (Packed at TOP after Hero)
        t_header = tk.Frame(parent, bg=self.C_CARD)
        t_header.pack(side="top", fill="x", padx=14, pady=(4, 2))

        tk.Label(t_header, text="Conversation Timeline", font=("Segoe UI", 11, "bold"),
                 fg=self.C_TEXT, bg=self.C_CARD).pack(side="left")

        # Clear Log Button
        tk.Button(t_header, text="🗑 Clear Log", font=("Segoe UI", 8), bg="#222C38", fg=self.C_MUTED,
                  activebackground="#2F3E50", activeforeground="#FFFFFF",
                  relief="flat", padx=8, pady=1, cursor="hand2", command=self.clear_log).pack(side="right", padx=(8, 0))

        # Font Scaling Controls [ A- ] [ A+ ]
        font_box = tk.Frame(t_header, bg=self.C_CARD)
        font_box.pack(side="right", padx=4)

        tk.Label(font_box, text="Font Size:", font=("Segoe UI", 8, "bold"), fg=self.C_MUTED, bg=self.C_CARD).pack(side="left", padx=(0, 4))

        tk.Button(font_box, text=" A- ", font=("Segoe UI", 8, "bold"), bg="#203040", fg="#80D8FF",
                  relief="flat", padx=6, pady=1, cursor="hand2",
                  command=lambda: self.change_font_scale(-1)).pack(side="left", padx=1)

        self.lbl_font_indicator = tk.Label(font_box, text="Large", font=("Segoe UI", 8), fg="#00E5FF", bg=self.C_CARD, width=12)
        self.lbl_font_indicator.pack(side="left", padx=2)

        tk.Button(font_box, text=" A+ ", font=("Segoe UI", 8, "bold"), bg="#203040", fg="#80D8FF",
                  relief="flat", padx=6, pady=1, cursor="hand2",
                  command=lambda: self.change_font_scale(1)).pack(side="left", padx=1)

        # 4. CONVERSATION TIMELINE LOG CONTAINER (Fills remaining middle space)
        log_container = tk.Frame(parent, bg=self.C_CARD)
        log_container.pack(side="top", fill="both", expand=True, padx=12, pady=2)

        self.txt_timeline = tk.Text(log_container, bg="#121820", fg=self.C_TEXT, font=("Segoe UI", 12),
                                    wrap="word", relief="flat", padx=12, pady=8, state="disabled",
                                    height=6, highlightthickness=1, highlightbackground="#202C3A")
        self.txt_timeline.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_container, orient="vertical", command=self.txt_timeline.yview)
        scrollbar.pack(side="right", fill="y")
        self.txt_timeline.config(yscrollcommand=scrollbar.set)

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

        # Quick preset words container (dynamically wrapped in rows)
        self.presets_box = tk.Frame(sec1, bg=self.C_CARD)
        self.presets_box.pack(fill="x", pady=4)

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
    # Accessibility Font Scaling Controller
    # ----------------------------------------------------------------------
    def change_font_scale(self, delta):
        """Increases or decreases font scale across conversation log, communicator, entry, and buttons."""
        self.font_scale_idx = max(0, min(len(FONT_LEVELS) - 1, self.font_scale_idx + delta))
        self._apply_font_scale()

    def _apply_font_scale(self):
        cfg = FONT_LEVELS[self.font_scale_idx]
        if hasattr(self, "lbl_font_indicator"):
            self.lbl_font_indicator.config(text=cfg["name"])

        # Timeline Text Tags
        if hasattr(self, "txt_timeline"):
            self.txt_timeline.config(font=("Segoe UI", cfg["log_body"]))
            self.txt_timeline.tag_config("time", foreground="#788899", font=("Consolas", cfg["log_time"]))
            self.txt_timeline.tag_config("signer", foreground="#00E5FF", font=("Segoe UI", cfg["log_signer"], "bold"))
            self.txt_timeline.tag_config("signer_body", foreground="#FFFFFF", font=("Segoe UI", cfg["log_signer"] + 1, "bold"))
            self.txt_timeline.tag_config("staff", foreground="#00E676", font=("Segoe UI", cfg["log_staff"], "bold"))
            self.txt_timeline.tag_config("staff_body", foreground="#E8F5E9", font=("Segoe UI", cfg["log_staff"]))
            self.txt_timeline.tag_config("system", foreground="#FFD600", font=("Segoe UI", cfg["log_time"] + 1, "bold"))
            self.txt_timeline.tag_config("system_body", foreground="#FFE082", font=("Segoe UI", cfg["log_time"] + 1, "italic"))

        # Two-Way Communicator Controls
        if hasattr(self, "two_way_frame"):
            self.two_way_frame.config(font=("Segoe UI", cfg["header_sub"], "bold"))
        if hasattr(self, "lbl_tw_hint"):
            self.lbl_tw_hint.config(font=("Segoe UI", cfg["header_sub"]))
        if hasattr(self, "entry_reply"):
            self.entry_reply.config(font=("Segoe UI", cfg["entry"]))
        if hasattr(self, "btn_send_reply"):
            self.btn_send_reply.config(font=("Segoe UI", cfg["btn_send"], "bold"))

        # Re-render quick preset buttons with new font and padding
        self._render_scenario_buttons()

    # ----------------------------------------------------------------------
    # AI Model & Scenario Context Switching
    # ----------------------------------------------------------------------
    def _on_scenario_switch(self, event=None):
        self.reload_scenario_presets()
        scenario = self.selected_scenario.get()
        self._append_to_timeline("⚙️ [System]", f"Switched context to [{scenario}]. Quick replies & recommended signs updated.", "system")

    def reload_scenario_presets(self):
        """Reloads scenario presets dynamically from data/scenario_presets.json."""
        self.scenario_staff_replies, self.scenario_gesture_presets, self.scenario_hints, self.scenario_list = load_scenario_data()
        if hasattr(self, "combo_scenario"):
            self.combo_scenario.config(values=self.scenario_list)
        self._render_scenario_buttons()

    def open_presets_settings(self):
        """Opens data/scenario_presets.json in default editor so user can customize words."""
        file_path = getattr(config, "PRESETS_CONFIG_PATH", config.DATA_DIR / "scenario_presets.json")
        if file_path.exists():
            try:
                os.startfile(str(file_path))
            except Exception:
                subprocess.run(["notepad.exe", str(file_path)])
            messagebox.showinfo(
                "Customize Scenario Presets",
                f"Opened preset configuration file:\n\n{file_path}\n\n"
                "• You can edit or add custom staff quick replies ('staff_replies')\n"
                "• You can edit or add recommended gesture words ('recommended_signs')\n"
                "• Save the file in Notepad / VS Code, then re-select the Scenario in the app to apply changes instantly!"
            )
        else:
            messagebox.showwarning("File Not Found", f"Cannot find: {file_path}")

    def _render_scenario_buttons(self):
        scenario = self.selected_scenario.get()
        hint_info = self.scenario_hints.get(scenario, ("General", self.C_ACCENT_CYAN, ""))
        cfg = FONT_LEVELS[self.font_scale_idx]

        # Update Scenario Badge and Hint in Hero Card
        if hasattr(self, "lbl_scenario_tag"):
            self.lbl_scenario_tag.config(text=f"[{scenario}]", fg=hint_info[1])
        if hasattr(self, "lbl_scenario_hint"):
            self.lbl_scenario_hint.config(text=hint_info[2])

        # 1. Update Staff Two-Way Replies with styled chip buttons wrapped nicely in rows of 3
        if hasattr(self, "quick_frame"):
            for widget in self.quick_frame.winfo_children():
                widget.destroy()
            replies = self.scenario_staff_replies.get(scenario, ["Please wait a moment", "Follow me please"])
            
            # Chunk replies into rows of at most 3 items
            chunk_size = 3
            reply_chunks = [replies[i:i + chunk_size] for i in range(0, len(replies), chunk_size)]
            for row_items in reply_chunks:
                row_f = tk.Frame(self.quick_frame, bg="#16202C")
                row_f.pack(fill="x", pady=2)
                for p in row_items:
                    btn = tk.Button(
                        row_f,
                        text=f"💬 {p}",
                        font=("Segoe UI", cfg["presets"], "bold"),
                        bg="#223648",
                        fg="#00E5FF",
                        activebackground="#00B0FF",
                        activeforeground="#FFFFFF",
                        relief="flat",
                        bd=0,
                        padx=cfg["padding_x"],
                        pady=cfg["padding_y"],
                        cursor="hand2",
                        command=lambda msg=p: self.send_staff_reply(msg)
                    )
                    btn.pack(side="left", padx=3, pady=1)

        # 2. Update Studio Suggested Gesture Words wrapped in rows of 4
        if hasattr(self, "presets_box"):
            for widget in self.presets_box.winfo_children():
                widget.destroy()
            words = self.scenario_gesture_presets.get(scenario, ["hello", "thank_you", "yes", "no"])
            chunk_size = 4
            word_chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
            for r_idx, row_words in enumerate(word_chunks):
                row_f = tk.Frame(self.presets_box, bg=self.C_CARD)
                row_f.pack(fill="x", pady=2)
                if r_idx == 0:
                    tk.Label(row_f, text="Presets:", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg=self.C_CARD).pack(side="left", padx=(0, 6))
                for w in row_words:
                    btn = tk.Button(row_f, text=w, font=("Segoe UI", cfg["presets"], "bold"),
                                    bg="#263238", fg="#80D8FF", activebackground="#37474F", activeforeground="#FFFFFF",
                                    relief="flat", padx=cfg["padding_x"], pady=cfg["padding_y"], cursor="hand2",
                                    command=lambda val=w: self.studio_target_word.set(val))
                    btn.pack(side="left", padx=3)
            if words:
                self.studio_target_word.set(words[0])

    def _on_model_switch(self, event=None):
        name = self.active_model_name.get()
        target_type = "gru" if "GRU" in name else "rf"
        self._load_model_safe(target_type)

    def _load_model_safe(self, preferred_type=None):
        """Loads trained models (PyTorch GRU or Scikit-learn RF) safely under lock."""
        with self.model_lock:
            try:
                if not config.SAVED_LABELS.exists():
                    self.lbl_model_status.config(text="● No labels found", fg=self.C_ACCENT_RED)
                    self.classifier = None
                    return

                label_map = load_label_mapping(config.SAVED_LABELS)
                self.id_to_label = {v: k for k, v in label_map.items()}

                if preferred_type is None:
                    preferred_type = self.model_type

                if preferred_type == "gru" and config.SAVED_TORCH_MODEL.exists() and torch is not None:
                    ckpt = torch.load(config.SAVED_TORCH_MODEL, map_location="cpu", weights_only=True)
                    model = SignGRU(
                        input_size=ckpt["input_size"],
                        hidden_size=ckpt["hidden_size"],
                        num_layers=ckpt["num_layers"],
                        num_classes=ckpt["num_classes"],
                    )
                    model.load_state_dict(ckpt["model_state"])
                    model.eval()
                    self.classifier = model
                    self.model_type = "gru"
                    self.active_model_name.set("PyTorch GRU")
                    self.lbl_model_status.config(text=f"● Model: GRU ({len(self.id_to_label)} gestures)", fg=self.C_ACCENT_GREEN)

                elif config.SAVED_SKLEARN_MODEL.exists() and joblib is not None:
                    self.classifier = joblib.load(config.SAVED_SKLEARN_MODEL)
                    self.model_type = "rf"
                    self.active_model_name.set("Random Forest")
                    self.lbl_model_status.config(text=f"● Model: RF ({len(self.id_to_label)} gestures)", fg=self.C_ACCENT_GREEN)

                else:
                    self.classifier = None
                    self.lbl_model_status.config(text="● Untrained (Run training)", fg="#FF9100")

            except Exception as e:
                self.classifier = None
                self.lbl_model_status.config(text=f"● Load error: {e}", fg=self.C_ACCENT_RED)

    def _predict(self, seq_24x112):
        """Predicts gesture class and confidence from (24, 112) matrix under lock."""
        with self.model_lock:
            if self.classifier is None:
                return "Untrained Model", 0.0

            if self.model_type == "rf":
                if hasattr(self.classifier, "predict_proba"):
                    flat = seq_24x112.reshape(1, -1)
                    probs = self.classifier.predict_proba(flat)[0]
                    idx = int(np.argmax(probs))
                    label = str(self.classifier.classes_[idx])
                    return label, float(probs[idx])
                return "RF Not Ready", 0.0

            elif self.model_type == "gru":
                if torch is not None and hasattr(self.classifier, "__call__"):
                    with torch.no_grad():
                        tensor_in = torch.tensor(seq_24x112[None, ...], dtype=torch.float32)
                        logits = self.classifier(tensor_in)
                        probs = torch.softmax(logits, dim=-1)[0].numpy()
                        idx = int(np.argmax(probs))
                        label = self.id_to_label.get(idx, f"Class {idx}")
                        return label, float(probs[idx])
                return "GRU Not Ready", 0.0

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
        if self.is_running:
            return
        try:
            self.cam = Camera()
            self.tracker = Tracker()
            self.is_running = True
            self.btn_camera.config(text="⏹ Stop Camera", bg=self.C_ACCENT_RED)
            self.lbl_camera_status.config(text="● Camera: Live", fg=self.C_ACCENT_GREEN)

            self.thread = threading.Thread(target=self._camera_worker, daemon=True)
            self.thread.start()
        except Exception as e:
            self.is_running = False
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
        self._render_pending = False

    def trigger_manual_record(self):
        """Triggers recording manually (via button or Spacebar)."""
        if not self.is_running:
            self.start_camera()
            if not self.is_running:
                return
            # Allow camera 200ms to initialize before arming
            self.after(200, self._arm_manual_record)
        else:
            self._arm_manual_record()

    def _arm_manual_record(self):
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

            # State Transitions (Thread-Safe check of variables)
            mode = self.trigger_mode_val
            is_translation_tab = (self.active_tab_idx == 0)

            if self.state == "idle":
                # Auto-trigger arms only in Tab 1 (Translation) and when not in Studio mode
                if mode == "auto" and hand_in_zone and is_translation_tab and not self.studio_recording:
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

            elif self.state == "countdown":
                # During studio countdown, auto-trigger is suppressed
                self.arm_start_time = None

            elif self.state == "recording":
                self.current_recording.append(frame_vec)
                elapsed = now - self.rec_start_time
                if elapsed >= 1.8:
                    # Recording finished
                    self.state = "result"
                    self._update_badge("PROCESSING", self.C_ACCENT_CYAN)

                    # Check if this was a Studio sample recording
                    if self.studio_recording:
                        rec_frames = list(self.current_recording)
                        self.studio_recording = False
                        # Dispatch sample save to main thread safely
                        self.after(0, self._save_studio_sample, rec_frames)
                    else:
                        if len(self.current_recording) >= 10:
                            seq_24 = resample_sequence(np.array(self.current_recording), target_len=config.TARGET_FRAMES)
                            label, conf = self._predict(seq_24)
                            self.after(0, self._on_prediction_ready, label, conf)
                        else:
                            self.after(0, self._on_prediction_ready, "(No motion)", 0.0)

            elif self.state == "result":
                # Cooldown: keep result visible for at least 1.4s
                elapsed = now - self.rec_start_time
                if elapsed >= 1.4:
                    if mode == "auto":
                        # In auto mode, wait until hand is lowered or 3.0s timeout to prevent immediate re-trigger
                        if not hand_in_zone or elapsed >= 3.0:
                            self.state = "idle"
                            self.arm_start_time = None
                    else:
                        self.state = "idle"

            # Render Overlays
            view = frame.copy()
            if self.show_skeleton_val:
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

            # Render to Tkinter Canvas efficiently with queue guard
            rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            if not self._render_pending:
                self._render_pending = True
                self.after(0, self._render_canvas, rgb)

    def _render_canvas(self, rgb_array):
        """
        Draws RGB frame onto canvas.
        Dynamically scales the image UP when window is maximized/large to fill the frame,
        and scales DOWN smoothly when window is smaller, preserving aspect ratio with zero clipping.
        """
        c_w = self.canvas_video.winfo_width()
        c_h = self.canvas_video.winfo_height()
        if c_w < 10 or c_h < 10:
            self._render_pending = False
            return

        orig_h, orig_w = rgb_array.shape[:2]
        # Aspect-ratio scaling that fits available canvas space
        scale = min(c_w / orig_w, c_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        img = Image.fromarray(rgb_array)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        img_tk = ImageTk.PhotoImage(image=img)

        if self._video_canvas_item is None:
            self._video_canvas_item = self.canvas_video.create_image(c_w // 2, c_h // 2, image=img_tk, anchor="center")
        else:
            self.canvas_video.coords(self._video_canvas_item, c_w // 2, c_h // 2)
            self.canvas_video.itemconfig(self._video_canvas_item, image=img_tk)

        self.canvas_video.image = img_tk  # Keep reference
        self._render_pending = False

    def _update_badge(self, text, bg_color):
        self.after(0, lambda: self.badge_state.config(text=text, bg=bg_color))

    def _update_diagnostics(self, dist_str, l_found, r_found, fps, lat):
        dist_color = self.C_ACCENT_GREEN if dist_str != "--" else self.C_MUTED
        self.lbl_distance.config(text=f"📏 Distance: {dist_str}", fg=dist_color)

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

        # Check against confidence threshold
        is_confident = (conf >= self.threshold) and (label not in ("(No motion)", "Unknown", "Untrained Model"))

        if is_confident:
            display_title = clean_label
            display_color = self.C_TEXT
            bar_color = self.C_ACCENT_GREEN
        else:
            if label in ("(No motion)", "Untrained Model"):
                display_title = clean_label
            else:
                display_title = f"{clean_label} (Uncertain)"
            display_color = "#FFB300"
            bar_color = "#FF9100"

        self.lbl_hero_word.config(text=display_title, fg=display_color)
        self.lbl_conf_tag.config(text=f"Confidence: {int(conf * 100)}%")

        # Draw Confidence Bar
        self.canvas_conf_bar.delete("all")
        bar_w = self.canvas_conf_bar.winfo_width()
        fill_w = int(bar_w * np.clip(conf, 0.0, 1.0))
        self.canvas_conf_bar.create_rectangle(0, 0, fill_w, 6, fill=bar_color, width=0)

        # Append to Conversation Timeline and Auto-speak ONLY if confident
        if is_confident:
            self._append_to_timeline("🤟 [Signer]", f"{clean_label} ({int(conf * 100)}%)", "signer")
            if self.auto_speak.get():
                speak_text_async(clean_label)

    def speak_current_sign(self):
        """Manual click to speak the current hero sign."""
        if self.last_sign and self.last_sign not in ("-", "(No motion)", "Untrained Model"):
            clean = self.last_sign.replace("_", " ").title()
            speak_text_async(clean)

    def send_staff_reply(self, message):
        """Sends reply from hearing staff/companion to display on screen."""
        msg = message.strip()
        if not msg:
            return
        self._append_to_timeline("👨‍💼 [Staff]", msg, "staff")
        self.entry_reply.delete(0, tk.END)

    def _append_to_timeline(self, speaker_badge, text_content, tag_type="signer"):
        """
        Appends styled, highly readable chat message into the timeline.
        tag_type: 'signer', 'staff', or 'system'
        """
        ts = time.strftime("%H:%M:%S")
        self.txt_timeline.config(state="normal")

        # Timestamp
        self.txt_timeline.insert(tk.END, f"[{ts}] ", "time")

        # Speaker Badge and Body
        if tag_type == "signer":
            self.txt_timeline.insert(tk.END, f"{speaker_badge} ", "signer")
            self.txt_timeline.insert(tk.END, f"{text_content}\n\n", "signer_body")
        elif tag_type == "staff":
            self.txt_timeline.insert(tk.END, f"{speaker_badge} ", "staff")
            self.txt_timeline.insert(tk.END, f"{text_content}\n\n", "staff_body")
        elif tag_type == "system":
            self.txt_timeline.insert(tk.END, f"{speaker_badge} ", "system")
            self.txt_timeline.insert(tk.END, f"{text_content}\n\n", "system_body")
        else:
            self.txt_timeline.insert(tk.END, f"{speaker_badge} {text_content}\n\n")

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
            if not self.is_running:
                return

        self.btn_studio_rec.config(state="disabled")
        self.state = "countdown"
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
        """Saves recorded frames directly into data/raw/<word>/ on main thread."""
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
        """Retrains Random Forest and PyTorch GRU in background thread with gradient clipping & evaluation."""
        self.btn_train_models.config(state="disabled")
        self.lbl_train_status.config(text="Training in progress... Please wait 3-5 seconds", fg=self.C_ACCENT_YELLOW)

        def _worker():
            try:
                # 1. Load dataset safely (skips empty directories)
                X, y, groups, label_to_id = load_raw_dataset()
                if len(label_to_id) < 2:
                    self.after(0, lambda: messagebox.showwarning("Training Error", "Need at least 2 distinct gesture classes to train!"))
                    self.after(0, self._on_train_finished, False, "Need at least 2 classes")
                    return

                # 2. Split dataset with safe stratification
                X_train, X_test, y_train, y_test = split_dataset(X, y, groups, test_size=0.2)

                # Augment compact training sets
                if len(X_train) < 50:
                    X_train, y_train = augment_dataset(X_train, y_train, num_copies=1, noise_std=0.015)

                # Train Random Forest
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)

                clf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42)
                clf.fit(X_train_flat, y_train)
                preds = clf.predict(X_test_flat)
                rf_acc = accuracy_score(y_test, preds)

                config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(clf, config.SAVED_SKLEARN_MODEL)
                save_label_mapping(label_to_id, config.SAVED_LABELS)

                # 3. Train PyTorch GRU Sequence Model
                y_encoded = np.array([label_to_id[name] for name in y], dtype=np.int64)
                X_train_g, X_test_g, y_train_g, y_test_g = split_dataset(X, y_encoded, groups, test_size=0.2)
                if len(X_train_g) < 50:
                    X_train_g, y_train_g = augment_dataset(X_train_g, y_train_g, num_copies=1, noise_std=0.015)

                train_ds = TensorDataset(torch.tensor(X_train_g, dtype=torch.float32), torch.tensor(y_train_g, dtype=torch.long))
                test_ds = TensorDataset(torch.tensor(X_test_g, dtype=torch.float32), torch.tensor(y_test_g, dtype=torch.long))

                loader = DataLoader(train_ds, batch_size=min(16, max(2, len(train_ds))), shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=min(16, max(2, len(test_ds))), shuffle=False)

                device = torch.device("cpu")
                model = SignGRU(input_size=config.NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=len(label_to_id)).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()

                best_gru_acc = 0.0
                best_model_state = None

                for epoch in range(40):
                    model.train()
                    for bx, by in loader:
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    # Evaluate on test set
                    model.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for bx, by in test_loader:
                            out = model(bx)
                            correct += (out.argmax(dim=-1) == by).sum().item()
                            total += len(by)
                    epoch_acc = (correct / total) if total > 0 else 0.0

                    if epoch_acc >= best_gru_acc or best_model_state is None:
                        best_gru_acc = epoch_acc
                        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

                # Save best GRU checkpoint
                torch.save({
                    "model_state": best_model_state or model.state_dict(),
                    "label_to_id": label_to_id,
                    "input_size": config.NUM_FEATURES,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "num_classes": len(label_to_id),
                }, config.SAVED_TORCH_MODEL)

                summary_msg = f"RF Acc: {rf_acc*100:.0f}%  |  GRU Acc: {best_gru_acc*100:.0f}%  |  {len(label_to_id)} classes ready"
                self.after(0, self._on_train_finished, True, summary_msg)

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
