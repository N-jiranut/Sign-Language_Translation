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
from collections import deque
import json
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# Local module imports
_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from dataset import load_label_mapping
from features import extract_frame_features
from landmarkers import Camera, Tracker, draw_hand, draw_pose, hands_by_side
from models import HybridEnsemblePredictor
from train_all import train_all_models, train_scenario_profile

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

ENGINE_OPTIONS = [
    ("Hybrid (Auto-Specialist)", "hybrid"),
    ("PyTorch GRU", "gru"),
    ("PyTorch 1D-CNN", "cnn1d"),
    ("RF (Kinematic)", "rf_kinematic"),
    ("RF (Legacy Flat)", "rf_legacy"),
]
ENGINE_NAME_TO_KEY = dict(ENGINE_OPTIONS)
ENGINE_KEY_TO_NAME = {k: name for name, k in ENGINE_OPTIONS}
ENGINE_SHORT_NAMES = {
    "hybrid": "Hybrid",
    "gru": "GRU",
    "cnn1d": "1D-CNN",
    "rf_kinematic": "RF-Kin",
    "rf_legacy": "RF-Leg",
}

# --------------------------------------------------------------------------
# Scenario Presets Loader (reads data/scenario_presets.json)
# --------------------------------------------------------------------------
def load_scenario_data():
    """Loads scenario presets from data/scenario_presets.json with resilient fallback."""
    cfg_file = getattr(config, "PRESETS_CONFIG_PATH", config.DATA_DIR / "scenario_presets.json")
    data = None
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to parse {cfg_file}: {e}")

    if not data:
        data = {
            "General Services": {
                "icon": "🏢",
                "tag_color": "#FFD600",
                "description": "General Counter & Everyday Communication",
                "staff_replies": [
                    "Please wait a moment",
                    "Follow me please",
                    "Take queue ticket",
                    "Thank you"
                ],
                "active_gestures": ["Done", "Dont_like", "How", "Me", "No", "Thank", "Tomorrow", "Worry", "Yesterday", "You"],
                "recommended_signs": ["Done", "Dont_like", "How", "Me", "No", "Thank", "Tomorrow", "Worry", "Yesterday", "You"]
            },
            "Hospital & Clinic": {
                "icon": "🏥",
                "tag_color": "#00E5FF",
                "description": "Medical Consultation, ER & Patient Triage",
                "staff_replies": [
                    "Wait for doctor",
                    "Take medication",
                    "Follow nurse",
                    "Emergency room",
                    "Where does it hurt?",
                    "Please sit & rest"
                ],
                "active_gestures": ["Me", "You", "No", "Thank", "Worry", "Dont_like", "How", "Emergency", "Doctor", "Pain"],
                "recommended_signs": ["Emergency", "Doctor", "Pain", "Worry", "Me", "No", "Thank"]
            },
        }

    staff_map = {}
    gesture_map = {}
    active_map = {}
    hints_map = {}
    for k, v in data.items():
        staff_map[k] = list(v.get("staff_replies", []))
        active_list = list(v.get("active_gestures", v.get("recommended_signs", [])))
        rec_list = list(v.get("recommended_signs", active_list))
        active_map[k] = active_list
        gesture_map[k] = rec_list

        desc = v.get("description") or v.get("category") or "General"
        color = v.get("tag_color") or v.get("accent_color") or "#00E5FF"
        hint_text = v.get("hint_text")
        if not hint_text:
            preview = ", ".join(active_list[:8]) + ("..." if len(active_list) > 8 else "")
            hint_text = f"{desc}  |  Database ({len(active_list)}): {preview}" if active_list else desc
        hints_map[k] = (desc, color, hint_text)

    return staff_map, gesture_map, active_map, hints_map, list(data.keys())


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
        (
            self.scenario_staff_replies,
            self.scenario_gesture_presets,
            self.scenario_active_gestures,
            self.scenario_hints,
            self.scenario_list,
        ) = load_scenario_data()
        self.model_type = "hybrid"
        self.predictor = None
        self.id_to_label = {}
        self.threshold = 0.60    # Unified recognition threshold
        self.auto_speak = tk.BooleanVar(value=True)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.trigger_mode = tk.StringVar(value="auto")  # 'auto' or 'manual'
        self.line_ratio = tk.DoubleVar(value=getattr(config, "READY_LINE_RATIO", 0.74))
        self.selected_scenario = tk.StringVar(value=self.scenario_list[0] if self.scenario_list else "Hospital & Clinic")
        self.active_model_name = tk.StringVar(value="Hybrid (Auto-Specialist)")

        # Thread-safe copies of control variables for camera worker
        self.trigger_mode_val = "auto"
        self.show_skeleton_val = True
        self.line_ratio_val = getattr(config, "READY_LINE_RATIO", 0.74)

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
        self.result_start_time = 0.0
        self.still_start_time = None
        self.rec_peak_speed = 0.0
        self.hand_dropped_frames = 0
        self.retrigger_move_frames = 0
        self.prev_wrist_xy = None
        self.wrist_speed = 0.0
        self.pre_roll_buffer = deque(maxlen=config.PRE_ROLL_FRAMES)
        self.current_recording = []
        self.phrase_words = []
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
        self.line_ratio.trace_add("write", lambda *args: self._sync_line_ratio())

        # Load AI Model in background
        self.after(200, self._load_model_safe)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _sync_trigger_mode(self):
        self.trigger_mode_val = self.trigger_mode.get()

    def _sync_skeleton(self):
        self.show_skeleton_val = self.show_skeleton.get()

    def _sync_line_ratio(self):
        try:
            self.line_ratio_val = float(self.line_ratio.get())
        except Exception:
            pass

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
        brand_frame.pack(side="left", padx=(12, 8), pady=10)

        tk.Label(brand_frame, text="🤟 TSL ASSIST", font=("Segoe UI", 14, "bold"), fg=self.C_ACCENT_CYAN, bg="#181818").pack(side="left")
        tk.Label(brand_frame, text=" | TSL Hub", font=("Segoe UI", 10), fg=self.C_MUTED, bg="#181818").pack(side="left", padx=(4, 0))

        # Scenario Context Switcher
        scenario_frame = tk.Frame(header_frame, bg="#181818")
        scenario_frame.pack(side="left", padx=8, pady=10)

        tk.Label(scenario_frame, text="Scenario:", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg="#181818").pack(side="left", padx=(0, 6))

        self.combo_scenario = ttk.Combobox(
            scenario_frame,
            textvariable=self.selected_scenario,
            values=self.scenario_list,
            state="readonly",
            width=19,
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
        btn_presets.pack(side="left", padx=(6, 0))

        # AI Engine Switcher (Hybrid + Individual Models)
        engine_frame = tk.Frame(header_frame, bg="#181818")
        engine_frame.pack(side="left", padx=8, pady=10)

        tk.Label(engine_frame, text="Engine:", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg="#181818").pack(side="left", padx=(0, 6))
        self.combo_model = ttk.Combobox(
            engine_frame,
            textvariable=self.active_model_name,
            values=[name for name, _ in ENGINE_OPTIONS],
            state="readonly",
            width=22,
            font=("Segoe UI", 9)
        )
        self.combo_model.pack(side="left")
        self.combo_model.bind("<<ComboboxSelected>>", self._on_model_switch)

        # Status Indicators (Right side of header)
        status_frame = tk.Frame(header_frame, bg="#181818")
        status_frame.pack(side="right", padx=(6, 12), pady=10)

        self.lbl_camera_status = tk.Label(status_frame, text="● Camera: Ready", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg="#181818")
        self.lbl_camera_status.pack(side="left", padx=6)

        self.lbl_model_status = tk.Label(status_frame, text="● Loading...", font=("Segoe UI", 9, "bold"), fg=self.C_ACCENT_YELLOW, bg="#181818")
        self.lbl_model_status.pack(side="left", padx=6)

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

        # Row 2: Detection Mode Selector & Ready Line Height
        row2 = tk.Frame(ctrl_frame, bg=self.C_CARD)
        row2.pack(fill="x", pady=(2, 0))

        tk.Label(row2, text="Trigger Mode:", font=("Segoe UI", 9, "bold"), fg=self.C_MUTED, bg=self.C_CARD).pack(side="left", padx=(0, 6))
        
        tk.Radiobutton(row2, text="Auto Line", variable=self.trigger_mode, value="auto",
                       bg=self.C_CARD, fg=self.C_TEXT, selectcolor=self.C_CARD_LIGHT, activebackground=self.C_CARD,
                       font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))
        tk.Radiobutton(row2, text="Manual", variable=self.trigger_mode, value="manual",
                       bg=self.C_CARD, fg=self.C_TEXT, selectcolor=self.C_CARD_LIGHT, activebackground=self.C_CARD,
                       font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))

        # Line Position Slider (Low <-> High)
        lbl_line = tk.Label(row2, text="Line Height:", font=("Segoe UI", 8, "bold"), fg=self.C_MUTED, bg=self.C_CARD)
        lbl_line.pack(side="left", padx=(4, 2))

        self.scale_line = ttk.Scale(row2, from_=0.55, to=0.88, variable=self.line_ratio, orient="horizontal", length=90)
        self.scale_line.pack(side="left", padx=(2, 4))

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
                                      fg=self.C_TEXT, bg="#15212E", pady=2)
        self.lbl_hero_word.pack()

        # Specialist / Active Engine Attribution Badge
        self.lbl_hero_engine = tk.Label(
            hero_frame,
            text="Engine: Hybrid (Auto-Specialist)",
            font=("Segoe UI", 8, "bold"),
            fg="#80D8FF",
            bg="#15212E",
        )
        self.lbl_hero_engine.pack(pady=(0, 2))

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
        self.canvas_conf_bar.pack(fill="x", padx=14, pady=(0, 6))

        # Hero Action Bar (Speak button, Auto-TTS toggle)
        h_actions = tk.Frame(hero_frame, bg="#15212E")
        h_actions.pack(fill="x", padx=12, pady=(0, 6))

        self.btn_speak = tk.Button(h_actions, text="🔊 Speak (Audio)", font=("Segoe UI", 9, "bold"),
                                   bg=self.C_ACCENT_CYAN, fg="#121212", activebackground="#00B0FF",
                                   relief="flat", padx=10, pady=2, command=self.speak_current_sign)
        self.btn_speak.pack(side="left")

        chk_tts = tk.Checkbutton(h_actions, text="Auto-Speak on detect", variable=self.auto_speak,
                                 bg="#15212E", fg=self.C_TEXT, selectcolor="#0D161F",
                                 activebackground="#15212E", font=("Segoe UI", 9))
        chk_tts.pack(side="left", padx=10)

        # Continuous Phrase Bar (accumulates fast sequential signs)
        phrase_frame = tk.Frame(parent, bg="#16222E", highlightbackground="#27394D", highlightthickness=1)
        phrase_frame.pack(side="top", fill="x", padx=12, pady=(2, 4))

        tk.Label(phrase_frame, text="🔗 Phrase:", font=("Segoe UI", 9, "bold"),
                 fg=self.C_ACCENT_CYAN, bg="#16222E").pack(side="left", padx=(8, 4), pady=4)
        self.lbl_phrase_text = tk.Label(
            phrase_frame,
            text="(Sign continuously to build a phrase...)",
            font=("Segoe UI", 11, "bold"),
            fg=self.C_MUTED,
            bg="#16222E",
            anchor="w",
        )
        self.lbl_phrase_text.pack(side="left", fill="x", expand=True, padx=4)

        tk.Button(
            phrase_frame, text="✖", font=("Segoe UI", 8, "bold"),
            bg="#22303D", fg="#FF8A80", relief="flat", padx=6, pady=2,
            cursor="hand2", command=self.clear_phrase
        ).pack(side="right", padx=(2, 6), pady=3)
        tk.Button(
            phrase_frame, text="🔊 Speak", font=("Segoe UI", 8, "bold"),
            bg="#1D3549", fg="#00E5FF", relief="flat", padx=8, pady=2,
            cursor="hand2", command=self.speak_phrase
        ).pack(side="right", padx=2, pady=3)

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

        self.lbl_train_status = tk.Label(row_train, text="Click to train RF (Legacy & Kinematic), GRU, 1D-CNN & Hybrid Ensemble",
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
        if hasattr(self, "lbl_phrase_text"):
            self.lbl_phrase_text.config(font=("Segoe UI", min(cfg["log_body"], 14), "bold"))
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
        self._load_model_safe()
        if hasattr(self, "txt_dataset_info"):
            self._refresh_dataset_summary()
        scenario = self.selected_scenario.get()
        n_gestures = len(self.id_to_label)
        ded = "dedicated" if (self.predictor is not None and getattr(self.predictor, "is_scenario_dedicated", False)) else "shared global"
        self._append_to_timeline(
            "⚙️ [System]",
            f"Switched context to [{scenario}] — Loaded {ded} AI model ({n_gestures} gestures).",
            "system",
        )

    def reload_scenario_presets(self):
        """Reloads scenario presets dynamically from data/scenario_presets.json."""
        (
            self.scenario_staff_replies,
            self.scenario_gesture_presets,
            self.scenario_active_gestures,
            self.scenario_hints,
            self.scenario_list,
        ) = load_scenario_data()
        if hasattr(self, "combo_scenario"):
            self.combo_scenario.config(values=self.scenario_list)
            if self.selected_scenario.get() not in self.scenario_list and self.scenario_list:
                self.selected_scenario.set(self.scenario_list[0])
        self._render_scenario_buttons()

    def open_presets_settings(self):
        """Opens interactive Scenario & Gesture Database Manager dialog."""
        cfg_file = getattr(config, "PRESETS_CONFIG_PATH", config.DATA_DIR / "scenario_presets.json")
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    presets_data = json.load(f)
            except Exception as e:
                messagebox.showerror("Preset Error", f"Failed to read {cfg_file}:\n{e}")
                return
        else:
            presets_data = {}

        # Gather all recorded gestures from shared data/raw/ bank
        raw_counts = {}
        if config.DATA_RAW_DIR.exists():
            for d in sorted(config.DATA_RAW_DIR.iterdir()):
                if d.is_dir():
                    raw_counts[d.name] = len(list(d.glob("*.npy")))

        dlg = tk.Toplevel(self)
        dlg.title("Scenario & Gesture Database Manager")
        dlg.geometry("780x560")
        dlg.minsize(700, 500)
        dlg.configure(bg=self.C_BG)
        dlg.transient(self)
        dlg.grab_set()

        # Left Pane: Scenario list + Add/Delete Location
        left_p = tk.Frame(dlg, bg=self.C_CARD, width=215, highlightbackground=self.C_BORDER, highlightthickness=1)
        left_p.pack(side="left", fill="y", padx=(12, 6), pady=12)
        left_p.pack_propagate(False)

        tk.Label(left_p, text="Locations / Scenarios", font=("Segoe UI", 10, "bold"),
                 fg=self.C_ACCENT_CYAN, bg=self.C_CARD).pack(anchor="w", padx=10, pady=(10, 6))

        lb_scenarios = tk.Listbox(
            left_p, bg="#141B24", fg=self.C_TEXT, selectbackground="#00838F",
            selectforeground="#FFFFFF", font=("Segoe UI", 10), relief="flat", exportselection=False
        )
        lb_scenarios.pack(fill="both", expand=True, padx=10, pady=4)

        btn_loc_row = tk.Frame(left_p, bg=self.C_CARD)
        btn_loc_row.pack(fill="x", padx=10, pady=(4, 10))

        # Right Pane: Selected Scenario Editor
        right_p = tk.Frame(dlg, bg=self.C_CARD, highlightbackground=self.C_BORDER, highlightthickness=1)
        right_p.pack(side="right", fill="both", expand=True, padx=(6, 12), pady=12)

        top_meta = tk.Frame(right_p, bg=self.C_CARD)
        top_meta.pack(fill="x", padx=12, pady=(10, 6))
        lbl_editing = tk.Label(top_meta, text="Editing: --", font=("Segoe UI", 11, "bold"),
                               fg=self.C_ACCENT_GREEN, bg=self.C_CARD)
        lbl_editing.pack(side="left")

        desc_row = tk.Frame(right_p, bg=self.C_CARD)
        desc_row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(desc_row, text="Description:", font=("Segoe UI", 9), fg=self.C_MUTED, bg=self.C_CARD).pack(side="left")
        var_desc = tk.StringVar()
        ent_desc = tk.Entry(desc_row, textvariable=var_desc, font=("Segoe UI", 9),
                            bg=self.C_CARD_LIGHT, fg=self.C_TEXT, insertbackground=self.C_TEXT, relief="flat")
        ent_desc.pack(side="left", fill="x", expand=True, padx=(8, 0), ipady=3)

        # Gesture Database Checklist Section
        gest_sec = tk.LabelFrame(
            right_p, text=" Active Gesture Database for this Location (Check to include in AI model) ",
            font=("Segoe UI", 9, "bold"), fg=self.C_ACCENT_CYAN, bg=self.C_CARD, padx=8, pady=6
        )
        gest_sec.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        add_word_row = tk.Frame(gest_sec, bg=self.C_CARD)
        add_word_row.pack(fill="x", pady=(0, 6))
        var_new_word = tk.StringVar()
        ent_new_word = tk.Entry(add_word_row, textvariable=var_new_word, font=("Segoe UI", 9),
                                bg=self.C_CARD_LIGHT, fg=self.C_TEXT, insertbackground=self.C_TEXT, relief="flat", width=22)
        ent_new_word.pack(side="left", padx=(0, 6), ipady=3)
        tk.Label(add_word_row, text="(e.g. Emergency, Doctor, Pain)", font=("Segoe UI", 8),
                 fg=self.C_MUTED, bg=self.C_CARD).pack(side="right")

        # Scrollable checkbox grid for gestures
        chk_canvas = tk.Canvas(gest_sec, bg="#141B24", highlightthickness=0, height=160)
        chk_scroll = ttk.Scrollbar(gest_sec, orient="vertical", command=chk_canvas.yview)
        chk_inner = tk.Frame(chk_canvas, bg="#141B24")
        chk_inner.bind("<Configure>", lambda e: chk_canvas.configure(scrollregion=chk_canvas.bbox("all")))
        chk_canvas.create_window((0, 0), window=chk_inner, anchor="nw")
        chk_canvas.configure(yscrollcommand=chk_scroll.set)
        chk_canvas.pack(side="left", fill="both", expand=True)
        chk_scroll.pack(side="right", fill="y")

        # Staff Quick Replies Section
        rep_sec = tk.LabelFrame(
            right_p, text=" Staff Quick Replies (One reply per line) ",
            font=("Segoe UI", 9, "bold"), fg=self.C_ACCENT_GREEN, bg=self.C_CARD, padx=8, pady=6
        )
        rep_sec.pack(fill="x", padx=12, pady=(0, 8))

        txt_replies = tk.Text(rep_sec, height=5, bg="#141B24", fg=self.C_TEXT, font=("Segoe UI", 9),
                              insertbackground=self.C_TEXT, relief="flat", padx=6, pady=4)
        txt_replies.pack(fill="x")

        # Bottom Action Buttons
        bot_row = tk.Frame(right_p, bg=self.C_CARD)
        bot_row.pack(fill="x", padx=12, pady=(4, 10))

        state = {"current": None, "check_vars": {}}

        def _commit_current_to_dict():
            cur = state["current"]
            if not cur or cur not in presets_data:
                return
            selected_gests = [w for w, var in state["check_vars"].items() if var.get()]
            replies = [ln.strip() for ln in txt_replies.get("1.0", tk.END).splitlines() if ln.strip()]
            presets_data[cur]["description"] = var_desc.get().strip() or cur
            presets_data[cur]["active_gestures"] = selected_gests
            presets_data[cur]["recommended_signs"] = selected_gests[:10]
            presets_data[cur]["staff_replies"] = replies

        def _render_checkboxes(sc_name):
            for w in chk_inner.winfo_children():
                w.destroy()
            state["check_vars"].clear()
            sc_cfg = presets_data.get(sc_name, {})
            active_list = list(sc_cfg.get("active_gestures", sc_cfg.get("recommended_signs", [])))
            active_lower = {g.lower() for g in active_list}

            # Merge shared bank gestures + scenario-specific custom words
            all_words = list(raw_counts.keys())
            existing_lower = {w.lower() for w in all_words}
            for g in active_list:
                if g.lower() not in existing_lower:
                    all_words.append(g)
                    existing_lower.add(g.lower())
            all_words.sort(key=lambda s: s.lower())

            for idx, word in enumerate(all_words):
                var = tk.BooleanVar(value=(word.lower() in active_lower))
                state["check_vars"][word] = var
                cnt = raw_counts.get(word, 0)
                badge = f"({cnt} samples)" if cnt > 0 else "(Not recorded yet)"
                fg_c = "#80D8FF" if cnt > 0 else "#FFB74D"
                cb = tk.Checkbutton(
                    chk_inner,
                    text=f"{word} {badge}",
                    variable=var,
                    bg="#141B24",
                    fg=fg_c,
                    selectcolor="#1E293B",
                    activebackground="#141B24",
                    activeforeground="#FFFFFF",
                    font=("Segoe UI", 9),
                    anchor="w",
                )
                cb.grid(row=idx // 2, column=idx % 2, sticky="w", padx=8, pady=2)

        def _load_scenario_into_editor(sc_name):
            _commit_current_to_dict()
            state["current"] = sc_name
            sc_cfg = presets_data.get(sc_name, {})
            lbl_editing.config(text=f"Editing: {sc_name}")
            var_desc.set(sc_cfg.get("description", ""))
            txt_replies.delete("1.0", tk.END)
            txt_replies.insert("1.0", "\n".join(sc_cfg.get("staff_replies", [])))
            _render_checkboxes(sc_name)

        def _on_lb_select(evt=None):
            sel = lb_scenarios.curselection()
            if not sel:
                return
            name = lb_scenarios.get(sel[0])
            if name != state["current"]:
                _load_scenario_into_editor(name)

        def _add_custom_word():
            cur = state["current"]
            raw_w = re.sub(r"\s+", "_", var_new_word.get().strip())
            if not cur or not raw_w:
                return
            _commit_current_to_dict()
            act = presets_data[cur].setdefault("active_gestures", [])
            if raw_w.lower() not in {g.lower() for g in act}:
                act.append(raw_w)
            var_new_word.set("")
            _render_checkboxes(cur)

        def _add_location():
            new_name = simpledialog.askstring("New Location Preset", "Enter location / scenario name (e.g. Emergency Room, Bank):", parent=dlg)
            if not new_name or not new_name.strip():
                return
            new_name = new_name.strip()
            if new_name in presets_data:
                messagebox.showwarning("Exists", f"Scenario '{new_name}' already exists.", parent=dlg)
                return
            _commit_current_to_dict()
            presets_data[new_name] = {
                "icon": "📍",
                "tag_color": "#00E676",
                "description": f"Custom Location: {new_name}",
                "staff_replies": ["Please wait a moment", "How can I help you?", "Thank you"],
                "active_gestures": list(raw_counts.keys()),
                "recommended_signs": list(raw_counts.keys())[:8],
            }
            lb_scenarios.insert(tk.END, new_name)
            lb_scenarios.selection_clear(0, tk.END)
            lb_scenarios.selection_set(tk.END)
            _load_scenario_into_editor(new_name)

        def _delete_location():
            if len(presets_data) <= 1:
                messagebox.showwarning("Cannot Delete", "At least one scenario preset must remain.", parent=dlg)
                return
            cur = state["current"]
            if cur and messagebox.askyesno("Delete Scenario", f"Delete preset '{cur}'?", parent=dlg):
                del presets_data[cur]
                state["current"] = None
                lb_scenarios.delete(0, tk.END)
                for k in presets_data.keys():
                    lb_scenarios.insert(tk.END, k)
                lb_scenarios.selection_set(0)
                _load_scenario_into_editor(lb_scenarios.get(0))

        def _save_to_disk(retrain_after=False):
            _commit_current_to_dict()
            try:
                cfg_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cfg_file, "w", encoding="utf-8") as f:
                    json.dump(presets_data, f, indent=2, ensure_ascii=False)
                active_sc = state["current"] or self.selected_scenario.get()
                self.reload_scenario_presets()
                if active_sc in self.scenario_list:
                    self.selected_scenario.set(active_sc)
                    self._render_scenario_buttons()
                self._load_model_safe()
                if hasattr(self, "txt_dataset_info"):
                    self._refresh_dataset_summary()
                dlg.destroy()
                if retrain_after:
                    self.retrain_models_async()
                else:
                    self._append_to_timeline("⚙️ [System]", f"Saved preset database for [{active_sc}].", "system")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save presets:\n{e}", parent=dlg)

        def _open_raw_json():
            if cfg_file.exists():
                try:
                    os.startfile(str(cfg_file))
                except Exception:
                    subprocess.run(["notepad.exe", str(cfg_file)])

        tk.Button(add_word_row, text="+ Add Word", font=("Segoe UI", 8, "bold"),
                  bg="#00838F", fg="#FFFFFF", relief="flat", padx=8, pady=2,
                  cursor="hand2", command=_add_custom_word).pack(side="left")

        tk.Button(btn_loc_row, text="+ New", font=("Segoe UI", 8, "bold"),
                  bg=self.C_ACCENT_GREEN, fg="#121212", relief="flat", padx=8, pady=3,
                  cursor="hand2", command=_add_location).pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(btn_loc_row, text="🗑 Delete", font=("Segoe UI", 8, "bold"),
                  bg="#3B1F24", fg="#FF8A80", relief="flat", padx=8, pady=3,
                  cursor="hand2", command=_delete_location).pack(side="right")

        tk.Button(bot_row, text="📄 Raw JSON", font=("Segoe UI", 8),
                  bg=self.C_CARD_LIGHT, fg=self.C_MUTED, relief="flat", padx=8, pady=4,
                  cursor="hand2", command=_open_raw_json).pack(side="left")
        tk.Button(bot_row, text="⚡ Save & Train Scenario", font=("Segoe UI", 9, "bold"),
                  bg=self.C_ACCENT_GREEN, fg="#121212", relief="flat", padx=12, pady=4,
                  cursor="hand2", command=lambda: _save_to_disk(retrain_after=True)).pack(side="right", padx=(6, 0))
        tk.Button(bot_row, text="💾 Save Only", font=("Segoe UI", 9, "bold"),
                  bg="#00838F", fg="#FFFFFF", relief="flat", padx=12, pady=4,
                  cursor="hand2", command=lambda: _save_to_disk(retrain_after=False)).pack(side="right")

        lb_scenarios.bind("<<ListboxSelect>>", _on_lb_select)
        names = list(presets_data.keys())
        for n in names:
            lb_scenarios.insert(tk.END, n)
        init_sc = self.selected_scenario.get()
        init_idx = names.index(init_sc) if init_sc in names else 0
        if names:
            lb_scenarios.selection_set(init_idx)
            _load_scenario_into_editor(names[init_idx])

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

    def _select_engine_locked(self, preferred_type):
        """Selects active engine (with safe fallback) assuming self.model_lock is held."""
        if self.predictor is None:
            return
        avail = self.predictor.available_engines()
        if not avail:
            self.lbl_model_status.config(text="● Untrained (Run training)", fg="#FF9100")
            return

        if preferred_type in avail:
            chosen = preferred_type
        else:
            for cand in ("hybrid", "gru", "cnn1d", "rf_kinematic", "rf_legacy"):
                if cand in avail:
                    chosen = cand
                    break

        self.model_type = chosen
        self.active_model_name.set(ENGINE_KEY_TO_NAME.get(chosen, "Hybrid (Auto-Specialist)"))
        short = ENGINE_SHORT_NAMES.get(chosen, chosen)
        self.lbl_model_status.config(text=f"● {short} ({len(self.id_to_label)} gestures)", fg=self.C_ACCENT_GREEN)
        if hasattr(self, "lbl_hero_engine"):
            self.lbl_hero_engine.config(text=f"Engine: {self.active_model_name.get()}")

    def _on_model_switch(self, event=None):
        name = self.active_model_name.get()
        target_type = ENGINE_NAME_TO_KEY.get(name, "hybrid")
        with self.model_lock:
            if self.predictor is not None:
                self._select_engine_locked(target_type)
                return
        self._load_model_safe(target_type)

    def _load_model_safe(self, preferred_type=None):
        """Loads scenario-specific trained models (with automatic fallback to global models/) under lock."""
        with self.model_lock:
            try:
                scenario = self.selected_scenario.get()
                scenario_dir = config.get_scenario_model_dir(scenario)
                self.predictor = HybridEnsemblePredictor.load_from_disk(model_dir=scenario_dir, fallback_global=True)

                if self.predictor is None or not self.predictor.available_engines():
                    self.predictor = None
                    self.id_to_label = {}
                    self.lbl_model_status.config(text="● Untrained (Run training)", fg="#FF9100")
                    return

                self.id_to_label = dict(self.predictor.id_to_label)
                target = preferred_type if preferred_type is not None else self.model_type
                self._select_engine_locked(target)

            except Exception as e:
                self.predictor = None
                self.lbl_model_status.config(text=f"● Load error: {e}", fg=self.C_ACCENT_RED)

    def _predict(self, raw_frames):
        """Predicts gesture label, confidence, and matched engine from raw (T, 112) frames."""
        with self.model_lock:
            if self.predictor is None:
                return "Untrained Model", 0.0, ""
            return self.predictor.predict(raw_frames, engine=self.model_type)

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
        self.still_start_time = None
        self.rec_peak_speed = 0.0
        self.hand_dropped_frames = 0
        self.retrigger_move_frames = 0
        self.prev_wrist_xy = None
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
            line_ratio = self.line_ratio_val
            ready_y = int(line_ratio * h)
            sh_w = 0.35
            if pose_lms is not None:
                try:
                    l_sh = pose_lms[config.POSE_KEEP["l_shoulder"]]
                    r_sh = pose_lms[config.POSE_KEEP["r_shoulder"]]
                    sh_w = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
                    # Lowered ready line (down toward rib/waist level ~0.92 * sh_w)
                    base_mult = getattr(config, "READY_LINE_SHOULDER_MULT", 0.92)
                    multiplier = base_mult * (line_ratio / 0.74)
                    chest_y = (l_sh.y + r_sh.y) / 2.0 + multiplier * max(sh_w, 0.15)
                    ready_y = int(np.clip(chest_y, 0.40, 0.90) * h)
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

            # Wrist velocity tracking for dynamic early-stop & continuous segmentation
            wrist_vec = frame_vec[10:14]
            if self.prev_wrist_xy is not None:
                wrist_speed = float(np.linalg.norm(wrist_vec - self.prev_wrist_xy))
            else:
                wrist_speed = 0.0
            self.prev_wrist_xy = wrist_vec.copy()
            self.wrist_speed = wrist_speed

            # State Transitions (Thread-Safe check of variables)
            mode = self.trigger_mode_val
            is_translation_tab = (self.active_tab_idx == 0)

            if self.state == "idle":
                self.pre_roll_buffer.append(frame_vec)
                # Auto-trigger arms only in Tab 1 (Translation) and when not in Studio mode
                if mode == "auto" and hand_in_zone and is_translation_tab and not self.studio_recording:
                    if self.arm_start_time is None:
                        self.arm_start_time = now
                        self._update_badge("ARMING...", self.C_ACCENT_YELLOW)
                    elif (now - self.arm_start_time) >= 0.25:
                        self.state = "recording"
                        self.rec_start_time = now
                        self.still_start_time = None
                        self.rec_peak_speed = wrist_speed
                        self.hand_dropped_frames = 0
                        # Include pre-roll frames captured during arming so fast signs never lose onset
                        self.current_recording = list(self.pre_roll_buffer)
                        self._update_badge("RECORDING", self.C_ACCENT_RED)
                else:
                    self.arm_start_time = None
                    self._update_badge("READY", self.C_ACCENT_GREEN)

            elif self.state == "countdown":
                # During studio countdown, auto-trigger is suppressed
                self.arm_start_time = None
                self.pre_roll_buffer.clear()

            elif self.state == "recording":
                self.current_recording.append(frame_vec)
                elapsed = now - self.rec_start_time
                self.rec_peak_speed = max(self.rec_peak_speed, wrist_speed)

                if wrist_speed < config.WRIST_STILL_THRESHOLD:
                    if self.still_start_time is None:
                        self.still_start_time = now
                else:
                    self.still_start_time = None

                self.hand_dropped_frames = (self.hand_dropped_frames + 1) if not hand_in_zone else 0

                if self.studio_recording:
                    done = elapsed >= config.MAX_SIGN_DURATION
                else:
                    can_early = (elapsed >= config.MIN_SIGN_DURATION) and (len(self.current_recording) >= 7)
                    has_moved = self.rec_peak_speed >= config.WRIST_ACTIVE_PEAK_THRESHOLD
                    still_done = has_moved and (self.still_start_time is not None) and ((now - self.still_start_time) >= config.END_HOLD_STILL_SEC)
                    drop_done = self.hand_dropped_frames >= 4
                    done = (elapsed >= config.MAX_SIGN_DURATION) or (can_early and (drop_done or still_done))

                if done:
                    # Recording finished
                    self.state = "result"
                    self.result_start_time = now
                    self.retrigger_move_frames = 0
                    self.pre_roll_buffer.clear()
                    self._update_badge("PROCESSING", self.C_ACCENT_CYAN)

                    # Check if this was a Studio sample recording
                    if self.studio_recording:
                        rec_frames = list(self.current_recording)
                        self.studio_recording = False
                        self.after(0, self._save_studio_sample, rec_frames)
                    else:
                        if len(self.current_recording) >= 7:
                            raw_seq = np.array(self.current_recording, dtype=np.float32)
                            label, conf, detail = self._predict(raw_seq)
                            self.after(0, self._on_prediction_ready, label, conf, detail)
                        else:
                            self.after(0, self._on_prediction_ready, "(No motion)", 0.0, "")

            elif self.state == "result":
                # Anti-loop cooldown + Instant Mid-Air Re-Trigger for continuous fast signing
                self.pre_roll_buffer.append(frame_vec)
                cd = now - self.result_start_time
                if mode == "auto":
                    if not hand_in_zone and cd >= 0.35:
                        self.state = "idle"
                        self.arm_start_time = None
                    elif hand_in_zone and cd >= 0.45:
                        if wrist_speed >= 0.055:
                            self.retrigger_move_frames += 1
                        else:
                            self.retrigger_move_frames = 0
                        if self.retrigger_move_frames >= 2:
                            # Directly transition to recording next sign mid-air (skipping 0.25s arming!)
                            self.state = "recording"
                            self.rec_start_time = now
                            self.still_start_time = None
                            self.rec_peak_speed = wrist_speed
                            self.hand_dropped_frames = 0
                            self.retrigger_move_frames = 0
                            self.current_recording = list(self.pre_roll_buffer)[-4:]
                            self._update_badge("RECORDING", self.C_ACCENT_RED)
                        elif cd >= 1.2:
                            self._update_badge("LOWER HAND / NEXT", self.C_MUTED)
                else:
                    if cd >= 0.40:
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
            cv2.putText(view, "READY LINE (Raise hand above line to start)", (w - 315, ready_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, line_color, 1, cv2.LINE_AA)

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
    def _on_prediction_ready(self, label, conf, engine_detail=""):
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
        if hasattr(self, "lbl_hero_engine"):
            badge_txt = f"Matched via: {engine_detail}" if engine_detail else f"Engine: {self.active_model_name.get()}"
            self.lbl_hero_engine.config(text=badge_txt)

        # Draw Confidence Bar
        self.canvas_conf_bar.delete("all")
        bar_w = self.canvas_conf_bar.winfo_width()
        fill_w = int(bar_w * np.clip(conf, 0.0, 1.0))
        self.canvas_conf_bar.create_rectangle(0, 0, fill_w, 6, fill=bar_color, width=0)

        # Append to Continuous Phrase Bar, Conversation Timeline & Auto-speak if confident
        if is_confident:
            self.phrase_words.append(clean_label)
            self.phrase_words = self.phrase_words[-8:]
            if hasattr(self, "lbl_phrase_text"):
                self.lbl_phrase_text.config(text=" ➔ ".join(self.phrase_words), fg=self.C_TEXT)

            detail_tag = f"  [{engine_detail}]" if engine_detail else ""
            self._append_to_timeline("🤟 [Signer]", f"{clean_label} ({int(conf * 100)}%){detail_tag}", "signer")
            if self.auto_speak.get():
                speak_text_async(clean_label)

    def speak_phrase(self):
        """Speaks the entire accumulated phrase sequentially."""
        if self.phrase_words:
            speak_text_async(" ".join(self.phrase_words))

    def clear_phrase(self):
        """Clears the accumulated continuous phrase bar."""
        self.phrase_words = []
        if hasattr(self, "lbl_phrase_text"):
            self.lbl_phrase_text.config(text="(Sign continuously to build a phrase...)", fg=self.C_MUTED)

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
            self.still_start_time = None
            self.hand_dropped_frames = 0
            self.prev_wrist_xy = None
            self.current_recording = []
            self._update_badge("STUDIO REC", self.C_ACCENT_RED)

    def _save_studio_sample(self, frames):
        """Saves recorded frames directly into data/raw/<word>/ on main thread."""
        raw_word = re.sub(r"\s+", "_", self.studio_target_word.get().strip())
        person = self.studio_person.get()

        # Match existing folder casing if already present in data/raw/
        word = raw_word
        if config.DATA_RAW_DIR.exists():
            for d in config.DATA_RAW_DIR.iterdir():
                if d.is_dir() and d.name.lower() == raw_word.lower():
                    word = d.name
                    break

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
        """Refreshes the sample count summary in Tab 2, highlighting active scenario gestures."""
        raw_dir = config.DATA_RAW_DIR
        self.txt_dataset_info.delete("1.0", tk.END)

        scenario = self.selected_scenario.get()
        active_list = self.scenario_active_gestures.get(scenario, [])
        active_lower = {g.lower() for g in active_list}

        if not raw_dir.exists():
            self.txt_dataset_info.insert(tk.END, "No 'data/raw' folder found.\n")
            return

        class_folders = sorted([d for d in raw_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
        existing_lower = {d.name.lower() for d in class_folders}

        total_samples = 0
        self.txt_dataset_info.insert(tk.END, f"Active Scenario: [{scenario}]\n")
        self.txt_dataset_info.insert(tk.END, f"{'GESTURE CLASS':<18} | {'SAMPLES':<7} | {'SCENARIO SCOPE'}\n")
        self.txt_dataset_info.insert(tk.END, "-" * 54 + "\n")

        for f in class_folders:
            cnt = len(list(f.glob("*.npy")))
            total_samples += cnt
            in_sc = (not active_lower) or (f.name.lower() in active_lower)
            scope = "★ Active" if in_sc else "Shared Bank"
            ready = "Ready" if cnt >= 5 else "Need ≥5"
            self.txt_dataset_info.insert(tk.END, f"{f.name:<18} | {cnt:<7} | {scope} ({ready})\n")

        # Also list gestures planned in this scenario that have 0 samples so far
        missing_in_sc = [g for g in active_list if g.lower() not in existing_lower]
        for g in missing_in_sc:
            self.txt_dataset_info.insert(tk.END, f"{g:<18} | {0:<7} | ★ Active (Record / Import video)\n")

        self.txt_dataset_info.insert(tk.END, "-" * 54 + "\n")
        self.txt_dataset_info.insert(tk.END, f"Recorded Classes: {len(class_folders)} | Total Samples: {total_samples}\n")

    def retrain_models_async(self):
        """Retrains global + active scenario dedicated models in background thread."""
        scenario = self.selected_scenario.get()
        self.btn_train_models.config(state="disabled")
        self.lbl_train_status.config(
            text=f"Training Global + [{scenario}] Hybrid models... (~6-10s)",
            fg=self.C_ACCENT_YELLOW,
        )

        def _worker():
            try:
                global_res = train_all_models(verbose=False)
                sc_res = None
                try:
                    sc_res = train_scenario_profile(scenario, verbose=False)
                except Exception:
                    sc_res = global_res

                c = sc_res["clean"]
                r = sc_res["robust"]
                summary_msg = (
                    f"[{sc_res['scenario']}] Hybrid: {c['hybrid']*100:.0f}% (Rob {r['hybrid']*100:.0f}%) | "
                    f"1D-CNN: {c['cnn1d']*100:.0f}% | GRU: {c['gru']*100:.0f}% | "
                    f"RF-Kin: {c['rf_kinematic']*100:.0f}% ({sc_res['num_classes']} gestures)"
                )
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
