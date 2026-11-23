"""
PyTorch Neural Network Models & Hybrid Specialist Ensemble for Sign Language Recognition.
โครงสร้างโมเดล Deep Learning (GRU, 1D-CNN) และระบบ Hybrid Specialist Ensemble
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from features import (
    add_velocity_channels,
    canonicalize_handedness,
    compute_motion_energy,
    extract_kinematic_features,
    resample_sequence,
    trim_active_motion,
)


_NNBase = nn.Module if nn is not None else object


class SignGRU(_NNBase):
    """Recurrent GRU sequence classifier for trajectory-dominant gestures."""

    def __init__(self, input_size=config.NUM_SEQ_FEATURES, hidden_size=64, num_layers=2, num_classes=2, dropout=0.25):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len=24, input_size)
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.classifier(last_out)


class SignCNN1D(_NNBase):
    """Multi-scale 1D-CNN (TCN) classifier for fast flicks & local temporal patterns."""

    def __init__(self, input_size=config.NUM_SEQ_FEATURES, num_classes=2, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.branch_dil = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        h = self.input_proj(x)
        h = torch.cat([self.branch3(h), self.branch5(h), self.branch_dil(h)], dim=1)
        h = self.conv2(self.drop1(h))
        avg_p = h.mean(dim=-1)
        max_p = h.max(dim=-1).values
        feat = torch.cat([avg_p, max_p], dim=-1)
        return self.classifier(feat)


class HybridEnsemblePredictor:
    """
    Loads all available sub-models (RF Kinematic, RF Legacy, PyTorch GRU, PyTorch 1D-CNN)
    and supports single-engine inference as well as Hybrid Specialist routing.
    """

    PRETTY_SPECIALIST = {
        "rf_kinematic": "Kinematic-RF",
        "gru": "GRU",
        "cnn1d": "1D-CNN",
    }

    def __init__(
        self,
        label_to_id,
        rf_kinematic=None,
        rf_legacy=None,
        gru=None,
        gru_input_size=config.NUM_SEQ_FEATURES,
        cnn1d=None,
        cnn_input_size=config.NUM_SEQ_FEATURES,
        meta=None,
    ):
        self.label_to_id = dict(label_to_id or {})
        self.id_to_label = {int(v): str(k) for k, v in self.label_to_id.items()}
        self.num_classes = len(self.id_to_label)

        self.rf_kinematic = rf_kinematic
        self.rf_legacy = rf_legacy
        self.gru = gru
        self.gru_input_size = gru_input_size
        self.cnn1d = cnn1d
        self.cnn_input_size = cnn_input_size
        self.meta = meta or {}

    @classmethod
    def load_from_disk(cls, model_dir=None, fallback_global=True):
        """
        Loads trained models and ensemble metadata safely from model_dir (defaults to config.MODEL_DIR).
        If model_dir is a scenario directory without a trained labels.json and fallback_global=True,
        gracefully falls back to global config.MODEL_DIR.
        """
        target_dir = Path(model_dir) if model_dir is not None else config.MODEL_DIR
        labels_path = target_dir / "labels.json"
        if not labels_path.exists():
            if fallback_global and target_dir.resolve() != config.MODEL_DIR.resolve():
                target_dir = config.MODEL_DIR
                labels_path = target_dir / "labels.json"
            if not labels_path.exists():
                return None

        with open(labels_path, "r", encoding="utf-8") as f:
            label_to_id = json.load(f)

        meta = {}
        meta_path = target_dir / "ensemble_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load ensemble meta: {e}")

        rf_legacy = None
        rf_leg_path = target_dir / "rf.pkl"
        if joblib is not None and rf_leg_path.exists():
            try:
                rf_legacy = joblib.load(rf_leg_path)
            except Exception as e:
                print(f"[Warning] Failed to load RF Legacy: {e}")

        rf_kinematic = None
        kin_path = target_dir / "rf_kinematic.pkl"
        if joblib is not None and kin_path.exists():
            try:
                rf_kinematic = joblib.load(kin_path)
            except Exception as e:
                print(f"[Warning] Failed to load RF Kinematic: {e}")

        gru, gru_in = None, config.NUM_SEQ_FEATURES
        gru_path = target_dir / "gru.pt"
        if torch is not None and gru_path.exists():
            try:
                ckpt = torch.load(gru_path, map_location="cpu", weights_only=True)
                gru_in = int(ckpt["input_size"])
                gru = SignGRU(
                    input_size=gru_in,
                    hidden_size=int(ckpt["hidden_size"]),
                    num_layers=int(ckpt["num_layers"]),
                    num_classes=int(ckpt["num_classes"]),
                )
                gru.load_state_dict(ckpt["model_state"])
                gru.eval()
            except Exception as e:
                print(f"[Warning] Failed to load GRU: {e}")
                gru = None

        cnn1d, cnn_in = None, config.NUM_SEQ_FEATURES
        cnn_path = target_dir / "cnn1d.pt"
        if torch is not None and cnn_path.exists():
            try:
                ckpt = torch.load(cnn_path, map_location="cpu", weights_only=True)
                cnn_in = int(ckpt["input_size"])
                cnn1d = SignCNN1D(
                    input_size=cnn_in,
                    num_classes=int(ckpt["num_classes"]),
                )
                cnn1d.load_state_dict(ckpt["model_state"])
                cnn1d.eval()
            except Exception as e:
                print(f"[Warning] Failed to load 1D-CNN: {e}")
                cnn1d = None

        inst = cls(
            label_to_id=label_to_id,
            rf_kinematic=rf_kinematic,
            rf_legacy=rf_legacy,
            gru=gru,
            gru_input_size=gru_in,
            cnn1d=cnn1d,
            cnn_input_size=cnn_in,
            meta=meta,
        )
        inst.model_dir = target_dir
        inst.is_scenario_dedicated = target_dir.resolve() != config.MODEL_DIR.resolve()
        return inst

    def available_engines(self):
        """Returns set of available engine keys loaded on disk."""
        avail = set()
        if self.rf_kinematic is not None:
            avail.add("rf_kinematic")
        if self.rf_legacy is not None:
            avail.add("rf_legacy")
        if self.gru is not None:
            avail.add("gru")
        if self.cnn1d is not None:
            avail.add("cnn1d")
        if avail & {"rf_kinematic", "gru", "cnn1d"}:
            avail.add("hybrid")
        return avail

    def has_engine(self, engine):
        return engine in self.available_engines()

    def _align_sklearn_probs(self, clf, feat_2d):
        """Aligns sklearn class probabilities strictly to self.label_to_id indexing."""
        aligned = np.zeros(self.num_classes, dtype=np.float32)
        if clf is None or self.num_classes == 0:
            return aligned
        probs = clf.predict_proba(feat_2d)[0]
        for cls_name, p in zip(clf.classes_, probs):
            idx = self.label_to_id.get(str(cls_name))
            if idx is not None and 0 <= idx < self.num_classes:
                aligned[idx] = float(p)
        return aligned

    def _predict_torch_probs(self, model, input_size, seq_24):
        """Runs PyTorch classifier on a (24, 112) sequence, adding velocity channels if expected."""
        if model is None or torch is None or self.num_classes == 0:
            return np.zeros(self.num_classes, dtype=np.float32)
        arr = add_velocity_channels(seq_24) if input_size == config.NUM_SEQ_FEATURES else np.asarray(seq_24, dtype=np.float32)
        with torch.no_grad():
            tensor_in = torch.tensor(arr[None, ...], dtype=torch.float32)
            logits = model(tensor_in)
            return torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)

    def _submodel_probs(self, seq_24):
        """Returns dict of aligned probability vectors across available modern sub-models."""
        probs = {}
        if self.rf_kinematic is not None:
            kin = extract_kinematic_features(seq_24).reshape(1, -1)
            probs["rf_kinematic"] = self._align_sklearn_probs(self.rf_kinematic, kin)
        if self.gru is not None:
            probs["gru"] = self._predict_torch_probs(self.gru, self.gru_input_size, seq_24)
        if self.cnn1d is not None:
            probs["cnn1d"] = self._predict_torch_probs(self.cnn1d, self.cnn_input_size, seq_24)
        return probs

    def _predict_hybrid_from_resampled(self, seq_24):
        """Computes Two-Stage Specialist Hybrid prediction on an already (24, 112) sequence."""
        probs_map = self._submodel_probs(seq_24)
        if not probs_map:
            return "Untrained Model", 0.0, "No Modern Models"

        available = list(probs_map.keys())
        motion_energy = compute_motion_energy(seq_24)
        motion_median = float(self.meta.get("motion_median", 0.15)) or 0.15
        per_class_weights = self.meta.get("per_class_weights", {})

        p_hybrid = np.zeros(self.num_classes, dtype=np.float32)
        class_weights_used = {}

        for idx in range(self.num_classes):
            lbl = self.id_to_label[idx]
            base_w = per_class_weights.get(lbl, {})
            raw_w = {}
            for m in available:
                w = float(base_w.get(m, 1.0 / len(available)))
                if motion_energy < 0.85 * motion_median and m == "rf_kinematic":
                    w *= 1.20
                elif motion_energy > 1.15 * motion_median and m in ("gru", "cnn1d"):
                    w *= 1.15
                raw_w[m] = max(w, 1e-4)

            total_w = sum(raw_w.values())
            norm_w = {m: v / total_w for m, v in raw_w.items()}
            class_weights_used[idx] = norm_w
            p_hybrid[idx] = sum(norm_w[m] * float(probs_map[m][idx]) for m in available)

        best_idx = int(np.argmax(p_hybrid))
        best_label = self.id_to_label.get(best_idx, f"Class {best_idx}")
        best_conf = float(p_hybrid[best_idx])

        w_best = class_weights_used[best_idx]
        winner = max(available, key=lambda m: w_best[m] * float(probs_map[m][best_idx]))
        pretty_winner = self.PRETTY_SPECIALIST.get(winner, winner)
        return best_label, best_conf, f"Hybrid [{pretty_winner} Specialist]"

    def predict_resampled(self, seq_24, engine="hybrid"):
        """Predicts using an already-resampled (24, 112) matrix (no trimming performed)."""
        seq_24 = np.asarray(seq_24, dtype=np.float32)

        if engine == "hybrid":
            return self._predict_hybrid_from_resampled(seq_24)

        if engine == "gru":
            if self.gru is None:
                return "GRU Not Ready", 0.0, "PyTorch GRU (Missing)"
            probs = self._predict_torch_probs(self.gru, self.gru_input_size, seq_24)
            idx = int(np.argmax(probs))
            return self.id_to_label.get(idx, f"Class {idx}"), float(probs[idx]), "PyTorch GRU"

        if engine == "cnn1d":
            if self.cnn1d is None:
                return "1D-CNN Not Ready", 0.0, "PyTorch 1D-CNN (Missing)"
            probs = self._predict_torch_probs(self.cnn1d, self.cnn_input_size, seq_24)
            idx = int(np.argmax(probs))
            return self.id_to_label.get(idx, f"Class {idx}"), float(probs[idx]), "PyTorch 1D-CNN"

        if engine == "rf_kinematic":
            if self.rf_kinematic is None:
                return "RF-Kin Not Ready", 0.0, "RF (Kinematic) Missing"
            kin = extract_kinematic_features(seq_24).reshape(1, -1)
            probs = self._align_sklearn_probs(self.rf_kinematic, kin)
            idx = int(np.argmax(probs))
            return self.id_to_label.get(idx, f"Class {idx}"), float(probs[idx]), "RF (Kinematic)"

        if engine == "rf_legacy":
            if self.rf_legacy is None:
                return "RF-Legacy Not Ready", 0.0, "RF (Legacy) Missing"
            flat = seq_24.reshape(1, -1)
            probs = self._align_sklearn_probs(self.rf_legacy, flat)
            idx = int(np.argmax(probs))
            return self.id_to_label.get(idx, f"Class {idx}"), float(probs[idx]), "RF (Legacy Flat)"

        return "Unknown Engine", 0.0, str(engine)

    def predict(self, raw_frames, engine="hybrid"):
        """
        End-to-end prediction from raw recorded (T, 112) frames.
        Applies Active Motion Trimming for all modern engines, while keeping
        Legacy Flat RF un-trimmed to preserve baseline behavior.
        Returns: (label, confidence, engine_detail_str)
        """
        frames = np.asarray(raw_frames, dtype=np.float32)
        if frames.ndim != 2 or len(frames) == 0:
            return "(No motion)", 0.0, engine

        if engine == "rf_legacy":
            seq_24 = resample_sequence(frames, target_len=config.TARGET_FRAMES)
        else:
            trimmed = canonicalize_handedness(trim_active_motion(frames))
            seq_24 = resample_sequence(trimmed, target_len=config.TARGET_FRAMES)

        return self.predict_resampled(seq_24, engine=engine)
