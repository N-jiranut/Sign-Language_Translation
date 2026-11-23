"""
Unified Multi-Model & Hybrid Specialist Ensemble Training Pipeline.
เทรนโมเดลครบทั้ง 4 ตัว (RF Legacy, RF Kinematic, PyTorch GRU, PyTorch 1D-CNN)
พร้อมสร้างตารางความเชี่ยวชาญรายท่า (Per-Class Specialist Routing) สำหรับโหมด Hybrid
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from dataset import augment_dataset, augment_sequence_rich, load_raw_dataset, save_label_mapping, split_indices
from features import add_velocity_batch, compute_motion_energy, extract_kinematic_batch
from models import HybridEnsemblePredictor, SignCNN1D, SignGRU


def _train_torch_classifier(model, X_tr_vel, y_tr_enc, X_te_vel, y_te_enc, epochs=50, lr=1e-3, weight_decay=2e-4, verbose=True, name="Model"):
    """Shared PyTorch training loop with gradient clipping and (acc, -loss) checkpointing."""
    train_ds = TensorDataset(torch.tensor(X_tr_vel, dtype=torch.float32), torch.tensor(y_tr_enc, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_te_vel, dtype=torch.float32), torch.tensor(y_te_enc, dtype=torch.long))

    batch_size = min(16, max(2, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=min(32, max(2, len(test_ds))), shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = (-1.0, float("-inf"))  # (test_acc, -test_loss)
    best_state = None
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        test_correct, test_total, test_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx)
                test_loss_sum += float(criterion(out, by).item()) * len(by)
                test_correct += int((out.argmax(dim=-1) == by).sum().item())
                test_total += len(by)

        test_acc = (test_correct / test_total) if test_total > 0 else 0.0
        test_loss = (test_loss_sum / test_total) if test_total > 0 else 0.0
        score = (test_acc, -test_loss)

        if score > best_score or best_state is None:
            best_score = score
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"  [{name}] Epoch {epoch:2d}/{epochs} | Test Acc: {test_acc * 100:5.1f}% | Test Loss: {test_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def train_all_models(data_dir=config.DATA_RAW_DIR, model_dir=None, include_classes=None, scenario_name=None, verbose=True):
    """
    Trains RF (Legacy Flat), RF (Kinematic), PyTorch GRU, and PyTorch 1D-CNN on a single
    consistent stratified/group split, computes Per-Class Specialist Routing weights,
    saves all artifacts to model_dir (default: models/), and returns an evaluation summary dict.
    """
    target_dir = Path(model_dir) if model_dir is not None else config.MODEL_DIR
    scope_label = f"Scenario: {scenario_name}" if scenario_name else "Global All-Gestures Bank"

    if verbose:
        print("=" * 64)
        print(f" Hybrid Specialist Training Pipeline [{scope_label}] ")
        print("=" * 64)

    X_trim, y, groups, label_to_id = load_raw_dataset(data_dir, trim_active=True, include_classes=include_classes)
    X_raw, y_raw, _, _ = load_raw_dataset(data_dir, trim_active=False, include_classes=include_classes)

    num_classes = len(label_to_id)
    if num_classes < 2:
        raise ValueError(f"Need at least 2 recorded gesture classes to train [{scope_label}] (found {num_classes}).")

    if verbose:
        print(f"Dataset loaded: {len(X_trim)} samples across {num_classes} classes: {list(label_to_id.keys())}")

    target_dir.mkdir(parents=True, exist_ok=True)
    save_label_mapping(label_to_id, target_dir / "labels.json")

    # Consistent train/test split indices across ALL models
    tr_idx, te_idx = split_indices(y, groups=groups, test_size=0.2, random_state=42, verbose=verbose)
    X_trim_tr, X_trim_te = X_trim[tr_idx], X_trim[te_idx]
    X_raw_tr, X_raw_te = X_raw[tr_idx], X_raw[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    y_enc_tr = np.array([label_to_id[name] for name in y_tr], dtype=np.int64)
    y_enc_te = np.array([label_to_id[name] for name in y_te], dtype=np.int64)

    # ------------------------------------------------------------------
    # 1. Train RF (Legacy Flat) — 2,688 raw un-trimmed features
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1/4] Training Random Forest (Legacy Flat, 2688 dims)...")
    X_leg_tr, y_leg_tr = X_raw_tr, y_tr
    if len(X_leg_tr) < 50:
        X_leg_tr, y_leg_tr = augment_dataset(X_leg_tr, y_leg_tr, num_copies=1, noise_std=0.015, rich=False, seed=42)

    rf_legacy = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_legacy.fit(X_leg_tr.reshape(len(X_leg_tr), -1), y_leg_tr)
    joblib.dump(rf_legacy, target_dir / "rf.pkl")

    # ------------------------------------------------------------------
    # Prepare Speed-Warped / Shifted Augmented Training Set for Modern Models
    # ------------------------------------------------------------------
    X_aug, y_aug = X_trim_tr, y_tr
    if len(X_aug) < 300:
        X_aug, y_aug = augment_dataset(X_aug, y_aug, num_copies=3, noise_std=0.015, rich=True, seed=42)
    y_aug_enc = np.array([label_to_id[name] for name in y_aug], dtype=np.int64)
    if verbose:
        print(f"[Augmentation] Training samples expanded: {len(X_trim_tr)} -> {len(X_aug)} (Time-warp + Shift + Scale)")

    # ------------------------------------------------------------------
    # 2. Train RF (Kinematic) — 1,008 statistical & velocity features
    # ------------------------------------------------------------------
    if verbose:
        print("\n[2/4] Training Random Forest (Kinematic Summary, 1008 dims)...")
    X_kin_tr = extract_kinematic_batch(X_aug)
    rf_kinematic = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_kinematic.fit(X_kin_tr, y_aug)
    joblib.dump(rf_kinematic, target_dir / "rf_kinematic.pkl")

    # ------------------------------------------------------------------
    # Prepare Velocity-Augmented 222-dim Sequences for Deep Learning Models
    # ------------------------------------------------------------------
    X_tr_vel = add_velocity_batch(X_aug)
    X_te_vel = add_velocity_batch(X_trim_te)

    # ------------------------------------------------------------------
    # 3. Train PyTorch GRU (input_size=222)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[3/4] Training PyTorch GRU (222-dim Position + Velocity)...")
    torch.manual_seed(42)
    gru_model = SignGRU(
        input_size=config.NUM_SEQ_FEATURES,
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.25,
    )
    gru_model, _ = _train_torch_classifier(
        gru_model, X_tr_vel, y_aug_enc, X_te_vel, y_enc_te, epochs=50, lr=1e-3, weight_decay=2e-4, verbose=verbose, name="GRU"
    )
    torch.save(
        {
            "model_state": gru_model.state_dict(),
            "label_to_id": label_to_id,
            "input_size": config.NUM_SEQ_FEATURES,
            "hidden_size": 64,
            "num_layers": 2,
            "num_classes": num_classes,
        },
        target_dir / "gru.pt",
    )

    # ------------------------------------------------------------------
    # 4. Train PyTorch 1D-CNN (input_size=222)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[4/4] Training PyTorch 1D-CNN (Multi-Scale Temporal Conv)...")
    torch.manual_seed(42)
    cnn_model = SignCNN1D(
        input_size=config.NUM_SEQ_FEATURES,
        num_classes=num_classes,
        dropout=0.3,
    )
    cnn_model, _ = _train_torch_classifier(
        cnn_model, X_tr_vel, y_aug_enc, X_te_vel, y_enc_te, epochs=50, lr=1.5e-3, weight_decay=2e-4, verbose=verbose, name="1D-CNN"
    )
    torch.save(
        {
            "model_state": cnn_model.state_dict(),
            "label_to_id": label_to_id,
            "input_size": config.NUM_SEQ_FEATURES,
            "num_classes": num_classes,
        },
        target_dir / "cnn1d.pt",
    )

    # ------------------------------------------------------------------
    # 5. Build Per-Class Specialist Routing Metadata (ensemble_meta.json)
    # ------------------------------------------------------------------
    predictor = HybridEnsemblePredictor(
        label_to_id=label_to_id,
        rf_kinematic=rf_kinematic,
        rf_legacy=rf_legacy,
        gru=gru_model,
        gru_input_size=config.NUM_SEQ_FEATURES,
        cnn1d=cnn_model,
        cnn_input_size=config.NUM_SEQ_FEATURES,
        meta={},
    )

    # Build speed/time-shift robustness validation suite from held-out X_trim_te
    rng_val = np.random.default_rng(123)
    rob_trim_list = list(X_trim_te)
    rob_raw_list = list(X_raw_te)
    rob_labels = list(y_te)
    for s_trim, s_raw, lbl in zip(X_trim_te, X_raw_te, y_te):
        for _ in range(2):
            rob_trim_list.append(augment_sequence_rich(s_trim, noise_std=0.015, rng=rng_val))
            rob_raw_list.append(augment_sequence_rich(s_raw, noise_std=0.015, rng=rng_val))
            rob_labels.append(lbl)

    X_rob_trim = np.stack(rob_trim_list, axis=0).astype(np.float32)
    X_rob_raw = np.stack(rob_raw_list, axis=0).astype(np.float32)
    y_rob = np.array(rob_labels)

    # Compute motion energy stats per gesture class on training set
    class_motion = {}
    for lbl in sorted(label_to_id.keys()):
        cls_seqs = X_trim_tr[y_tr == lbl]
        class_motion[lbl] = float(np.mean([compute_motion_energy(s) for s in cls_seqs])) if len(cls_seqs) > 0 else 0.15
    motion_median = float(np.median(list(class_motion.values())))

    # Score modern specialists on each class's robustness suite
    modern_models = ("rf_kinematic", "gru", "cnn1d")
    per_class_weights = {}
    per_class_specialist = {}

    for lbl in sorted(label_to_id.keys()):
        c_idx = label_to_id[lbl]
        mask = y_rob == lbl
        cls_rob = X_rob_trim[mask]
        sub_probs_list = [predictor._submodel_probs(s) for s in cls_rob]

        scores = {}
        is_low_motion = class_motion[lbl] <= motion_median
        for m in modern_models:
            if len(sub_probs_list) > 0:
                preds_m = np.array([int(np.argmax(sp[m])) for sp in sub_probs_list])
                true_probs_m = np.array([float(sp[m][c_idx]) for sp in sub_probs_list])
                acc_mc = float(np.mean(preds_m == c_idx))
                prob_mc = float(np.mean(true_probs_m))
                raw_score = 0.5 * acc_mc + 0.5 * prob_mc
            else:
                raw_score = 0.5

            # Apply domain prior (Static/Posture vs Dynamic/Trajectory)
            if is_low_motion:
                if m == "rf_kinematic":
                    raw_score += 0.08
                elif m == "cnn1d":
                    raw_score += 0.04
            else:
                if m in ("gru", "cnn1d"):
                    raw_score += 0.08

            scores[m] = max(raw_score, 0.05) ** 2

        total_s = sum(scores.values())
        norm_weights = {m: float(scores[m] / total_s) for m in modern_models}
        per_class_weights[lbl] = norm_weights
        per_class_specialist[lbl] = max(norm_weights, key=norm_weights.get)

    predictor.meta = {
        "motion_median": motion_median,
        "class_motion": class_motion,
        "per_class_weights": per_class_weights,
        "per_class_specialist": per_class_specialist,
    }

    # Evaluate all 5 engines cleanly on held-out clean & robustness validation suites
    clean_metrics = {}
    robust_metrics = {}
    for eng in ("rf_legacy", "rf_kinematic", "gru", "cnn1d", "hybrid"):
        eval_clean_X = X_raw_te if eng == "rf_legacy" else X_trim_te
        eval_rob_X = X_rob_raw if eng == "rf_legacy" else X_rob_trim
        preds_c = [predictor.predict_resampled(s, engine=eng)[0] for s in eval_clean_X]
        preds_r = [predictor.predict_resampled(s, engine=eng)[0] for s in eval_rob_X]
        clean_metrics[eng] = float(accuracy_score(y_te, preds_c))
        robust_metrics[eng] = float(accuracy_score(y_rob, preds_r))

    predictor.meta["scenario"] = scenario_name or "Global"
    predictor.meta["metrics"] = {
        "clean": clean_metrics,
        "robust": robust_metrics,
    }

    meta_out = target_dir / "ensemble_meta.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(predictor.meta, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n" + "=" * 64)
        print(f" Evaluation Summary [{scope_label}]: Clean vs Speed/Shift Robust ")
        print("=" * 64)
        print(f"{'Engine':<26} | {'Clean Acc':<11} | {'Robust Acc (Speed/Shift)'}")
        print("-" * 64)
        pretty_eng = {
            "rf_legacy": "RF (Legacy Flat)",
            "rf_kinematic": "RF (Kinematic)",
            "gru": "PyTorch GRU",
            "cnn1d": "PyTorch 1D-CNN",
            "hybrid": "Hybrid (Auto-Specialist)",
        }
        for eng in ("rf_legacy", "rf_kinematic", "gru", "cnn1d", "hybrid"):
            print(f"{pretty_eng[eng]:<26} | {clean_metrics[eng]*100:6.1f}%    | {robust_metrics[eng]*100:6.1f}%")

        print("\nPer-Gesture Specialist Assignment:")
        print(f"{'Gesture':<14} | {'Motion':<7} | {'Specialist':<14} | {'Weights (RF-Kin / GRU / 1D-CNN)'}")
        print("-" * 64)
        for lbl in sorted(label_to_id.keys()):
            w = per_class_weights[lbl]
            sp = HybridEnsemblePredictor.PRETTY_SPECIALIST.get(per_class_specialist[lbl], per_class_specialist[lbl])
            print(f"{lbl:<14} | {class_motion[lbl]:<7.3f} | {sp:<14} | {w['rf_kinematic']:.2f} / {w['gru']:.2f} / {w['cnn1d']:.2f}")
        print(f"\n[Saved] Ensemble metadata saved to: {meta_out}")
        print("=" * 64)

    return {
        "scenario": scenario_name or "Global",
        "model_dir": str(target_dir),
        "num_classes": num_classes,
        "num_samples": len(X_trim),
        "clean": clean_metrics,
        "robust": robust_metrics,
        "per_class_specialist": per_class_specialist,
    }


def train_scenario_profile(scenario_name, presets_path=config.PRESETS_CONFIG_PATH, verbose=True):
    """Trains the dedicated AI model folder for a single scenario defined in scenario_presets.json."""
    include_classes = None
    if Path(presets_path).exists():
        with open(presets_path, "r", encoding="utf-8") as f:
            presets = json.load(f)
        sc_cfg = presets.get(scenario_name, {})
        active = sc_cfg.get("active_gestures") or sc_cfg.get("recommended_signs")
        if active:
            include_classes = active

    return train_all_models(
        data_dir=config.DATA_RAW_DIR,
        model_dir=config.get_scenario_model_dir(scenario_name),
        include_classes=include_classes,
        scenario_name=scenario_name,
        verbose=verbose,
    )


def train_all_scenarios(presets_path=config.PRESETS_CONFIG_PATH, verbose=True):
    """Trains the global model plus a dedicated model for every scenario in scenario_presets.json."""
    results = {}
    results["Global"] = train_all_models(verbose=verbose)

    if Path(presets_path).exists():
        with open(presets_path, "r", encoding="utf-8") as f:
            presets = json.load(f)
        for sc_name in presets.keys():
            try:
                results[sc_name] = train_scenario_profile(sc_name, presets_path=presets_path, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[Skip Scenario '{sc_name}'] {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Global and Per-Scenario Hybrid Sign Language Models")
    parser.add_argument("--scenario", type=str, default=None, help="Train only a specific scenario name")
    parser.add_argument("--global-only", action="store_true", help="Train only the global all-gestures model")
    args = parser.parse_args()

    try:
        if args.scenario:
            train_scenario_profile(args.scenario, verbose=True)
        elif args.global_only:
            train_all_models(verbose=True)
        else:
            train_all_scenarios(verbose=True)
    except Exception as e:
        print(f"[Error] Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
