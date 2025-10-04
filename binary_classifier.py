import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from art.estimators.classification import PyTorchClassifier
from art.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, DeepFool
from types import SimpleNamespace

# ---------------------------
# Config
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 384
BATCH = 16
SAMPLES_PER_CLASS = 25
EPOCHS = 20
ATTACKS = [
    {"name": "FGSM", "eps": 0.03},
    {"name": "FGSM", "eps": 0.01},
    {"name": "FGSM", "eps": 0.02},
    {"name": "FGSM", "eps": 0.04},
    {"name": "FGSM", "eps": 0.05},
    {"name": "PGD",  "eps": 0.03, "eps_step": 0.01, "max_iter": 10},
]
MODEL_DIR = "models"
DATA_ROOT = "testing"

# ---------------------------
# Helpers: interactive prompts
# ---------------------------

def _fmt_default(val):
    return f" [default: {val}]" if val is not None else ""


def ask_str(prompt: str, default: str | None = None) -> str:
    s = input(f"{prompt}{_fmt_default(default)}: ").strip()
    return s if s else (default if default is not None else "")


def ask_float(prompt: str, default: float | None = None, allow_none: bool = False) -> float | None:
    while True:
        s = input(f"{prompt}{_fmt_default(default)}: ").strip()
        if not s:
            return default if (default is not None or allow_none) else default
        s = s.replace(",", ".")  # accept comma decimal
        try:
            return float(s)
        except Exception:
            if allow_none and s.lower() in {"none", "null"}:
                return None
            print("Invalid number, try again (e.g., 0.03 or 0,03).")


def ask_int(prompt: str, default: int | None = None) -> int:
    while True:
        s = input(f"{prompt}{_fmt_default(default)}: ").strip()
        if not s and default is not None:
            return default
        try:
            return int(s)
        except Exception:
            print("Invalid integer, try again.")


def ask_bool(prompt: str, default: bool = False) -> bool:
    suffix = " [y/N]" if not default else " [Y/n]"
    while True:
        s = input(f"{prompt}{suffix}: ").strip().lower()
        if not s:
            return default
        if s in {"y", "yes", "t", "true", "1"}:
            return True
        if s in {"n", "no", "f", "false", "0"}:
            return False
        print("Please answer y/n.")


def ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    choices_disp = "/".join(choices)
    while True:
        s = input(f"{prompt} ({choices_disp}){_fmt_default(default)}: ").strip()
        if not s and default:
            return default
        s = s or ""
        s = s.upper()
        # allow some flexibility for DeepFool
        if s in {"DEEPFOOL", "DEEP", "DF"}:
            s = "DEEPFOOL"
        if s in [c.upper() for c in choices]:
            return s
        print(f"Invalid choice. Pick one of: {choices_disp}")

# ---------------------------
# Load models
# ---------------------------

def get_model_by_type(t):
    model_names = {
        'resnet': 'resnet34.tv_in1k',
        'efficientnet': 'efficientnet_b0.ra_in1k',
        'densenet': 'densenet121.tv_in1k',
        'convnext': 'convnext_tiny.in12k_ft_in1k_384',
        'visiontransformer': 'vit_small_patch16_384',
    }
    if t in model_names:
        return lambda num_classes: timm.create_model(model_names[t], pretrained=True, num_classes=num_classes)
    return None


def load_model(model, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    if 'pretrained_cfg' in checkpoint and hasattr(model, 'pretrained_cfg'):
        model.pretrained_cfg = checkpoint['pretrained_cfg']
    model.to(device).eval()
    print(f"[INFO] Model loaded: {filename}")
    return model


def choose_file(dir_path):
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    if not files:
        raise FileNotFoundError(f"Directory '{dir_path}' is empty.")
    print("Choose model file:")
    for i, f in enumerate(files, 1):
        print(f"{i} - {f}")
    idx = ask_int("File number", 1)
    if idx < 1 or idx > len(files):
        raise ValueError("Invalid file number")
    return os.path.join(dir_path, files[idx - 1])

# ---------------------------
# Transform and clip_values
# ---------------------------

def set_transform(model, input_size_override=384):
    cfg = resolve_data_config({}, model=model, pretrained_cfg=getattr(model, 'pretrained_cfg', None))
    cfg['input_size'] = (3, input_size_override, input_size_override)
    transform = create_transform(**cfg)
    mean, std = cfg['mean'], cfg['std']
    print("[DEBUG] Transform cfg:", cfg)
    return transform, mean, std


def compute_clip_values_from_mean_std(mean, std):
    min_vals = np.array([(0.0 - m) / s for m, s in zip(mean, std)], dtype=np.float32).reshape(3, 1, 1)
    max_vals = np.array([(1.0 - m) / s for m, s in zip(mean, std)], dtype=np.float32).reshape(3, 1, 1)
    return (min_vals, max_vals)


def eps_pixel_to_normalized(eps_in_pixel, std):
    std_t = torch.tensor(std, dtype=torch.float32)
    if isinstance(eps_in_pixel, (float, int)):
        return float((eps_in_pixel / std_t).max())
    elif isinstance(eps_in_pixel, list):
        return [float((torch.tensor(e) / std_t).max()) for e in eps_in_pixel]
    else:
        raise ValueError("eps_in_pixel must be float/int/list")

# ---------------------------
# Dataset operations
# ---------------------------

def create_subset_per_class(dataset, samples_per_class=20):
    """Pick up to `samples_per_class` items from each class for a balanced subset."""
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
    selected = [i for lst in class_indices.values() for i in lst]
    return Subset(dataset, selected)


def loader_to_numpy(dl):
    X, y = [], []
    for batch, labels in dl:
        X.append(batch.numpy())
        y.append(labels.numpy())
    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.concatenate(y, axis=0).astype(np.int64)
    return X, y

# ---------------------------
# Logit-based detector (0=clean, 1=adv)
# ---------------------------

class LogitDetectorMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):  # x: (B, K) where K = nb_classes of base model
        return self.net(x)


def build_art_classifier_on_logits(model, input_shape, nb_classes, device):
    """
    ART classifier for a detector operating on logit vectors.
    Note: clip_values/preprocessing are not needed here.
    """
    loss = nn.CrossEntropyLoss()
    optim_ = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    clf = PyTorchClassifier(
        model=model.to(device),
        loss=loss,
        optimizer=optim_,
        input_shape=input_shape,  # (K,) where K = nb_classes of the base model
        nb_classes=nb_classes,     # 2 (clean/adv)
        clip_values=None,
        device_type="gpu" if str(device).startswith("cuda") else "cpu",
        preprocessing=None,
    )
    return clf

@torch.no_grad()
def get_logits_from_base(base_clf: PyTorchClassifier, X_np: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """
    Return logits from the base model for inputs X_np (already normalized).
    We call base_clf.predict() in batches.
    """
    outs = []
    for i in range(0, len(X_np), batch_size):
        outs.append(base_clf.predict(X_np[i:i + batch_size]))
    return np.concatenate(outs, axis=0).astype(np.float32)  # shape: (N, K)

# ---------------------------
# Generate adversarial samples
# ---------------------------

def generate_adversarials(base_clf, X_clean_norm, y_true, mean, std, spec):
    name = spec["name"].upper()
    if name == "FGSM":
        atk = FastGradientMethod(
            estimator=base_clf,
            eps=eps_pixel_to_normalized(spec.get("eps", 0.03), std),
        )
    elif name == "PGD":
        eps = eps_pixel_to_normalized(spec.get("eps", 0.03), std)
        step = eps_pixel_to_normalized(spec.get("eps_step", spec.get("eps", 0.03) / 3), std)
        atk = ProjectedGradientDescent(
            estimator=base_clf, eps=eps, eps_step=step, max_iter=spec.get("max_iter", 10)
        )
    elif name == "CW":
        atk = CarliniL2Method(
            classifier=base_clf,
            targeted=False,
            max_iter=spec.get("max_iter", 20),
            confidence=spec.get("confidence", 0.0),
            initial_const=spec.get("initial_const", 0.1),
            binary_search_steps=spec.get("binary_search_steps", 5),
        )
    elif name == "DEEPFOOL":
        atk = DeepFool(classifier=base_clf, max_iter=spec.get("max_iter", 50), nb_grads=spec.get("nb_grads", 10))
    else:
        raise ValueError(f"Unsupported attack: {name}")
    y_oh = to_categorical(y_true, base_clf.nb_classes)
    return atk.generate(x=X_clean_norm, y=y_oh)

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, balanced_accuracy_score


def pick_threshold_for_fpr(y_true, p_adv, target_fpr=0.10):
    fpr, tpr, thr = roc_curve(y_true, p_adv)
    i = np.argmin(np.abs(fpr - target_fpr))
    return float(thr[i]), float(fpr[i]), float(tpr[i])


def eval_at_threshold(y_true, p_adv, thr):
    y_pred = (p_adv >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # [[TN,FP],[FN,TP]]
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"thr": thr, "acc": acc, "bal_acc": bal_acc, "prec": prec, "rec": rec, "f1": f1, "fpr": fpr, "tpr": tpr, "cm": cm}

# ---------------------------
# Add Gaussian noise
# ---------------------------

def add_noisy_clean_imgs(X_clean_norm, std, clip_values, sigma_px=0.02, frac=1.0, seed=42):
    rng = np.random.default_rng(seed)
    N = len(X_clean_norm)
    n_aug = int(np.clip(np.ceil(frac * N), 0, N))
    if n_aug == 0:
        return np.empty((0, *X_clean_norm.shape[1:]), dtype=np.float32)

    idx = rng.choice(N, size=n_aug, replace=False)
    X_sel = X_clean_norm[idx].copy()

    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    sigma_norm = sigma_px / std_arr  # per-channel
    noise = rng.normal(0.0, 1.0, size=X_sel.shape).astype(np.float32) * sigma_norm

    X_noisy = X_sel + noise
    clip_min, clip_max = clip_values  # (3,1,1)
    X_noisy = np.clip(X_noisy, clip_min, clip_max)  # keep in model's input range
    return X_noisy

# ---------------------------
# Training and evaluation
# ---------------------------

def train_and_eval_detector_logits(X_tr, y_tr, X_va, y_va, input_dim, device, args):
    # 1) Train the detector on logits
    det_torch = LogitDetectorMLP(in_dim=input_dim)
    det = build_art_classifier_on_logits(det_torch, input_shape=(input_dim,), nb_classes=2, device=device)
    det.fit(X_tr, to_categorical(y_tr, 2), batch_size=64, nb_epochs=EPOCHS)

    # 2) TEST: generate adversarials on CLEAN validation images, then take logits
    spec = {
        "name": args.test_attack,
        "eps": args.eps,
        "eps_step": args.eps_step if args.eps_step is not None else (args.eps / 3 if args.eps is not None else None),
        "max_iter": args.max_iter,
        "confidence": args.confidence,
        "binary_search_steps": args.binary_search_steps,
        "nb_grads": args.nb_grads,
    }
    print(f"[TEST] Generating {spec['name']} adversarials for evaluation (images)...")
    X_adv_va_imgs = generate_adversarials(base_clf, X_val_norm, y_val, MEAN, STD, spec)

    # Attack success rate on the base model (info)
    logits_adv = base_clf.predict(X_adv_va_imgs)
    pred_adv = logits_adv.argmax(axis=1)
    y_true_adv = y_val[:len(pred_adv)]
    succ_mask = (pred_adv != y_true_adv)

    sr_te = float(succ_mask.mean())
    ok, tot = int(succ_mask.sum()), int(len(succ_mask))
    print(f"[INFO] Attack success rate on base model (TEST): {sr_te*100:.1f}% ({ok}/{tot})")

    # Evaluate only on successful adversarials if requested
    if getattr(args, "eval_only_success", False):
        X_adv_va_imgs = X_adv_va_imgs[succ_mask]
        if len(X_adv_va_imgs) == 0:
            print("[WARN] No successful adversarials kept for test. Increase attack strength.")
            return det

    # Keep only successful adversarials (non-targeted: pred != y_true)
    logits_adv = base_clf.predict(X_adv_va_imgs)  # (M,K)
    pred_adv = logits_adv.argmax(axis=1)
    y_true_adv = y_val[:len(pred_adv)]
    succ_mask = (pred_adv != y_true_adv)

    print(
        f"[INFO] Attack success rate on base model: {succ_mask.mean()*100:.1f}% "
        f"({succ_mask.sum()}/{len(succ_mask)})"
    )

    X_adv_va_imgs = X_adv_va_imgs[succ_mask]
    if len(X_adv_va_imgs) == 0:
        print("[WARN] No successful adversarials kept. Increase attack strength.")
        return det

    # Balance test 1:1: as many clean as successful adversarials
    N = min(len(X_val_norm), len(X_adv_va_imgs))
    rng = np.random.default_rng(42)
    idx_clean = rng.choice(len(X_val_norm), size=N, replace=False)

    # logits for clean and adversarial
    X_clean_test_logits = get_logits_from_base(base_clf, X_val_norm[idx_clean])
    X_adv_test_logits = get_logits_from_base(base_clf, X_adv_va_imgs[:N])

    X_te = np.concatenate([X_clean_test_logits, X_adv_test_logits], axis=0)
    y_te = np.concatenate([np.zeros(N, dtype=np.int64), np.ones(N, dtype=np.int64)], axis=0)

    # 3) Threshold calibrated on validation under FPR ≤ target
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix, balanced_accuracy_score

    # (A) Threshold-independent metrics on TEST
    probs_test = det.predict(X_te)  # (2N, 2)
    p_adv_test = probs_test[:, 1]
    auc_roc = roc_auc_score(y_te, p_adv_test)
    ap = average_precision_score(y_te, p_adv_test)
    print(f"[AUC] ROC={auc_roc:.3f}  PR(AP)={ap:.3f}")

    # (B) Calibrate threshold tau on VALIDATION to meet FPR ≤ target
    probs_val = det.predict(X_va)
    p_adv_val = probs_val[:, 1]
    fpr_va, tpr_va, thr_va = roc_curve(y_va, p_adv_val)

    mask = fpr_va <= args.target_fpr + 1e-12
    if np.any(mask):
        idx = np.argmax(tpr_va[mask])
        tau = float(thr_va[mask][idx])
        fpr_va_tau = float(fpr_va[mask][idx])
        tpr_va_tau = float(tpr_va[mask][idx])
    else:
        idx = np.argmin(fpr_va)
        tau = float(thr_va[idx])
        fpr_va_tau = float(fpr_va[idx])
        tpr_va_tau = float(tpr_va[idx])

    print(f"[THR τ] chosen on VAL: τ={tau:.6f}  FPR_val={fpr_va_tau*100:.2f}%  TPR_val={tpr_va_tau*100:.2f}%")

    # (C) Evaluate on TEST at tau
    y_pred_tau = (p_adv_test >= tau).astype(int)

    acc = accuracy_score(y_te, y_pred_tau)
    bal_acc = balanced_accuracy_score(y_te, y_pred_tau)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred_tau, average="binary", zero_division=1)
    cm = confusion_matrix(y_te, y_pred_tau, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(
        f"[THR τ on TEST] Acc={acc*100:.2f}%  BalAcc={bal_acc*100:.2f}%  "
        f"P={prec*100:.2f}%  R/TPR={rec*100:.2f}%  F1={f1*100:.2f}%  "
        f"FPR={fpr*100:.2f}%  TPR={tpr*100:.2f}%  CM={cm.tolist()}"
    )

    return det


def filter_success_adv_images(X_adv_imgs, y_true, base_clf):
    """Keep only successful (non-targeted) adversarials."""
    logits = base_clf.predict(X_adv_imgs)
    pred = logits.argmax(1)
    y_true = y_true[:len(pred)]
    mask = (pred != y_true)
    return X_adv_imgs[mask], float(mask.mean()), int(mask.sum()), int(len(mask))


def balance_adv_per_attack(X_adv_logits_tr_by_attack, oversample=False, seed=42):
    """
    Return a balanced (equal-count) pool of adversarial logits.
    - oversample=False: undersampling to the MIN count,
    - oversample=True : oversample with replacement to the MAX count.
    """
    rng = np.random.default_rng(seed)
    sizes = {k: len(v) for k, v in X_adv_logits_tr_by_attack.items()}
    if len(sizes) == 0:
        return np.empty((0, 0), dtype=np.float32)

    target = max(sizes.values()) if oversample else min(sizes.values())

    parts = []
    for name, X in X_adv_logits_tr_by_attack.items():
        n = len(X)
        if n == 0:
            continue
        if oversample and n < target:
            idx = rng.choice(n, size=target, replace=True)
        else:
            idx = rng.permutation(n)[:target]
        parts.append(X[idx])
    if len(parts) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(parts, axis=0)

# ---------------------------
# Interactive args collection
# ---------------------------

def collect_args_interactive() -> SimpleNamespace:
    print("\n=== Interactive configuration ===")
    test_attack = ask_choice("Choose test attack", ["FGSM", "PGD", "CW", "DeepFool"], default="PGD")

    # Common params
    eps = None
    eps_step = None
    max_iter = ask_int("Max iterations (for PGD/CW/DeepFool)", 20)
    confidence = ask_float("CW confidence (ignored if not CW)", 0.0)
    binary_search_steps = ask_int("CW binary search steps (ignored if not CW)", 5)
    nb_grads = ask_int("DeepFool nb_grads (ignored if not DeepFool)", 10)
    target_fpr = ask_float("Target FPR for threshold calibration (0..1)", 0.10)

    # Attack-specific eps/eps_step
    if test_attack in {"FGSM", "PGD"}:
        eps = ask_float("Epsilon (pixel scale 0..1)", 0.03)
        if test_attack == "PGD":
            eps_step = ask_float("PGD step size (pixel scale) or blank for eps/3", None, allow_none=True)

    add_noisy_clean = ask_bool("Add noisy-but-clean images to detector training?", False)
    noise_sigma = ask_float("Gaussian noise sigma in pixel scale [0..1]", 0.02)
    noise_frac = ask_float("Fraction of clean training samples to noise (0..1)", 0.5)

    train_only_success = ask_bool("Train detector on successful adversarials only?", False)
    eval_only_success = ask_bool("Evaluate metrics only on successful adversarials?", False)
    balance_per_attack = ask_bool("Balance adversarial pool equally per attack?", False)
    oversample_per_attack = False
    if balance_per_attack:
        oversample_per_attack = ask_bool("Use oversampling (instead of undersampling) when balancing per attack?", False)

    print("=== Configuration set. ===\n")

    return SimpleNamespace(
        test_attack=test_attack,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
        confidence=confidence,
        binary_search_steps=binary_search_steps,
        nb_grads=nb_grads,
        target_fpr=target_fpr,
        add_noisy_clean=add_noisy_clean,
        noise_sigma=noise_sigma,
        noise_frac=noise_frac,
        train_only_success=train_only_success,
        eval_only_success=eval_only_success,
        balance_per_attack=balance_per_attack,
        oversample_per_attack=oversample_per_attack,
    )

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Threads
    torch.set_num_threads(12)
    os.environ["OMP_NUM_THREADS"] = "12"

    # 1) Select and load a model
    model_path = choose_file(MODEL_DIR)
    base_type = os.path.basename(model_path).split('_')[0]
    tmp_loader = DataLoader(datasets.ImageFolder(root=DATA_ROOT), batch_size=1)
    num_classes = len(tmp_loader.dataset.classes)

    ctor = get_model_by_type(base_type)
    if ctor is None:
        raise RuntimeError(f"Unknown model type in filename: {base_type}")
    base_model = ctor(num_classes=num_classes)
    base_model = load_model(base_model, model_path, DEVICE)

    # 2) Transforms and DataLoader
    transform, MEAN, STD = set_transform(base_model, INPUT_SIZE)
    full_ds = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
    subset = create_subset_per_class(full_ds, samples_per_class=SAMPLES_PER_CLASS)

    # split 80/20
    idxs = np.arange(len(subset))
    np.random.shuffle(idxs)
    split = int(0.8 * len(idxs))
    train_idx, val_idx = idxs[:split], idxs[split:]
    train_ds = Subset(subset, train_idx)
    val_ds = Subset(subset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=False)

    # 3) ART classifier for the base model
    clip_values = compute_clip_values_from_mean_std(MEAN, STD)
    base_clf = PyTorchClassifier(
        model=base_model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(base_model.parameters(), lr=1e-4),
        input_shape=(3, INPUT_SIZE, INPUT_SIZE),
        nb_classes=num_classes,
        clip_values=clip_values,
        device_type="gpu" if str(DEVICE).startswith("cuda") else "cpu",
        preprocessing=None,
    )

    # 4) Dump data from loaders to numpy
    X_train_norm, y_train = loader_to_numpy(train_loader)
    X_val_norm, y_val = loader_to_numpy(val_loader)

    # === Gather interactive arguments ===
    args = collect_args_interactive()

    # 5) Generate adversaries PER ATTACK (train/val), optionally filter successful ones for training
    adv_imgs_tr_by_attack = {}
    adv_imgs_va_by_attack = {}
    for spec in ATTACKS:
        name = spec["name"].upper()
        print(f"[ATTACK] Generating {name} (train)...")
        X_adv_tr_img = generate_adversarials(base_clf, X_train_norm, y_train, MEAN, STD, spec)
        if args.train_only_success:
            X_adv_tr_img, sr, ok, tot = filter_success_adv_images(X_adv_tr_img, y_train, base_clf)
            print(f"[INFO] {name} train success rate: {sr*100:.1f}% ({ok}/{tot})")
        adv_imgs_tr_by_attack[name] = X_adv_tr_img

        print(f"[ATTACK] Generating {name} (val)...")
        X_adv_va_img = generate_adversarials(base_clf, X_val_norm, y_val, MEAN, STD, spec)
        # For training-time validation we usually DO NOT filter, but you could add a switch if desired
        adv_imgs_va_by_attack[name] = X_adv_va_img

    # 5a) Convert adv images → logits, per attack
    X_adv_logits_tr_by_attack = {name: get_logits_from_base(base_clf, X) for name, X in adv_imgs_tr_by_attack.items()}
    X_adv_logits_va_by_attack = {name: get_logits_from_base(base_clf, X) for name, X in adv_imgs_va_by_attack.items()}

    # 5b) (INFO) Counts per attack
    print("[INFO] Train adv per attack:", {k: len(v) for k, v in X_adv_logits_tr_by_attack.items()})
    print("[INFO] Val   adv per attack:", {k: len(v) for k, v in X_adv_logits_va_by_attack.items()})

    # 5c) Equal-count per attack for TRAIN (undersample by default, oversample if flag)
    X_adv_logits_tr_bal = (
        balance_adv_per_attack(
            X_adv_logits_tr_by_attack, oversample=args.oversample_per_attack, seed=42
        )
        if args.balance_per_attack
        else np.concatenate(list(X_adv_logits_tr_by_attack.values()), axis=0)
    )

    # 5d) For validation just concatenate (you can also balance if you wish)
    X_adv_logits_va = np.concatenate(list(X_adv_logits_va_by_attack.values()), axis=0)

    # Clean images to logits
    X_clean_logits_tr = get_logits_from_base(base_clf, X_train_norm)  # (N_tr, K)
    X_clean_logits_va = get_logits_from_base(base_clf, X_val_norm)    # (N_va, K)
    K = X_clean_logits_tr.shape[1]

    # (NEW) Noisy clean to logits (training only)
    if args.add_noisy_clean:
        X_noisy_imgs_tr = add_noisy_clean_imgs(
            X_train_norm, std=STD, clip_values=clip_values, sigma_px=args.noise_sigma, frac=args.noise_frac, seed=42
        )
        X_noisy_logits_tr = get_logits_from_base(base_clf, X_noisy_imgs_tr)
        print(f"[INFO] Noisy clean generated: {len(X_noisy_logits_tr)}")
    else:
        X_noisy_logits_tr = np.empty((0, K), dtype=np.float32)

    clean_pool_tr = np.concatenate([X_clean_logits_tr, X_noisy_logits_tr], axis=0)
    n_clean_tr = len(clean_pool_tr)
    n_adv_tr = len(X_adv_logits_tr_bal)
    n_take_tr = min(n_clean_tr, n_adv_tr)

    rng = np.random.default_rng(42)
    idx_clean = rng.permutation(n_clean_tr)[:n_take_tr]
    idx_adv = rng.permutation(n_adv_tr)[:n_take_tr]

    X_tr = np.concatenate([clean_pool_tr[idx_clean], X_adv_logits_tr_bal[idx_adv]], axis=0)
    y_tr = np.concatenate([np.zeros(n_take_tr, dtype=np.int64), np.ones(n_take_tr, dtype=np.int64)], axis=0)

    n_clean_va = len(X_clean_logits_va)
    n_adv_va = len(X_adv_logits_va)
    n_take_va = min(n_clean_va, n_adv_va)
    idx_c_va = rng.permutation(n_clean_va)[:n_take_va]
    idx_a_va = rng.permutation(n_adv_va)[:n_take_va]
    X_va = np.concatenate([X_clean_logits_va[idx_c_va], X_adv_logits_va[idx_a_va]], axis=0)
    y_va = np.concatenate([np.zeros(n_take_va, dtype=np.int64), np.ones(n_take_va, dtype=np.int64)], axis=0)

    def shuffle_xy(X, y):
        p = np.random.permutation(len(y))
        return X[p], y[p]

    X_tr, y_tr = shuffle_xy(X_tr, y_tr)
    X_va, y_va = shuffle_xy(X_va, y_va)

    print(
        f"[INFO] Train clean logits: {len(X_clean_logits_tr)}, noisy clean logits: {len(X_noisy_logits_tr)}, "
        f"adv logits (balanced): {len(X_adv_logits_tr_bal)} -> final per class: {n_take_tr}"
    )

    det = train_and_eval_detector_logits(
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        input_dim=K,
        device=DEVICE,
        args=args,
    )

    print("[DONE]")