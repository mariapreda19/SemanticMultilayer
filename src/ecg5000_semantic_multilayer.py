import json
import os
import random
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.signal import periodogram, find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

from types import SimpleNamespace

CONFIG = {
    "train_path": "/kaggle/input/datasets/salsabilahmid/ecg50000/ECG5000_TRAIN.txt",
    "test_path": "/kaggle/input/datasets/salsabilahmid/ecg50000/ECG5000_TEST.txt",
    "csv_path": None,
    "label_position": "first",
    "label_col": "auto",
    "normal_label": 1,
    "out_dir": "./semantic_ecg5000_results",
    "test_size": 0.25,
    "seed": 42,
    "aggregator": "ensemble",
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_leaf": 6,
    "logreg_C": 1.0,
    "use_ae": True,
    "use_masked_ae": True,
    "epochs": 30,
    "batch_size": 128,
    "latent_dim": 20,
    "lr": 1e-3,
    "mask_ratio": 0.25,
    "denoise_noise": 0.03,
    "device": "auto",
    "use_shapelets": True,
    "shapelet_lengths": "16,32,48",
    "shapelets_per_class": 24,
    "use_supervised_embedding": True,
    "embedding_epochs": 40,
    "embedding_dim": 24,
    "use_unsup_vector_detector": True,
    "unsup_detector": "iforest",
    "contamination": 0.15,
    "supervised_weight": 0.90,
    "threshold_metric": "precision_recall_balance",
    "cv": False,
    "cv_folds": 5,
    "explain_top_k": 5,
}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def resolve_device():
    # In unele medii notebook, CUDA este vizibila, dar PyTorch nu are kernels
    # pentru placa disponibila (ex. Tesla P100 / sm_60). Pentru acest proiect
    # folosesc CPU ca varianta reproductibila.
    return torch.device("cpu") if TORCH_AVAILABLE else None


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_table_auto(path: str) -> pd.DataFrame:
    if path.lower().endswith((".txt", ".tsv")):
        return pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, header=None)
    try:
        return pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except Exception:
        return pd.read_csv(path, header=None)


def table_to_X_y(df: pd.DataFrame, label_position: str, normal_label: int):
    arr = df.apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float32)
    if label_position == "first":
        y_raw = arr[:, 0].astype(int)
        X = arr[:, 1:]
    elif label_position == "last":
        y_raw = arr[:, -1].astype(int)
        X = arr[:, :-1]
    else:
        raise ValueError("label_position must be first or last")
    y = (y_raw != int(normal_label)).astype(int)
    return X.astype(np.float32), y.astype(int), y_raw.astype(int)


def load_ecg_data(
    train_path,
    test_path,
    csv_path,
    label_position,
    label_col,
    normal_label,
    test_size,
    seed,
):
    if train_path and test_path:
        train_df = read_table_auto(train_path)
        test_df = read_table_auto(test_path)

        X_train, y_train, yraw_train = table_to_X_y(train_df, label_position, normal_label)
        X_test, y_test, yraw_test = table_to_X_y(test_df, label_position, normal_label)

        X_raw = np.vstack([X_train, X_test])
        y = np.r_[y_train, y_test]
        y_raw = np.r_[yraw_train, yraw_test]

        train_idx = np.arange(len(X_train))
        test_idx = np.arange(len(X_train), len(X_train) + len(X_test))

        return X_raw, y, y_raw, train_idx, test_idx, "ucr_fixed_train_test"

    if csv_path:
        df = pd.read_csv(csv_path)

        if label_col == "auto":
            label_col = df.columns[-1]
        else:
            try:
                label_col = df.columns[int(label_col)]
            except Exception:
                pass

        y_raw = pd.to_numeric(df[label_col], errors="coerce").astype(int).to_numpy()
        feat_cols = [c for c in df.columns if c != label_col]
        X_raw = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        good = np.isfinite(X_raw).all(axis=1) & np.isfinite(y_raw)
        X_raw, y_raw = X_raw[good], y_raw[good]

        uniq = sorted(np.unique(y_raw).tolist())
        if normal_label in uniq:
            y = (y_raw != int(normal_label)).astype(int)
        else:
            mapping = {uniq[0]: 0, uniq[-1]: 1}
            y = np.array([mapping[v] for v in y_raw], dtype=int)

        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )

        return X_raw, y, y_raw, train_idx, test_idx, "csv_stratified_split"

    raise ValueError("Trebuie setat fie train_path si test_path, fie csv_path.")


def robust_normalize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    scaler = RobustScaler()
    scaler.fit(X_train.reshape(-1, 1))
    Xt = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape).astype(np.float32)
    Xv = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape).astype(np.float32)
    return Xt, Xv, scaler


def per_sample_center_scale(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True) + eps
    return ((X - med) / mad).astype(np.float32)


def z_norm_segment(seg: np.ndarray, eps: float = 1e-6):
    return (seg - np.mean(seg)) / (np.std(seg) + eps)

def spectral_features(x: np.ndarray) -> Dict[str, float]:
    freqs, psd = periodogram(x)
    psd = np.maximum(psd, 1e-12)
    psd_norm = psd / psd.sum()
    entropy = -float(np.sum(psd_norm * np.log(psd_norm)))
    centroid = float(np.sum(freqs * psd_norm))
    q35, q65, q85 = np.quantile(freqs, [0.35, 0.65, 0.85]) if len(freqs) > 4 else (0, 0, 0)
    return {
        "freq_entropy": entropy,
        "freq_centroid": centroid,
        "freq_high_ratio": float(psd[freqs > q65].sum() / psd.sum()) if len(freqs) > 4 else 0.0,
        "freq_low_ratio": float(psd[freqs <= q35].sum() / psd.sum()) if len(freqs) > 4 else 0.0,
        "freq_very_high_ratio": float(psd[freqs > q85].sum() / psd.sum()) if len(freqs) > 4 else 0.0,
    }


def morphology_features(x: np.ndarray) -> Dict[str, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)
    peaks, _ = find_peaks(x, distance=4)
    troughs, _ = find_peaks(-x, distance=4)

    if len(peaks) > 0:
        peak_vals = x[peaks]
        main_peak_idx = int(peaks[np.argmax(peak_vals)])
        main_peak_pos = float(main_peak_idx / max(len(x) - 1, 1))
        main_peak_val = float(np.max(peak_vals))
    else:
        main_peak_idx = int(np.argmax(x))
        main_peak_pos = float(main_peak_idx / max(len(x) - 1, 1))
        main_peak_val = float(np.max(x))

    if len(troughs) > 0:
        trough_vals = x[troughs]
        main_trough_idx = int(troughs[np.argmin(trough_vals)])
        main_trough_val = float(np.min(trough_vals))
    else:
        main_trough_idx = int(np.argmin(x))
        main_trough_val = float(np.min(x))

    amp = float(np.max(x) - np.min(x))
    energy = float(np.mean(x ** 2))
    slope_energy = float(np.mean(dx ** 2)) if len(dx) else 0.0
    curvature_energy = float(np.mean(ddx ** 2)) if len(ddx) else 0.0

    peak_trough_distance = abs(main_peak_idx - main_trough_idx) / max(len(x) - 1, 1)
    zero_crossings = float(np.sum(np.diff(np.signbit(x)).astype(int)))
    slope_zero_crossings = float(np.sum(np.diff(np.signbit(dx)).astype(int))) if len(dx) else 0.0

    return {
        "amp_range": amp,
        "energy": energy,
        "mean_abs": float(np.mean(np.abs(x))),
        "std": float(np.std(x)),
        "skew_proxy": float(np.mean((x - np.mean(x)) ** 3) / (np.std(x) ** 3 + 1e-6)),
        "kurt_proxy": float(np.mean((x - np.mean(x)) ** 4) / (np.std(x) ** 4 + 1e-6)),
        "slope_energy": slope_energy,
        "curvature_energy": curvature_energy,
        "n_peaks": float(len(peaks)),
        "n_troughs": float(len(troughs)),
        "main_peak_pos": main_peak_pos,
        "main_peak_val": main_peak_val,
        "main_trough_val": main_trough_val,
        "peak_to_trough": float(main_peak_val - main_trough_val),
        "peak_trough_distance": float(peak_trough_distance),
        "zero_crossings": zero_crossings,
        "slope_zero_crossings": slope_zero_crossings,
    }


def window_morphology_features(x: np.ndarray, parts: int = 5) -> Dict[str, float]:
    out = {}
    chunks = np.array_split(x, parts)
    for i, c in enumerate(chunks):
        out[f"part{i}_mean"] = float(np.mean(c))
        out[f"part{i}_std"] = float(np.std(c))
        out[f"part{i}_energy"] = float(np.mean(c ** 2))
        out[f"part{i}_max"] = float(np.max(c))
        out[f"part{i}_min"] = float(np.min(c))
        out[f"part{i}_range"] = float(np.max(c) - np.min(c))
    return out


def wavelet_like_features(x: np.ndarray, scales=(1, 2, 4, 8, 12)) -> Dict[str, float]:
    out = {}
    prev = x.astype(float)
    total_energy = np.mean(x ** 2) + 1e-9
    for s in scales:
        smooth = gaussian_filter1d(x, sigma=s)
        detail = prev - smooth
        out[f"wavelet_detail_energy_s{s}"] = float(np.mean(detail ** 2))
        out[f"wavelet_detail_ratio_s{s}"] = float(np.mean(detail ** 2) / total_energy)
        out[f"wavelet_detail_max_s{s}"] = float(np.max(np.abs(detail)))
        prev = smooth
    out["wavelet_low_energy"] = float(np.mean(prev ** 2) / total_energy)
    return out


def build_statistical_feature_matrix(X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    rows, names = [], None
    for x in X:
        d = {}
        d.update(morphology_features(x))
        d.update(spectral_features(x))
        d.update(window_morphology_features(x, parts=5))
        d.update(wavelet_like_features(x))
        if names is None:
            names = list(d.keys())
        rows.append([d[k] for k in names])
    return np.asarray(rows, dtype=np.float32), names


def extract_random_shapelets(X: np.ndarray, y: np.ndarray, lengths: List[int], per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    prototypes = []
    labels = []
    n, T = X.shape
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        for L in lengths:
            if L >= T:
                continue
            chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=len(idx) < per_class)
            for i in chosen:
                start = int(rng.integers(0, T - L + 1))
                proto = z_norm_segment(X[i, start:start+L])
                prototypes.append(proto.astype(np.float32))
                labels.append(cls)
    return prototypes, np.array(labels, dtype=int)


def min_subseq_dist(x: np.ndarray, proto: np.ndarray) -> float:
    L = len(proto)
    best = np.inf
    for s in range(0, len(x) - L + 1):
        seg = z_norm_segment(x[s:s+L])
        d = np.sqrt(np.mean((seg - proto) ** 2))
        if d < best:
            best = d
    return float(best)


def shapelet_response_features(X: np.ndarray, prototypes: List[np.ndarray], labels: np.ndarray):
    if len(prototypes) == 0:
        return np.zeros((len(X), 3), dtype=np.float32), ["shapelet_normal_min", "shapelet_abnormal_min", "shapelet_margin"]
    rows = []
    for x in tqdm(X, desc="Shapelet responses", leave=False):
        dists = np.array([min_subseq_dist(x, p) for p in prototypes], dtype=np.float32)
        dn = np.min(dists[labels == 0]) if np.any(labels == 0) else np.min(dists)
        da = np.min(dists[labels == 1]) if np.any(labels == 1) else np.min(dists)
        rows.append([dn, da, dn - da])
    return np.asarray(rows, dtype=np.float32), ["shapelet_normal_min", "shapelet_abnormal_min", "shapelet_margin"]


class ECGAutoencoder(nn.Module):
    def __init__(self, length=140, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(length, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, length),
        )

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec, z


class ECGEmbeddingNet(nn.Module):
    def __init__(self, length=140, latent=24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(length, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent), nn.ReLU(),
        )
        self.head = nn.Linear(latent, 1)

    def forward(self, x):
        z = self.encoder(x)
        logit = self.head(z).squeeze(-1)
        return logit, z


def train_autoencoder(
    X_train_normal: np.ndarray,
    latent_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    mask_ratio: float,
    denoise_noise: float,
    masked: bool = False,
):
    if not TORCH_AVAILABLE:
        return None

    device = resolve_device()
    model = ECGAutoencoder(X_train_normal.shape[1], latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    loop_name = "Masked AE" if masked else "AE"

    for _ in tqdm(range(epochs), desc=loop_name, leave=False):
        for (xb,) in loader:
            xb = xb.to(device)

            if masked:
                mask = (torch.rand_like(xb) < mask_ratio).float()
                x_in = xb * (1.0 - mask)
                rec, _ = model(x_in)
                loss = (((rec - xb) ** 2) * (mask + 0.1)).mean()
            else:
                if denoise_noise > 0:
                    noisy = xb + denoise_noise * torch.randn_like(xb)
                else:
                    noisy = xb
                rec, _ = model(noisy)
                loss = F.mse_loss(rec, xb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def ae_responses(model, X: np.ndarray, batch_size=256, device="auto"):
    if model is None:
        return np.zeros(len(X), dtype=np.float32), np.zeros((len(X), 1), dtype=np.float32)
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval(); errs, latents = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=dev)
            rec, z = model(xb)
            errs.append(torch.mean((rec - xb) ** 2, dim=1).cpu().numpy())
            latents.append(z.cpu().numpy())
    return np.concatenate(errs).astype(np.float32), np.vstack(latents).astype(np.float32)


def train_embedding_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    embedding_dim: int,
    lr: float,
    batch_size: int,
    embedding_epochs: int,
):
    if not TORCH_AVAILABLE:
        return None

    device = resolve_device()
    model = ECGEmbeddingNet(X_train.shape[1], embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    model.train()
    for _ in tqdm(range(embedding_epochs), desc="Supervised embedding", leave=False):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit, _ = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def embedding_responses(model, X_train, y_train, X_all, batch_size: int):
    if model is None:
        return np.zeros((len(X_all), 3), dtype=np.float32), ["emb_prob", "emb_normal_dist", "emb_margin"]

    dev = resolve_device()
    model.eval()

    probs, Z = [], []

    with torch.no_grad():
        for i in range(0, len(X_all), batch_size):
            xb = torch.tensor(X_all[i:i + batch_size], dtype=torch.float32, device=dev)
            logit, z = model(xb)
            probs.append(torch.sigmoid(logit).cpu().numpy())
            Z.append(z.cpu().numpy())

    prob = np.concatenate(probs).reshape(-1, 1)
    Z = np.vstack(Z).astype(np.float32)

    Z_train = Z[:len(X_train)]
    zn = Z_train[y_train == 0] if np.any(y_train == 0) else Z_train
    za = Z_train[y_train == 1] if np.any(y_train == 1) else Z_train

    cn = np.median(zn, axis=0, keepdims=True)
    ca = np.median(za, axis=0, keepdims=True)

    dn = np.sqrt(np.mean((Z - cn) ** 2, axis=1, keepdims=True))
    da = np.sqrt(np.mean((Z - ca) ** 2, axis=1, keepdims=True))
    margin = dn - da

    return np.column_stack([prob, dn, margin]).astype(np.float32), ["emb_prob", "emb_normal_dist", "emb_margin"]


def robust_response_from_train(values_train: np.ndarray, values_all: np.ndarray, positive_only=True) -> np.ndarray:
    med = np.median(values_train, axis=0, keepdims=True)
    mad = np.median(np.abs(values_train - med), axis=0, keepdims=True) + 1e-6
    z = (values_all - med) / mad
    z = np.maximum(z, 0.0) if positive_only else np.abs(z)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def nearest_centroid_distance(X_train_normal: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    center = np.median(X_train_normal, axis=0, keepdims=True)
    mad = np.median(np.abs(X_train_normal - center), axis=0, keepdims=True) + 1e-6
    z = (X_all - center) / mad
    return np.sqrt(np.mean(z ** 2, axis=1)).astype(np.float32)


def build_semantic_responses(X_train_raw, y_train, X_test_raw, seed):
    X_train_scaled, X_test_scaled, _ = robust_normalize_train_test(X_train_raw, X_test_raw)
    X_train_shape = per_sample_center_scale(X_train_scaled)
    X_test_shape = per_sample_center_scale(X_test_scaled)
    X_all_shape = np.vstack([X_train_shape, X_test_shape])

    X_train_normal = X_train_shape[y_train == 0]
    if len(X_train_normal) == 0:
        X_train_normal = X_train_shape

    response_parts, response_names = [], []

    F_train, stat_names = build_statistical_feature_matrix(X_train_shape)
    F_test, _ = build_statistical_feature_matrix(X_test_shape)

    F_all = np.vstack([F_train, F_test])
    F_ref = F_train[y_train == 0] if np.any(y_train == 0) else F_train
    F_resp = robust_response_from_train(F_ref, F_all, positive_only=False)

    response_parts.append(F_resp)
    response_names += [f"stat_{n}" for n in stat_names]

    raw_dist = nearest_centroid_distance(X_train_normal, X_all_shape).reshape(-1, 1)
    response_parts.append(raw_dist)
    response_names.append("raw_shape_distance")

    response_parts.append(F_resp.max(axis=1, keepdims=True))
    response_parts.append(np.sort(F_resp, axis=1)[:, -min(5, F_resp.shape[1]):].mean(axis=1, keepdims=True))
    response_parts.append((F_resp > 3.0).mean(axis=1, keepdims=True))
    response_names += ["stat_top1", "stat_top5", "stat_active_frac"]

    # Raspunsuri de tip shapelet.
    # Lungimile sunt alese pentru seria ECG5000 de 140 puncte: scurt, mediu, mai larg.
    use_shapelets = True
    if use_shapelets:
        shapelet_lengths = [16, 32, 48]
        shapelets_per_class = 24

        protos, plabels = extract_random_shapelets(
            X_train_shape,
            y_train,
            shapelet_lengths,
            shapelets_per_class,
            seed,
        )

        S_all, S_names = shapelet_response_features(X_all_shape, protos, plabels)
        S_resp = robust_response_from_train(S_all[:len(X_train_shape)], S_all, positive_only=False)

        response_parts.append(S_resp)
        response_names += S_names

    # Setarile pentru retele sunt aici, fiindca sunt folosite doar in blocurile AE/embedding.
    batch_size = 128
    latent_dim = 20
    lr = 1e-3
    ae_epochs = 30
    mask_ratio = 0.25
    denoise_noise = 0.03

    use_ae = True
    if use_ae:
        ae = train_autoencoder(
            X_train_normal,
            latent_dim=latent_dim,
            lr=lr,
            batch_size=batch_size,
            epochs=ae_epochs,
            mask_ratio=mask_ratio,
            denoise_noise=denoise_noise,
            masked=False,
        )

        ae_err_train, ae_lat_train = ae_responses(ae, X_train_shape, batch_size, "cpu")
        ae_err_test, ae_lat_test = ae_responses(ae, X_test_shape, batch_size, "cpu")

        ae_err_all = np.r_[ae_err_train, ae_err_test].reshape(-1, 1)
        ae_ref = ae_err_train[y_train == 0].reshape(-1, 1) if np.any(y_train == 0) else ae_err_train.reshape(-1, 1)

        response_parts.append(robust_response_from_train(ae_ref, ae_err_all, positive_only=True))
        response_names.append("ae_reconstruction_error")

        lat_all = np.vstack([ae_lat_train, ae_lat_test])
        lat_ref = ae_lat_train[y_train == 0] if np.any(y_train == 0) else ae_lat_train

        response_parts.append(nearest_centroid_distance(lat_ref, lat_all).reshape(-1, 1))
        response_names.append("ae_latent_distance")

    use_masked_ae = True
    if use_masked_ae:
        mae = train_autoencoder(
            X_train_normal,
            latent_dim=latent_dim,
            lr=lr,
            batch_size=batch_size,
            epochs=ae_epochs,
            mask_ratio=mask_ratio,
            denoise_noise=denoise_noise,
            masked=True,
        )

        mae_err_train, _ = ae_responses(mae, X_train_shape, batch_size, "cpu")
        mae_err_test, _ = ae_responses(mae, X_test_shape, batch_size, "cpu")

        mae_err_all = np.r_[mae_err_train, mae_err_test].reshape(-1, 1)
        mae_ref = mae_err_train[y_train == 0].reshape(-1, 1) if np.any(y_train == 0) else mae_err_train.reshape(-1, 1)

        response_parts.append(robust_response_from_train(mae_ref, mae_err_all, positive_only=True))
        response_names.append("masked_ae_error")

    use_supervised_embedding = True
    if use_supervised_embedding:
        embedding_dim = 24
        embedding_epochs = 40

        emb = train_embedding_net(
            X_train_shape,
            y_train,
            embedding_dim=embedding_dim,
            lr=lr,
            batch_size=batch_size,
            embedding_epochs=embedding_epochs,
        )

        E_all, E_names = embedding_responses(emb, X_train_shape, y_train, X_all_shape, batch_size)
        E_prob = E_all[:, :1]
        E_dist = robust_response_from_train(E_all[:len(X_train_shape), 1:], E_all[:, 1:], positive_only=False)

        response_parts.append(np.column_stack([E_prob, E_dist]))
        response_names += E_names

    R_all = np.column_stack(response_parts).astype(np.float32)
    R_all = np.log1p(np.maximum(R_all, 0.0))
    R_all = np.nan_to_num(R_all, nan=0.0, posinf=0.0, neginf=0.0)

    n_train = len(X_train_raw)

    return R_all[:n_train], R_all[n_train:], response_names


def train_fusion_model(X_train, y_train, aggregator, seed):
    n_estimators = 500
    max_depth = 10
    min_samples_leaf = 6
    logreg_C = 1.0

    if aggregator == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=logreg_C,
                random_state=seed,
            ),
        ).fit(X_train, y_train)

    if aggregator == "extra":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ).fit(X_train, y_train)

    if aggregator == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ).fit(X_train, y_train)

    if aggregator == "ensemble":
        models = []
        for name in ["extra", "rf", "logreg"]:
            models.append(train_fusion_model(X_train, y_train, name, seed))
        return models

    raise ValueError(aggregator)


def predict_fusion(model, X):
    if isinstance(model, list):
        return np.mean(np.column_stack([predict_fusion(m, X) for m in model]), axis=1)
    return model.predict_proba(X)[:, 1]


def train_unsupervised_detector(R_train, y_train, unsup_detector, contamination, seed):
    R_normal = R_train[y_train == 0]
    if len(R_normal) == 0:
        R_normal = R_train

    if unsup_detector == "iforest":
        return IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        ).fit(R_normal)

    return make_pipeline(
        StandardScaler(),
        OneClassSVM(nu=contamination, kernel="rbf", gamma="scale"),
    ).fit(R_normal)


def predict_unsupervised(model, R):
    if model is None:
        return np.zeros(len(R), dtype=np.float32)
    s = -model.decision_function(R) if hasattr(model, "decision_function") else -model.score_samples(R)
    s = np.asarray(s, dtype=np.float32)
    return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-9)


def choose_threshold_from_train(score_train, y_train, metric="f1"):
    qs = np.linspace(0.01, 0.99, 200)
    candidates = np.unique(np.quantile(score_train, qs))
    best = (-1.0, 0.5)
    for thr in candidates:
        pred = (score_train >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_train, pred, average="binary", zero_division=0)
        if metric == "precision_recall_balance":
            val = 0.5 * f1 + 0.25 * p + 0.25 * r
        elif metric == "precision":
            val = p
        elif metric == "recall":
            val = r
        else:
            val = f1
        if val > best[0]:
            best = (float(val), float(thr))
    return best[1], best[0]


def eval_binary(y_true, score, thr) -> Dict[str, float]:
    pred = (score >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, pred)
    try:
        roc = roc_auc_score(y_true, score)
    except Exception:
        roc = np.nan
    try:
        pr = average_precision_score(y_true, score)
    except Exception:
        pr = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1),
        "roc_auc": float(roc), "pr_auc": float(pr), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp), "threshold": float(thr),
        "pred_positive_ratio": float(pred.mean()), "true_positive_ratio": float(y_true.mean()),
    }


def run_one_split(X_raw, y, train_idx, test_idx, split_name, seed):
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    R_train, R_test, response_names = build_semantic_responses(
        X_train_raw,
        y_train,
        X_test_raw,
        seed,
    )

    aggregator = "ensemble"
    fusion = train_fusion_model(R_train, y_train, aggregator, seed)

    s_train_sup = predict_fusion(fusion, R_train)
    s_test_sup = predict_fusion(fusion, R_test)

    use_unsup_vector_detector = True
    if use_unsup_vector_detector:
        unsup_detector = "iforest"
        contamination = 0.15
        supervised_weight = 0.90

        unsup = train_unsupervised_detector(R_train, y_train, unsup_detector, contamination, seed)

        s_train_unsup = predict_unsupervised(unsup, R_train)
        s_test_unsup = predict_unsupervised(unsup, R_test)

        score_train = supervised_weight * s_train_sup + (1.0 - supervised_weight) * s_train_unsup
        score_test = supervised_weight * s_test_sup + (1.0 - supervised_weight) * s_test_unsup
    else:
        score_train, score_test = s_train_sup, s_test_sup

    threshold_metric = "precision_recall_balance"
    thr, train_obj = choose_threshold_from_train(score_train, y_train, metric=threshold_metric)

    train_metrics = eval_binary(y_train, score_train, thr)
    test_metrics = eval_binary(y_test, score_test, thr)

    train_metrics.update({"split": split_name, "part": "train", "train_threshold_objective": train_obj})
    test_metrics.update({"split": split_name, "part": "test", "train_threshold_objective": train_obj})

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "score_train": score_train,
        "score_test": score_test,
        "R_train": R_train,
        "R_test": R_test,
        "response_names": response_names,
        "threshold": thr,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_train": y_train,
        "y_test": y_test,
    }


def approximate_feature_importance(R_train, y_train, response_names, seed):
    model = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(R_train, y_train)

    return pd.DataFrame(
        {"feature": response_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)


def semantic_group(feature: str) -> str:
    f = feature.lower()
    if "shapelet" in f or "template" in f:
        return "forme asemanatoare cu exemple din train"
    if "ae_reconstruction" in f or "masked_ae" in f:
        return "reconstructie cu autoencoder"
    if "ae_latent" in f or "emb_" in f:
        return "spatiu latent / embedding"
    if "wavelet" in f:
        return "detalii pe scale diferite"
    if "freq" in f:
        return "frecvente"
    if "part" in f:
        return "bucati locale din ECG"
    if "slope" in f or "curvature" in f:
        return "panta si curbura semnalului"
    if "peak" in f or "trough" in f or "zero" in f:
        return "varfuri si forma ECG"
    if "raw" in f or "shape" in f:
        return "forma globala"
    if "top" in f or "active" in f:
        return "scoruri agregate"
    return "statistici simple"


def humanize_feature(feature: str) -> str:
    f = feature.lower()
    if "shapelet_abnormal" in f:
        return "are o bucata care seamana mai mult cu exemple anormale"
    if "shapelet_normal" in f:
        return "nu seamana prea bine cu shapelet-urile normale"
    if "shapelet_margin" in f:
        return "e mai aproape de shapelet-uri anormale decat normale"
    if "ae_reconstruction_error" in f:
        return "autoencoderul antrenat pe normale nu o reconstruieste prea bine"
    if "masked_ae" in f:
        return "cand maschez parti din semnal, modelul nu le ghiceste bine"
    if "ae_latent" in f:
        return "in latent space pica mai departe de grupul normal"
    if "emb_prob" in f:
        return "embedding-ul supravegheat ii da probabilitate mare de anormal"
    if "emb_normal_dist" in f:
        return "embedding-ul e cam departe de centrul normal"
    if "emb_margin" in f:
        return "embedding-ul pare mai apropiat de abnormal decat de normal"
    if "wavelet_detail" in f:
        return "detaliile fine/multiscale arata diferit fata de normale"
    if "wavelet_low" in f:
        return "componenta mai neteda a semnalului e diferita"
    if "freq_entropy" in f:
        return "frecventele sunt mai imprastiate/neobisnuite"
    if "freq_centroid" in f:
        return "energia din frecvente e mutata fata de normal"
    if "freq_high" in f or "freq_very_high" in f:
        return "are mai multa energie pe frecvente inalte"
    if "freq_low" in f:
        return "are alta energie pe frecvente joase"
    if "part" in f:
        return "o zona locala din ECG iese din tiparul normal"
    if "main_peak_pos" in f:
        return "varful principal apare in alta zona decat de obicei"
    if "main_peak_val" in f:
        return "varful principal are amplitudine cam neobisnuita"
    if "main_trough_val" in f:
        return "minimul/trough-ul e cam diferit de normal"
    if "peak_to_trough" in f:
        return "diferenta varf-minim e cam mare/mica fata de normal"
    if "peak_trough_distance" in f:
        return "distanta dintre varf si minim nu prea arata normal"
    if "n_peaks" in f:
        return "numarul de varfuri nu prea seamana cu cel normal"
    if "n_troughs" in f:
        return "numarul de minime/trough-uri pare neobisnuit"
    if "slope_energy" in f:
        return "semnalul se schimba mai brusc decat in normale"
    if "curvature_energy" in f:
        return "curbura semnalului e mai ciudata"
    if "zero_crossings" in f:
        return "trece prin zero de un numar cam neobisnuit de ori"
    if "raw_shape_distance" in f:
        return "forma generala e departe de forma normala mediana"
    if "stat_top1" in f:
        return "cel mai mare raspuns semantic e ridicat"
    if "stat_top5" in f:
        return "mai multe raspunsuri semantice sunt ridicate in acelasi timp"
    if "active_frac" in f:
        return "multe feature-uri zic simultan ca ceva nu e ok"
    if "amp_range" in f:
        return "range-ul amplitudinii e cam diferit"
    if "energy" in f:
        return "energia semnalului e diferita"
    if "mean_abs" in f:
        return "amplitudinea medie absoluta e neobisnuita"
    if "std" in f:
        return "variatia semnalului e diferita"
    if "skew" in f:
        return "semnalul e asimetric fata de normale"
    if "kurt" in f:
        return "are valori mai extreme decat in normale"
    return "feature-ul asta are valoare neobisnuita fata de train"


def _importance_vector(importance_df: pd.DataFrame, response_names: List[str]) -> np.ndarray:
    imp_map = dict(zip(importance_df["feature"], importance_df["importance"]))
    imp = np.array([float(imp_map.get(n, 0.0)) for n in response_names], dtype=np.float64)
    if imp.sum() <= 0:
        imp = np.ones(len(response_names), dtype=np.float64) / max(len(response_names), 1)
    else:
        imp = imp / imp.sum()
    return imp


def strength_word(value: float, all_values: np.ndarray) -> str:
    q50 = float(np.quantile(all_values, 0.50))
    q80 = float(np.quantile(all_values, 0.80))
    q95 = float(np.quantile(all_values, 0.95))
    if value >= q95:
        return "foarte mare"
    if value >= q80:
        return "mare"
    if value >= q50:
        return "mediu"
    return "mic"


def compact_reason(feature: str, value: float, contrib: float, all_feature_values: np.ndarray) -> str:
    level = strength_word(value, all_feature_values)
    return f"{humanize_feature(feature)} (raspuns {level}, val={value:.3f}, contrib={contrib:.5f})"


def make_student_explanation(prediction: int,
                             score: float,
                             threshold: float,
                             top_reasons: List[str],
                             top_groups: List[str],
                             correct: int) -> str:
    pred_text = "anormal" if prediction == 1 else "normal"
    margin = score - threshold

    if prediction == 1:
        start = f"Modelul a zis {pred_text}, scor {score:.4f} peste pragul {threshold:.4f}."
    else:
        start = f"Modelul a zis {pred_text}, scor {score:.4f} sub pragul {threshold:.4f}."

    if abs(margin) < 0.03:
        confidence = "Nu e o decizie super clara, e destul de aproape de prag."
    elif abs(margin) < 0.10:
        confidence = "Decizia pare ok, dar nu e la foarte mare distanta de prag."
    else:
        confidence = "Decizia pare destul de clara dupa scor."

    groups = []
    for g in top_groups:
        if g not in groups:
            groups.append(g)

    reason_text = "; ".join(top_reasons[:3])
    group_text = ", ".join(groups[:3])

    if correct:
        ending = "Pe eticheta din dataset, predictia iese corecta."
    else:
        ending = "Pe eticheta din dataset, aici modelul greseste, deci explicatia trebuie privita cu grija."

    return (
        f"{start} {confidence} Cel mai mult au contat zonele/feature-urile din: {group_text}. "
        f"Pe scurt: {reason_text}. {ending}"
    )


def build_explanations_table(R: np.ndarray,
                             response_names: List[str],
                             scores: np.ndarray,
                             threshold: float,
                             y_true: np.ndarray,
                             y_raw: np.ndarray,
                             indices: np.ndarray,
                             importance_df: pd.DataFrame,
                             top_k: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    imp = _importance_vector(importance_df, response_names)
    contrib = R.astype(np.float64) * imp.reshape(1, -1)
    pred = (scores >= threshold).astype(int)

    rows = []
    long_rows = []
    group_rows = []

    groups = [semantic_group(n) for n in response_names]
    unique_groups = sorted(set(groups))

    for i in range(len(R)):
        order = np.argsort(contrib[i])[::-1]
        top = order[:top_k]

        top_features = []
        top_groups = []
        top_values = []
        top_contribs = []
        top_meanings = []

        for j in top:
            feature = response_names[j]
            value = float(R[i, j])
            contribution = float(contrib[i, j])
            top_features.append(feature)
            top_groups.append(groups[j])
            top_values.append(value)
            top_contribs.append(contribution)
            top_meanings.append(compact_reason(feature, value, contribution, R[:, j]))

        correct = int(pred[i] == y_true[i])
        sentence = make_student_explanation(
            prediction=int(pred[i]),
            score=float(scores[i]),
            threshold=float(threshold),
            top_reasons=top_meanings,
            top_groups=top_groups,
            correct=correct,
        )

        rows.append({
            "index": int(indices[i]),
            "y_raw": int(y_raw[i]),
            "y_true_binary": int(y_true[i]),
            "prediction": int(pred[i]),
            "score": float(scores[i]),
            "threshold": float(threshold),
            "distance_from_threshold": float(scores[i] - threshold),
            "correct": correct,
            "top_features": " | ".join(top_features),
            "top_groups": " | ".join(top_groups),
            "top_values": " | ".join(f"{v:.4f}" for v in top_values),
            "top_contributions": " | ".join(f"{v:.6f}" for v in top_contribs),
            "plain_reasons": " | ".join(top_meanings),
            "explanation": sentence,
        })

        for rank, j in enumerate(top, start=1):
            feature = response_names[j]
            value = float(R[i, j])
            contribution = float(contrib[i, j])
            long_rows.append({
                "index": int(indices[i]),
                "rank": rank,
                "feature": feature,
                "group": groups[j],
                "response_value": value,
                "response_level": strength_word(value, R[:, j]),
                "global_importance": float(imp[j]),
                "local_contribution": contribution,
                "meaning": humanize_feature(feature),
                "student_style_reason": compact_reason(feature, value, contribution, R[:, j]),
                "y_true_binary": int(y_true[i]),
                "prediction": int(pred[i]),
                "score": float(scores[i]),
                "threshold": float(threshold),
            })

        for g in unique_groups:
            ids = [j for j, gg in enumerate(groups) if gg == g]
            group_contribution = float(contrib[i, ids].sum())
            group_max = float(R[i, ids].max())
            group_mean = float(R[i, ids].mean())
            group_rows.append({
                "index": int(indices[i]),
                "group": g,
                "group_contribution": group_contribution,
                "group_max_response": group_max,
                "group_mean_response": group_mean,
                "y_true_binary": int(y_true[i]),
                "prediction": int(pred[i]),
                "score": float(scores[i]),
                "threshold": float(threshold),
            })

    local_df = pd.DataFrame(rows)
    long_df = pd.DataFrame(long_rows)
    group_df = pd.DataFrame(group_rows)
    return local_df, long_df, group_df


def global_group_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    tmp = importance_df.copy()
    tmp["group"] = tmp["feature"].map(semantic_group)
    return tmp.groupby("group", as_index=False).agg(
        total_importance=("importance", "sum"),
        mean_importance=("importance", "mean"),
        n_features=("feature", "count"),
    ).sort_values("total_importance", ascending=False)


def main():
    seed = 42
    seed_everything(seed)

    out_dir = "./semantic_ecg5000_improved_results"
    os.makedirs(out_dir, exist_ok=True)

    # Datele ECG5000 in format UCR: prima coloana este eticheta, restul sunt punctele seriei.
    X_raw, y, y_raw, train_idx, test_idx, split_mode = load_ecg_data(
        train_path="/kaggle/input/datasets/salsabilahmid/ecg50000/ECG5000_TRAIN.txt",
        test_path="/kaggle/input/datasets/salsabilahmid/ecg50000/ECG5000_TEST.txt",
        csv_path=None,
        label_position="first",
        label_col="auto",
        normal_label=1,
        test_size=0.25,
        seed=seed,
    )

    print("Split mode:", split_mode)
    print("Samples:", len(X_raw), "Length:", X_raw.shape[1])
    print("Binary abnormal ratio:", float(y.mean()))
    print("Raw labels:", dict(zip(*np.unique(y_raw, return_counts=True))))
    print("Torch available:", TORCH_AVAILABLE)

    cv = False

    if cv and split_mode == "csv_stratified_split":
        cv_folds = 5
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        all_rows = []
        last_result = None

        for fold, (tr, te) in enumerate(skf.split(X_raw, y), start=1):
            print(f"\n===== CV fold {fold}/{cv_folds} =====")

            result = run_one_split(X_raw, y, tr, te, f"fold_{fold}", seed)
            all_rows += [result["train_metrics"], result["test_metrics"]]

            print("TRAIN:", {k: round(v, 4) if isinstance(v, float) else v for k, v in result["train_metrics"].items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"]})
            print("TEST :", {k: round(v, 4) if isinstance(v, float) else v for k, v in result["test_metrics"].items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"]})

            last_result = result

        metrics_df = pd.DataFrame(all_rows)
        metrics_df.to_csv(os.path.join(out_dir, "cv_metrics.csv"), index=False)

        test_mean = metrics_df[metrics_df["part"] == "test"].mean(numeric_only=True).to_dict()
        test_std = metrics_df[metrics_df["part"] == "test"].std(numeric_only=True).to_dict()

        with open(os.path.join(out_dir, "cv_summary.json"), "w") as f:
            json.dump(
                {
                    "mean": {k: float(v) for k, v in test_mean.items()},
                    "std": {k: float(v) for k, v in test_std.items()},
                },
                f,
                indent=2,
            )

        result = last_result

    else:
        result = run_one_split(X_raw, y, train_idx, test_idx, "fixed_or_single", seed)

        metrics_df = pd.DataFrame([result["train_metrics"], result["test_metrics"]])
        metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

        print("\nTRAIN:", {k: round(v, 4) if isinstance(v, float) else v for k, v in result["train_metrics"].items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "tp", "fp", "tn", "fn"]})
        print("TEST :", {k: round(v, 4) if isinstance(v, float) else v for k, v in result["test_metrics"].items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "tp", "fp", "tn", "fn"]})

        pred_test = (result["score_test"] >= result["threshold"]).astype(int)

        pd.DataFrame(
            {
                "index": result["test_idx"],
                "y_raw": y_raw[result["test_idx"]],
                "y_true_binary": result["y_test"],
                "score": result["score_test"],
                "prediction": pred_test,
            }
        ).to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False)

        pd.DataFrame(
            {
                "index": np.r_[result["train_idx"], result["test_idx"]],
                "split": ["train"] * len(result["train_idx"]) + ["test"] * len(result["test_idx"]),
            }
        ).to_csv(os.path.join(out_dir, "split_indices.csv"), index=False)

    try:
        imp = approximate_feature_importance(
            result["R_train"],
            result["y_train"],
            result["response_names"],
            seed,
        )

        imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

        group_imp = global_group_importance(imp)
        group_imp.to_csv(os.path.join(out_dir, "semantic_group_importance.csv"), index=False)

        print("\nTop semantic responses:")
        print(imp.head(20).to_string(index=False))

        print("\nSemantic group importance:")
        print(group_imp.to_string(index=False))

        explain_top_k = 5

        local_test, long_test, group_test = build_explanations_table(
            R=result["R_test"],
            response_names=result["response_names"],
            scores=result["score_test"],
            threshold=result["threshold"],
            y_true=result["y_test"],
            y_raw=y_raw[result["test_idx"]],
            indices=result["test_idx"],
            importance_df=imp,
            top_k=explain_top_k,
        )

        local_test.to_csv(os.path.join(out_dir, "explanations_test.csv"), index=False)
        long_test.to_csv(os.path.join(out_dir, "explanations_test_long.csv"), index=False)
        group_test.to_csv(os.path.join(out_dir, "explanations_test_by_group.csv"), index=False)

        local_train, long_train, group_train = build_explanations_table(
            R=result["R_train"],
            response_names=result["response_names"],
            scores=result["score_train"],
            threshold=result["threshold"],
            y_true=result["y_train"],
            y_raw=y_raw[result["train_idx"]],
            indices=result["train_idx"],
            importance_df=imp,
            top_k=explain_top_k,
        )

        local_train.to_csv(os.path.join(out_dir, "explanations_train.csv"), index=False)
        long_train.to_csv(os.path.join(out_dir, "explanations_train_long.csv"), index=False)
        group_train.to_csv(os.path.join(out_dir, "explanations_train_by_group.csv"), index=False)

        print("\nExample local explanations from TEST:")
        show_cols = ["index", "y_raw", "y_true_binary", "prediction", "score", "top_features", "explanation"]
        print(local_test.head(8)[show_cols].to_string(index=False))

    except Exception as e:
        print("Could not compute feature importance / explanations:", e)

    print("\nSaved outputs in:", out_dir)


if __name__ == "__main__":
    main()