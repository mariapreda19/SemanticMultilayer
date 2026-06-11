import argparse
import copy
import itertools
import json
import os
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d, binary_closing, binary_opening
from scipy.signal import periodogram
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x



def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)



def _norm_key(path: str) -> str:
    return str(path).replace("\\", "/").lstrip("/")


def _strip_prefix(path: str, prefix: str) -> str:
    return path[len(prefix):] if path.startswith(prefix) else path


def list_series(data_path: str) -> List[str]:
    p = Path(data_path)
    out = []
    if p.is_dir():
        roots = [p]
        if (p / "data").is_dir():
            roots.append(p / "data")
        for root in roots:
            for f in root.rglob("*.csv"):
                rel = _norm_key(f.relative_to(root))
                out.append(_strip_prefix(rel, "data/"))
        return sorted(set(out))
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as z:
            for name in z.namelist():
                name = _norm_key(name)
                if name.endswith(".csv"):
                    out.append(_strip_prefix(name, "data/"))
        return sorted(set(out))
    raise ValueError(f"data_path must be folder or .zip: {data_path}")


def load_csv(data_path: str, series_key: str) -> pd.DataFrame:
    p = Path(data_path)
    series_key = _norm_key(series_key)

    if p.is_dir():
        candidates = [p / series_key, p / "data" / series_key]
        csv_path = next((c for c in candidates if c.is_file()), None)
        if csv_path is None:
            examples = list_series(str(p))[:20]
            raise FileNotFoundError(f"Nu gasesc seria {series_key}. Exemple:\n" + "\n".join(examples))
        df = pd.read_csv(csv_path)
    elif p.suffix.lower() == ".zip":
        candidates = ["data/" + series_key, series_key]
        with zipfile.ZipFile(p, "r") as z:
            names = set(_norm_key(n) for n in z.namelist())
            internal = next((c for c in candidates if c in names), None)
            if internal is None:
                examples = [n for n in names if n.endswith(".csv")][:20]
                raise FileNotFoundError(f"Nu gasesc seria {series_key}. Exemple:\n" + "\n".join(examples))
            with z.open(internal) as f:
                df = pd.read_csv(f)
    else:
        raise ValueError(f"data_path must be folder or .zip: {data_path}")

    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{series_key} nu are timestamp,value. Coloane: {list(df.columns)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df


def load_label_json(labels_path: str, filename: str) -> Dict:
    p = Path(labels_path)
    if p.is_dir():
        candidates = [p / filename, p / "labels" / filename]
        json_path = next((c for c in candidates if c.is_file()), None)
        if json_path is None:
            raise FileNotFoundError(f"Nu gasesc {filename} in {labels_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if p.suffix.lower() == ".zip":
        candidates = ["labels/" + filename, filename]
        with zipfile.ZipFile(p, "r") as z:
            names = set(_norm_key(n) for n in z.namelist())
            internal = next((c for c in candidates if c in names), None)
            if internal is None:
                raise FileNotFoundError(f"Nu gasesc {filename} in {labels_path}")
            with z.open(internal) as f:
                return json.load(f)
    raise ValueError(f"labels_path must be folder or .zip: {labels_path}")


def labels_to_sparse_point_mask(df: pd.DataFrame, label_times: List[str]) -> np.ndarray:
    label_ts = set(pd.to_datetime(label_times))
    return df["timestamp"].isin(label_ts).astype(int).to_numpy()


def windows_to_point_mask(df: pd.DataFrame, windows: List[List[str]]) -> np.ndarray:
    ts = df["timestamp"]
    y = np.zeros(len(df), dtype=int)
    for pair in windows:
        if len(pair) != 2:
            continue
        a, b = pd.to_datetime(pair[0]), pd.to_datetime(pair[1])
        y[((ts >= a) & (ts <= b)).to_numpy()] = 1
    return y


def mask_to_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask).astype(int)
    segs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i] == 0:
            i += 1
            continue
        j = i
        while j < n and mask[j] == 1:
            j += 1
        segs.append((i, j - 1))
        i = j
    return segs



def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def robust_z(values: np.ndarray, train_end: int, positive_only: bool = True) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    train_end = max(3, min(train_end, len(v)))
    tr = v[:train_end]
    med = np.nanmedian(tr)
    mad = np.nanmedian(np.abs(tr - med)) + 1e-9
    z = (v - med) / mad
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(z, 0.0) if positive_only else np.abs(z)


def robust_scale_signal(x_raw: np.ndarray, train_end: int, use_log: bool = True) -> np.ndarray:
    x = np.asarray(x_raw, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x), posinf=np.nanmedian(x), neginf=np.nanmedian(x))

    if use_log and np.nanmin(x) >= 0:
        q99 = np.nanquantile(x[:train_end], 0.99)
        q50 = np.nanquantile(x[:train_end], 0.50)
        if q99 > 10 * max(abs(q50), 1e-6):
            x = np.log1p(x)

    scaler = RobustScaler().fit(x[:train_end].reshape(-1, 1))
    y = scaler.transform(x.reshape(-1, 1)).ravel().astype(float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def clean_mask(mask: np.ndarray, min_segment: int = 1, close_gap: int = 0) -> np.ndarray:
    out = mask.astype(bool)
    if close_gap and close_gap > 1:
        out = binary_closing(out, structure=np.ones(close_gap)).astype(bool)
    if min_segment and min_segment > 1:
        out = binary_opening(out, structure=np.ones(min_segment)).astype(bool)
    return out.astype(int)


def expand_events(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask).astype(int)
    if radius <= 0 or mask.sum() == 0:
        return mask
    out = np.zeros_like(mask)
    idx = np.where(mask == 1)[0]
    n = len(mask)
    for i in idx:
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        out[a:b] = 1
    return out


def choose_clean_train_prefix(y_window: np.ndarray, n: int, train_ratio: float, min_train: int, use_labels: bool) -> int:
    default_end = min(max(min_train, int(n * train_ratio)), n - 1)
    if not use_labels:
        return default_end
    idx = np.where(y_window == 1)[0]
    if len(idx) == 0:
        return default_end
    first = int(idx[0])
    if first > min_train:
        return min(default_end, first - 1)
    return default_end



def rolling_stats(x: np.ndarray, w: int, causal: bool) -> Dict[str, np.ndarray]:
    s = pd.Series(x)
    minp = max(3, w // 4)
    r = s.rolling(w, center=not causal, min_periods=minp)
    out = {
        "mean": r.mean(),
        "median": r.median(),
        "std": r.std(),
        "min": r.min(),
        "max": r.max(),
        "q10": r.quantile(0.10),
        "q25": r.quantile(0.25),
        "q75": r.quantile(0.75),
        "q90": r.quantile(0.90),
    }
    res = {}
    for k, v in out.items():
        res[k] = v.bfill().ffill().fillna(0.0).to_numpy(dtype=float)
    res["range"] = res["max"] - res["min"]
    res["iqr"] = res["q75"] - res["q25"]
    res["qspread"] = res["q90"] - res["q10"]
    return res


def seasonal_residual_layer(x: np.ndarray, train_end: int, periods: List[int]) -> Tuple[np.ndarray, int]:
    n = len(x)
    train = x[:train_end]
    best_p, best_corr = -1, -np.inf
    for p in periods:
        if p < 2 or train_end <= 2 * p:
            continue
        a, b = train[p:], train[:-p]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr, best_p = corr, p
    out = np.zeros(n, dtype=float)
    if best_p == -1:
        return out, -1
    out[best_p:] = np.abs(x[best_p:] - x[:-best_p])
    out[:best_p] = out[best_p]
    return out, best_p


def ar_residual_layer(x: np.ndarray, train_end: int, p: int = 12) -> np.ndarray:
    n = len(x)
    out = np.zeros(n, dtype=float)
    if train_end <= p + 10:
        return out
    X = np.vstack([x[i-p:i] for i in range(p, train_end)])
    y = x[p:train_end]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    for i in range(p, n):
        out[i] = abs(x[i] - float(np.dot(x[i-p:i], coef)))
    return out


def ewma_residual_layer(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    m = x[0]
    for i in range(1, len(x)):
        pred = m
        out[i] = abs(x[i] - pred)
        m = alpha * x[i] + (1 - alpha) * m
    return out


def cusum_layer(x: np.ndarray, train_end: int, drift: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.median(x[:train_end])
    mad = np.median(np.abs(x[:train_end] - mu)) + 1e-9
    z = (x - mu) / mad
    up = np.zeros(len(x))
    down = np.zeros(len(x))
    for i in range(1, len(x)):
        up[i] = max(0.0, up[i-1] + z[i] - drift)
        down[i] = max(0.0, down[i-1] - z[i] - drift)
    return up, down


def spectral_entropy_layer(x: np.ndarray, w: int, causal: bool) -> np.ndarray:
    n = len(x)
    out = np.zeros(n, dtype=float)
    if w >= n:
        return out
    if causal:
        rng = range(w, n)
        for i in rng:
            win = x[i-w:i]
            _, psd = periodogram(win)
            psd = psd / (psd.sum() + 1e-12)
            out[i] = -np.sum(psd * np.log(psd + 1e-12))
        out[:w] = out[w]
    else:
        h = w // 2
        for i in range(h, n-h):
            win = x[i-h:i+h]
            _, psd = periodogram(win)
            psd = psd / (psd.sum() + 1e-12)
            out[i] = -np.sum(psd * np.log(psd + 1e-12))
        out[:h] = out[h]
        out[n-h:] = out[n-h-1]
    return out


def build_semantic_layers(x_raw: np.ndarray, train_end: int, windows: List[int], seasonal_periods: List[int],
                          layer_smooth: float, causal: bool, use_log: bool) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    x = robust_scale_signal(x_raw, train_end, use_log=use_log)
    base_med = np.median(x[:train_end])
    base_mean = np.mean(x[:train_end])
    layers, names = [], []

    def add(name: str, values: np.ndarray, positive_only: bool = True):
        z = robust_z(values, train_end, positive_only=positive_only)
        if layer_smooth > 0:
            z = gaussian_filter1d(z, sigma=layer_smooth)
        layers.append(z)
        names.append(name)

    dx = np.r_[0.0, np.diff(x)]
    rel_dx = np.r_[0.0, np.abs(np.diff(x)) / (np.abs(x[:-1]) + 1.0)]

    add("amp_abs", np.abs(x - base_med))
    add("point_up", x - base_med)
    add("point_down", base_med - x)
    add("diff_abs", np.abs(dx))
    add("diff_up", dx)
    add("diff_down", -dx)
    add("relative_change", rel_dx)

    seas, period = seasonal_residual_layer(x, train_end, seasonal_periods)
    add(f"seasonal_residual_p{period}", seas)
    if period > 0:
        signed = np.zeros_like(x)
        signed[period:] = x[period:] - x[:-period]
        add(f"seasonal_up_p{period}", signed)
        add(f"seasonal_down_p{period}", -signed)

    add("ar12_residual", ar_residual_layer(x, train_end, p=12))
    add("ewma_residual_fast", ewma_residual_layer(x, alpha=0.3))
    add("ewma_residual_slow", ewma_residual_layer(x, alpha=0.05))
    cu, cd = cusum_layer(x, train_end, drift=0.2)
    add("cusum_up", cu)
    add("cusum_down", cd)

    for w in windows:
        feats = rolling_stats(x, w, causal)
        add(f"level_abs_w{w}", np.abs(feats["median"] - base_med))
        add(f"level_up_w{w}", feats["median"] - base_med)
        add(f"level_down_w{w}", base_med - feats["median"])
        add(f"mean_abs_w{w}", np.abs(feats["mean"] - base_mean))
        add(f"var_high_w{w}", feats["std"])
        add(f"iqr_high_w{w}", feats["iqr"])
        add(f"range_high_w{w}", feats["range"])
        add(f"flat_std_low_w{w}", np.median(feats["std"][:train_end]) - feats["std"])
        add(f"flat_range_low_w{w}", np.median(feats["range"][:train_end]) - feats["range"])

        train_med = np.median(feats["median"][:train_end])
        train_q10 = np.quantile(feats["median"][:train_end], 0.10)
        train_q90 = np.quantile(feats["median"][:train_end], 0.90)
        add(f"local_quantile_break_w{w}", np.maximum(feats["median"] - train_q90, train_q10 - feats["median"]))

        if w >= 8:
            half = max(2, w // 2)
            s = pd.Series(x)
            if causal:
                recent = s.rolling(half, min_periods=max(2, half//3)).mean()
                past = recent.shift(half)
                cp = (recent - past).bfill().ffill().fillna(0).to_numpy(dtype=float)
            else:
                left = s.rolling(half, center=False, min_periods=max(2, half//3)).mean()
                right = s[::-1].rolling(half, center=False, min_periods=max(2, half//3)).mean()[::-1]
                cp = (right - left).bfill().ffill().fillna(0).to_numpy(dtype=float)
            add(f"changepoint_abs_w{w}", np.abs(cp))
            add(f"changepoint_up_w{w}", cp)
            add(f"changepoint_down_w{w}", -cp)

    for w in [64, 128, 256]:
        if w < len(x):
            add(f"spectral_entropy_w{w}", spectral_entropy_layer(x, w, causal))

    R = np.column_stack(layers)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R, names, {"seasonal_period": float(period)}


def aggregate_modes(R: np.ndarray, names: List[str], train_end: int, score_smooth: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    def idx(words):
        return [i for i, name in enumerate(names) if any(w in name for w in words)]

    groups = {
        "all_peak": list(range(R.shape[1])),
        "up": idx(["up", "cusum_up"]),
        "down": idx(["down", "cusum_down"]),
        "flat": idx(["flat"]),
        "level": idx(["level", "mean_abs", "local_quantile"]),
        "variance": idx(["var_high", "iqr_high", "range_high"]),
        "change": idx(["changepoint", "diff", "relative_change", "cusum"]),
        "frequency": idx(["spectral", "seasonal"]),
        "residual": idx(["residual", "ewma", "ar12"]),
    }
    scores = {}
    for g, ids in groups.items():
        if not ids:
            continue
        if g == "all_peak":
            raw = R[:, ids].max(axis=1)
        else:
            k = min(4, len(ids))
            raw = np.sort(R[:, ids], axis=1)[:, -k:].mean(axis=1)
        z = robust_z(raw, train_end)
        if score_smooth > 0:
            z = gaussian_filter1d(z, sigma=score_smooth)
        scores[g] = z

    M = np.column_stack(list(scores.values()))
    hybrid = np.sort(M, axis=1)[:, -min(3, M.shape[1]):].mean(axis=1)
    hybrid = robust_z(hybrid, train_end)
    if score_smooth > 0:
        hybrid = gaussian_filter1d(hybrid, sigma=score_smooth)
    scores["hybrid"] = hybrid
    return hybrid, scores


def build_response_table(args, series_key: str, labels_all: Dict, windows_all: Dict, split_name: str):
    df = load_csv(args.data_path, series_key)
    x_raw = df["value"].to_numpy(dtype=float)
    y_sparse = labels_to_sparse_point_mask(df, labels_all[series_key])
    y_window = windows_to_point_mask(df, windows_all[series_key])

    train_end = choose_clean_train_prefix(
        y_window=y_window,
        n=len(df),
        train_ratio=args.train_ratio,
        min_train=max(args.windows),
        use_labels=not args.no_label_clean_train,
    )

    R, layer_names, meta = build_semantic_layers(
        x_raw=x_raw,
        train_end=train_end,
        windows=args.windows,
        seasonal_periods=args.seasonal_periods,
        layer_smooth=args.layer_smooth,
        causal=args.causal,
        use_log=not args.no_log_preprocess,
    )
    hybrid, modes = aggregate_modes(R, layer_names, train_end, args.score_smooth)

    mode_names = ["hybrid", "all_peak", "up", "down", "flat", "level", "variance", "change", "frequency", "residual"]
    X_parts = []
    feature_names = []
    for m in mode_names:
        if m in modes:
            X_parts.append(modes[m].reshape(-1, 1))
            feature_names.append(f"mode_{m}")

    top1 = R.max(axis=1)
    top3 = np.sort(R, axis=1)[:, -min(3, R.shape[1]):].mean(axis=1)
    top8 = np.sort(R, axis=1)[:, -min(8, R.shape[1]):].mean(axis=1)
    active = (R > 3.0).mean(axis=1)
    X_parts += [top1.reshape(-1,1), top3.reshape(-1,1), top8.reshape(-1,1), active.reshape(-1,1)]
    feature_names += ["layer_top1", "layer_top3", "layer_top8", "layer_active_frac"]

    X = np.column_stack(X_parts).astype(np.float32)
    X = np.log1p(np.maximum(X, 0.0))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    info = {
        "series": series_key,
        "category": series_key.split("/")[0] if "/" in series_key else "unknown",
        "split": split_name,
        "df": df,
        "X": X,
        "feature_names": feature_names,
        "y_window": y_window.astype(int),
        "y_sparse": y_sparse.astype(int),
        "train_end": train_end,
        "meta": meta,
        "modes": modes,
        "hybrid": hybrid,
    }
    return info



def make_stratified_series_split(keys: List[str], windows_all: Dict, labels_all: Dict, test_size: float, seed: int) -> pd.DataFrame:
    rows = []
    for k in keys:
        category = k.split("/")[0] if "/" in k else "unknown"
        has_anom = 1 if len(windows_all.get(k, [])) > 0 else 0
        rows.append({"series": k, "category": category, "has_anom": has_anom})
    df = pd.DataFrame(rows)
    df["stratum"] = df["category"] + "__" + df["has_anom"].astype(str)

    counts = df["stratum"].value_counts()
    df["stratum_safe"] = df["stratum"]
    rare = set(counts[counts < 2].index)
    df.loc[df["stratum_safe"].isin(rare), "stratum_safe"] = df.loc[df["stratum_safe"].isin(rare), "category"]

    counts2 = df["stratum_safe"].value_counts()
    rare2 = set(counts2[counts2 < 2].index)
    df.loc[df["stratum_safe"].isin(rare2), "stratum_safe"] = "anom_" + df.loc[df["stratum_safe"].isin(rare2), "has_anom"].astype(str)

    y = df["stratum_safe"].to_numpy()
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df["series"], y))
    except Exception:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df["series"], df["category"]))

    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df.sort_values(["split", "category", "series"]).reset_index(drop=True)



def sample_training_points(infos: List[Dict], max_points_per_series: int, neg_pos_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    Xs, ys, groups = [], [], []
    for info in infos:
        X = info["X"]
        y = info["y_window"]
        n = len(y)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        if len(pos_idx) > max_points_per_series // 2:
            pos_idx = rng.choice(pos_idx, size=max_points_per_series // 2, replace=False)
        n_neg = int(max(len(pos_idx) * neg_pos_ratio, max_points_per_series // 3))
        n_neg = min(n_neg, len(neg_idx), max_points_per_series)
        if n_neg > 0:
            neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)
        idx = np.concatenate([pos_idx, neg_idx]) if len(pos_idx) or len(neg_idx) else np.array([], dtype=int)
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        Xs.append(X[idx])
        ys.append(y[idx])
        groups.extend([info["series"]] * len(idx))
    return np.vstack(Xs), np.concatenate(ys).astype(int), np.array(groups)


def make_model(args):
    if args.aggregator == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=args.logreg_C, random_state=args.seed)
        )
    if args.aggregator == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        )
    if args.aggregator == "extra":
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    if args.aggregator == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=args.seed,
        )
    raise ValueError(args.aggregator)


def model_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]



def parse_float_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_grid(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def train_metric_value(y_true: np.ndarray, pred: np.ndarray, score: np.ndarray, args) -> Dict[str, float]:
    win = evaluate(y_true, pred, score)
    ev = event_window_f1(y_true, pred)
    soft = soft_event_metrics(y_true, pred)
    nab = nab_like_score(y_true, pred, args.nab_fp_weight, args.nab_fn_weight)
    return {
        "window_f1": win["f1"],
        "event_f1": ev["event_f1"],
        "soft_event_f1": soft["soft_event_f1"],
        "nab_like": nab["nab_like_score"],
        "pred_ratio": win["pred_ratio"],
        "gt_ratio": win["gt_ratio"],
        "event_fp": ev["event_fp"],
        "event_fn": ev["event_fn"],
    }


def objective_from_metrics(metrics: Dict[str, float], args) -> float:
    if args.threshold_metric == "event_f1":
        return metrics["event_f1"]
    if args.threshold_metric == "soft_event_f1":
        return metrics["soft_event_f1"]
    if args.threshold_metric == "nab_like":
        return metrics["nab_like"]
    if args.threshold_metric == "window_f1":
        return metrics["window_f1"]

    nab01 = (metrics["nab_like"] + 1.0) / 2.0
    pred_penalty = max(0.0, metrics["pred_ratio"] - args.target_max_pred_ratio)
    return (
        args.obj_event_weight * metrics["event_f1"]
        + args.obj_soft_weight * metrics["soft_event_f1"]
        + args.obj_nab_weight * nab01
        + args.obj_window_weight * metrics["window_f1"]
        - args.obj_pred_ratio_penalty * pred_penalty
    )


def find_train_threshold(train_infos: List[Dict], train_scores: Dict[str, np.ndarray], args) -> float:
    qs = np.linspace(args.thr_q_min, args.thr_q_max, args.thr_steps)
    all_scores = np.concatenate([train_scores[i["series"]] for i in train_infos])
    candidate_thr = np.unique(np.quantile(all_scores, qs))

    best = (-1e18, candidate_thr[0])
    for thr in candidate_thr:
        vals = []
        for info in train_infos:
            score = train_scores[info["series"]]
            pred = postprocess_score(score, thr, args)
            vals.append(objective_from_metrics(train_metric_value(info["y_window"], pred, score, args), args))
        m = float(np.nanmean(vals))
        if m > best[0]:
            best = (m, float(thr))
    if getattr(args, "verbose_thresholds", False):
        print(f"Selected threshold on TRAIN only: {best[1]:.6f} using {args.threshold_metric} objective={best[0]:.4f}", flush=True)
    return best[1]


def tune_postprocess_on_train(train_infos: List[Dict], train_scores: Dict[str, np.ndarray], args):
    if not args.auto_tune_postprocess:
        thr = find_train_threshold(train_infos, train_scores, args)
        return args, thr, {
            "auto_tune_postprocess": False,
            "threshold": float(thr),
            "threshold_metric": args.threshold_metric,
        }

    grid = list(itertools.product(
        parse_int_grid(args.grid_min_segment),
        parse_int_grid(args.grid_close_gap),
        parse_int_grid(args.grid_expand_radius),
        parse_float_grid(args.grid_max_pred_ratio),
        parse_float_grid(args.grid_min_pred_ratio),
        parse_float_grid(args.grid_nab_fp_weight),
    ))

    original_grid_size = len(grid)
    if args.autotune_max_combinations > 0 and len(grid) > args.autotune_max_combinations:
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(grid), size=args.autotune_max_combinations, replace=False)
        grid = [grid[i] for i in chosen]

    print(f"\nAUTOTUNE postprocess: testing {len(grid)} / {original_grid_size} combinations "
          f"x {args.thr_steps} threshold quantiles on {len(train_infos)} train series", flush=True)

    best = {"objective": -1e18}
    best_args = None
    best_thr = None

    for combo_i, (min_segment, close_gap, expand_radius, max_pred_ratio, min_pred_ratio, nab_fp_weight) in enumerate(
        tqdm(grid, desc="Autotune postprocess", unit="combo"), start=1
    ):
        cand = copy.copy(args)
        cand.min_segment = min_segment
        cand.close_gap = close_gap
        cand.expand_radius = expand_radius
        cand.max_pred_ratio = max_pred_ratio
        cand.min_pred_ratio = min_pred_ratio
        cand.nab_fp_weight = nab_fp_weight

        thr = find_train_threshold(train_infos, train_scores, cand)
        vals = []
        debug = []
        for info in train_infos:
            score = train_scores[info["series"]]
            pred = postprocess_score(score, thr, cand)
            m = train_metric_value(info["y_window"], pred, score, cand)
            vals.append(objective_from_metrics(m, cand))
            debug.append(m)
        obj = float(np.nanmean(vals))
        mean_event = float(np.nanmean([d["event_f1"] for d in debug]))
        mean_soft = float(np.nanmean([d["soft_event_f1"] for d in debug]))
        mean_nab = float(np.nanmean([d["nab_like"] for d in debug]))
        no_anom = [d for d in debug if d["gt_ratio"] == 0.0]
        no_anom_acc = float(np.mean([d["pred_ratio"] == 0.0 for d in no_anom])) if no_anom else float("nan")

        tie = (obj, mean_nab, mean_soft, no_anom_acc, -max_pred_ratio, -expand_radius, -close_gap)
        best_tie = best.get("tie", (-1e18,))
        if tie > best_tie:
            best = {
                "objective": obj,
                "tie": tie,
                "threshold": float(thr),
                "threshold_metric": cand.threshold_metric,
                "min_segment": int(min_segment),
                "close_gap": int(close_gap),
                "expand_radius": int(expand_radius),
                "max_pred_ratio": float(max_pred_ratio),
                "min_pred_ratio": float(min_pred_ratio),
                "nab_fp_weight": float(nab_fp_weight),
                "train_mean_event_f1": mean_event,
                "train_mean_soft_event_f1": mean_soft,
                "train_mean_nab_like": mean_nab,
                "train_no_anomaly_accuracy": no_anom_acc,
                "grid_size": len(grid),
            }
            best_args = cand
            best_thr = thr
            print(
                f"[new best {combo_i}/{len(grid)}] obj={obj:.4f} "
                f"event={mean_event:.4f} soft={mean_soft:.4f} nab={mean_nab:.4f} "
                f"noanom={no_anom_acc:.3f} thr={thr:.4f} "
                f"minseg={min_segment} close={close_gap} expand={expand_radius} "
                f"maxratio={max_pred_ratio} fpw={nab_fp_weight}",
                flush=True,
            )

    print("\nSELECTED POSTPROCESSING ON TRAIN ONLY")
    for k, v in best.items():
        if k != "tie":
            print(f"  {k}: {v}")
    return best_args, best_thr, best


def postprocess_score(score: np.ndarray, thr: float, args) -> np.ndarray:
    raw = (score >= thr).astype(int)
    pred = clean_mask(raw, min_segment=args.min_segment, close_gap=args.close_gap)
    if pred.sum() > 0 and args.expand_radius > 0:
        pred = expand_events(pred, args.expand_radius)
        pred = clean_mask(pred, min_segment=args.min_segment, close_gap=args.close_gap)

    if pred.mean() > args.max_pred_ratio:
        pred = cap_by_top_components(score, pred, args.max_pred_ratio, args.min_segment)
    if pred.mean() < args.min_pred_ratio:
        pred[:] = 0
    return pred.astype(int)


def cap_by_top_components(score: np.ndarray, mask: np.ndarray, max_ratio: float, min_segment: int) -> np.ndarray:
    n = len(mask)
    cap = max(min_segment, int(round(n * max_ratio)))
    segs = mask_to_segments(mask)
    if not segs or mask.sum() <= cap:
        return mask.astype(int)
    ranked = []
    for a, b in segs:
        length = b - a + 1
        mass = float(np.sum(score[a:b+1]))
        peak = float(np.max(score[a:b+1]))
        ranked.append((mass / np.sqrt(length) + 0.1 * peak, a, b, length))
    ranked.sort(reverse=True)
    out = np.zeros(n, dtype=int)
    used = 0
    for _, a, b, length in ranked:
        if used + length > cap and used > 0:
            continue
        out[a:b+1] = 1
        used += length
        if used >= cap:
            break
    return clean_mask(out, min_segment=min_segment, close_gap=0)

def safe_auc(y_true: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_true)) <= 1:
        return np.nan, np.nan
    return float(roc_auc_score(y_true, score)), float(average_precision_score(y_true, score))


def evaluate(y_true: np.ndarray, pred: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    gt_ratio = float(np.mean(y_true))
    pred_ratio = float(np.mean(pred))
    if gt_ratio == 0.0:
        if pred_ratio == 0.0:
            p = r = f1 = 1.0
        else:
            p = r = f1 = 0.0
    else:
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    roc, pr = safe_auc(y_true, score)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "pred_ratio": pred_ratio,
            "gt_ratio": gt_ratio, "roc_auc": roc, "pr_auc": pr}


def event_window_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    gt_segments = mask_to_segments(y_true)
    pred_segments = mask_to_segments(y_pred)
    if len(gt_segments) == 0:
        if len(pred_segments) == 0:
            return {"event_precision": 1.0, "event_recall": 1.0, "event_f1": 1.0,
                    "event_tp": 0, "event_fp": 0, "event_fn": 0}
        return {"event_precision": 0.0, "event_recall": 0.0, "event_f1": 0.0,
                "event_tp": 0, "event_fp": len(pred_segments), "event_fn": 0}
    detected = []
    for a, b in gt_segments:
        detected.append(any(not (q < a or p > b) for p, q in pred_segments))
    tp = int(sum(detected))
    fn = len(gt_segments) - tp
    fp = 0
    for p, q in pred_segments:
        if not any(not (q < a or p > b) for a, b in gt_segments):
            fp += 1
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"event_precision": float(precision), "event_recall": float(recall), "event_f1": float(f1),
            "event_tp": tp, "event_fp": fp, "event_fn": fn}


def soft_event_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    gt_segments = mask_to_segments(y_true)
    pred_segments = mask_to_segments(y_pred)
    if len(gt_segments) == 0:
        v = 1.0 if len(pred_segments) == 0 else 0.0
        return {"soft_event_precision": v, "soft_event_recall": v, "soft_event_f1": v}
    if len(pred_segments) == 0:
        return {"soft_event_precision": 0.0, "soft_event_recall": 0.0, "soft_event_f1": 0.0}

    def iou(s1, s2):
        a, b = s1; p, q = s2
        inter = max(0, min(b, q) - max(a, p) + 1)
        union = (b - a + 1) + (q - p + 1) - inter
        return inter / max(union, 1)

    recall = np.mean([max(iou(g, p) for p in pred_segments) for g in gt_segments])
    precision = np.mean([max(iou(p, g) for g in gt_segments) for p in pred_segments])
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"soft_event_precision": float(precision), "soft_event_recall": float(recall), "soft_event_f1": float(f1)}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nab_like_score(y_true: np.ndarray, y_pred: np.ndarray, fp_weight=0.22, fn_weight=1.0) -> Dict[str, float]:
    gt_segments = mask_to_segments(y_true)
    pred_segments = mask_to_segments(y_pred)
    if len(gt_segments) == 0:
        raw = -fp_weight * len(pred_segments)
        return {"nab_like_score": float(raw), "nab_like_raw_score": float(raw),
                "nab_like_gt_events": 0, "nab_like_pred_events": len(pred_segments),
                "nab_like_tp_events": 0, "nab_like_fp_events": len(pred_segments), "nab_like_fn_events": 0}
    score = 0.0; tp = 0; fn = 0; used = set()
    for gi, (a, b) in enumerate(gt_segments):
        best_reward, best_pi = None, None
        for pi, (p, q) in enumerate(pred_segments):
            if q < a or p > b:
                continue
            hit = max(p, a)
            rel = (hit - a) / max((b - a + 1), 1)
            reward = 2.0 * sigmoid(5.0 * (1.0 - rel)) - 1.0
            if best_reward is None or reward > best_reward:
                best_reward, best_pi = reward, pi
        if best_reward is None:
            score -= fn_weight; fn += 1
        else:
            score += best_reward; used.add(best_pi); tp += 1
    fp = 0
    for pi, (p, q) in enumerate(pred_segments):
        if pi in used:
            continue
        if not any(not (q < a or p > b) for a, b in gt_segments):
            score -= fp_weight; fp += 1
    return {"nab_like_score": float(score / max(len(gt_segments), 1)), "nab_like_raw_score": float(score),
            "nab_like_gt_events": len(gt_segments), "nab_like_pred_events": len(pred_segments),
            "nab_like_tp_events": tp, "nab_like_fp_events": fp, "nab_like_fn_events": fn}


def evaluate_info(info: Dict, score: np.ndarray, pred: np.ndarray, args, out_dir: str) -> Dict[str, float]:
    metrics = evaluate(info["y_window"], pred, score)
    sparse = evaluate(info["y_sparse"], pred, score)
    event = event_window_f1(info["y_window"], pred)
    soft = soft_event_metrics(info["y_window"], pred)
    nab = nab_like_score(info["y_window"], pred, args.nab_fp_weight, args.nab_fn_weight)

    safe = info["series"].replace("/", "__").replace(".csv", "")
    out_csv = os.path.join(out_dir, f"{safe}_supervised_scores.csv")
    out = info["df"].copy()
    out["label_window"] = info["y_window"]
    out["label_sparse_point"] = info["y_sparse"]
    out["score_supervised"] = score
    out["prediction"] = pred
    for j, name in enumerate(info["feature_names"]):
        out[f"response_{name}"] = info["X"][:, j]
    out.to_csv(out_csv, index=False)

    row = {
        "series": info["series"],
        "category": info["category"],
        "split": info["split"],
        "n": len(info["df"]),
        "train_end": info["train_end"],
        "seasonal_period": int(info["meta"]["seasonal_period"]),
        "out_csv": out_csv,
    }
    row.update({f"window_{k}": v for k, v in metrics.items()})
    row.update({f"sparse_{k}": v for k, v in sparse.items()})
    row.update(event)
    row.update(soft)
    row.update(nab)
    return row


def global_report(summary: pd.DataFrame, out_dir: str, name: str):
    rows = []
    with_anom = summary[summary["window_gt_ratio"] > 0]
    no_anom = summary[summary["window_gt_ratio"] == 0]
    rows.append({
        "split": name,
        "count": len(summary),
        "mean_window_f1": summary["window_f1"].mean(),
        "median_window_f1": summary["window_f1"].median(),
        "mean_event_f1": summary["event_f1"].mean(),
        "median_event_f1": summary["event_f1"].median(),
        "mean_soft_event_f1": summary["soft_event_f1"].mean(),
        "median_soft_event_f1": summary["soft_event_f1"].median(),
        "mean_nab_like": summary["nab_like_score"].mean(),
        "median_nab_like": summary["nab_like_score"].median(),
        "no_anomaly_accuracy": (no_anom["window_pred_ratio"] == 0).mean() if len(no_anom) else np.nan,
        "anomalous_mean_event_f1": with_anom["event_f1"].mean() if len(with_anom) else np.nan,
    })
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(out_dir, f"global_scores_{name}.csv"), index=False)
    cat = summary.groupby("category").agg(
        count=("series", "count"),
        mean_window_f1=("window_f1", "mean"),
        mean_event_f1=("event_f1", "mean"),
        mean_soft_event_f1=("soft_event_f1", "mean"),
        mean_nab_like=("nab_like_score", "mean"),
        mean_pred_ratio=("window_pred_ratio", "mean"),
        mean_gt_ratio=("window_gt_ratio", "mean"),
    ).reset_index()
    cat.to_csv(os.path.join(out_dir, f"category_scores_{name}.csv"), index=False)
    print(f"\nGLOBAL {name.upper()} SCORES")
    print(g.to_string(index=False))
    print(f"\nCATEGORY {name.upper()} SCORES")
    print(cat.sort_values("mean_event_f1", ascending=False).to_string(index=False))


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/kaggle/input/datasets/mariapreda/nab-data/data")
    parser.add_argument("--labels_path", type=str, default="/kaggle/input/datasets/mariapreda/nab-labels/labels")
    parser.add_argument("--out_dir", type=str, default="./semantic_supervised_stratified_autotune_results")
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--windows", type=parse_int_list, default=parse_int_list("8,16,32,64,128,256"))
    parser.add_argument("--seasonal_periods", type=parse_int_list, default=parse_int_list("24,48,96,288,1440"))
    parser.add_argument("--train_ratio", type=float, default=0.15)
    parser.add_argument("--no_label_clean_train", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--no_log_preprocess", action="store_true")
    parser.add_argument("--layer_smooth", type=float, default=1.0)
    parser.add_argument("--score_smooth", type=float, default=2.0)

    parser.add_argument("--aggregator", type=str, default="extra", choices=["logreg", "rf", "extra", "hgb"])
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--logreg_C", type=float, default=1.0)
    parser.add_argument("--max_points_per_series", type=int, default=5000)
    parser.add_argument("--neg_pos_ratio", type=float, default=3.0)

    parser.add_argument("--threshold_metric", type=str, default="balanced", choices=["balanced", "event_f1", "soft_event_f1", "nab_like", "window_f1"])
    parser.add_argument("--thr_q_min", type=float, default=0.70)
    parser.add_argument("--thr_q_max", type=float, default=0.995)
    parser.add_argument("--thr_steps", type=int, default=60)
    parser.add_argument("--verbose_thresholds", action="store_true")

    parser.add_argument("--min_segment", type=int, default=8)
    parser.add_argument("--close_gap", type=int, default=48)
    parser.add_argument("--expand_radius", type=int, default=32)
    parser.add_argument("--min_pred_ratio", type=float, default=0.0005)
    parser.add_argument("--max_pred_ratio", type=float, default=0.20)
    parser.add_argument("--nab_fp_weight", type=float, default=0.33)
    parser.add_argument("--nab_fn_weight", type=float, default=1.0)

    parser.add_argument("--auto_tune_postprocess", action="store_true", default=True)
    parser.add_argument("--no_auto_tune_postprocess", dest="auto_tune_postprocess", action="store_false")
    parser.add_argument("--autotune_max_combinations", type=int, default=48,
                        help="Max random grid combinations to test. Use 0 for full grid.")
    parser.add_argument("--grid_min_segment", type=str, default="8,16")
    parser.add_argument("--grid_close_gap", type=str, default="16,32,48")
    parser.add_argument("--grid_expand_radius", type=str, default="0,16,32")
    parser.add_argument("--grid_max_pred_ratio", type=str, default="0.08,0.12,0.16,0.20")
    parser.add_argument("--grid_min_pred_ratio", type=str, default="0.0,0.0005")
    parser.add_argument("--grid_nab_fp_weight", type=str, default="0.33,0.50,0.75")

    parser.add_argument("--obj_event_weight", type=float, default=0.35)
    parser.add_argument("--obj_soft_weight", type=float, default=0.25)
    parser.add_argument("--obj_nab_weight", type=float, default=0.30)
    parser.add_argument("--obj_window_weight", type=float, default=0.10)
    parser.add_argument("--obj_pred_ratio_penalty", type=float, default=0.25)
    parser.add_argument("--target_max_pred_ratio", type=float, default=0.15)
    return parser


def main(args=None):
    parser = build_argparser()
    if args is None:
        args, _ = parser.parse_known_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    labels_all = load_label_json(args.labels_path, "combined_labels.json")
    windows_all = load_label_json(args.labels_path, "combined_windows.json")
    keys = [k for k in list_series(args.data_path) if k in labels_all and k in windows_all]
    split_df = make_stratified_series_split(keys, windows_all, labels_all, args.test_size, args.seed)
    split_df.to_csv(os.path.join(args.out_dir, "split_series.csv"), index=False)

    print("DATA PATH:", args.data_path)
    print("LABELS PATH:", args.labels_path)
    print("SERIES COUNT:", len(keys))
    print("AGGREGATOR:", args.aggregator)
    print("SPLIT:")
    print(split_df.groupby(["split", "category", "has_anom"]).size().to_string())

    infos = []
    for _, r in split_df.iterrows():
        try:
            info = build_response_table(args, r["series"], labels_all, windows_all, r["split"])
            infos.append(info)
            print(f"built {r['split']:5s} | {r['series']} | n={len(info['df'])} | gt={info['y_window'].mean():.4f}", flush=True)
        except Exception as e:
            print(f"[SKIP/ERROR] {r['series']}: {e}")

    train_infos = [i for i in infos if i["split"] == "train"]
    test_infos = [i for i in infos if i["split"] == "test"]

    X_train, y_train, groups = sample_training_points(
        train_infos,
        max_points_per_series=args.max_points_per_series,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=args.seed,
    )
    print("\nTRAIN AGGREGATOR DATA:", X_train.shape, "positive ratio:", y_train.mean())

    model = make_model(args)
    if args.aggregator == "hgb":
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)

    scores = {info["series"]: model_scores(model, info["X"]) for info in infos}

    tuned_args, thr, selected = tune_postprocess_on_train(train_infos, scores, args)
    with open(os.path.join(args.out_dir, "selected_hyperparams.json"), "w") as f:
        json.dump(selected, f, indent=2)

    rows_train, rows_test = [], []
    for info in infos:
        score = scores[info["series"]]
        pred = postprocess_score(score, thr, tuned_args)
        row = evaluate_info(info, score, pred, tuned_args, args.out_dir)
        fmt = lambda d: {k: round(v, 4) if isinstance(v, float) and not np.isnan(v) else v for k, v in d.items()}
        print(f"\n=== {info['split'].upper()} | {info['series']} ===")
        print(f"n={len(info['df'])} | gt_ratio={info['y_window'].mean():.4f} | pred_ratio={pred.mean():.4f}")
        print("WINDOW:", fmt({k.replace('window_', ''): v for k, v in row.items() if k.startswith('window_')}))
        print("EVENT:", fmt({k: row[k] for k in ['event_precision','event_recall','event_f1','event_tp','event_fp','event_fn']}))
        print("SOFT:", fmt({k: row[k] for k in ['soft_event_precision','soft_event_recall','soft_event_f1']}))
        print("NAB-like:", fmt({k: row[k] for k in ['nab_like_score','nab_like_gt_events','nab_like_pred_events','nab_like_tp_events','nab_like_fp_events','nab_like_fn_events']}))
        if info["split"] == "train":
            rows_train.append(row)
        else:
            rows_test.append(row)

    train_summary = pd.DataFrame(rows_train)
    test_summary = pd.DataFrame(rows_test)
    train_summary.to_csv(os.path.join(args.out_dir, "summary_train.csv"), index=False)
    test_summary.to_csv(os.path.join(args.out_dir, "summary_test.csv"), index=False)

    global_report(train_summary, args.out_dir, "train")
    global_report(test_summary, args.out_dir, "test")

    print("\nSaved outputs in:", args.out_dir)


if __name__ == "__main__":
    import sys
    sys.argv = [
        "semantic_multilayer_supervised_stratified_AUTOTUNE_nab.py",
        "--data_path", "/kaggle/input/datasets/mariapreda/nab-data/data",
        "--labels_path", "/kaggle/input/datasets/mariapreda/nab-labels/labels",
        "--out_dir", "./semantic_supervised_stratified_autotune_results",
        "--test_size", "0.30",
        "--aggregator", "extra",
        "--threshold_metric", "balanced",
        "--auto_tune_postprocess",
        "--grid_min_segment", "8,16",
        "--grid_close_gap", "16,32,48",
        "--grid_expand_radius", "0,16,32",
        "--grid_max_pred_ratio", "0.08,0.12,0.16,0.20",
        "--grid_min_pred_ratio", "0.0,0.0005",
        "--grid_nab_fp_weight", "0.33,0.50,0.75",
        "--autotune_max_combinations", "48",
        "--thr_steps", "60",
    ]
    main()