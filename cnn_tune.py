"""
sEMG 1D-CNN 超参数自动搜索（segment_id + train-only z-score 版）
================================================
输入：timeseries_all_sets_unified.csv（预处理后，未 z-score，且包含 segment_id）
方法：Optuna 贝叶斯优化，自动搜索最佳 CNN 结构和训练参数

本版修正：
  1. 滑窗按 set_id + segment_id 分组，避免跨断点假窗口
  2. z-score 挪到 CNN 内部，只用训练集拟合
  3. 验证集 / 测试集严格使用训练集统计量
  4. 多 set 时预留严格独立测试集
  5. LOSO 下 Optuna pruning 使用唯一 step
  6. final_train 避免同一 batch 双重前向
  7. 增加空集、列名、标签范围检查

安装依赖：
  pip install optuna torch scikit-learn pandas numpy

使用：
  python cnn_tune.py
  python cnn_tune.py --trials 100
  python cnn_tune.py --trials 50 --epochs 40
================================================
"""

import time
import argparse
import json
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")

# =================配置区域=================
CSV_FILE = "timeseries_all_sets_unified_with_ch4_current.csv"

WINDOW_SIZE = 300          # 100 ms @ 2000 Hz
STRIDE = WINDOW_SIZE // 4   # = 75
MIN_LABEL_PURITY = 0.90
# 使用全部 4 通道（Raw + Env）
CHANNELS = ["CH1", "CH2", "CH3", "CH4_Band_Current"]

N_CLASSES = 7
RANDOM_STATE = 42
VAL_CHECK_EVERY = 5

GESTURE_NAMES = {
    0: "静息",
    1: "握拳",
    2: "伸掌",
    3: "内屈",
    4: "外伸",
    5: "内旋",
    6: "外旋",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ========================================


# ========================================
# 工具函数
# ========================================
def set_seed(seed: int = RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_py_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def validate_dataframe(df: pd.DataFrame):
    required_cols = ["set_id", "segment_id", "Label"] + CHANNELS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    if df.empty:
        raise ValueError("CSV 为空，无法训练")

    labels = df["Label"].dropna().astype(int).unique().tolist()
    invalid = sorted(set(labels) - set(range(N_CLASSES)))
    if invalid:
        raise ValueError(
            f"发现非法标签 {invalid}。请先确保 Label 只包含 0 到 {N_CLASSES - 1}"
        )

    bad_numeric = []
    for c in CHANNELS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            bad_numeric.append(c)
    if bad_numeric:
        raise ValueError(f"以下通道列不是数值类型: {bad_numeric}")


def majority_label_and_purity(lbl_seg: np.ndarray):
    """
    返回:
      maj_label: 窗口内众数标签
      purity   : 众数标签占比, 范围 0 到 1
    """
    counts = np.bincount(lbl_seg, minlength=N_CLASSES)
    maj = int(np.argmax(counts))
    purity = float(counts[maj] / max(len(lbl_seg), 1))
    return maj, purity
def split_time_order(X, y, ratio=0.8):
    """
    单个 set 场景，按时间顺序切分
    """
    if len(y) < 2:
        raise ValueError("窗口数量太少，无法做时间切分")

    split = int(len(y) * ratio)
    if split <= 0 or split >= len(y):
        raise ValueError(
            f"时间切分失败。总窗口数={len(y)}，切分点={split}。请增加数据量或减小 WINDOW_SIZE"
        )

    return X[:split], y[:split], X[split:], y[split:]


def fit_channel_zscore(X: np.ndarray):
    """
    对窗口数据按通道拟合 z-score 参数
    X 形状: (N, T, C)
    对 (N, T) 两个维度求统计量，得到每个通道一个 mean/std
    """
    if X.ndim != 3:
        raise ValueError(f"X 维度错误，应为 3 维，实际为 {X.ndim}")

    mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)  # (1,1,C)
    std = X.std(axis=(0, 1), keepdims=True).astype(np.float32)    # (1,1,C)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_channel_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    使用给定通道统计量对窗口数据做 z-score
    """
    return ((X.astype(np.float32) - mean) / std).astype(np.float32)


def standardise_by_train(X_tr: np.ndarray, X_other: np.ndarray):
    """
    只用训练集拟合 z-score，并应用到训练集与另一个数据集
    返回:
      X_tr_z, X_other_z, mean, std
    """
    mean, std = fit_channel_zscore(X_tr)
    X_tr_z = apply_channel_zscore(X_tr, mean, std)
    X_other_z = apply_channel_zscore(X_other, mean, std)
    return X_tr_z, X_other_z, mean, std


# ========================================
# 1. 滑动窗口构建数据集
# ========================================
def build_windows(df: pd.DataFrame):
    """
    按 (set_id, segment_id) 分组滑窗，返回：
      X      : (N, WINDOW_SIZE, n_channels)
      y      : (N,)
      groups : (N,) 对应的 set_id，用于 LOSO
    """
    n_dropped_impure = 0
    X, y, groups = [], [], []

    n_dropped_nan = 0
    n_skipped_short_segments = 0

    # 关键修复：必须按 set_id + segment_id 分组
    for (set_id, segment_id), group in df.groupby(["set_id", "segment_id"], sort=True):
        vals = group[CHANNELS].to_numpy(dtype=np.float32)   # (T, C)
        labels = group["Label"].to_numpy(dtype=np.int64)    # (T,)
        n = len(group)

        if n < WINDOW_SIZE:
            print(
                f"  跳过 set {set_id}, segment {segment_id}，"
                f"长度 {n} < WINDOW_SIZE {WINDOW_SIZE}"
            )
            n_skipped_short_segments += 1
            continue

        # off-by-one 修复
        for i in range(0, n - WINDOW_SIZE + 1, STRIDE):
            seg = vals[i:i + WINDOW_SIZE]
            lbl_seg = labels[i:i + WINDOW_SIZE]

            if np.isnan(seg).any():
                n_dropped_nan += 1
                continue

            seg_label, purity = majority_label_and_purity(lbl_seg)

            if purity < MIN_LABEL_PURITY:
                n_dropped_impure += 1
                continue

            X.append(seg)
            y.append(seg_label)
            groups.append(int(set_id))
    if len(X) == 0:
        raise ValueError(
            "没有生成任何窗口。请检查数据长度、WINDOW_SIZE、STRIDE，或是否有大量 NaN"
        )
    print(f"  总窗口: {len(X)}")
    print(f"  丢弃 NaN 窗口: {n_dropped_nan}")
    print(f"  丢弃低纯度窗口(<{MIN_LABEL_PURITY:.2f}): {n_dropped_impure}")
    print(f"  跳过过短 segment: {n_skipped_short_segments}")

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(groups, dtype=np.int64),
    )


def make_loaders(X_tr, y_tr, X_te, y_te, batch_size):
    """
    X: (N, T, C) -> (N, C, T)
    """
    if len(X_tr) == 0 or len(y_tr) == 0:
        raise ValueError("训练集为空")
    if len(X_te) == 0 or len(y_te) == 0:
        raise ValueError("验证集或测试集为空")

    def to_tensor(X, y):
        Xt = torch.from_numpy(X.transpose(0, 2, 1)).float()   # (N, C, T)
        yt = torch.from_numpy(y.astype(np.int64))
        return TensorDataset(Xt, yt)

    tr_loader = DataLoader(
        to_tensor(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    te_loader = DataLoader(
        to_tensor(X_te, y_te),
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return tr_loader, te_loader


# ========================================
# 2. 动态 CNN 模型
# ========================================
class DynamicEMGNet(nn.Module):
    """
    可配置的 1D-CNN：
      输入 (B, C, T)
      N 层 Conv1d -> BN -> ReLU -> MaxPool(2)
      AdaptiveAvgPool -> Flatten
      FC -> ReLU -> Dropout -> FC(n_classes)
    """

    def __init__(self, n_channels, n_classes, params: dict):
        super().__init__()

        n_layers = params["n_layers"]
        filters = params["filters"]
        kernel = params["kernel"]
        fc_hidden = params["fc_hidden"]
        dropout1 = params["dropout1"]
        dropout2 = params["dropout2"]

        layers = []
        in_ch = n_channels

        for i in range(n_layers):
            out_ch = filters[i]
            pad = kernel // 2

            layers += [
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel,
                    padding=pad,
                    bias=False,
                ),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x


def expand_params(best_params: dict) -> dict:
    """
    由 Optuna 的 best_params 重建完整参数
    """
    n_layers = int(best_params["n_layers"])
    filter_base = int(best_params["filter_base"])

    filters = []
    f = filter_base
    for _ in range(n_layers):
        filters.append(f)
        f = min(f * 2, 256)

    return {
        "n_layers": n_layers,
        "filters": filters,
        "kernel": int(best_params["kernel"]),
        "fc_hidden": int(best_params["fc_hidden"]),
        "dropout1": float(best_params["dropout1"]),
        "dropout2": float(best_params["dropout2"]),
        "lr": float(best_params["lr"]),
        "weight_decay": float(best_params["weight_decay"]),
        "batch_size": int(best_params["batch_size"]),
    }


# ========================================
# 3. 训练与评估
# ========================================
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        logits = model(xb)
        pred = logits.argmax(dim=1)

        correct += (pred == yb).sum().item()
        total += len(yb)

    if total == 0:
        raise ValueError("评估集为空，无法计算准确率")

    return correct / total


def train_eval(X_tr, y_tr, X_va, y_va, params, n_epochs, trial=None, report_offset=0):
    """
    用给定 params 训练一个 CNN，返回验证集最佳准确率
    这里的 z-score 只用 X_tr 拟合，再应用到 X_tr / X_va
    """
    set_seed(RANDOM_STATE)

    # 关键修复：只用训练集拟合 z-score
    X_tr_z, X_va_z, _, _ = standardise_by_train(X_tr, X_va)

    loader_tr, loader_va = make_loaders(
        X_tr_z, y_tr, X_va_z, y_va, params["batch_size"]
    )

    model = DynamicEMGNet(
        n_channels=len(CHANNELS),
        n_classes=N_CLASSES,
        params=params,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6,
    )

    best_acc = 0.0

    for ep in range(n_epochs):
        model.train()

        for xb, yb in loader_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

        scheduler.step()

        if (ep + 1) % VAL_CHECK_EVERY == 0 or ep == n_epochs - 1:
            acc = evaluate_accuracy(model, loader_va)
            best_acc = max(best_acc, acc)

            if trial is not None:
                step = report_offset + ep + 1
                trial.report(acc, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_acc


# ========================================
# 4. Optuna 目标函数
# ========================================
def make_objective(X, y, groups, n_epochs):
    """
    对调参集构建 objective
      内层如果是多 set -> LOSO
      内层如果只有 1 个 set -> 时间顺序 80/20
    """
    inner_set_ids = np.unique(groups)
    inner_multi = len(inner_set_ids) > 1

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 4)
        filter_base = trial.suggest_categorical("filter_base", [16, 32, 64])

        filters = []
        f = filter_base
        for _ in range(n_layers):
            filters.append(f)
            f = min(f * 2, 256)

        params = {
            "n_layers": n_layers,
            "filters": filters,
            "kernel": trial.suggest_categorical("kernel", [3, 5, 7, 11]),
            "fc_hidden": trial.suggest_categorical("fc_hidden", [64, 128, 256]),
            "dropout1": trial.suggest_float("dropout1", 0.2, 0.6, step=0.1),
            "dropout2": trial.suggest_float("dropout2", 0.1, 0.4, step=0.1),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        }

        if inner_multi:
            fold_accs = []

            for fold_idx, val_sid in enumerate(inner_set_ids):
                mask_va = (groups == val_sid)
                mask_tr = ~mask_va

                if not mask_va.any():
                    raise ValueError(f"fold 验证集为空，set_id={val_sid}")
                if not mask_tr.any():
                    raise ValueError(f"fold 训练集为空，set_id={val_sid}")

                acc = train_eval(
                    X[mask_tr], y[mask_tr],
                    X[mask_va], y[mask_va],
                    params=params,
                    n_epochs=n_epochs,
                    trial=trial,
                    report_offset=fold_idx * n_epochs,
                )
                fold_accs.append(acc)

            return float(np.mean(fold_accs))

        else:
            X_tr, y_tr, X_va, y_va = split_time_order(X, y, ratio=0.8)

            return train_eval(
                X_tr, y_tr,
                X_va, y_va,
                params=params,
                n_epochs=n_epochs,
                trial=trial,
                report_offset=0,
            )

    return objective


# ========================================
# 5. 最终训练
# ========================================
def final_train(
    X_tr, y_tr, X_te, y_te,
    best_params,
    n_epochs,
    save_path="model_cnn_tuned.pt",
    extra_meta=None
):
    """
    用最佳超参数完整训练，打印详细结果并保存模型
    这里会只用训练集拟合 z-score，并把统计量保存到模型里
    """
    set_seed(RANDOM_STATE)

    params = expand_params(best_params)

    # 关键修复：只用训练集拟合 z-score
    X_tr_z, X_te_z, z_mean, z_std = standardise_by_train(X_tr, X_te)

    loader_tr, loader_te = make_loaders(
        X_tr_z, y_tr, X_te_z, y_te, params["batch_size"]
    )

    model = DynamicEMGNet(
        n_channels=len(CHANNELS),
        n_classes=N_CLASSES,
        params=params,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6,
    )

    best_acc = -1.0
    best_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    print(f"\n最终训练 ({n_epochs} epochs) ...")
    print("按训练集拟合 z-score，并应用到训练集 / 测试集")

    for ep in range(n_epochs):
        model.train()

        ep_loss = 0.0
        n_correct = 0
        n_total = 0

        for xb, yb in loader_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # 同一 batch 只前向一次
            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            ep_loss += loss.item() * len(yb)
            n_correct += (logits.detach().argmax(dim=1) == yb).sum().item()
            n_total += len(yb)

        scheduler.step()

        if (ep + 1) % 10 == 0 or ep == n_epochs - 1:
            train_acc = n_correct / max(n_total, 1)

            print(
                f"  Epoch {ep + 1:>3}/{n_epochs}  "
                f"loss={ep_loss / max(n_total, 1):.4f}  "
                f"train={train_acc:.4f}"
            )

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_true = [], []

    with torch.no_grad():
        for xb, yb in loader_te:
            xb = xb.to(DEVICE, non_blocking=True)

            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(pred)
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    final_acc = accuracy_score(y_true, y_pred)

    print(f"\n{'=' * 55}")
    print(f"  最终测试准确率: {final_acc:.4f}")
    print(f"{'=' * 55}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(N_CLASSES)),
            target_names=[GESTURE_NAMES[i] for i in range(N_CLASSES)],
            digits=4,
            zero_division=0,
        )
    )

    save_payload = {
        "model_state_dict": best_state,
        "model_params": params,
        "n_channels": len(CHANNELS),
        "n_classes": N_CLASSES,
        "channels": CHANNELS,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "gesture_names": GESTURE_NAMES,
        "best_val_acc": float(best_acc),
        "best_optuna_params": best_params,
        "min_label_purity": MIN_LABEL_PURITY,
        # 保存 z-score 参数，后续推理必须用
        "zscore_mean": z_mean.reshape(-1).astype(np.float32).tolist(),
        "zscore_std": z_std.reshape(-1).astype(np.float32).tolist(),
    }

    if extra_meta is not None:
        save_payload.update(extra_meta)

    torch.save(save_payload, save_path)
    print(f"模型已保存: {save_path}")

    return final_acc


# ========================================
# 6. 主流程
# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMG CNN 超参数搜索（segment_id + train-only z-score 版）")
    parser.add_argument(
        "--trials",
        default=50,
        type=int,
        help="Optuna 试验次数（默认 50）",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        help="每次试验的训练 epoch 数（默认 30）",
    )
    parser.add_argument(
        "--final-epochs",
        default=80,
        type=int,
        help="最终训练 epoch 数（默认 80）",
    )
    parser.add_argument(
        "--csv",
        default=CSV_FILE,
        help="预处理后的 CSV 文件路径",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Optuna 试验次数: {args.trials}，每次 {args.epochs} epochs")
    print(f"最终训练: {args.final_epochs} epochs\n")

    # 1) 读取数据
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    validate_dataframe(df)

    print(f"总行数: {len(df):,}")
    print(f"set 数量: {df['set_id'].nunique()}")
    print(f"segment 数量: {df[['set_id', 'segment_id']].drop_duplicates().shape[0]}")

    # 2) 滑窗
    print("\nBuilding windows ...")
    X, y, groups = build_windows(df)

    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(
        "各手势窗口数:",
        {
            GESTURE_NAMES[int(k)]: int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        }
    )

    set_ids = np.unique(groups)
    n_sets = len(set_ids)

    # 3) 外层严格测试集切分
    if n_sets >= 2:
        holdout_sid = set_ids[-1]

        mask_tune = (groups != holdout_sid)
        mask_test = (groups == holdout_sid)

        if not mask_tune.any():
            raise ValueError("调参集为空")
        if not mask_test.any():
            raise ValueError("独立测试集为空")

        X_tune, y_tune, g_tune = X[mask_tune], y[mask_tune], groups[mask_tune]
        X_test, y_test = X[mask_test], y[mask_test]

        inner_set_ids = np.unique(g_tune)

        print(f"\n检测到 {n_sets} 个 set")
        print(f"外层严格测试集: set {holdout_sid}")
        print(f"调参集 set: {list(inner_set_ids)}")

        if len(inner_set_ids) > 1:
            print("内层调参策略: Leave-One-Set-Out")
        else:
            print("内层调参策略: 单 set 时间顺序 80/20")

    else:
        holdout_sid = None
        X_tune, y_tune, g_tune = X, y, groups
        X_test, y_test = None, None

        print("\n仅检测到 1 个 set")
        print("无法预留严格独立测试集")
        print("调参与最终评估都将使用时间顺序 80/20")

    # 4) Optuna 搜索
    print(f"\n{'=' * 55}")
    print("  开始 Optuna 超参数搜索")
    print(f"{'=' * 55}")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(10, args.trials),
        n_warmup_steps=max(5, VAL_CHECK_EVERY),
    )
    sampler = TPESampler(seed=RANDOM_STATE)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(
        X_tune,
        y_tune,
        g_tune,
        n_epochs=args.epochs,
    )

    t0 = time.time()
    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if len(complete_trials) == 0:
        raise RuntimeError("所有试验都被剪枝或失败，没有可用的 best_params")

    print(f"\n{'=' * 55}")
    print(f"  搜索完成  耗时: {elapsed / 60:.1f} 分钟")
    print(f"{'=' * 55}")
    print(f"  最佳验证准确率: {study.best_value:.4f}")
    print("  最佳超参数:")
    for k, v in study.best_params.items():
        print(f"    {k:<20}: {v}")

    result_summary = {
        "best_val_acc": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": int(args.trials),
        "search_epochs": int(args.epochs),
        "elapsed_min": round(elapsed / 60, 2),
        "n_total_sets": int(n_sets),
        "holdout_set_id": None if holdout_sid is None else to_py_scalar(holdout_sid),
    }

    with open("cnn_tune_results.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)

    print("\n搜索结果已保存: cnn_tune_results.json")

    # 5) 最终训练
    print(f"\n{'=' * 55}")
    print(f"  用最佳参数进行最终训练 ({args.final_epochs} epochs)")
    print(f"{'=' * 55}")

    if n_sets >= 2:
        X_tr, y_tr = X_tune, y_tune
        X_te, y_te = X_test, y_test

        print(f"最终训练集: {len(y_tr)} 个窗口")
        print(f"最终测试集: set {holdout_sid}，共 {len(y_te)} 个窗口")

        extra_meta = {
            "outer_holdout_set_id": to_py_scalar(holdout_sid),
            "tuning_set_ids": [to_py_scalar(s) for s in np.unique(g_tune)],
        }

    else:
        X_tr, y_tr, X_te, y_te = split_time_order(X_tune, y_tune, ratio=0.8)

        print(f"最终切分: 前 {len(y_tr)} 训练，后 {len(y_te)} 测试")

        extra_meta = {
            "outer_holdout_set_id": None,
            "tuning_set_ids": [to_py_scalar(s) for s in np.unique(g_tune)],
        }

    final_train(
        X_tr, y_tr,
        X_te, y_te,
        best_params=study.best_params,
        n_epochs=args.final_epochs,
        save_path="model_cnn_tuned.pt",
        extra_meta=extra_meta,
    )

    # 6) Top 5 试验
    print(f"\n{'=' * 55}")
    print("  Top 5 试验结果")
    print(f"{'=' * 55}")

    top5 = sorted(
        complete_trials,
        key=lambda t: t.value,
        reverse=True,
    )[:5]

    for i, t in enumerate(top5, start=1):
        print(
            f"  #{i}  "
            f"val_acc={t.value:.4f}  "
            f"layers={t.params.get('n_layers')}  "
            f"filter_base={t.params.get('filter_base')}  "
            f"kernel={t.params.get('kernel')}  "
            f"lr={t.params.get('lr', 0):.2e}"
        )