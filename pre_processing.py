import os
import glob
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal

# =========================================================
# 全通道预处理 ── 修复版
# 改动摘要（相对原版）：
#   FIX-1  segment 切分同时感知 Label 跳变，避免跨手势窗口污染
#   FIX-2  过滤 Label 后再切 segment（顺序很重要）
#   FIX-3  输出每个 segment 的点数统计，方便决定窗口大小上限
# =========================================================

FS               = 2000.0
ADC_MID          = 2048.0
VALID_LABELS     = {0, 1, 2, 3, 4, 5, 6}
GAP_THRESHOLD_MS = 100
ENVELOPE_CUTOFF_HZ = 10.0

CH4_SPECIAL_RULE_LAST_SET = 16
TIME_COL      = "Time_ms"
LABEL_COL     = "Label"
CH123_COLS    = ["CH1", "CH2", "CH3"]
CH4_COL       = "CH4"
CH123_ENV_COLS = ["CH1_Env", "CH2_Env", "CH3_Env"]
CH4_BAND_COL  = "CH4_Band_Current"
CH4_ENV_COL   = "CH4_Env_Current"

DATA_DIR     = "."
FILE_PATTERN = "EMG_Dataset_V9*.csv"
OUT_CSV      = "timeseries_all_sets_unified_with_ch4_current.csv"

FILTER_MODE  = "range"
SET_RANGE    = (1, 16)
SET_ID_LIST  = {1, 3, 5, 8}

CH4_INPUT_MODE  = "old_fw_filtered_20_400"
ENVELOPE_MODE   = "ac_centered"

GESTURE_NAMES = {
    0: "静息 (Static Rest)",
    1: "握拳 (Fist)",
    2: "伸掌 (Spread)",
    3: "内屈 (Flexion)",
    4: "外伸 (Extension)",
    5: "内旋 (Pronation)",
    6: "外旋 (Supination)",
}

HP20_B     = np.array([0.9565432, -1.9130865, 0.9565432], dtype=float)
HP20_A     = np.array([1.0, -1.9111971, 0.9149758],       dtype=float)
NOTCH50_B  = np.array([0.9845975, -1.9449509, 0.9845975], dtype=float)
NOTCH50_A  = np.array([1.0, -1.9449509, 0.9691950],       dtype=float)
LP250_S1_B = np.array([0.01020948, 0.02041896, 0.01020948], dtype=float)
LP250_S1_A = np.array([1.0, -0.85539793, 0.20971536],       dtype=float)
LP250_S2_B = np.array([1.0, 2.0, 1.0],                      dtype=float)
LP250_S2_A = np.array([1.0, -1.11302985, 0.57406192],        dtype=float)


# =========================
# 工具函数
# =========================

def compute_envelope(signal: np.ndarray, fs: float = FS,
                     cutoff: float = ENVELOPE_CUTOFF_HZ) -> np.ndarray:
    x_abs = np.abs(signal.astype(float))
    nyq   = fs / 2.0
    wn    = cutoff / nyq
    if not (0.0 < wn < 1.0):
        raise ValueError(f"包络截止频率非法: cutoff={cutoff}, fs={fs}")
    b, a = scipy.signal.butter(4, wn, btype="low")
    return scipy.signal.filtfilt(b, a, x_abs)


def extract_set_num(path: str) -> Optional[int]:
    m = re.search(r"_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def is_selected_set(set_num: int) -> bool:
    if FILTER_MODE == "all":
        return True
    if FILTER_MODE == "range":
        return SET_RANGE[0] <= set_num <= SET_RANGE[1]
    if FILTER_MODE == "list":
        return set_num in SET_ID_LIST
    raise ValueError(f"未知 FILTER_MODE: {FILTER_MODE}")


# ─────────────────────────────────────────────────────────
# FIX-1  同时按时间跳变 + Label 变化切 segment
# 原版只按时间跳变，导致同一 segment 内可能横跨多个手势
# ─────────────────────────────────────────────────────────
def assign_segment_ids(time_ms: np.ndarray,
                       labels:   np.ndarray,
                       gap_threshold_ms: float = GAP_THRESHOLD_MS) -> np.ndarray:
    """
    以下任一条件发生时开启新 segment：
      1. 相邻时间差 > gap_threshold_ms（采集中断）
      2. Label 发生变化（手势切换）
    保证每个 segment 内 Label 单一，滑窗不会横跨手势边界。
    """
    if len(time_ms) == 0:
        return np.array([], dtype=np.int64)

    dt           = np.diff(time_ms.astype(float))
    label_change = np.diff(labels.astype(int)) != 0          # FIX-1 新增
    breaks       = (dt > gap_threshold_ms) | label_change    # FIX-1 新增

    segment_id = np.ones(len(time_ms), dtype=np.int64)
    if len(time_ms) > 1:
        segment_id[1:] += np.cumsum(breaks).astype(np.int64)

    return segment_id


def compute_envelope_by_segments(signal: np.ndarray,
                                  segment_ids: np.ndarray,
                                  fs: float = FS,
                                  envelope_mode: str = ENVELOPE_MODE) -> np.ndarray:
    env         = np.zeros_like(signal, dtype=float)
    min_seg_len = int(fs * 0.05)

    for sid in np.unique(segment_ids):
        idx     = np.where(segment_ids == sid)[0]
        seg_sig = signal[idx].astype(float)

        if envelope_mode == "ac_centered":
            env_in = seg_sig - ADC_MID
        elif envelope_mode == "legacy_direct_abs":
            env_in = seg_sig
        else:
            raise ValueError(f"未知 ENVELOPE_MODE: {envelope_mode}")

        env[idx] = np.abs(env_in) if len(env_in) < min_seg_len \
                   else compute_envelope(env_in, fs=fs)

    return env


def biquad_process_segment(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros_like(x, dtype=float)
    zi    = scipy.signal.lfilter_zi(b, a) * x[0]
    y, _  = scipy.signal.lfilter(b, a, x, zi=zi)
    return y


def apply_current_ch4_from_true_raw_by_segments(ch4_raw_adc: np.ndarray,
                                                 segment_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(ch4_raw_adc, dtype=float)
    for sid in np.unique(segment_ids):
        idx = np.where(segment_ids == sid)[0]
        x   = ch4_raw_adc[idx].astype(float) - ADC_MID
        x   = biquad_process_segment(x, HP20_B,     HP20_A)
        x   = biquad_process_segment(x, NOTCH50_B,  NOTCH50_A)
        x   = biquad_process_segment(x, LP250_S1_B, LP250_S1_A)
        x   = biquad_process_segment(x, LP250_S2_B, LP250_S2_A)
        out[idx] = np.clip(x + ADC_MID, 0.0, 4095.0)
    return out


def apply_current_ch4_from_old_fw_by_segments(ch4_old_adc: np.ndarray,
                                               segment_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(ch4_old_adc, dtype=float)
    for sid in np.unique(segment_ids):
        idx = np.where(segment_ids == sid)[0]
        x   = ch4_old_adc[idx].astype(float) - ADC_MID
        x   = biquad_process_segment(x, LP250_S1_B, LP250_S1_A)
        x   = biquad_process_segment(x, LP250_S2_B, LP250_S2_A)
        out[idx] = np.clip(x + ADC_MID, 0.0, 4095.0)
    return out


# =========================
# 单文件处理
# =========================
def preprocess_one_file(csv_path: str, set_id: int) -> Tuple[Optional[pd.DataFrame], dict]:
    print(f"处理: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[错误] 无法读取文件: {e}")
        return None, {}

    required = [TIME_COL, LABEL_COL] + CH123_COLS + [CH4_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[错误] 缺少必要列: {missing}")
        return None, {}

    print(f"  原始行数: {len(df):,}")
    print(f"  原始标签分布: {dict(df[LABEL_COL].value_counts().sort_index())}")

    # ─────────────────────────────────────────────────────
    # FIX-2  先过滤 Label，再切 segment
    # 原版顺序反了：若先切 segment 再过滤，静息段和手势段
    # 可能共享同一 segment_id，导致后续滑窗跨越 Label 边界
    # ─────────────────────────────────────────────────────
    df = df[df[LABEL_COL].isin(VALID_LABELS)].copy()
    df.reset_index(drop=True, inplace=True)

    print(f"  过滤后行数: {len(df):,}")
    if df.empty:
        print("  [警告] 过滤后数据为空，跳过")
        return None, {}

    # FIX-1 + FIX-2：用新函数切 segment
    df["segment_id"] = assign_segment_ids(
        df[TIME_COL].to_numpy(dtype=float),
        df[LABEL_COL].to_numpy(dtype=int),     # 新增 label 参数
        gap_threshold_ms=GAP_THRESHOLD_MS,
    )
    seg_ids = df["segment_id"].to_numpy(dtype=np.int64)

    # CH1 / CH2 / CH3 envelope
    for raw_col, env_col in zip(CH123_COLS, CH123_ENV_COLS):
        sig        = df[raw_col].to_numpy(dtype=float)
        df[env_col] = compute_envelope_by_segments(sig, seg_ids, fs=FS,
                                                   envelope_mode=ENVELOPE_MODE)

    # CH4
    ch4_sig = df[CH4_COL].to_numpy(dtype=float)
    print(f"  CH4处理模式: {'专属逻辑' if set_id <= CH4_SPECIAL_RULE_LAST_SET else '与CH1-CH3相同'}")

    if set_id <= CH4_SPECIAL_RULE_LAST_SET:
        if CH4_INPUT_MODE == "true_raw_adc":
            df[CH4_BAND_COL] = apply_current_ch4_from_true_raw_by_segments(ch4_sig, seg_ids)
        elif CH4_INPUT_MODE == "old_fw_filtered_20_400":
            df[CH4_BAND_COL] = apply_current_ch4_from_old_fw_by_segments(ch4_sig, seg_ids)
        else:
            raise ValueError(f"未知 CH4_INPUT_MODE: {CH4_INPUT_MODE}")
        df[CH4_ENV_COL] = compute_envelope_by_segments(
            df[CH4_BAND_COL].to_numpy(dtype=float), seg_ids,
            fs=FS, envelope_mode=ENVELOPE_MODE)
    else:
        df[CH4_BAND_COL] = ch4_sig
        df[CH4_ENV_COL]  = compute_envelope_by_segments(
            ch4_sig, seg_ids, fs=FS, envelope_mode=ENVELOPE_MODE)

    out_cols = [TIME_COL, "segment_id", LABEL_COL,
                *CH123_COLS, CH4_COL, CH4_BAND_COL,
                *CH123_ENV_COLS, CH4_ENV_COL]
    df = df[out_cols].copy()
    df.insert(0, "set_id", int(set_id))

    seg_lengths = df.groupby("segment_id").size()

    # ─────────────────────────────────────────────────────
    # FIX-3  打印每个 segment 的点数分位数
    # 用于判断安全的窗口大小上限
    # ─────────────────────────────────────────────────────
    pct = seg_lengths.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).astype(int)
    print(f"  分段数: {df['segment_id'].nunique()}")
    print(f"  segment 长度分位数（点数）:")
    print(f"    p10={pct[0.10]}  p25={pct[0.25]}  p50={pct[0.50]}"
          f"  p75={pct[0.75]}  p90={pct[0.90]}")
    print(f"  最短: {int(seg_lengths.min())}  最长: {int(seg_lengths.max())}")

    # 针对候选窗口大小，打印能贡献窗口的 segment 比例
    for ws in [200, 300, 400]:
        usable = (seg_lengths >= ws).sum()
        total  = len(seg_lengths)
        print(f"  window_size={ws}: {usable}/{total} 个 segment 能贡献至少 1 个窗口")

    print("  各手势样本数:")
    for lbl, cnt in df[LABEL_COL].value_counts().sort_index().items():
        print(f"    Label {lbl} ({GESTURE_NAMES.get(lbl, '?')}): {cnt:,} 行")

    info = {
        "n_rows":          int(len(df)),
        "n_segments":      int(df["segment_id"].nunique()),
        "min_segment_len": int(seg_lengths.min()),
        "max_segment_len": int(seg_lengths.max()),
        "p50_segment_len": int(pct[0.50]),
    }
    return df, info


# =========================
# 批量处理（不变）
# =========================
def run_all(data_dir: str = DATA_DIR, pattern: str = FILE_PATTERN,
            out_csv: str = OUT_CSV) -> Optional[pd.DataFrame]:
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        print(f"[错误] 在 {data_dir} 下未找到匹配 '{pattern}' 的文件")
        return None

    filtered_files, skipped_bad_name = [], []
    for f in files:
        set_num = extract_set_num(f)
        if set_num is None:
            skipped_bad_name.append(os.path.basename(f))
            continue
        if is_selected_set(set_num):
            filtered_files.append((set_num, f))

    filtered_files.sort(key=lambda x: x[0])

    if skipped_bad_name:
        print(f"[警告] 文件名格式不符，已跳过: {skipped_bad_name}")
    if not filtered_files:
        print("[错误] 没有符合筛选条件的文件")
        return None

    print(f"待处理文件数: {len(filtered_files)}\n")

    all_dfs, summary = [], {}
    for set_num, fpath in filtered_files:
        df, info = preprocess_one_file(fpath, set_num)
        if df is not None:
            all_dfs.append(df)
            summary[str(set_num)] = info
            print(f"  ✓ set {set_num} 完成\n")

    if not all_dfs:
        print("[错误] 没有成功处理任何文件")
        return None

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(out_csv, index=False)

    print("===== 最终汇总 =====")
    print(f"总行数: {len(final_df):,}")
    print(f"set 数量: {final_df['set_id'].nunique()}")
    seg_len_all = final_df.groupby(["set_id", "segment_id"]).size()
    print(f"segment 总数: {len(seg_len_all)}")
    print(f"全局 segment 长度中位数: {int(seg_len_all.median())} 点")
    print("各手势总行数:")
    for lbl, cnt in final_df[LABEL_COL].value_counts().sort_index().items():
        print(f"  Label {lbl} ({GESTURE_NAMES.get(lbl, '?')}): {cnt:,} 行")
    print(f"\n✓ 已保存: {out_csv}")
    return final_df


if __name__ == "__main__":
    run_all()
