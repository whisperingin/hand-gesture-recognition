"""
predict_interactive.py

流程：浏览器「开始」→ 采集 N 秒 → 推理 → 置信度 ≥ 阈值则输出，否则重试
HTTP API:  POST /api/control  {"cmd": "start" | "stop"}
静态文件:  GET  /hand_model.html, GET /gesture_state.json 等
"""

import argparse
import functools
import json
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import serial
import torch
import torch.nn as nn


# ================================================================
# 模型定义（与训练脚本一致）
# ================================================================
class DynamicEMGNet(nn.Module):
    def __init__(self, n_channels, n_classes, params):
        super().__init__()
        n_layers  = params["n_layers"]
        filters   = params["filters"]
        kernel    = params["kernel"]
        fc_hidden = params["fc_hidden"]
        dropout1  = params["dropout1"]
        dropout2  = params["dropout2"]

        layers, in_ch = [], n_channels
        for _ in range(n_layers):
            out_ch = filters[len(layers) // 4]
            pad    = kernel // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ]
            in_ch = out_ch

        self.features   = nn.Sequential(*layers)
        self.gap        = nn.AdaptiveAvgPool1d(1)
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
        return self.classifier(x)


def expand_params_from_optuna(best_params):
    n_layers = int(best_params["n_layers"])
    f        = int(best_params["filter_base"])
    filters  = []
    for _ in range(n_layers):
        filters.append(f)
        f = min(f * 2, 256)
    return {
        "n_layers":     n_layers,
        "filters":      filters,
        "kernel":       int(best_params["kernel"]),
        "fc_hidden":    int(best_params["fc_hidden"]),
        "dropout1":     float(best_params["dropout1"]),
        "dropout2":     float(best_params["dropout2"]),
        "lr":           float(best_params["lr"]),
        "weight_decay": float(best_params["weight_decay"]),
        "batch_size":   int(best_params["batch_size"]),
    }


# ================================================================
# 工具函数
# ================================================================
def normalise_gesture_names(raw_map):
    out = {}
    if not isinstance(raw_map, dict):
        return out
    for k, v in raw_map.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out


def atomic_write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_csv4(line: str):
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 4:
        return None
    try:
        return np.asarray([float(parts[i]) for i in range(4)], dtype=np.float32)
    except ValueError:
        return None


def build_raw_feature_window(raw_arr, channel_names):
    mapping = {f"CH{i+1}": raw_arr[:, i] for i in range(4)}
    invalid = [n for n in channel_names if n not in mapping]
    if invalid:
        raise KeyError(f"不支持的通道列: {invalid}")
    return np.stack([mapping[n] for n in channel_names], axis=1).astype(np.float32)


# ================================================================
# 推理控制器
# ================================================================
class InferenceController:
    def __init__(self, *, ser, model, channels, window_size,
                 z_mean, z_std_safe, gesture_names, device,
                 collect_seconds, subtract_mid, adc_mid,
                 confidence_threshold, state_path):
        self.ser                  = ser
        self.model                = model
        self.channels             = channels
        self.window_size          = window_size
        self.z_mean               = z_mean
        self.z_std_safe           = z_std_safe
        self.gesture_names        = gesture_names
        self.device               = device
        self.collect_seconds      = collect_seconds
        self.subtract_mid         = subtract_mid
        self.adc_mid              = adc_mid
        self.confidence_threshold = confidence_threshold
        self.state_path           = state_path

        self._lock       = threading.Lock()
        self._active     = False
        self._stop_event = threading.Event()
        self._thread     = None

    # ---------- 公开接口 ----------

    def start(self):
        with self._lock:
            if self._active:
                return
            self._active = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        print("[CTRL] 推理循环已启动")

    def stop(self):
        self._stop_event.set()
        with self._lock:
            self._active = False
        self._write({"phase": "idle", "timestamp": time.time()})
        print("[CTRL] 推理循环已停止")

    # ---------- 私有方法 ----------

    def _write(self, data: dict):
        atomic_write_json(self.state_path, data)

    def _collect(self):
        """
        从串口采集 collect_seconds 秒，
        每 0.1 s 向 gesture_state.json 写一次倒计时。
        若中途收到停止信号返回 None。
        """
        samples   = []
        deadline  = time.time() + self.collect_seconds
        next_tick = time.time()

        while time.time() < deadline:
            if self._stop_event.is_set():
                return None

            now = time.time()
            if now >= next_tick:
                self._write({
                    "phase":           "collecting",
                    "countdown":       round(max(0.0, deadline - now), 1),
                    "collect_seconds": self.collect_seconds,
                    "timestamp":       now,
                })
                next_tick = now + 0.1

            line = self.ser.readline()   # timeout=0.1 保证循环快速
            if not line:
                continue
            try:
                text = line.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text:
                continue
            sample = parse_csv4(text)
            if sample is None:
                continue
            if self.subtract_mid:
                sample = sample - self.adc_mid
            samples.append(sample)

        return samples

    def _infer(self, samples):
        if not samples:
            return None

        raw_arr = np.asarray(samples, dtype=np.float32)  # (N, 4)
        n = len(raw_arr)

        # ── 滑窗：stride 和训练时保持一致 ──────────────────────────
        INFER_STRIDE = 50  # 对应训练脚本里的 STRIDE = 50

        windows = []
        for start in range(0, n - self.window_size + 1, INFER_STRIDE):
            seg = raw_arr[start: start + self.window_size]  # (window_size, 4)

            # 用训练集拟合的 z-score 参数归一化
            feat = build_raw_feature_window(seg, self.channels)
            feat = (feat - self.z_mean[None, :]) / self.z_std_safe[None, :]
            windows.append(feat)

        # 采集时间太短，连一个窗口都凑不齐时的兜底
        if not windows:
            print(f"[WARN] 样本数 {n} < window_size {self.window_size}，跳过本轮")
            return None

        # ── 批量推理所有窗口 ────────────────────────────────────────
        # stack: (W, window_size, C) → transpose → (W, C, window_size)
        X = torch.from_numpy(
            np.stack(windows).transpose(0, 2, 1)
        ).float().to(self.device)

        with torch.no_grad():
            all_probs = torch.softmax(
                self.model(X), dim=1
            ).cpu().numpy()  # (W, n_classes)

        # ── 概率平均（比硬投票更稳，置信度估计更准）─────────────────
        mean_probs = all_probs.mean(axis=0)  # (n_classes,)

        pid = int(np.argmax(mean_probs))
        conf = float(mean_probs[pid])

        return {
            "gesture_id": pid,
            "name": self.gesture_names.get(pid, str(pid)),
            "confidence": conf,
            "all_probs": {
                self.gesture_names.get(i, str(i)): float(p)
                for i, p in enumerate(mean_probs)
            },
            "n_samples": n,
            "n_windows": len(windows),  # 新增，方便调试时确认用了多少窗口
        }

    def _run(self):
        attempt = 0
        try:
            while not self._stop_event.is_set():
                attempt += 1

                # ---- 采集 ----
                samples = self._collect()
                if samples is None or self._stop_event.is_set():
                    break

                # ---- 推理 ----
                result = self._infer(samples)
                if result is None:
                    continue

                conf = result["confidence"]

                # ---- 置信度判断 ----
                if conf >= self.confidence_threshold:
                    # 输出结果
                    self._write({
                        "phase":                "result",
                        "gesture_id":           result["gesture_id"],
                        "name":                 result["name"],
                        "confidence":           conf,
                        "all_probs":            result["all_probs"],
                        "n_samples":            result["n_samples"],
                        "attempt":              attempt,
                        "confidence_threshold": self.confidence_threshold,
                        "timestamp":            time.time(),
                    })
                    print(
                        f"[OK  #{attempt:>3}] {result['name']:<12} "
                        f"conf={conf:.3f}  samples={result['n_samples']}"
                    )
                    attempt = 0  # 下一轮重新计数
                    # 短暂停顿让用户看清结果再采集下一轮
                    if self._stop_event.wait(timeout=0.8):
                        break

                else:
                    # 置信度不足，重试
                    self._write({
                        "phase":                "retrying",
                        "gesture_id":           result["gesture_id"],
                        "name":                 result["name"],
                        "confidence":           conf,
                        "all_probs":            result["all_probs"],
                        "attempt":              attempt,
                        "confidence_threshold": self.confidence_threshold,
                        "timestamp":            time.time(),
                    })
                    print(
                        f"[RETRY #{attempt:>2}] {result['name']:<12} "
                        f"conf={conf:.3f} < {self.confidence_threshold:.2f}，重试..."
                    )
                    if self._stop_event.wait(timeout=0.25):
                        break

        finally:
            with self._lock:
                self._active = False


# ================================================================
# HTTP 处理器
# ================================================================
def make_handler(controller: InferenceController, web_dir: Path):
    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path.startswith("/api/control"):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body   = json.loads(self.rfile.read(length))
                    cmd    = body.get("cmd", "")
                except Exception:
                    cmd = ""

                if cmd == "start":
                    controller.start()
                elif cmd == "stop":
                    controller.stop()

                resp = json.dumps({"ok": True, "cmd": cmd}).encode()
                self.send_response(200)
                self.send_header("Content-Type",   "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            else:
                self.send_error(404)

        def log_message(self, *_):
            pass  # 屏蔽 HTTP 访问日志

    return functools.partial(Handler, directory=str(web_dir))


# ================================================================
# 主入口
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="交互式 EMG 手势预测（按需采集 + 置信度门槛）")
    parser.add_argument("--model",                default="model_cnn_tuned.pt")
    parser.add_argument("--port",                 required=True, help="串口号，如 COM5")
    parser.add_argument("--baud",                 type=int,   default=921600)
    parser.add_argument("--html",                 default="hand_model_2.html")
    parser.add_argument("--host",                 default="127.0.0.1")
    parser.add_argument("--http-port",            type=int,   default=8000)
    parser.add_argument("--no-browser",           action="store_true")
    parser.add_argument("--adc-mid",              type=float, default=2048.0)
    parser.add_argument("--collect-seconds",      type=float, default=3.0,
                        help="每轮采集时长（秒），默认 3.0")
    parser.add_argument("--confidence-threshold", type=float, default=0.95,
                        help="输出置信度阈值，低于此值重试，默认 0.95")
    parser.add_argument("--subtract-mid-from-raw", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---------- 加载模型 ----------
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    channels = [str(x) for x in (ckpt.get("channels") or [])]
    if not channels:
        raise KeyError("模型文件里没有 channels 字段")

    allowed = {"CH1", "CH2", "CH3", "CH4"}
    if any(n not in allowed for n in channels):
        raise ValueError(f"模型通道不是标准 Raw 通道: {channels}")

    n_channels  = int(ckpt.get("n_channels",  len(channels)))
    n_classes   = int(ckpt.get("n_classes",   7))
    window_size = int(ckpt.get("window_size", 200))

    model_params = ckpt.get("model_params") or expand_params_from_optuna(ckpt["best_optuna_params"])

    gesture_names = normalise_gesture_names(
        ckpt.get("gesture_names",
                 {0:"Rest", 1:"Fist", 2:"Spread", 3:"Flexion",
                  4:"Extension", 5:"Pronation", 6:"Supination"})
    )

    z_mean = np.asarray(
        ckpt.get("zscore_mean", np.zeros(n_channels)), dtype=np.float32
    ).reshape(-1)
    z_std  = np.asarray(
        ckpt.get("zscore_std",  np.ones(n_channels)),  dtype=np.float32
    ).reshape(-1)
    z_std_safe = np.where(np.abs(z_std) < 1e-8, 1.0, z_std)

    if len(z_mean) != n_channels:
        raise ValueError("zscore_mean 维度与通道数不匹配")

    model = DynamicEMGNet(
        n_channels=n_channels, n_classes=n_classes, params=model_params
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] 模型加载完成 | 通道: {channels} | window_size={window_size}")

    # ---------- 路径 ----------
    html_path = Path(args.html).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"找不到 HTML 文件: {html_path}")
    web_dir    = html_path.parent
    state_path = web_dir / "gesture_state.json"

    atomic_write_json(state_path, {"phase": "idle", "timestamp": time.time()})

    # ---------- 串口（timeout=0.1 保证倒计时刷新流畅）----------
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    print(f"[INFO] 串口: {args.port} @ {args.baud}")

    # ---------- 控制器 ----------
    controller = InferenceController(
        ser=ser,
        model=model,
        channels=channels,
        window_size=window_size,
        z_mean=z_mean,
        z_std_safe=z_std_safe,
        gesture_names=gesture_names,
        device=device,
        collect_seconds=args.collect_seconds,
        subtract_mid=args.subtract_mid_from_raw,
        adc_mid=args.adc_mid,
        confidence_threshold=args.confidence_threshold,
        state_path=state_path,
    )

    # ---------- HTTP 服务 ----------
    handler_class = make_handler(controller, web_dir)
    httpd         = ThreadingHTTPServer((args.host, args.http_port), handler_class)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://{args.host}:{args.http_port}/{html_path.name}"
    print(f"[INFO] HTTP 服务: {url}")
    print(f"[INFO] 置信度阈值: {args.confidence_threshold:.0%} | 采集时长: {args.collect_seconds}s")
    print("[INFO] 在浏览器中点击「▶ 开始预测」按钮即可")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] 正在退出...")
    finally:
        controller.stop()
        try:
            ser.close()
        except Exception:
            pass
        httpd.shutdown()


if __name__ == "__main__":
    main()
