# sEMG 单通道采集与软件降噪建议（STM32）

> 目标：在不改硬件的前提下，把当前 4 通道处理链路改成只保留 `Channel 4`，并尽量降低工频、运动伪迹和宽带噪声。

## 1) 代码层：改成单通道（只采 CH4）

下面按你贴出的 `main.c` 结构给出最小改动点。

### 1.1 通道数改为 1

```c
#define N_CH 1
```

并保持：

```c
#define FRAMES_PER_BLOCK 20
#define ADC_HALF_LEN (N_CH * FRAMES_PER_BLOCK)
#define ADC_DMA_LEN  (2 * ADC_HALF_LEN)
```

这样 DMA 缓冲长度会自动缩小，CPU 和串口负担都降低。

### 1.2 仅保留 ADC 的 CH4（序列长度=1）

在 `MX_ADC1_Init()` 里做三处关键修改：

1. `hadc1.Init.NbrOfConversion = 1;`
2. 仅保留这段配置：

```c
sConfig.Channel = ADC_CHANNEL_4;
sConfig.Rank = 1;
sConfig.SamplingTime = ADC_SAMPLETIME_480CYCLES;
if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
{
    Error_Handler();
}
```

3. 删除/注释 `ADC_CHANNEL_0 / 1 / 8` 的配置块。

### 1.3 CSV 输出改为单列

你当前 `append_u16_csv4_line()` 是四列输出。单通道后建议改成单列（更省带宽）：

```c
static inline int append_u16_csv1_line(char *dst, int remaining, uint16_t a)
{
    int pos = 0;
    int n = u16_to_ascii(a, &dst[pos]);
    pos += n;
    if (pos >= remaining) return -1;
    dst[pos++] = '\n';
    if (pos > remaining) return -1;
    return pos;
}
```

发送时改成：

```c
int n = append_u16_csv1_line((char*)&tx_buf[sel][pos], remaining,
                             out[i*N_CH + 0]);
```

### 1.4 建议增加 UART 丢包计数

当 `uart_tx_busy` 为 1 时你当前直接 `return`，会静默丢帧。建议加：

```c
volatile uint32_t uart_drop_count;

if (uart_tx_busy) {
    uart_drop_count++;
    return;
}
```

便于判断“噪声像增益放大”是否其实是上位机抽样断续导致的视觉假象。

---

## 2) 不改硬件时的软件降噪方案（按优先级）

你描述“噪声很大，且随着放大而放大”，这通常说明前端噪声被真实放大后进入 ADC。软件无法逆转前端 SNR，但可以明显提升可用度。

## 2.1 先做“带通 + 陷波”的稳定组合

sEMG 常用频段大致在 20–450 Hz。你现在链路：

- 20 Hz 高通（去漂移/运动伪迹）
- 50 Hz 陷波（工频）
- 400 Hz 低通（抑制高频）

方向正确。建议进一步：

1. **高通可试 25–30 Hz**：若运动伪迹很重，适当提高截止频率。
2. **保留 50 Hz 陷波，Q 不要过高**：Q 太高会振铃。
3. **低通设在 350–450 Hz**：你现在 400 Hz 合理。

> 一句话：先保证“20~400Hz 左右带通 + 50Hz notch”稳定，再做后处理。

## 2.2 加一个轻量平滑（包络通道）

若你用于手势识别，很多时候不直接用原始波形，而是用包络特征：

1. 对滤波后信号做 `abs(x)`（全波整流）
2. 再做一个 5–10 Hz 的低通（得到肌电包络）

这样能极大降低视觉噪声和瞬时毛刺对特征的影响。

## 2.3 增加稳健特征而非只看时域波形

推荐每个 100–200 ms 窗口计算：

- RMS
- MAV (mean absolute value)
- WL (waveform length)
- ZC (zero crossing，可加阈值)

这些特征比“看一条噪声较大的原始曲线”更抗噪。

## 2.4 软件工频自适应（可选）

如果现场工频干扰随环境变化大，可考虑：

- 自适应陷波（LMS/ANF）
- 或在上位机做频谱估计后动态微调 notch 频点（49–51 Hz）

MCU 端先用固定 50 Hz notch 就足够实用。

## 2.5 采样与量化策略优化

1. **定点/浮点一致化**：滤波内部保持 float 没问题，但最终特征建议用更稳定的统计窗口。
2. **避免可视化误判**：上位机画图时固定 y 轴范围，别自动缩放（自动缩放会让噪声看起来“被放大”）。
3. **串口数据完整性监控**：显示 `adc_half_overrun/adc_full_overrun/uart_drop_count`。

---

## 3) 针对你当前现象的排查顺序（纯软件）

1. **先单通道 CH4 + 保留现有滤波**，确认数据链路稳定。
2. 打开计数器：`adc_half_overrun`、`adc_full_overrun`、`uart_drop_count`。
3. 上位机固定 y 轴，再观察噪声随“模拟增益档位”变化。
4. 加入包络通道（abs + 5~10Hz LPF），比较识别特征稳定性。
5. 仍不够时，再考虑提高高通到 25~30Hz 与 notch Q 微调。

---

## 4) 结论

- 你现在的 DSP 主框架是对的。先改成 **单通道 CH4**，减少系统复杂度与吞吐压力。
- 不改硬件时，最有效的软件路径是：
  - 稳定带通+陷波；
  - 包络提取；
  - 用窗口特征做识别；
  - 加 overruns/drop 可观测性防止误诊。
- 但请记住：如果前端噪声底本身很高，软件只能“改善可用性”，不能真正提升前端 SNR。
