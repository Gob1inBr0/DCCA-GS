#!/usr/bin/env python3
"""jr2 轮 BD-rate / matched-rate 复核脚本（2026-09-19，随报告 v1.1）。

数据来源：docs/data/experiments.csv 的 ph0c / ph1v3 / jr2 组（硬编码自 CSV 已核行）。
方法：
- 逐格（同 λ 同种子）原始 delta：PSNR 差与 MB 差；
- BD-PSNR / BD-rate：对数码率轴 pchip（两点退化为线性）插值，
  在两条曲线的公共码率 / 公共 PSNR 区间积分，Bjontegaard 口径；
- matched-rate ΔPSNR：在被比较曲线的码率点上，用比较曲线自身的
  相邻两点做对数码率线性插值，读出同码率下的 PSNR 差。

注意：λ=0.0005（高码率端）存在已登记的异常（base 比 λ=0.002 又大又差），
含该点的全区间 BD 会被污染，只算 λ0.004–λ0.002 紧/中码率区间。
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

# (MB, PSNR) — 全部来自 experiments.csv
base_s42 = [(11.0835, 26.9829), (14.4526, 27.7298), (21.4410, 27.5974)]
a2_s42   = [(11.0557, 27.5038), (14.4323, 27.8058), (20.5475, 27.7186)]
base_s104= [(11.7290, 27.2617), (20.9158, 27.4487)]            # λ0002 base 缺
a2_s104  = [(11.2760, 27.4046), (14.4984, 27.6764), (20.1871, 27.2804)]  # λ0002 用 s104r
fusA_s104= [(11.0215, 27.1936), (14.7530, 27.5225)]
base_s104_l4 = (11.7290, 27.2617)
fisher_s104_l4 = (11.0045, 26.7443)

def cell_delta(a, b):
    return a[1] - b[1], a[0] - b[0]

def interp_psnr(pts, mb):
    x = np.log10([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    order = np.argsort(x)
    if len(pts) == 2:
        return float(np.interp(np.log10(mb), x[order], y[order]))
    return float(PchipInterpolator(x[order], y[order])(np.log10(mb)))

def bd(curve_a, curve_b):
    """BD-PSNR (dB) 与 BD-rate (%)，curve 相对 curve_b（负 BD-rate = 更省码率）。"""
    la = np.log10([p[0] for p in curve_a]); pa = np.array([p[1] for p in curve_a])
    lb = np.log10([p[0] for p in curve_b]); pb = np.array([p[1] for p in curve_b])
    # BD-PSNR：公共码率区间
    lo, hi = max(la.min(), lb.min()), min(la.max(), lb.max())
    if lo < hi:
        fa = PchipInterpolator(la, pa); fb = PchipInterpolator(lb, pb)
        xs = np.linspace(lo, hi, 200)
        bd_psnr = float(np.mean(fa(xs) - fb(xs)))
    else:
        bd_psnr = float("nan")
    # BD-rate：公共 PSNR 区间（对 PSNR 轴积分 log-rate 差）
    plo, phi = max(pa.min(), pb.min()), min(pa.max(), pb.max())
    if plo < phi:
        ga = PchipInterpolator(pa, la); gb = PchipInterpolator(pb, lb)
        xs = np.linspace(plo, phi, 200)
        bd_rate = float((10 ** np.mean(ga(xs) - gb(xs)) - 1) * 100)
    else:
        bd_rate = float("nan")
    return bd_psnr, bd_rate

print("== G1 逐格（A2 vs 同λ同种子 base；λ0002 s104 无 base，用 s42 base 代用并标注）==")
rows = [
    ("λ0002 s42 ", (14.4323, 27.8058), (14.4526, 27.7298), ""),
    ("λ0002 s104", (14.4984, 27.6764), (14.4526, 27.7298), "跨种子代用"),
    ("λ0004 s42 ", (11.0557, 27.5038), (11.0835, 26.9829), ""),
    ("λ0004 s104", (11.2760, 27.4046), (11.7290, 27.2617), ""),
    ("λ0005 s42 ", (20.5475, 27.7186), (21.4410, 27.5974), "该档 base 异常"),
    ("λ0005 s104", (20.1871, 27.2804), (20.9158, 27.4487), "该档 base 异常"),
]
deltas = []
for name, a, b, note in rows:
    d, dm = cell_delta(a, b); deltas.append(d)
    print(f"{name}: {d:+.3f} dB @ {dm:+.3f} MB  {note}")
lam = {0.002: [deltas[0], deltas[1]], 0.004: [deltas[2], deltas[3]], 0.0005: [deltas[4], deltas[5]]}
print("\n== 逐λ两种子均值 ==")
for k, v in lam.items():
    print(f"λ{k}: {np.mean(v):+.3f} dB")
print(f"总体（6 格平均）: {np.mean(deltas):+.3f} dB")
print(f"总体（逐λ均值再平均）: {np.mean([np.mean(v) for v in lam.values()]):+.3f} dB")

print("\n== BD（紧/中码率区间 λ0.004–λ0.002，两点=对数码率线性）==")
p, r = bd(a2_s42[:2], base_s42[:2])
print(f"s42  A2 vs base : BD-PSNR {p:+.3f} dB, BD-rate {r:+.1f}%  (区间 11.08–14.43 MB)")

print("\n== matched-rate ΔPSNR（在 base 的码率点上插值 A2/fusA）==")
m = interp_psnr(a2_s104, 11.7290) - 27.2617
print(f"s104 A2   @11.729 MB: {m:+.3f} dB (raw λ0004 格 {cell_delta((11.2760,27.4046), base_s104_l4)[0]:+.3f})")
m2 = interp_psnr(fusA_s104, 11.7290) - 27.2617
print(f"s104 fusA @11.729 MB: {m2:+.3f} dB (raw λ0004 格 {cell_delta((11.0215,27.1936), base_s104_l4)[0]:+.3f})")
print(f"s104 fisher vs fusA 同码率点: {fisher_s104_l4[1] - 27.1936:+.3f} dB (尺寸差 {fisher_s104_l4[0]-11.0215:+.3f} MB)")
