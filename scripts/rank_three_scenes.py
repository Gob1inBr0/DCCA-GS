"""按用户公式对 allMothod.csv 的 3 个数据集（1-78 / 2-6 / 4-10）求方法排名。

公式（每个数据集内，rank 1 = 最好）：
    Dataset rank = rank(PSNR)/6 + rank(SSIM)/6 + rank(LPIPS)/6 + rank(Size [MB])/2
其中 PSNR/SSIM 越高 rank 越小，LPIPS/Size 越低 rank 越小；并列值取平均 rank。
某指标 N/A 时，该项不计入，按剩余项权重归一（加权平均 rank）。
最终排名 = 三个数据集 Dataset rank 的均值（越小越好）。

另算一版「插入 DCCA-GS（fusion，covfix@0.004）」的对照排名，数字取自
docs/data/experiments.csv（gen_matrix 组，fp32 解码口径，30k·seed42 协议）。

用法：python scripts/rank_three_scenes.py
"""

from __future__ import annotations

import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "data" / "allMothod.csv"
SCENES = ["1-78", "2-6", "4-10"]
METRICS = ["PSNR", "SSIM", "LPIPS", "Size [MB]"]
WEIGHT = {"PSNR": 1 / 6, "SSIM": 1 / 6, "LPIPS": 1 / 6, "Size [MB]": 1 / 2}
HIGHER_BETTER = {"PSNR", "SSIM"}

# DCCA-GS（fusion 曲线，统一取 λ=0.002 档，与表内"每方法每场景一个点"口径对应）:
# 数字来自 docs/data/experiments.csv gen_matrix 组（fp32 解码口径，30k·seed42）。
DCCA_ALT = {
    "1-78": {"PSNR": 27.817, "SSIM": 0.8747, "LPIPS": 0.1672, "Size [MB]": 15.3021},
    "2-6": {"PSNR": 28.0561, "SSIM": 0.8749, "LPIPS": 0.1456, "Size [MB]": 16.2233},
    "4-10": {"PSNR": 28.7391, "SSIM": 0.8534, "LPIPS": 0.1798, "Size [MB]": 15.9056},
}


def load() -> dict[str, list[dict]]:
    """返回 {scene: [{Method, PSNR, SSIM, LPIPS, Size}, ...]}。"""
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header_scenes, header_metrics = rows[0], rows[1]
    # 场景名在第一行组首列（后 3 列为空），指标名在第二行 → 场景名向前填充
    scene_cols: dict[str, dict[str, int]] = {}
    cur = ""
    for i, (s, m) in enumerate(zip(header_scenes, header_metrics)):
        s, m = s.strip(), m.strip()
        if s:
            cur = s
        if m and cur:
            scene_cols.setdefault(cur, {})[m] = i
    assert all(len(v) == 4 for v in scene_cols.values()), scene_cols

    out: dict[str, list[dict]] = {s: [] for s in SCENES}
    for row in rows[2:]:
        method = (row[0] or "").strip()
        if not method:
            continue
        for s in SCENES:
            rec = {"Method": method}
            ok = True
            for m in METRICS:
                raw = (row[scene_cols[s][m]] or "").strip()
                try:
                    rec[m] = float(raw)
                except ValueError:
                    rec[m] = None
                    if m != "Size [MB]":
                        ok = ok and False
            out[s].append(rec)
    return out


def avg_rank(values: list[tuple[int, float]], higher_better: bool) -> dict[int, float]:
    """values: [(idx, val)]（val 非 None）。返回 {idx: average_rank}，rank 1 = 最好。"""
    valid = [(i, v) for i, v in values if v is not None]
    order = sorted(valid, key=lambda t: t[1], reverse=higher_better)
    ranks: dict[int, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        avg = (i + j) / 2 + 1  # 并列取平均（1-based）
        for k in range(i, j + 1):
            ranks[order[k][0]] = avg
        i = j + 1
    return ranks


def scene_scores(data: list[dict], extra: dict | None = None) -> dict[str, float]:
    """单个数据集的 Dataset rank 分（加权 rank 和，N/A 归一）。extra 追加为最后一个方法。"""
    recs = list(data)
    if extra is not None:
        recs = recs + [dict(extra, Method="__DCCA__")]
    n = len(recs)
    score = [0.0] * n
    wsum = [0.0] * n
    for m in METRICS:
        ranks = avg_rank([(i, r[m]) for i, r in enumerate(recs)], m in HIGHER_BETTER)
        for i, rk in ranks.items():
            score[i] += WEIGHT[m] * rk
            wsum[i] += WEIGHT[m]
    out = {}
    for i, r in enumerate(recs):
        out[r["Method"]] = score[i] / wsum[i] if wsum[i] else float("nan")
    return out


def render(title: str, data: dict[str, list[dict]], extra: dict | None) -> str:
    per_scene = {s: scene_scores(data[s], (extra or {}).get(s)) for s in SCENES}
    methods = [r["Method"] for r in data["1-78"]]
    if extra is not None:
        methods = methods + ["DCCA-GS (ours)"]
    rows = []
    for m in methods:
        key = "__DCCA__" if m == "DCCA-GS (ours)" else m
        vals = [per_scene[s][key] for s in SCENES]
        mean = sum(vals) / len(vals)
        rows.append((m, vals, mean))
    rows.sort(key=lambda t: t[2])
    lines = [f"\n## {title}\n", "| 排名 | 方法 | 1-78 | 2-06 | 4-10 | 均值(Dataset rank) |", "|---:|---|---:|---:|---:|---:|"]
    for i, (m, vals, mean) in enumerate(rows, 1):
        name = m if m != "__DCCA__" else "**DCCA-GS (ours)**"
        cells = " | ".join(f"{v:.2f}" for v in vals)
        star = " ⭐" if m == "__DCCA__" else ""
        lines.append(f"| {i} | {name}{star} | {cells} | **{mean:.2f}** |")
    return "\n".join(lines)


def main() -> None:
    data = load()
    out = ["# 三个数据集（1-78 / 2-06 / 4-10）方法排名", "",
           "公式：Dataset rank = rank(PSNR)/6 + rank(SSIM)/6 + rank(LPIPS)/6 + rank(Size)/2；"
           "rank 1=最好；均值为三数据集平均，越小越好。",
           "N/A 指标按剩余项加权归一。",]
    out.append(render("A. allMothod.csv 内 18 个方法", data, None))
    out.append(render("B. 插入 DCCA-GS（fusion·λ=0.002 档，ours）", data, DCCA_ALT))
    print("\n".join(out))


if __name__ == "__main__":
    main()
