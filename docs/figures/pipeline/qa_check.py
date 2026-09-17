# -*- coding: utf-8 -*-
"""几何质检：instrument svgkit，记录所有文字/面板/盒子/chip 的包围盒，
检测：画布越界、文字溢出所在面板/盒子、文字互压、chip 互压。
用法：python3 qa_check.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import svgkit
from svgkit import SVG, text_width

TEXTS = []
PANELS = []
BOXES = []
CHIPS = []
ARROWS = []          # (fig, x, y, kind)
CANVAS = []          # (fig, W, H)
CUR = {"fig": "?"}

_orig_arrow = SVG.arrow
_orig_elbow = SVG.elbow


def _arrow(self, x1, y1, x2, y2, **kw):
    ARROWS.append((CUR["fig"], x1, y1, "start"))
    ARROWS.append((CUR["fig"], x2, y2, "end"))
    return _orig_arrow(self, x1, y1, x2, y2, **kw)


def _elbow(self, pts, **kw):
    if kw.get("arrow", True):
        ARROWS.append((CUR["fig"], pts[0][0], pts[0][1], "start"))
        ARROWS.append((CUR["fig"], pts[-1][0], pts[-1][1], "end"))
    else:
        ARROWS.append((CUR["fig"], pts[0][0], pts[0][1], "start"))
    return _orig_elbow(self, pts, **kw)


_orig_text = SVG.text
_orig_panel = SVG.panel
_orig_box = SVG.box
_orig_chip = SVG.chip
_orig_save = SVG.save
_orig_init = SVG.__init__


def _init(self, width, height, title, subtitle=""):
    CUR["fig"] = "?"
    _orig_init(self, width, height, title, subtitle)
    self._qa_fig = str(len(CANVAS)).zfill(2)
    CUR["fig"] = self._qa_fig
    CANVAS.append((self._qa_fig, width, height))


def _text(self, x, y, s, size=13, color=None, anchor="start", weight="normal",
          style="", opacity=1.0):
    w = text_width(str(s), size) * (1.04 if weight == "bold" else 1.0)
    h = size * 1.05
    if anchor == "middle":
        x0 = x - w / 2
    elif anchor == "end":
        x0 = x - w
    else:
        x0 = x
    TEXTS.append((CUR["fig"], x0, y - h * 0.8, x0 + w, y + h * 0.25, str(s), size))
    return _orig_text(self, x, y, s, size, color, anchor, weight, style, opacity)


def _panel(self, x, y, w, h, title, kind="blue", rx=12):
    PANELS.append((CUR["fig"], x, y, w, h, title))
    return _orig_panel(self, x, y, w, h, title, kind, rx)


def _box(self, x, y, w, title, subs=(), kind="plain", **kw):
    min_h = kw.get("min_h")
    tsize = kw.get("tsize", 14)
    ssize = kw.get("ssize", 11.5)
    pad_v = kw.get("pad_v", 11)
    subs = list(subs)
    h = pad_v * 2 + tsize + 7 + (len(subs) * (ssize + 4.5) if subs else 0)
    if min_h:
        h = max(h, min_h)
    BOXES.append((CUR["fig"], x, y, w, h, title))
    return _orig_box(self, x, y, w, title, subs, kind=kind, **kw)


def _chip(self, x, y, s, kind="amber", size=11):
    w = text_width(s, size) + 14
    h = size + 9
    CHIPS.append((CUR["fig"], x, y, w, h, s))
    return _orig_chip(self, x, y, s, kind, size)


def _save(self, path):
    return _orig_save(self, path)


SVG.text = _text
SVG.arrow = _arrow
SVG.elbow = _elbow
SVG.panel = _panel
SVG.box = _box
SVG.chip = _chip
SVG.save = _save
SVG.__init__ = _init

FIG_ORDER = ["fig00", "fig01", "fig02", "fig03", "fig04", "fig05",
             "fig06", "fig07", "fig08", "fig09", "fig10", "fig11", "fig12"]


def figname(tag):
    try:
        return FIG_ORDER[int(tag)]
    except Exception:
        return tag


def overlap_r(a, b, tol=1.5):
    return not (a[2] < b[0] + tol or b[2] < a[0] + tol or
                a[3] < b[1] + tol or b[3] < a[1] + tol)


def main():
    import gen_a, gen_b, gen_c, gen_d
    for m in (gen_a, gen_b, gen_c, gen_d):
        for name in sorted(dir(m)):
            if name.startswith("fig") and callable(getattr(m, name)):
                getattr(m, name)(HERE)

    sizes = {f: (w, h) for f, w, h in CANVAS}
    issues = []

    # 1) 画布越界（排除顶部主/副标题）
    for f, x0, y0, x1, y1, s, size in TEXTS:
        if y0 < 64 and size >= 12.5:
            continue
        W, H = sizes.get(f, (0, 0))
        if x0 < 12 or x1 > W - 12 or y0 < 8 or y1 > H - 8:
            issues.append(f"[{figname(f)}] 画布越界: '{s[:40]}' bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) canvas=({W},{H})")

    # 2) 文字溢出其中心所在的最小面板/盒子
    containers = [(p[0], p[1], p[2], p[3], p[4], f"面板'{p[5][:16]}'") for p in PANELS]
    containers += [(b[0], b[1], b[2], b[3], b[4], f"盒子'{b[5][:16]}'") for b in BOXES]
    for f, x0, y0, x1, y1, s, size in TEXTS:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cands = [c for c in containers if c[0] == f and
                 c[1] < cx < c[1] + c[3] and c[2] < cy < c[2] + c[4]]
        if not cands:
            continue
        c = min(cands, key=lambda c: c[3] * c[4])
        if x0 < c[1] - 4 or x1 > c[1] + c[3] + 4 or y0 < c[2] - 4 or y1 > c[2] + c[4] + 4:
            issues.append(f"[{figname(f)}] 溢出{c[5]}: '{s[:36]}' text=({x0:.0f}..{x1:.0f}) 容器=({c[1]:.0f},{c[2]:.0f},{c[3]:.0f},{c[4]:.0f})")

    # 3) 文字互压
    byfig = {}
    for t in TEXTS:
        byfig.setdefault(t[0], []).append(t)
    for f, ts in byfig.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                h_ov = min(a[3], b[3]) - max(a[1], b[1])
                v_ov = min(a[4], b[4]) - max(a[2], b[2])
                if h_ov > 6 and v_ov > min(a[6], b[6]) * 0.45:
                    issues.append(f"[{figname(f)}] 文字互压: '{a[5][:22]}' × '{b[5][:22]}' @({max(a[1], b[1]):.0f},{max(a[2], b[2]):.0f})")

    # 4) chip 互压
    byfigc = {}
    for c in CHIPS:
        byfigc.setdefault(c[0], []).append(c)
    for f, cs in byfigc.items():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if overlap_r(cs[i][1:5], cs[j][1:5], tol=0.5):
                    issues.append(f"[{figname(f)}] chip互压: '{cs[i][4][:18]}' × '{cs[j][4][:18]}'")

    # 5) 箭头端点悬空：每个端点须在某盒子/面板边界 16px 内或其内部
    rects = [(b[0], b[1], b[2], b[3], b[4], "盒" + b[5][:10]) for b in BOXES]
    rects += [(p[0], p[1], p[2], p[3], p[4], "板" + p[5][:10]) for p in PANELS]
    for f, x, y, kind in ARROWS:
        near = False
        for r in rects:
            if r[0] != f:
                continue
            x0, y0, w, h = r[1], r[2], r[3], r[4]
            if x0 - 16 <= x <= x0 + w + 16 and y0 - 16 <= y <= y0 + h + 16:
                near = True
                break
        if not near:
            issues.append(f"[{figname(f)}] 箭头{kind}悬空 @({x:.0f},{y:.0f})")

    print(f"texts={len(TEXTS)} panels={len(PANELS)} boxes={len(BOXES)} chips={len(CHIPS)} arrows={len(ARROWS)}")
    if issues:
        print(f"== {len(issues)} 个疑似问题 ==")
        for it in issues:
            print(" ", it)
    else:
        print("== 无几何问题 ==")


if __name__ == "__main__":
    main()
