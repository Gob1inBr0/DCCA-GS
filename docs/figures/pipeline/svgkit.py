# -*- coding: utf-8 -*-
"""极简 SVG 流程图工具库（DCCA-GS 全流程配图专用）。

设计目标：手写布局但自动测量文字宽度，避免中文溢出；
统一配色：蓝=HAC++原有，绿=创新①复杂度乘子，橙=创新②融合剪枝，
紫=码流内容，灰虚线=训练期专用信号，红=失败模式/警示。
"""
import html
import math

FONT = "'PingFang SC','Hiragino Sans GB','Microsoft YaHei',Helvetica,Arial,sans-serif"

# ---- 统一配色 ----
C = {
    "ink":      "#263238",   # 正文
    "sub":      "#546e7a",   # 次要文字
    "blue":     "#1565c0",   # HAC++ 原有机制
    "blue_bg":  "#e8f1fd",
    "blue_pn":  "#f4f9ff",
    "green":    "#2e7d32",   # 创新① 复杂度乘子（量化路径）
    "green_bg": "#e9f5ea",
    "green_pn": "#f2f8f2",
    "orange":   "#e65100",   # 创新② 覆盖感知融合剪枝
    "orange_bg":"#fdeede",
    "orange_pn":"#fff6ec",
    "purple":   "#6a1b9a",   # 码流
    "purple_bg":"#f3e8f8",
    "purple_pn":"#faf3fc",
    "grey":     "#607d8b",   # 训练期专用信号（不进码流）
    "grey_bg":  "#eceff1",
    "red":      "#c62828",   # 失败/警示
    "red_bg":   "#fdecea",
    "amber":    "#b26a00",   # 配置值 chip
    "amber_bg": "#fff3d6",
    "frame":    "#cfd8dc",
    "paper":    "#fdfefe",
}


def text_width(s, fs):
    """粗略估宽：全角≈1.0fs，半角≈0.56fs。"""
    w = 0.0
    for ch in s:
        o = ord(ch)
        if o > 0x2E80 or (0xFF00 <= o <= 0xFFEF):  # CJK / 全角
            w += fs
        elif ch in "iIl.,:;'|!()[]":
            w += 0.34 * fs
        elif ch in "mwMW—":
            w += 0.9 * fs
        elif ch.isupper():
            w += 0.68 * fs
        elif ch.isdigit():
            w += 0.56 * fs
        elif ch == " ":
            w += 0.30 * fs
        else:
            w += 0.54 * fs
    return w


def esc(s):
    return html.escape(str(s), quote=True)


class SVG:
    def __init__(self, width, height, title, subtitle=""):
        self.W, self.H = width, height
        self.parts = []
        self._title = title
        self._subtitle = subtitle
        self._markers = {}          # color -> marker id
        self.body = []              # 延迟写入，便于先收集 marker

    # ---------- 基元 ----------
    def _marker(self, color):
        key = f"ar_{color.lstrip('#')}"
        if key not in self._markers:
            self._markers[color] = key
        return key

    def text(self, x, y, s, size=13, color=None, anchor="start", weight="normal",
             style="", opacity=1.0):
        color = color or C["ink"]
        extra = f" font-style='{style}'" if style else ""
        self.parts.append(
            f"<text x='{x}' y='{y}' text-anchor='{anchor}' font-size='{size}' "
            f"fill='{color}' font-weight='{weight}' opacity='{opacity}'{extra}>{esc(s)}</text>"
        )

    def multiline(self, x, y, lines, size=12, lh=None, color=None, anchor="start",
                  weight="normal"):
        lh = lh or size + 5
        color = color or C["ink"]
        for i, ln in enumerate(lines):
            self.text(x, y + i * lh, ln, size=size, color=color, anchor=anchor,
                      weight=weight)

    def rrect(self, x, y, w, h, rx=10, fill="#ffffff", stroke=None, sw=1.5,
              dashed=False, opacity=1.0):
        stroke = stroke or C["frame"]
        dash = " stroke-dasharray='7,4'" if dashed else ""
        self.parts.append(
            f"<rect x='{x}' y='{y}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' "
            f"fill='{fill}' stroke='{stroke}' stroke-width='{sw}'{dash} opacity='{opacity}'/>"
        )

    def line(self, x1, y1, x2, y2, color=None, sw=1.5, dashed=False):
        color = color or C["sub"]
        dash = " stroke-dasharray='7,4'" if dashed else ""
        self.parts.append(
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' "
            f"stroke-width='{sw}'{dash}/>"
        )

    def path(self, d, color=None, sw=2.0, dashed=False, arrow=False, fill="none"):
        color = color or C["sub"]
        dash = " stroke-dasharray='7,4'" if dashed else ""
        mk = f" marker-end='url(#{self._marker(color)})'" if arrow else ""
        self.parts.append(
            f"<path d='{d}' fill='{fill}' stroke='{color}' stroke-width='{sw}'{dash}{mk}/>"
        )

    def arrow(self, x1, y1, x2, y2, color=None, dashed=False, label=None,
              label_dy=-7, label_size=11, sw=2.0, label_dx=0, label_anchor="middle",
              label_color=None):
        color = color or C["sub"]
        mk = self._marker(color)
        dash = " stroke-dasharray='7,4'" if dashed else ""
        self.parts.append(
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' "
            f"stroke-width='{sw}'{dash} marker-end='url(#{mk})'/>"
        )
        if label:
            mx, my = (x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy
            self.text(mx, my, label, size=label_size, color=label_color or color,
                      anchor=label_anchor)

    def elbow(self, pts, color=None, dashed=False, arrow=True, label=None,
              label_at=None, label_size=11):
        """折线箭头 pts=[(x,y),...]"""
        color = color or C["sub"]
        d = f"M{pts[0][0]},{pts[0][1]}" + "".join(f"L{x},{y}" for x, y in pts[1:])
        mk = f" marker-end='url(#{self._marker(color)})'" if arrow else ""
        dash = " stroke-dasharray='7,4'" if dashed else ""
        self.parts.append(
            f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'{dash}{mk}/>"
        )
        if label and label_at:
            self.text(label_at[0], label_at[1], label, size=label_size, color=color,
                      anchor="middle")

    # ---------- 组合件 ----------
    def panel(self, x, y, w, h, title, kind="blue", rx=12):
        k = {"blue": ("blue_pn", "blue"), "green": ("green_pn", "green"),
             "orange": ("orange_pn", "orange"), "purple": ("purple_pn", "purple"),
             "grey": ("#f5f7f8", "grey"), "plain": ("paper", "#78909c"),
             "red": ("#fff5f4", "red")}[kind]
        fill = k[0] if k[0].startswith("#") else C[k[0]]
        stroke = k[1] if k[1].startswith("#") else C[k[1]]
        self.rrect(x, y, w, h, rx=rx, fill=fill, stroke=stroke, sw=1.4)
        tc = C["ink"] if kind in ("plain", "grey") else stroke
        self.text(x + 14, y + 24, title, size=15, weight="bold", color=tc)
        return (x, y)

    def box(self, x, y, w, title, subs=(), kind="plain", dashed=False, rx=10,
            tsize=14, ssize=11.5, min_h=None, pad_v=11, stroke=None, tcolor=None):
        """带标题+副行的盒子。返回 (x,y,w,h)。高度自适应副行数。"""
        palette = {
            "blue":  ("blue_bg", "blue"),
            "green": ("green_bg", "green"),
            "orange":("orange_bg", "orange"),
            "purple":("purple_bg", "purple"),
            "grey":  ("grey_bg", "grey"),
            "red":   ("red_bg", "red"),
            "amber": ("amber_bg", "amber"),
            "plain": ("#ffffff", None),
        }
        fill, sc = palette[kind]
        if stroke:
            sc = stroke
        subs = list(subs)
        th = tsize + 7
        sh = len(subs) * (ssize + 4.5)
        h = pad_v * 2 + th + (sh if subs else 0)
        if min_h:
            h = max(h, min_h)
        self.rrect(x, y, w, h, rx=rx, fill=fill, stroke=sc, sw=1.8, dashed=dashed)
        cy = y + pad_v + tsize
        self.text(x + w / 2, cy, title, size=tsize, weight="bold",
                  color=tcolor or (sc or C["ink"]), anchor="middle")
        if subs:
            yy = cy + ssize + 2
            for s in subs:
                self.text(x + w / 2, yy, s, size=ssize, color=C["sub"], anchor="middle")
                yy += ssize + 4.5
        return (x, y, w, h)

    def hflow(self, x, y, items, gap=46, kind="plain", arrow_color=None, tsize=14,
              ssize=11.5, labels=None, min_w=None):
        """水平流程：items=[(title, subs) or str]，自动测宽、画箭头。返回末端 x。"""
        labels = labels or [None] * len(items)
        boxes = []
        for it in items:
            t, s = (it, []) if isinstance(it, str) else (it[0], list(it[1:]))
            boxes.append((t, s))
        widths = []
        for t, s in boxes:
            w = text_width(t, tsize)
            for ln in s:
                w = max(w, text_width(ln, ssize))
            widths.append(max(w + 26, min_w or 96))
        cx = x
        prev_end = None
        for (t, s), w, lb in zip(boxes, widths, labels):
            if prev_end is not None:
                self.arrow(prev_end, y + 30, cx - 5, y + 30,
                           color=arrow_color or C["sub"], label=lb)
            self.box(cx, y, w, t, s, kind=kind, tsize=tsize, ssize=ssize)
            prev_end = cx + w
            cx = prev_end + gap
        return cx - gap + widths[-1]

    def chip(self, x, y, s, kind="amber", size=11):
        w = text_width(s, size) + 14
        h = size + 9
        palette = {"amber": ("amber_bg", "amber"), "grey": ("grey_bg", "grey"),
                   "red": ("red_bg", "red"), "green": ("green_bg", "green"),
                   "blue": ("blue_bg", "blue"), "purple": ("purple_bg", "purple"),
                   "orange": ("orange_bg", "orange")}
        fill, sc = palette[kind]
        self.rrect(x, y, w, h, rx=h / 2, fill=fill, stroke=C[sc], sw=1.1)
        self.text(x + w / 2, y + h - 4, s, size=size, color=C[sc], anchor="middle")
        return w

    def chips(self, x, y, items, gap=10, **kw):
        cx = x
        for it in items:
            if isinstance(it, tuple):
                w = self.chip(cx, y, it[0], kind=it[1], **kw)
            else:
                w = self.chip(cx, y, it, **kw)
            cx += w + gap
        return cx - x - gap

    def legend(self, x, y, entries, size=11.5, gap=16, title=None):
        """entries=[(label, colorkey, dashed?)] 横排图例。"""
        cx = x
        if title:
            self.text(cx, y + 4, title, size=size, weight="bold", color=C["sub"])
            cx += text_width(title, size) + 12
        for label, ck, *rest in entries:
            dashed = bool(rest and rest[0])
            fill = C.get(f"{ck}_bg", "#fff")
            self.rrect(cx, y - 8, 26, 14, rx=4, fill=fill, stroke=C[ck], sw=1.4,
                       dashed=dashed)
            self.text(cx + 32, y + 3, label, size=size, color=C["sub"])
            cx += 32 + text_width(label, size) + gap
        return cx

    def footer(self, code_refs, note="", y=None):
        y = y or self.H - 16
        self.line(16, y - 18, self.W - 16, y - 18, color=C["frame"], sw=1)
        self.text(18, y - 4, "代码: " + code_refs, size=10.5, color=C["sub"])
        if note:
            self.text(self.W - 18, y - 4, note, size=10.5, color=C["sub"],
                      anchor="end")

    # ---------- 输出 ----------
    def save(self, path):
        defs = []
        for color, key in self._markers.items():
            defs.append(
                f"<marker id='{key}' markerWidth='9' markerHeight='9' refX='7' "
                f"refY='4.5' orient='auto'><path d='M0,0 L8,4.5 L0,9 z' "
                f"fill='{color}'/></marker>"
            )
        head = (
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.W}' "
            f"height='{self.H}' viewBox='0 0 {self.W} {self.H}' font-family=\"{FONT}\">"
            f"<defs>{''.join(defs)}</defs>"
        )
        bg = (f"<rect x='8' y='8' width='{self.W-16}' height='{self.H-16}' rx='14' "
              f"fill='{C['paper']}' stroke='{C['frame']}' stroke-width='1.5'/>")
        # 标题
        title = (
            f"<text x='{self.W/2}' y='40' text-anchor='middle' font-size='21' "
            f"font-weight='bold' fill='{C['ink']}'>{esc(self._title)}</text>"
        )
        sub = ""
        if self._subtitle:
            sub = (f"<text x='{self.W/2}' y='62' text-anchor='middle' font-size='12.5' "
                   f"fill='{C['sub']}'>{esc(self._subtitle)}</text>")
        svg = "\n".join([head, bg, title, sub] + self.parts) + "\n</svg>"
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        return path
