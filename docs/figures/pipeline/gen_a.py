# -*- coding: utf-8 -*-
"""fig00 总览 / fig01 训练主循环 / fig02 锚点→神经高斯。"""
from svgkit import SVG, C, text_width


# =====================================================================
def fig00(outdir):
    s = SVG(1400, 900, "DCCA-GS 全流程总览：训练 → 编码 → 码流 → 解码 → 评估",
            "两个创新都是“解码端可复算”：①复杂度乘子活在量化路径里（绿），②覆盖感知融合剪枝活在训练侧投影里（橙）；两者都不往码流写任何额外比特")
    # 图例
    s.legend(30, 84, [
        ("HAC++ 原有机制", "blue"), ("创新① 复杂度乘子（量化路径）", "green"),
        ("创新② 覆盖感知融合剪枝（训练侧）", "orange"), ("码流内容", "purple"),
        ("训练期专用信号（永不进码流）", "grey", True), ("失败/警示", "red"),
    ])

    # ---------- 行 1 ----------
    # ① 数据
    s.panel(26, 110, 148, 360, "① 数据", "blue")
    s.box(38, 148, 124, "UAV 影像", ["4 航道照片", "+ 位姿"], "blue")
    s.box(38, 244, 124, "大规模场景", ["1-78 · 2-06", "4-10"], "blue")
    s.text(100, 372, "离线 DINO 语义特征", size=10, color=C["sub"], anchor="middle")
    s.text(100, 388, "（可选 · 已判负）", size=10, color=C["sub"], anchor="middle")

    # ② 训练
    s.panel(196, 110, 636, 360, "② 训练（30,000 迭代 · seed 42 · λ = 0.0005 / 0.002 / 0.004）", "plain")
    s.text(362, 146, "每步执行", size=12, color=C["sub"], anchor="middle", weight="bold")
    s.text(672, 146, "周期 / 阶段事件", size=12, color=C["sub"], anchor="middle", weight="bold")
    b1 = s.box(212, 158, 300, "渲染 + 总损失",
               ["L = D + λ·R + λsens·Lsens + Lspa",
                "D = (1−0.2)L1 + 0.2(1−SSIM) + 尺度正则"], "blue")
    b2 = s.box(212, 244, 300, "反向传播 → 敏感度 EMA",
               ["‖∂L/∂a‖ 逐锚点 · α=0.99", "（训练期专用信号）"], "grey", dashed=True)
    b3 = s.box(212, 330, 300, "量化仿真",
               ["3k–10k：固定步长 + 均匀噪声", ">10k：AQM 预测 Q + 噪声"], "blue")
    b4 = s.box(528, 158, 288, "每 100 步：生长 + SPA 投影",
               ["1500 < it < 15000 · 预算 κ ramp",
                "融合分数 + 覆盖约束（创新②）"], "orange")
    b5 = s.box(528, 244, 288, "@15000：深度再初始化",
               ["8 视角深度反投影 · 预算不增", "候选须与旧锚竞争"], "orange")
    b6 = s.box(528, 330, 288, "Q = Q0·(1+tanh q)·m",
               ["m = 复杂度乘子（创新①）", "mlp_complexity: 4→16→3"], "green")
    s.arrow(362, b1[1] + b1[3], 362, b2[1] - 3, color=C["blue"])
    s.arrow(362, b2[1] + b2[3], 362, b3[1] - 3, color=C["blue"])
    s.arrow(672, b4[1] + b4[3], 672, b5[1] - 3, color=C["orange"])
    s.arrow(672, b5[1] + b5[3], 672, b6[1] - 3, color=C["orange"], label="m 进入 Q")

    # ③ 编码
    s.panel(856, 110, 272, 360, "③ 编码", "blue")
    s.box(868, 150, 248, "坐标 voxel 对齐 + Morton 排序", [], "blue", tsize=13)
    s.box(868, 210, 248, "共享 Q 路径 → STE 量化",
          ["ctx → mlp_grid → mlp_complexity → Q", "与训练同一路径"], "blue", tsize=13)
    s.box(868, 296, 248, "各字段熵编码",
          ["G-PCC 坐标 · 算术编码属性", "（feat 混合熵 / scaling、offsets 高斯熵）"], "blue", tsize=13)
    s.box(868, 382, 248, "hac_meta.json 体积分解", [], "blue", tsize=13)

    # ④ 码流
    s.panel(1152, 110, 222, 360, "④ 码流", "purple")
    fb = s.box(1164, 150, 198, "码流文件", [
        "xyz_gpcc.npz（坐标）", "hash.b（哈希网格 1bit）",
        "masks.b", "feat / scaling / offsets 码流",
        "attributes.pth · hac_meta.json", "MLP 权重（32bit/参数）"], "purple", tsize=13)
    s.text(1263, fb[1] + fb[3] + 26, "✗ 不含：敏感度 · 覆盖面积 ·", size=11.5,
           color=C["red"], anchor="middle")
    s.text(1263, fb[1] + fb[3] + 44, "融合分数 · ADMM 状态 · 语义", size=11.5,
           color=C["red"], anchor="middle")

    # 行间箭头
    s.arrow(174, 290, 194, 290, color=C["blue"])
    s.arrow(832, 290, 854, 290, color=C["blue"])
    s.arrow(1128, 290, 1150, 290, color=C["purple"])

    # ---------- 行 2 ----------
    # ⑤ 解码
    s.panel(196, 510, 436, 300, "⑤ 解码（无需任何预训练模型）", "blue")
    d1 = s.box(210, 550, 190, "GPCC 解坐标", ["SHA-256 校验"], "blue", tsize=13)
    d2 = s.box(424, 550, 196, "重算哈希 ctx", ["网格参数来自码流"], "blue", tsize=13)
    d3 = s.box(210, 640, 190, "mlp_grid + 熵解码", ["feat 通道组上下文", "可由已解码通道重算"], "blue", tsize=13)
    d4 = s.box(424, 640, 196, "Q 同式重算 → 反量化", ["m = mlp_complexity(x)", "与编码端逐位一致"], "green", tsize=13)
    d5 = s.box(210, 730, 410, "generate_gaussians（跳过量化）→ gsplat 渲染", [], "blue", tsize=13)
    s.arrow(d1[0] + d1[2], 585, d2[0] - 3, 585, color=C["blue"])
    s.arrow(d2[0] + d2[2] / 2, d2[1] + d2[3], 522, d4[1] - 3, color=C["blue"], label=None)
    s.arrow(305, d1[1] + d1[3], 305, d3[1] - 3, color=C["blue"])
    s.arrow(d3[0] + d3[2], 687, d4[0] - 3, 687, color=C["blue"])
    s.arrow(305, d3[1] + d3[3], 305, d5[1] - 3, color=C["blue"])
    s.arrow(522, d4[1] + d4[3], 522, d5[1] - 3, color=C["blue"])

    # ⑥ 评估
    s.panel(656, 510, 368, 300, "⑥ 评估", "blue")
    s.box(670, 550, 156, "RD 曲线", ["PSNR/SSIM/LPIPS", "vs total_MB", "（log / linear）"], "blue", tsize=13)
    s.box(846, 550, 164, "BD-rate", ["vs HAC++", "（共享质量区）"], "blue", tsize=13)
    s.box(670, 660, 340, "bit-exact 逐位回传校验", [
        "整数锚点与 masks 的 SHA-256 编/解码前后一致",
        "codec_roundtrip_diagnostics.json：mismatch = 0"], "blue", tsize=13)
    s.chips(670, 756, ["30k 迭代", "seed 42", "fp32 体积口径", "HAC++ 官方实现同协议"])

    # 关键结果
    s.panel(1048, 510, 326, 300, "关键结果", "green")
    s.text(1064, 548, "· 平均 58% 的 HAC++ 体积", size=13, color=C["ink"])
    s.text(1064, 570, "  （匹配 λ，PSNR −0.27 ~ −0.61 dB）", size=12, color=C["sub"])
    s.text(1064, 600, "· 消融：去掉复杂度乘子", size=13, color=C["ink"])
    s.text(1064, 622, "  −0.41 dB（λ=0.002，场景 1-78）", size=12, color=C["sub"])
    s.text(1064, 652, "· 覆盖设计消除 1.62 dB 低码率塌缩", size=13, color=C["ink"])
    s.text(1064, 674, "  （w=0.5 无帽：27.51→25.95 dB）", size=12, color=C["sub"])
    s.text(1064, 704, "· 训练侧两创新 → 码流零额外比特", size=13, color=C["ink"])
    s.text(1064, 726, "  解码端只重算，不读侧信息", size=12, color=C["sub"])
    s.text(1064, 756, "详见 fig04/07/08（机制）与 fig11（协议）", size=11, color=C["green"])

    # 行 2 箭头
    s.elbow([(1263, 470), (1263, 492), (414, 492), (414, 508)], color=C["purple"],
            label="读码流", label_at=(838, 487))
    s.arrow(632, 660, 654, 660, color=C["blue"])
    s.arrow(1024, 660, 1046, 660, color=C["green"])

    s.footer("train.py · scaffold_gs/{trainer,hacpp}.py · hacplus/scene/gaussian_model.py · scripts/rd_sweep.py",
             note="DCCA-GS 全流程总览 · 2026-09-08 · 配色全局统一")
    s.save(f"{outdir}/fig00-overview.svg")


# =====================================================================
def fig01(outdir):
    s = SVG(1360, 1010, "DCCA-GS 训练主循环：单步流水 + 迭代时间轴",
            "每步：渲染 → 损失 → 反传 → 敏感度累积 →（周期）生长/SPA 投影 →（@15k）深度再初始化 → 参数更新；机制按迭代区间分阶段启用")

    # ---------- A. 单步流水 ----------
    s.panel(20, 80, 1320, 420, "A. 单步流水（scaffold_gs/trainer.py:218–329，按执行顺序）", "plain")
    row1 = [
        ("① 学习率更新\n取随机相机", []),
        ("② render", ["is_training=True", "retain_grad（敏感度用）"]),
        ("③ 计算总损失 L", ["五项分解见下"]),
        ("④ loss.backward()", []),
        ("⑤ 敏感度 EMA 累积", ["t ≥ sensitivity_start", "‖∂L/∂a‖ · α=0.99"]),
        ("⑥ 训练统计累积", ["500 < t < 15000", "opacity / offset 梯度"]),
    ]
    row2 = [
        ("⑦ 每 100 步 adjust_anchor", ["1500 < t < 15000", "生长 + SPA 投影 → fig06/07/08"]),
        ("⑧ @15000 深度再初始化", ["一次性 · 预算不增 → fig09"]),
        ("⑨ optimizer.step()", ["zero_grad()"]),
        ("⑩ 评估 + 保存", ["@{15000, 30000}", "PLY / MLP / 检查点"]),
    ]
    x0, y1, y2 = 44, 130, 296
    # 第一行（手排，保证宽度）
    w1 = [168, 172, 150, 158, 196, 186]
    xs = []
    cx = x0
    for w in w1:
        xs.append(cx); cx += w + 48
    kinds1 = ["blue", "blue", "blue", "blue", "grey", "grey"]
    dashes = [False, False, False, False, True, True]
    subs1 = [r[1] for r in row1]
    titles1 = [r[0].replace("\n", " ") for r in row1]
    for i in range(6):
        s.box(xs[i], y1, w1[i], titles1[i], subs1[i], kind=kinds1[i], dashed=dashes[i], tsize=13.5)
        if i:
            s.arrow(xs[i-1] + w1[i-1] + 3, y1 + 32, xs[i] - 4, y1 + 32, color=C["sub"])
    # ② 的副注
    s.text(xs[1] + w1[1]/2, y1 - 8, "gsplat packed 渲染 · 分块 16,384 锚点", size=10.5,
           color=C["sub"], anchor="middle")
    # 第二行
    w2 = [236, 216, 168, 190]
    xs2 = []
    cx = x0
    for w in w2:
        xs2.append(cx); cx += w + 56
    kinds2 = ["orange", "orange", "blue", "blue"]
    for i, (t, sub) in enumerate(row2):
        s.box(xs2[i], y2, w2[i], t, sub, kind=kinds2[i], tsize=13.5)
        if i:
            s.arrow(xs2[i-1] + w2[i-1] + 3, y2 + 32, xs2[i] - 4, y2 + 32,
                    color=C["orange"] if i < 2 else C["sub"])
    # 行间连接：⑥ → ⑦
    s.elbow([(xs[5] + w1[5]/2, y1 + 78), (xs[5] + w1[5]/2, y2 - 40),
             (xs2[0] + w2[0]/2, y2 - 40), (xs2[0] + w2[0]/2, y2 - 4)],
            color=C["sub"], label=None)
    s.text(xs[5] + w1[5]/2 + 8, y1 + 102, "继续（含周期事件）", size=10.5, color=C["sub"])

    # 损失分解
    lb = s.box(44, 386, 1272, "总损失分解（③ 中每一项 · 默认权重如注）", [
        "L = L_render + λ·R + λsens·L_sens + L_spa （+ L_sem：DINO 语义监督，方向一已判负，默认关）",
        "L_render = (1−0.2)·L1 + 0.2·(1−SSIM) + 0.01·mean(∏scaling)          λ_dssim=0.2 · scale_reg=0.01",
        "R = bit_per_param（5% 采样估计）+ bit_hash/锚点数          λ=λ_rate=0.004（RD 扫描 0.0005 / 0.002 / 0.004）",
        "L_sens = 1e-3·MSE(pred_m, target_m)（t≥20000，→ fig05）          L_spa = (1e-3/2)·mean‖a−z+u‖²（ADMM 稀疏项，→ fig06）",
    ], "plain", tsize=13.5, ssize=12, stroke=C["blue"])
    s.elbow([(xs[2] + w1[2]/2, y1 + 64), (xs[2] + w1[2]/2, 380)],
            color=C["blue"], dashed=True, label=None)
    s.text(xs[2] + w1[2]/2 + 10, 372, "展开", size=10.5, color=C["blue"])

    # ---------- B. 时间轴 ----------
    s.panel(20, 520, 1320, 420, "B. 迭代时间轴 0 → 30,000（轴为等距事件间距，非线性刻度）", "plain")
    ticks = [0, 1500, 3000, 10000, 15000, 20000, 30000]
    ax0, ax1 = 320, 1290
    def tx(t):
        i = min(bisect_(ticks, t), len(ticks) - 1)
        i0 = max(i - 1, 0)
        if t <= ticks[0]:
            return ax0
        if t >= ticks[-1]:
            return ax1
        # 线性插值所在段
        for k in range(len(ticks) - 1):
            if ticks[k] <= t <= ticks[k + 1]:
                frac = (t - ticks[k]) / (ticks[k + 1] - ticks[k])
                return ax0 + (ax1 - ax0) * (k + frac) / (len(ticks) - 1)
        return ax1
    def bisect_(arr, v):
        for i, a in enumerate(arr):
            if a > v:
                return i
        return len(arr)

    axis_y = 880
    s.line(ax0 - 20, axis_y, ax1 + 16, axis_y, color=C["ink"], sw=2)
    for t in ticks:
        x = tx(t)
        s.line(x, axis_y - 5, x, axis_y + 5, color=C["ink"], sw=2)
        s.text(x, axis_y + 22, f"{t:,}", size=12, color=C["ink"], anchor="middle", weight="bold")
    s.text(ax0 - 24, axis_y + 6, "迭代", size=11, color=C["sub"], anchor="end")

    lanes = [
        ("训练统计累积", [(500, 15000)], "blue", False, ""),
        ("生长 + SPA 投影（每100步）", [(1500, 15000)], "orange", False, "最后一次投影 @14900（15000 不含）"),
        ("量化仿真：固定 Q0 + 噪声", [(3000, 10000)], "blue", False, ""),
        ("量化仿真：AQM 预测 Q + 噪声", [(10000, 30000)], "blue", False, ""),
        ("深度再初始化（一次性）", [(15000, 15000)], "orange", False, "锁预算 · 竞争淘汰"),
        ("敏感度监督 L_sens", [(20000, 30000)], "grey", True, ""),
        ("复杂度乘子 strength ramp", [(20000, 30000)], "green", False, "0 → 0.35"),
        ("评估 / 保存", [(15000, 15000), (30000, 30000)], "purple", False, ""),
    ]
    ly = 566
    for name, spans, ck, dashed, note in lanes:
        s.text(300, ly + 14, name, size=11.5, color=C[ck], anchor="end")
        for a, b in spans:
            x, xe = tx(a), tx(b)
            if a == b:  # 事件：菱形标记
                s.parts.append(f"<path d='M{x},{ly-2} L{x+7},{ly+11} L{x},{ly+24} L{x-7},{ly+11} Z' "
                               f"fill='{C[ck+'_bg']}' stroke='{C[ck]}' stroke-width='2'/>")
                s.text(x + 12, ly + 10, note, size=10.5, color=C["sub"]) if note else None
            else:
                s.rrect(x, ly + 2, xe - x, 18, rx=8, fill=C.get(ck + "_bg", "#fff"),
                        stroke=C[ck], sw=1.6, dashed=dashed)
                if xe - x > 120:
                    s.text((x + xe) / 2, ly + 15.5, note, size=10.5, color=C[ck], anchor="middle")
        ly += 36
    s.text(320, 556, "（悬停区间见各机制专图：fig04/05 乘子与敏感度 · fig06 SPA · fig07/08 融合与覆盖 · fig09 深度再初始化）",
           size=11, color=C["sub"])

    s.footer("scaffold_gs/trainer.py:157–366 · scaffold_gs/hacpp.py:563–644（量化阶段） · config.py 默认值",
             note="机制生效窗口以 config 默认为准")
    s.save(f"{outdir}/fig01-training-loop.svg")


# =====================================================================
def fig02(outdir):
    s = SVG(1240, 800, "锚点 → 神经高斯 → gsplat 渲染（HAC++ 解码路径）",
            "每个锚点派生 K=10 个子高斯候选：三个共享 MLP 从 [feat‖视角信息] 解码不透明度/尺度旋转/颜色；masks 训练时软乘、评估时硬筛")

    # 左：锚点属性
    s.panel(24, 88, 292, 560, "锚点的编码属性（N 个锚点）", "blue")
    attrs = [
        ("xyz [N,3]", "锚点坐标（G-PCC 编码）"),
        ("feat [N,32]", "锚点特征（feat_dim=32，默认 50）"),
        ("scaling [N,6]", "exp 域：前 3 维锚尺度 / 后 3 维子高斯基准"),
        ("offsets [N,K=10,3]", "子高斯偏移（显式潜变量）"),
        ("masks [N,K,1]", "偏移掩码（sigmoid STE，阈值 0.01）"),
    ]
    ay = 128
    for t, sub in attrs:
        s.box(38, ay, 264, t, [sub], "blue", tsize=13.5)
        ay += 84
    s.text(170, ay + 6, "_opacity 恒为 inverse_sigmoid(0.1)，", size=11.5, color=C["red"], anchor="middle")
    s.text(170, ay + 24, "不编码——神经不透明度由 MLP 解码", size=11.5, color=C["red"], anchor="middle")

    # 中：共享 MLP
    s.panel(340, 88, 570, 560, "共享 MLP 解码（generate_gaussians，分块 16,384 锚点）", "plain")
    inp = s.box(356, 128, 538, "视角条件输入 cat = [ feat 32 ‖ ob_view 3 ‖ ob_dist 1 ] = 36 维", [
        "ob_view = 归一化(相机中心 − 锚点) · ob_dist = ‖相机中心 − 锚点‖"], "blue", tsize=13.5)
    m1 = s.box(356, 224, 166, "mlp_opacity", ["36→32→K, tanh", "→ 神经不透明度"], "blue", tsize=13.5)
    m2 = s.box(542, 224, 172, "mlp_cov", ["36→32→7K", "→ 尺度 + 四元数"], "blue", tsize=13.5)
    m3 = s.box(734, 224, 160, "mlp_color", ["36→32→3K, sigmoid", "→ RGB"], "blue", tsize=13.5)
    s.elbow([(625, inp[1] + inp[3]), (625, 204), (439, 204), (439, 220)], color=C["blue"])
    s.elbow([(625, 204), (628, 204), (628, 220)], color=C["blue"])
    s.elbow([(625, 204), (814, 204), (814, 220)], color=C["blue"])
    f1 = s.box(356, 330, 166, "候选筛选", ["神经不透明度 > 0", "才生成该子高斯"], "plain", tsize=13)
    f2 = s.box(542, 330, 172, "子高斯属性", ["尺度 = s[:,3:]·sigmoid(o)", "旋转 = normalize(o[3:7])"], "plain", tsize=13)
    f3 = s.box(734, 330, 160, "颜色", ["RGB ∈ (0,1)"], "plain", tsize=13)
    for a, b in ((m1, f1), (m2, f2), (m3, f3)):
        s.arrow(a[0] + a[2]/2, a[1] + a[3], b[0] + b[2]/2, b[1] - 3, color=C["blue"])
    mk = s.box(356, 438, 538, "masks 的作用", [
        "训练：软乘——opacity ← opacity·m，尺度 ← 尺度·m（可导，推动掩码稀疏化）",
        "评估/编码：硬筛——只保留二值化后 m=1 的子高斯（offsets 也仅对 m=1 编码）"], "grey", tsize=13.5, dashed=True)
    pos = s.box(356, 546, 538, "子高斯位置  xyz = 锚点 + offsets · scaling[:,:3]", [
        "锚尺度前 3 维调制偏移幅度：锚尺度大 → 子高斯散布更开"], "plain", tsize=13.5)

    # 右：渲染
    s.panel(934, 88, 282, 560, "组装与渲染", "blue")
    g1 = s.box(948, 128, 254, "有效神经高斯集合", ["≈ N × K × 活跃 mask", "× 不透明度筛选"], "blue", tsize=13.5)
    g2 = s.box(948, 226, 254, "gsplat rasterization", ["packed 模式", "means2d.retain_grad", "（生长统计用）"], "blue", tsize=13.5)
    g3 = s.box(948, 340, 254, "渲染图像", ["→ L_render（fig01）", "→ 深度图（fig09 再初始化）"], "blue", tsize=13.5)
    s.arrow(g1[0] + 127, g1[1] + g1[3], 948 + 127, g2[1] - 3, color=C["blue"])
    s.arrow(g2[0] + 127, g2[1] + g2[3], 948 + 127, g3[1] - 3, color=C["blue"])
    g4 = s.box(948, 452, 254, "反传信息流", [
        "means2d 梯度 → 生长统计（每 100 步）",
        "pre_quant 属性梯度 → 敏感度 EMA（fig05）"], "grey", tsize=13, dashed=True)
    g5 = s.box(948, 566, 254, "工程细节", [
        "锚点分块 16,384，峰值显存与总锚数无关",
        "packed 光栅化：只保留有效子高斯"], "plain", tsize=13)
    s.arrow(948 + 127, g3[1] + g3[3], 948 + 127, g4[1] - 3, color=C["blue"], dashed=True)
    s.arrow(948 + 127, g4[1] + g4[3], 948 + 127, g5[1] - 3, color=C["grey"], dashed=True)

    # 连接左右
    s.elbow([(302, 268), (322, 268), (322, 170), (352, 170)], color=C["blue"])
    s.elbow([(894, 467), (920, 467), (920, 165), (942, 165)], color=C["blue"])
    s.elbow([(894, 575), (928, 575), (928, 190), (942, 190)], color=C["blue"])

    s.text(620, 692, "要点：颜色/不透明度/尺度旋转全部由共享 MLP 从量化后的 feat + 视角信息即时解码——因此码流只需存锚点属性，不需存每个子高斯",
           size=12.5, color=C["ink"], anchor="middle")
    s.text(620, 714, "量化发生在锚点属性上（feat / scaling / offsets，Q 见 fig04），MLP 输入用的就是量化-重建后的属性",
           size=12.5, color=C["sub"], anchor="middle")

    s.footer("scaffold_gs/hacpp.py:522–822（generate_gaussians） · hacplus/scene/gaussian_model.py:407–448（MLP 定义） · renderer.py:105–168",
             note="K = n_offsets = 10")
    s.save(f"{outdir}/fig02-anchor-to-gaussians.svg")
