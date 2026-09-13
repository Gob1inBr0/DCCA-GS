# -*- coding: utf-8 -*-
"""fig03 熵编码 / fig04 复杂度乘子（创新①）/ fig05 敏感度信号与监督（创新①）。"""
from svgkit import SVG, C, text_width


# =====================================================================
def fig03(outdir):
    s = SVG(1400, 880, "HAC++ 熵编码管线：上下文建模 → 逐字段算术编码 → 体积构成",
            "所有熵参数由 mlp_grid 从哈希上下文预测；坐标走 G-PCC；feat 用通道组自回归 + 双高斯混合熵；MLP 权重按 32bit/参数计入体积")

    # ---------- 上：上下文路径 ----------
    s.panel(24, 84, 1352, 250, "上下文生成路径（编码/解码共用，解码端可从码流完全重算）", "blue")
    c1 = s.box(40, 130, 170, "锚点坐标", ["活锚 xyz [N,3]"], "blue", tsize=13.5)
    c2 = s.box(246, 130, 210, "voxel 对齐 + Morton 排序", ["round(x/v)·v", "（与解码端一致 → bit-exact）"], "blue", tsize=13.5)
    c3 = s.box(488, 130, 252, "多分辨率二值哈希网格", ["3D 18–514 · 2D 平面 130–1026", "每参数 1bit（存码流 hash.b）"], "blue", tsize=13.5)
    c4 = s.box(772, 130, 174, "ctx 向量", ["逐锚点上下文"], "blue", tsize=13.5)
    s.arrow(c1[0]+c1[2], 175, c2[0]-4, 175, color=C["blue"])
    s.arrow(c2[0]+c2[2], 175, c3[0]-4, 175, color=C["blue"])
    s.arrow(c3[0]+c3[2], 175, c4[0]-4, 175, color=C["blue"])

    g = s.box(976, 122, 190, "mlp_grid(ctx)", ["Linear-ReLU-Linear"], "blue", tsize=13.5)
    s.arrow(c4[0]+c4[2], 175, g[0]-4, 175, color=C["blue"])
    outs = [
        ("feat 熵参数", "mean / scale / prob（3×32）", 1180),
    ]
    o1 = s.box(1180, 108, 186, "feat 熵参数", ["mean/scale/prob 3×32"], "blue", tsize=12.5)
    o2 = s.box(1180, 168, 186, "scaling/offsets 熵参数", ["mean/scale 6+6 / 3K+3K"], "blue", tsize=12.5)
    o3 = s.box(1180, 232, 186, "AQM 步长调整 qa·qs·qo", ["3 标量/锚点 → 基础 Q"], "green", tsize=12.5)
    s.elbow([(g[0]+g[2], 160), (1160, 160), (1160, 130), (1176, 130)], color=C["blue"])
    s.elbow([(g[0]+g[2], 175), (1168, 175), (1168, 190), (1176, 190)], color=C["blue"])
    s.elbow([(g[0]+g[2], 190), (1160, 190), (1160, 248), (1176, 248)], color=C["green"])

    s.text(492, 292, "可选 I1 层级父网格 + 3 维层级 one-hot（config 开关，默认关）", size=10.5, color=C["sub"])

    # ---------- 中：逐字段编码 ----------
    s.panel(24, 352, 1352, 330, "逐字段编码方式（encode_attributes：先 STE_multistep 量化，再算术编码）", "plain")
    cells = [
        ("坐标 xyz", ["G-PCC 八树编码", "→ xyz_gpcc.npz"], "blue", 40, 396, 288),
        ("哈希网格参数", ["p∈{±1} → (p+1)/2", "算术编码 → hash.b"], "blue", 364, 396, 288),
        ("masks [N,K,1]", ["二值化后算术编码", "→ masks.b"], "blue", 688, 396, 288),
        ("feat [N,32]", ["通道组自回归（cg=8）：mlp_deform 由已解码", "通道组预测 (mean,scale,prob) 调整 → 双高斯", "混合熵 EG_mix_prob_2 逐块算术编码"], "purple", 40, 508, 440),
        ("scaling [N,6]", ["高斯熵模型（mlp_grid 预测 mean/scale）", "→ encoder_gaussian_chunk"], "purple", 512, 508, 300),
        ("offsets [N,3K]", ["高斯熵模型；仅对 mask=1 的元素编码", "（掩码即稀疏性来源）"], "purple", 844, 508, 300),
        ("可选：attr_ctx R4 预测器", ["用已解码 feat 调整", "scaling/offsets 熵参数", "载荷计入 total_MB（默认关）"], "grey", 1180, 508, 186),
    ]
    for t, subs, ck, x, y, w in cells:
        s.box(x, y, w, t, subs, ck, tsize=13.5, dashed=(ck == "grey"))

    # ---------- 下：体积构成 ----------
    s.panel(24, 700, 1352, 130, "体积构成 total_MB（hac_meta.json 逐项记录，fp32 口径）", "purple")
    s.text(40, 734, "total_bits = bit_anchor(G-PCC) + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + bit_attr_ctx(可选)", size=13, color=C["ink"])
    s.text(40, 758, "             + bit_mlp（全部 mlp_* 参数 × 32bit，含 mlp_complexity）+ bit_bounds（2×[3] fp32）+ header", size=13, color=C["ink"])
    s.text(40, 786, "total_MB = total_bits / 8×1024×1024      ·      后处理旋钮：q_scale_{feat,scaling,offsets} 与 mask_keep_ratio（编码时再丢低分锚）", size=12.5, color=C["sub"])

    s.footer("scaffold_gs/hacpp.py:1344–1711（encode_attributes） · hacplus/utils/entropy_models.py · gpcc_utils.py · hacpp.py:1235–1342（5% 采样率估计）",
             note="量化步长 Q 的来源见 fig04")
    s.save(f"{outdir}/fig03-entropy-coding.svg")


# =====================================================================
def fig04(outdir):
    s = SVG(1360, 820, "创新①：复杂度乘子——解码端可复算的内容自适应量化步长",
            "四个结构统计量 → 共享小 MLP → m = 1+tanh(·)·strength 重缩放 AQM 步长；训练/编码/解码三端走同一路径，权重随熵模型进码流，零侧信息")

    # 左：四个统计量
    s.panel(24, 92, 400, 560, "结构统计量 x [N,4]（全部由已解码量重算）", "green")
    st = [
        ("局部密度", ["ρ = exp(−d_nn / voxel)", "d_nn = 最近邻锚点距离", "（N>4096 时 4096 采样近似）"]),
        ("尺度各向异性", ["std( 预测平均 scaling 前三维 )", "预测值来自 mlp_grid（码流内）"]),
        ("偏移能量", ["mean | 预测平均 offsets |", "（K×3，预测值来自 mlp_grid）"]),
        ("活跃掩码率", ["mean( masks )", "存活子高斯比例"]),
    ]
    yy = 132
    for t, sub in st:
        s.box(40, yy, 368, t, sub, "green", tsize=13.5)
        yy += 106
    s.text(214, yy + 8, "来源：解码后的坐标 / 熵参数 / 掩码", size=11.5, color=C["green"], anchor="middle")
    s.text(214, yy + 26, "→ 解码端零成本获得，无需任何侧信息", size=11.5, color=C["green"], anchor="middle")

    # 中：MLP 与乘子
    s.panel(452, 92, 380, 560, "乘子生成（mlp_complexity）", "green")
    n1 = s.box(468, 136, 348, "mlp_complexity", ["Linear(4,16) → ReLU → Linear(16,3)", "hidden = feat_dim/2 = 16"], "green", tsize=14)
    n2 = s.box(468, 232, 348, "m = 1 + tanh(logits) · strength", [
        "logits [N,3] → 逐字段乘子 m₀ m₁ m₂",
        "strength = complexity_scale × ramp", "= 0.35 × clamp((t−20000)/10000, 0, 1)"], "green", tsize=14)
    s.arrow(642, n1[1]+n1[3], 642, n2[1]-3, color=C["green"])
    # ramp 小图
    rx, ry, rw, rh = 468, 352, 348, 110
    s.rrect(rx, ry, rw, rh, rx=8, fill="#ffffff", stroke=C["green"], sw=1.4)
    s.text(rx+16, ry+20, "strength ramp（t: 20k → 30k 线性升 0 → 0.35）", size=11.5, color=C["green"])
    s.line(rx+40, ry+rh-22, rx+rw-30, ry+rh-22, color=C["sub"], sw=1.2)
    s.line(rx+40, ry+rh-22, rx+40, ry+18, color=C["sub"], sw=1.2)
    s.path(f"M{rx+40},{ry+rh-22} L{rx+120},{ry+rh-22} L{rx+300},{ry+26}", color=C["green"], sw=2.2, arrow=False)
    s.parts.append(f"<path d='M{rx+300},{ry+26} l-9,2 l3,9 z' fill='{C['green']}'/>")
    s.text(rx+300, ry+rh-8, "t=30k →0.35", size=10.5, color=C["green"], anchor="middle")
    s.text(rx+40, ry+rh-8, "t=20k →0", size=10.5, color=C["green"], anchor="middle")
    n3 = s.box(468, 490, 348, "为何要 ramp", [
        "20k 前保持 HAC++ 原行为（m≡1）",
        "量化感知训练从原行为平滑出发，避免突变"], "plain", tsize=12.5)

    # 右：Q 公式
    s.panel(860, 92, 476, 560, "最终量化步长（Q0 = HAC++ 基础步长）", "plain")
    q1 = s.box(876, 132, 444, "Q_feat = 1.0 · (1+tanh qa) · m₀", ["基础 1.0 · AQM 调整 · 内容乘子"], "green", tsize=14)
    q2 = s.box(876, 220, 444, "Q_scaling = 0.001 · (1+tanh qs) · m₁", ["基础 0.001"], "green", tsize=14)
    q3 = s.box(876, 308, 444, "Q_offsets = 0.2 · (1+tanh qo) · m₂", ["基础 0.2"], "green", tsize=14)
    q4 = s.box(876, 396, 444, "使用位置", [
        "训练 >10k：属性 + (rand−0.5)·Q 噪声仿真",
        "评估/编码：STE_multistep(x, Q, 均值) 硬量化",
        "解码：同一公式重算 Q → 反量化"], "plain", tsize=12.5)
    s.text(1098, 560, "直觉", size=13, color=C["ink"], anchor="middle", weight="bold")
    s.text(876, 584, "密集 · 各向异性 · 高偏移能量 · 多活跃偏移  →  m<1  →  细 Q（费比特保质量）", size=12.5, color=C["ink"])
    s.text(876, 606, "稀疏 · 各向同性 · 低能量（平滑大区域）  →  m>1  →  粗 Q（省比特）", size=12.5, color=C["ink"])

    # 箭头
    s.arrow(424, 372, 466, 372, color=C["green"])
    s.elbow([(832, 372), (852, 372), (852, 340), (872, 340)], color=C["green"], label=None)
    s.text(852, 362, "m", size=12, color=C["green"], anchor="middle")

    # 顶部契约条
    s.panel(24, 672, 1312, 104, "三端一致契约（zero-side-information）", "green")
    s.text(40, 712, "训练渲染 = 率估计 = 编码 = 解码，四处调用同一 build_formula_complexity_input → mlp_complexity → Q 路径（hacpp.py:610/683/1271/1526/1845）", size=12.5, color=C["ink"])
    s.text(40, 736, "mlp_complexity 权重按 mlp_* 参数计入 bit_mlp 随码流传给解码端；输入 x 在解码端由已解码量重算 → 编解码 Q 逐位一致（bit-exact 校验通过）", size=12.5, color=C["ink"])
    s.text(40, 760, "消融：去掉乘子（content_aware_quant=False）在 1-78 λ=0.002 掉 0.41 dB；注意：论文正文的 8→32→3 为旧版，PHG v2 删除 4 个恒零图像统计后现为 4→16→3（旧检查点 8 维切片兼容）", size=12, color=C["red"])

    s.footer("hacplus/scene/gaussian_model.py:674–808 · hacplus/utils/codec_consistency.py:77–137 · scaffold_gs/config.py:94–103",
             note="敏感度如何监督 mlp_complexity → fig05")
    s.save(f"{outdir}/fig04-complexity-multiplier.svg")


# =====================================================================
def fig05(outdir):
    s = SVG(1340, 800, "创新①（监督侧）：渲染敏感度信号链——只在训练里存在的“重要性标注员”",
            "反传后取各锚点属性的梯度范数做 EMA；以相对归一化 z 构造有界 target，回归监督 mlp_complexity；解码端从头到尾不需要它")

    # 上：信号采集链
    s.panel(24, 92, 1292, 250, "信号采集（每步、反传之后）", "grey")
    a1 = s.box(40, 134, 210, "retain_grad", ["渲染前对 feat /", "grid_scaling / grid_offsets", "保留梯度"], "blue", tsize=13)
    a2 = s.box(294, 134, 190, "loss.backward()", ["（L 含全部项）"], "blue", tsize=13)
    a3 = s.box(528, 134, 250, "g = ‖∂L/∂a‖₂", ["逐锚点 · 逐字段梯度范数", "（按 anchor_indices 归并到 N）"], "grey", tsize=13, dashed=True)
    a4 = s.box(822, 134, 240, "EMA  α = 0.99", ["s ← 0.99·s + 0.01·g", "index_add 仅本轮可见锚"], "grey", tsize=13, dashed=True)
    a5 = s.box(1106, 134, 194, "敏感度状态", ["sensitivity_feat /", "scaling / offsets", "各 [N,1]（+mean/var EMA）"], "grey", tsize=12.5, dashed=True)
    for a, b in ((a1, a2), (a2, a3), (a3, a4), (a4, a5)):
        s.arrow(a[0]+a[2]+3, 176, b[0]-4, 176, color=C["grey"], dashed=True)
    s.chips(40, 262, [("生效条件: sensitivity_enabled ∧ is_training ∧ t ≥ sensitivity_start_iter（默认 20000）", "amber")])

    # 中：监督支路
    s.panel(24, 366, 780, 300, "监督支路（t ≥ 20000，L_sens 项）", "green")
    b1 = s.box(40, 408, 340, "z = (s − mean) / mean", [
        "相对归一化（以均值为中心）",
        "注：方差 z-score 在 α=0.99 下不收敛，", "信号会被压平 → 改用相对归一化"], "plain", tsize=13)
    b2 = s.box(420, 408, 372, "target = clamp(1 + tanh(−z), 0.1, 2.0)", [
        "高敏感 z>0 → target<1（要细 Q）",
        "低敏感 z<0 → target>1（可粗 Q）",
        "clamp 防止 Q→0 时算术编码 CDF 出 NaN"], "green", tsize=13)
    s.arrow(380, 452, 416, 452, color=C["green"])
    b3 = s.box(40, 530, 340, "pred = 1 + tanh(mlp_complexity logits)", [
        "可微 → 梯度直达 mlp_complexity",
        "target 侧 detach（只当标签用）"], "green", tsize=13)
    b4 = s.box(420, 530, 372, "L_sens = 1e-3 · MSE(pred, target)", [
        "sensitivity_weight = λsens = 1e-3",
        "“从结构预测重要性”：结构在 x 里，", "重要性标签来自渲染梯度"], "green", tsize=13)
    s.arrow(380, 574, 416, 574, color=C["green"])
    s.elbow([(b2[0]+186, b2[1]+b2[3]), (b2[0]+186, 520), (b4[0]+186, b4[1]-4)], color=C["green"], dashed=True, label=None)
    s.text(606, 516, "target", size=10.5, color=C["green"], anchor="middle")

    # 右：下游用户 + 契约
    s.panel(830, 366, 486, 300, "两个下游用途（都只发生在训练侧）", "plain")
    u1 = s.box(846, 408, 454, "① 监督 mlp_complexity（fig04）", [
        "让 m 学会“哪里该细/粗量化”",
        "训练结束后标签即弃"], "green", tsize=13)
    u2 = s.box(846, 530, 454, "② 融合剪枝的敏感度项 t_i（fig07）", [
        "s = log1p(s_f+s_s+s_o) / max",
        "作为剪枝分数的次要成分（份额 ≤0.3）"], "orange", tsize=13)
    s.elbow([(822, 460), (830, 460)], color=C["grey"], dashed=True, arrow=True, label=None)
    s.elbow([(822, 580), (830, 580)], color=C["grey"], dashed=True, arrow=True, label=None)
    s.text(826, 336, "s", size=12, color=C["grey"], anchor="middle")

    # 底部强调
    s.panel(24, 690, 1292, 66, "", "red")
    s.text(40, 716, "契约审计：EMA 状态、梯度、target、z——全部只活在训练进程里；码流与解码路径不含任何一个（解码端只运行 mlp_complexity(x)，见 fig04/Table I）", size=13, color=C["ink"])
    s.text(40, 740, "方向性：敏感度高 = 该锚点属性一动画面就变 = 对图像重要 = 值得细量化 / 值得留下（fig07）", size=12.5, color=C["sub"])

    s.footer("scaffold_gs/hacpp.py:713–721（retain_grad） · 1152–1183（EMA） · 1011–1047（L_sens） · hacpp.py:90–97（clamp 理由）",
             note="α=0.99 · λsens=1e-3 · strength=1.0")
    s.save(f"{outdir}/fig05-sensitivity.svg")
