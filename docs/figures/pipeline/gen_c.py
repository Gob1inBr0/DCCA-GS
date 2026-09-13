# -*- coding: utf-8 -*-
"""fig06 SPA-ADMM / fig07 融合分数（创新②）/ fig08 覆盖约束（创新②）/ fig09 深度再初始化。"""
from svgkit import SVG, C, text_width


# =====================================================================
def fig06(outdir):
    s = SVG(1340, 860, "基础机制：SPA-ADMM 预算稀疏化（GaussianSpa，决定“留多少锚”）",
            "软掩码分数 a + 对偶 u 做 TopK 硬投影，预算 κ 随训练从 N_max 线性压到 0.85·N_max；创新②只替换“排序分数”与“投影规则”（→ fig07/08），ADMM 骨架原样保留")

    # 时机条
    s.panel(24, 88, 1292, 88, "调用时机", "plain")
    s.text(40, 128, "每 100 步（1500 < it < 15000，update_interval=100）在 backward 之后、optimizer.step 之前执行 adjust_anchor：先生长（offset 梯度 > 2e-4 处补锚），再 SPA 投影", size=12.5, color=C["ink"])
    s.text(40, 152, "15000 不含 → 最后一次投影发生在 @14900；训练损失侧每步都有 ADMM 二次项 L_spa = (ρ/2)·mean‖a−z+u‖²（ρ=1e-3）把 a 拉向 z", size=12.5, color=C["ink"])

    # 主流程
    s.panel(24, 196, 700, 400, "投影步（每 100 步）", "blue")
    p1 = s.box(40, 240, 320, "a = mean(get_mask, dim=1) [N,1]", [
        "软锚点分数：K 个偏移掩码的均值",
        "（sigmoid STE，可导，被 L_spa 优化）"], "blue", tsize=13)
    p2 = s.box(40, 348, 320, "scores = a + u", ["u：ADMM 对偶变量 [N,1]", "（把历史未入选的惩罚抬进分数）"], "blue", tsize=13)
    p3 = s.box(40, 456, 320, "z = TopK(scores, κ)", ["硬稀疏指示 z∈{0,1}[N]", "（或覆盖约束版 → fig08）"], "orange", tsize=13)
    s.arrow(200, p1[1]+p1[3], 200, p2[1]-3, color=C["blue"])
    s.arrow(200, p2[1]+p2[3], 200, p3[1]-3, color=C["blue"])
    p4 = s.box(392, 240, 316, "u ← clamp(u + a − z, ±1)", ["对偶更新（±1 截断保稳定）"], "blue", tsize=13)
    p5 = s.box(392, 348, 316, "prune：删去 z=0 的锚", ["物理删除属性与优化器状态"], "blue", tsize=13)
    p6 = s.box(392, 456, 316, "状态张量同步切片", ["见下方清单（长度必须对齐）"], "red", tsize=13)
    s.elbow([(360, 500), (376, 500), (376, 276), (388, 276)], color=C["blue"])
    s.elbow([(475, 348), (475, 340)], color=C["blue"], arrow=True)
    s.arrow(550, p4[1]+p4[3], 550, p5[1]-3, color=C["blue"])
    s.arrow(550, p5[1]+p5[3], 550, p6[1]-3, color=C["blue"])

    # 预算 ramp
    s.panel(748, 196, 568, 400, "预算 κ 的 ramp（锚定历史最大锚数，防反馈塌缩）", "orange")
    s.text(764, 232, "spa_ref_n = max(spa_ref_n, N)   （历史最大锚数）", size=12.5, color=C["ink"])
    s.text(764, 258, "progress = clamp((t − 1500) / 13500, 0, 1)", size=12.5, color=C["ink"])
    s.text(764, 284, "κ = round( spa_ref_n · (1 − 0.15 · progress) )     （1500 → 15000：κ/N 1.0 → 0.85）", size=12.5, color=C["ink"])
    s.text(764, 310, "t ≥ 15000：κ = round( spa_final_n · 0.85 )（spa_final_n 在首次越过 15000 时锁存；", size=12.5, color=C["sub"])
    s.text(764, 332, "深度再初始化会把 spa_final_n 钉在再初始化前的锚数 → fig09）", size=12.5, color=C["sub"])
    rx, ry, rw, rh = 764, 356, 536, 150
    s.rrect(rx, ry, rw, rh, rx=8, fill="#ffffff", stroke=C["orange"], sw=1.4)
    s.text(rx+14, ry+20, "κ(t) 示意", size=11.5, color=C["orange"])
    s.line(rx+50, ry+rh-26, rx+rw-24, ry+rh-26, color=C["sub"], sw=1.2)
    s.line(rx+50, ry+rh-26, rx+50, ry+20, color=C["sub"], sw=1.2)
    x0, x1 = rx+50, rx+rw-24
    txm = lambda t: x0 + (x1-x0) * t
    s.line(txm(0), ry+rh-26, txm(0), ry+40, color=C["sub"], sw=1, dashed=True)
    s.line(txm(1), ry+rh-26, txm(1), ry+52, color=C["sub"], sw=1, dashed=True)
    s.path(f"M{x0},{ry+40} L{txm(1)},{ry+52} L{txm(1)},{ry+rh-26}", fill="none", color=C["orange"], sw=2.4)
    s.text(x0, ry+rh-10, "t=1500（κ=N_max）", size=10.5, color=C["sub"], anchor="middle")
    s.text(txm(1), ry+rh-10, "t=15000 → 0.85·N_max", size=10.5, color=C["sub"], anchor="end")
    s.text(txm(1)-6, ry+48, "之后 0.85·spa_final_n", size=10.5, color=C["sub"], anchor="end")

    # 状态切片清单
    s.panel(24, 618, 1292, 160, "prune 时的状态切片清单（全部按布尔掩码同步切片，新增 per-anchor 状态必须加进来，否则长度错位 → 运行期报错/静默错位）", "red")
    from svgkit import text_width as _tw
    items = ["offset_denom", "offset_gradient_accum", "opacity_accum", "anchor_demon",
             "sensitivity_feat/scaling/offsets", "spa_z", "spa_u", "mini_splat_importance（覆盖面积）"]
    cx, cy = 44, 668
    for it in items:
        w = _tw(it, 11.5) + 14
        if cx + w > 1296:
            cx, cy = 44, cy + 38
        s.chip(cx, cy, it, "grey", size=11.5)
        cx += w + 12
    s.text(44, 754, "生长新增的锚同样要 pad 这些状态（新锚敏感度置 0、统计清零、SPA 状态初始化）", size=12, color=C["sub"])

    s.footer("hacplus/scene/gaussian_model.py:1579–1806（adjust_anchor） · scaffold_gs/hacpp.py:1143–1150（L_spa） · config.py:161–168",
             note="spa_ratio=0.85 · ρ=1e-3 · u∈±1")
    s.save(f"{outdir}/fig06-spa-admm.svg")


# =====================================================================
def fig07(outdir):
    s = SVG(1380, 900, "创新②（软的部分）：覆盖主导的融合剪枝分数——决定“留哪些锚”",
            "覆盖面积为主、敏感度为辅（份额硬帽 ≤0.3）、ADMM 分数为 tiebreaker：scores = imp/max + 0.25·(a+u)，其中 imp = (1−eff)·cov + eff·s")

    # 信号源
    s.panel(24, 92, 1332, 190, "三个信号源（训练侧，每 100 步投影前刷新）", "plain")
    c1 = s.box(40, 130, 400, "覆盖 c_i（主力）", [
        "8 个均匀采样训练视角 · 每像素 argmax 贡献者权重",
        "逐锚取最大贡献像素面积 / (H·W) → cov = c / c_max",
        "“这个锚替画面撑了多大一块地盘”"], "orange", tsize=13.5)
    c2 = s.box(476, 130, 396, "敏感度 s_i（辅助，硬帽 ≤0.3）", [
        "s = log1p(s_f+s_s+s_o) / max（EMA 状态，→ fig05）",
        "“这个锚一动，画面变多少”"], "grey", tsize=13.5, dashed=True)
    c3 = s.box(908, 130, 432, "ADMM 分数 a + u（tiebreaker）", [
        "软掩码分数 + 对偶（→ fig06）",
        "保留原有 SPA 行为的连续性"], "blue", tsize=13.5)

    # 融合
    s.panel(24, 302, 1332, 210, "融合（在 fig06 的 TopK 之前替换排序分数）", "orange")
    f1 = s.box(40, 344, 300, "eff = min(w, 0.3)", [
        "硬帽！无论配置 w 多大",
        "敏感度份额不得超过 0.3"], "orange", tsize=14)
    f2 = s.box(376, 344, 380, "imp = (1−eff)·cov + eff·s", ["覆盖恒为主导项（≥0.7）"], "orange", tsize=14)
    f3 = s.box(792, 344, 300, "imp ← imp / imp.max()", ["归一化到 [0,1]"], "orange", tsize=14)
    f4 = s.box(40, 442, 700, "最终排序分数  scores = imp + 0.25 · (a + u)", [
        "mini_splat_importance_weight = 0.25：ADMM 分数只作次级调节"], "orange", tsize=14)
    f5 = s.box(776, 442, 564, "→ 送入 TopK / 覆盖约束投影（fig08）", [
        "“先别把地盘拆了，再谈省比特”"], "orange", tsize=14)
    s.arrow(340, 388, 372, 388, color=C["orange"])
    s.arrow(756, 388, 788, 388, color=C["orange"])
    s.elbow([(942, 396), (942, 420), (744, 420), (744, 438)], color=C["orange"])
    s.arrow(744, 486, 772, 486, color=C["orange"])
    # 信号源箭头
    s.elbow([(240, 282), (240, 296), (566, 296), (566, 340)], color=C["orange"], label=None)
    s.elbow([(674, 282), (674, 340)], color=C["grey"], dashed=True)
    s.elbow([(1124, 282), (1124, 428), (712, 428), (712, 438)], color=C["blue"], label=None)
    s.text(690, 298, "s", size=11.5, color=C["grey"])

    # 失败模式
    s.panel(24, 532, 700, 268, "为什么必须有硬帽：实测的失败模式", "red")
    s.text(40, 566, "w = 0.5（无帽）@ 1-78 · λ=0.004：", size=13, color=C["red"], weight="bold")
    s.text(40, 590, "解码 PSNR 27.51 → 25.95 dB；锚点数几乎不变（507.8k vs 506.0k）", size=12.5, color=C["ink"])
    s.text(40, 612, "低码率段塌缩幅度 1.62 dB（1-78，论文口径）", size=12.5, color=C["ink"])
    s.text(40, 638, "机理：敏感度主导 → 留高敏感“细节锚”、删大覆盖背景锚", size=12.5, color=C["ink"])
    s.text(40, 660, "→ 覆盖洞——不是“剪多了”，是“剪错了”", size=12.5, color=C["red"], weight="bold")

    def scene(x, y, title, hole):
        s.rrect(x, y, 300, 86, rx=8, fill="#ffffff", stroke=C["frame"], sw=1.2)
        s.text(x + 150, y + 16, title, size=11.5, color=C["sub"], anchor="middle")
        bg = [(x+45, y+42, 14), (x+105, y+38, 12), (x+165, y+46, 13), (x+245, y+40, 13)]
        dt = [(x+65, y+68, 3.5), (x+125, y+70, 3.2), (x+185, y+68, 3.5), (x+225, y+72, 3.2), (x+262, y+68, 3.5)]
        keep_bg = [bg[0], bg[2], bg[3]] if hole else bg
        for (bx, by, r) in keep_bg:
            s.parts.append(f"<circle cx='{bx}' cy='{by}' r='{r}' fill='{C['orange_bg']}' stroke='{C['orange']}' stroke-width='1.6'/>")
        if hole:
            s.parts.append(f"<circle cx='{bg[1][0]}' cy='{bg[1][1]}' r='{bg[1][2]}' fill='none' stroke='{C['red']}' stroke-width='1.6' stroke-dasharray='4,3'/>")
            s.text(bg[1][0], bg[1][1] + 28, "覆盖洞!", size=10.5, color=C["red"], anchor="middle", weight="bold")
        for (bx, by, r) in dt:
            s.parts.append(f"<circle cx='{bx}' cy='{by}' r='{r}' fill='{C['grey_bg']}' stroke='{C['grey']}' stroke-width='1.2'/>")
    scene(46, 676, "w ≤ 0.3（覆盖主导）：大覆盖锚都在", hole=False)
    scene(376, 676, "w = 0.5（敏感度主导）：大覆盖锚被删", hole=True)
    s.text(366, 786, "大圆 = 大覆盖锚 · 小点 = 高敏感细节锚", size=10.5, color=C["sub"], anchor="middle")

    # 接线与降级
    s.panel(748, 532, 608, 268, "接线条件与降级（审计后加固）", "plain")
    s.text(764, 566, "· fusion_prune 需同时启用 mini_splat：覆盖 provider 才会在 trainer 接线", size=12.5, color=C["ink"])
    s.text(764, 592, "· 敏感度状态长度不符（如刚生长后）→ 自动降级 coverage-only，", size=12.5, color=C["ink"])
    s.text(776, 614, "打印 [FusionPrune] SENSITIVITY_UNAVAILABLE（不再静默）", size=12, color=C["sub"])
    s.text(764, 640, "· 每次投影打印 [FusionPrune] engaged=1 eff=… cov_mean=… sens_mean=…，", size=12.5, color=C["ink"])
    s.text(776, 662, "engaged=0 即视为无效运行（→ fig12 静默门控审计）", size=12, color=C["sub"])
    s.text(764, 692, "· 默认 fusion_sensitivity_weight = 0.3（正好在帽上）", size=12.5, color=C["ink"])

    s.footer("hacplus/scene/gaussian_model.py:1670–1721（融合分数） · scaffold_gs/mini_splat.py:282–305（覆盖 provider） · trainer.py:192–208（接线）",
             note="覆盖约束（硬保证）→ fig08")
    s.save(f"{outdir}/fig07-fusion-score.svg")


# =====================================================================
def fig08(outdir):
    s = SVG(1340, 800, "创新②（硬的部分）：ADMM 覆盖约束——每个被占用粗胞至少留 1 锚",
            "在 TopK 之前先“保护”每个被占用胞的最高分锚：无论融合分数说什么，场景任何区域都不会失去代表；解码零成本")

    # 算法步骤
    s.panel(24, 92, 640, 560, "投影步的修改（对照 fig06 的 z = TopK）", "orange")
    st1 = s.box(40, 132, 608, "① 粗胞划分：cell = floor(anchor_xyz / 0.01) [N,3]", [
        "0.01 为场景单位的胞尺寸（spa_coverage_cell_size）"], "orange", tsize=13.5)
    st2 = s.box(40, 224, 608, "② unique(cell) → 被占用胞集合", [], "orange", tsize=13.5)
    st3 = s.box(40, 300, 608, "③ scores 降序排序，每个占用胞保前 mn=1 名 → mandated 集合", [
        "“胞内最优”而不是“全局最优”"], "orange", tsize=13.5)
    st4 = s.box(40, 388, 608, "④ keep = |mandated| ≥ κ ? mandated 前 κ : mandated ∪ 按分数补齐到 κ", [
        "补齐（fill）仍然按全局分数从高到低"], "orange", tsize=13)
    st5 = s.box(40, 476, 608, "⑤ z[keep]=1 → 对偶更新 / 切片 / 生长统计照旧（fig06）", [], "orange", tsize=13.5)
    for a, b in ((st1, st2), (st2, st3), (st3, st4), (st4, st5)):
        s.arrow(344, a[1]+a[3], 344, b[1]-3, color=C["orange"])

    # 网格示意
    s.panel(688, 92, 628, 340, "2D 示意（实际为 3D 粗胞；圆点=锚，大小示意覆盖/分数）", "plain")
    gx, gy, cw, chh = 716, 132, 138, 150
    cells = [(gx, "高分聚集胞", "也只强制保 1（胞内最高分）"),
             (gx + cw + 8, "普通胞", "按分数正常参与 TopK"),
             (gx + 2 * (cw + 8), "独占胞", "低分但被 mandated 保护"),
             (gx + 3 * (cw + 8), "若被删（无保护）", "该区域再无代表 → 覆盖洞")]
    for cx0, t, sub in cells:
        s.rrect(cx0, gy, cw, chh, rx=6, fill="#ffffff", stroke=C["frame"], sw=1.2)
        s.text(cx0 + cw / 2, gy + chh + 18, t, size=11.5, color=C["ink"], anchor="middle", weight="bold")
        s.text(cx0 + cw / 2, gy + chh + 36, sub, size=10, color=C["sub"], anchor="middle")
    # 胞1：高分聚集
    x0 = gx
    ptsA = [(x0+34, gy+38), (x0+64, gy+64), (x0+100, gy+36), (x0+44, gy+104), (x0+92, gy+108)]
    for (px, py) in ptsA:
        s.parts.append(f"<circle cx='{px}' cy='{py}' r='7' fill='{C['orange_bg']}' stroke='{C['orange']}' stroke-width='1.6'/>")
    s.parts.append(f"<circle cx='{ptsA[0][0]}' cy='{ptsA[0][1]}' r='11.5' fill='none' stroke='{C['green']}' stroke-width='2.2'/>")
    # 胞2：普通
    x0 = gx + cw + 8
    for (px, py) in [(x0+38, gy+44), (x0+82, gy+72), (x0+34, gy+102), (x0+100, gy+104)]:
        s.parts.append(f"<circle cx='{px}' cy='{py}' r='5.5' fill='{C['grey_bg']}' stroke='{C['grey']}' stroke-width='1.4'/>")
    # 胞3：独占低分锚（红星）
    x0 = gx + 2 * (cw + 8)
    bx, by = x0 + cw / 2, gy + 78
    s.parts.append(f"<circle cx='{bx}' cy='{by}' r='7' fill='{C['grey_bg']}' stroke='{C['grey']}' stroke-width='1.6'/>")
    s.parts.append(f"<path d='M{bx},{by-26} L{bx+6},{by-14} L{bx+19},{by-12} L{bx+10},{by-2} L{bx+12},{by+11} L{bx},{by+5} L{bx-12},{by+11} L{bx-10},{by-2} L{bx-19},{by-12} L{bx-6},{by-14} Z' fill='{C['red']}'/>")
    # 胞4：空洞
    x0 = gx + 3 * (cw + 8)
    s.parts.append(f"<circle cx='{x0+cw/2}' cy='{gy+78}' r='13' fill='#fdecea' stroke='{C['red']}' stroke-width='1.8' stroke-dasharray='5,4'/>")
    s.text(x0 + cw / 2, gy + 82, "✗", size=16, color=C["red"], anchor="middle", weight="bold")

    # 特性
    s.panel(688, 452, 628, 200, "特性", "green")
    s.text(704, 486, "· 无条件：不依赖分数是否合理——失败模式在构造上不可能", size=12.5, color=C["ink"])
    s.text(704, 512, "· 解码零成本：纯训练侧投影规则，码流不变", size=12.5, color=C["ink"])
    s.text(704, 538, "· 单独启用时锚点数几乎不变（4-10：671,570 vs 671,497）——它是保险，不是旋钮", size=12.5, color=C["ink"])
    s.text(704, 564, "· 与 fig07 软帽互补：软帽防分数失衡，硬约束兜底任何残余风险；", size=12.5, color=C["ink"])
    s.text(716, 586, "正是有了它，融合剪枝才敢在低码率档位启用", size=12.5, color=C["ink"])
    s.text(704, 614, "注：论文正文 δ=0.05 为旧值，代码/config 实际默认 0.01 · mn=1", size=11.5, color=C["red"])

    # 底部关系
    s.panel(24, 672, 1292, 80, "与 fig06/fig07 的关系", "plain")
    s.text(40, 712, "fig06 ADMM 决定“留多少”（κ）；fig07 融合分数决定“按什么排序”；本图决定“无论怎么排，每个区域都有人留守”。三者叠加 = 覆盖感知融合剪枝", size=13, color=C["ink"])
    s.text(40, 736, "默认关闭（spa_coverage_constraint=False），低码率档位与融合剪枝一起启用", size=12, color=C["sub"])

    s.footer("hacplus/scene/gaussian_model.py:1725–1757（胞划分/mandated/fill） · config.py:193–198",
             note="cell=0.01 · min_per_cell=1")
    s.save(f"{outdir}/fig08-coverage-constraint.svg")


# =====================================================================
def fig09(outdir):
    s = SVG(1360, 860, "深度再初始化：@15000 用深度反投影给锚点“搬家”（预算不增）",
            "SfM 初始点聚集偏离真实表面 → 停止生长时渲染深度、反投影出表面候选锚；预算 κ 钉死，候选必须挤掉旧锚才能活——向表面迁移而不长大模型")

    # 步骤流
    s.panel(24, 92, 700, 580, "流程（mini_splat_reinit，一次性）", "orange")
    k1 = s.box(40, 132, 668, "① 触发：it = 15000（mini_splat_reinit_iter，恰为生长停止刻）", [], "orange", tsize=13)
    k2 = s.box(40, 196, 668, "② 锁预算：spa_final_n ← 当前锚数 N_before（此后 κ 不再上涨）", [
        "再初始化不给“免费名额”"], "red", tsize=13)
    k3 = s.box(40, 274, 668, "③ 均匀采样 mini_splat_views = 8 个训练相机", [], "orange", tsize=13)
    k4 = s.box(40, 338, 668, "④ 逐视角：gsplat 渲染深度（render_mode=\"D\"）", [
        "有效像素：depth>0 ∧ alpha>0.05 · 用 K、c2w 反投影为表面点"], "orange", tsize=13)
    k5 = s.box(40, 418, 668, "⑤ 体素化：每占用体素取均值点 → _voxel_subsample 上限 4000", [
        "seed=42 确定性子采样（mini_splat_max_new）"], "orange", tsize=13)
    k6 = s.box(40, 492, 668, "⑥ append_depth_anchors：新锚 scaling=log(voxel)，offsets=0，", [
        "masks=1，opacity=0.1，feat=0；全部 per-anchor 状态 pad（fig06 清单）"], "orange", tsize=13)
    for a, b in ((k1, k2), (k2, k3), (k3, k4), (k4, k5), (k5, k6)):
        s.arrow(374, a[1]+a[3], 374, b[1]-3, color=C["orange"])

    # 右：竞争 + full 模式
    s.panel(748, 92, 588, 300, "之后：竞争淘汰（核心设计）", "orange")
    s.box(764, 132, 556, "候选 ≠ 保留", [
        "后续 SPA 投影 κ 不变（0.85·spa_final_n）",
        "新表面候选必须在融合分数（fig07）下挤掉旧锚",
        "→ 锚点集合向真实表面迁移，模型体积不增长",
        "等效于“用更好的位置换掉差的位置”"], "orange", tsize=13.5)

    s.panel(748, 412, 588, 260, "full 模式（mini_splat_full=True，默认关）", "grey")
    s.box(764, 452, 556, "附加两步", [
        "· blur split：贡献面积 > 0.01 的锚在其子高斯 xyz 处分裂新锚",
        "· 按贡献面积 top-κ 简化，spa_final_n=κ、spa_ratio=1.0",
        "（借鉴 Mini-Splatting 的 blur split；主路径只取深度反投影思想）"], "grey", tsize=12.5, dashed=True)

    # 底部动机
    s.panel(24, 692, 1312, 100, "动机与边界", "plain")
    s.text(40, 732, "动机：锚点初始位置来自 SfM 稀疏点，天然聚集在纹理区；生长停止后位置不再更新 → 深度反投影是唯一低成本的“纠位”机会", size=13, color=C["ink"])
    s.text(40, 758, "边界：只借 Mini-Splatting 的“深度反投影再初始化”思想；不采其无预算分裂——所有新锚进入既有 ADMM 预算竞争（这是与 Mini-Splatting 的本质区别）", size=13, color=C["ink"])

    s.footer("scaffold_gs/hacpp.py:878–995（mini_splat_reinit） · scaffold_gs/mini_splat.py:29–174/282–392 · hacplus/scene/gaussian_model.py:1472–1577（append）",
             note="views=8 · max_new=4000 · seed=42")
    s.save(f"{outdir}/fig09-depth-reinit.svg")
