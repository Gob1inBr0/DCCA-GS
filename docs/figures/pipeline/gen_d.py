# -*- coding: utf-8 -*-
"""fig10 编码↔解码 bit-exact / fig11 评估协议与结果 / fig12 审计与负结果。"""
from svgkit import SVG, C, text_width


# =====================================================================
def fig10(outdir):
    s = SVG(1420, 900, "编码 ↔ 解码：同一条 Q 路径跑两端，bit-exact 回传校验",
            "编码端先把坐标 voxel 对齐 + Morton 排序再算上下文（与训练一致）；解码端从码流重算 ctx → mlp_grid → Q → 反量化；整数锚与 masks 的 SHA-256 两端一致 → mismatch=0")

    # 共享路径条（顶部贯穿）
    s.panel(24, 88, 1372, 96, "共享 Q 路径（绿色 = 两端逐位一致的前提）", "green")
    s.text(40, 122, "round(x/voxel)·voxel → Morton 排序 → 哈希 ctx → mlp_grid → (qa,qs,qo) 与熵参数 → x=[ρ, 各向异性, 偏移能量, 掩码率] → mlp_complexity → m → Q", size=13, color=C["ink"])
    s.text(40, 150, "编码端在量化前跑这一整条；解码端在反量化前原样再跑一遍（mlp 权重与 strength=0.35 都由码流/头文件携带）→ 两端 Q 逐位相同", size=12.5, color=C["green"])

    # 编码端
    s.panel(24, 206, 430, 540, "编码端（encode_attributes）", "blue")
    e1 = s.box(40, 246, 398, "训练完成模型", ["活锚属性 · 哈希网格 · 全部 mlp_* 权重"], "blue", tsize=13.5)
    e2 = s.box(40, 330, 398, "坐标 voxel 对齐 + Morton 排序", ["round(x/v)·v（关键：先做这步，上下文才与训练/解码一致）"], "blue", tsize=13.5)
    e3 = s.box(40, 428, 398, "STE_multistep 硬量化", ["x_q = Q·round((x−均值)/Q)，均值取锚点均值", "得整数残差 → 算术编码"], "blue", tsize=13.5)
    e4 = s.box(40, 540, 398, "逐字段熵编码（fig03）", ["feat 混合熵 · scaling/offsets 高斯熵 · offsets 只编码 mask=1"], "blue", tsize=13.5)
    e5 = s.box(40, 638, 398, "体积核算 hac_meta.json", ["bit_anchor/feat/scaling/offsets/hash/masks/mlp/bounds/header"], "purple", tsize=13.5)
    for a, b in ((e1, e2), (e2, e3), (e3, e4), (e4, e5)):
        s.arrow(239, a[1]+a[3], 239, b[1]-3, color=C["blue"])

    # 码流
    s.panel(478, 206, 380, 540, "码流（全部文件）", "purple")
    files = [
        ("xyz_gpcc.npz", "G-PCC 八树坐标"),
        ("hash.b", "哈希网格 1bit 参数"),
        ("masks.b", "偏移掩码"),
        ("feat / scaling / offsets 码流", "算术编码整数"),
        ("attributes.pth", "锚点属性与状态"),
        ("hac_meta.json", "体积分解 + 配置"),
        ("content_aware_q.json · header", "strength/格式/公式版本"),
    ]
    fy = 242
    for t, sub in files:
        s.box(494, fy, 348, t, [sub], "purple", tsize=12.5, ssize=11, pad_v=9)
        fy += 64
    s.text(668, fy + 10, "✗ 敏感度 EMA · 覆盖面积 · 融合分数", size=12, color=C["red"], anchor="middle")
    s.text(668, fy + 30, "✗ spa_u/z · 语义 · 优化器状态（一律不进码流）", size=12, color=C["red"], anchor="middle")

    # 解码端
    s.panel(882, 206, 514, 540, "解码端（decode_attributes，无需预训练模型）", "blue")
    d1 = s.box(898, 246, 482, "读 header：codec / 公式版本 / strength 一致性校验", [], "blue", tsize=13)
    d2 = s.box(898, 310, 482, "GPCC 解坐标 → 整数锚 SHA-256 校验", ["masks 解码后同样校验"], "blue", tsize=13)
    d3 = s.box(898, 374, 482, "重算哈希 ctx → mlp_grid → 熵参数", ["网格参数来自 hash.b"], "blue", tsize=13)
    d4 = s.box(898, 438, 482, "逐字段算术解码", ["feat 通道组上下文用已解码通道组即时重建（mlp_deform）"], "blue", tsize=13)
    d5 = s.box(898, 502, 482, "Q 同式重算 → 反量化", ["m = 1+tanh(mlp_complexity(x))·0.35 与编码端逐位一致"], "green", tsize=13)
    d6 = s.box(898, 566, 482, "generate_gaussians(decoded=True)", ["跳过一切量化仿真，直接组装神经高斯"], "blue", tsize=13)
    d7 = s.box(898, 630, 482, "gsplat 渲染 → 评估（fig11）", [], "blue", tsize=13)
    for a, b in ((d1, d2), (d2, d3), (d3, d4), (d4, d5), (d5, d6), (d6, d7)):
        s.arrow(1139, a[1]+a[3], 1139, b[1]-3, color=C["blue"])
    s.arrow(862, 470, 896, 470, color=C["purple"], label="读码流", label_dy=-8)

    # 编码→码流
    s.arrow(462, 470, 476, 470, color=C["purple"], label="写")

    # 底部 bit-exact
    s.panel(24, 766, 1372, 96, "bit-exact 回传诊断（每个上报的码率点都跑）", "green")
    s.text(40, 806, "校验点：① 整数锚坐标 GPCC 往返 SHA-256 一致 ② masks 算术往返 SHA-256 一致 ③ header 的 codec/公式版本/strength 与编码端一致", size=13, color=C["ink"])
    s.text(40, 834, "产物 codec_roundtrip_diagnostics.json：bit_exact_roundtrip=true · mismatch=0 —— 编码端量到的体积/画质就是解码端实际得到的（论文协议的硬性前提）", size=13, color=C["ink"])

    s.footer("scaffold_gs/hacpp.py:1344–1711（encode） · 1745–1993（decode+诊断） · hacplus/utils/codec_consistency.py · scripts/rd_sweep.py",
             note="评估 → fig11")
    s.save(f"{outdir}/fig10-encode-decode.svg")


# =====================================================================
def fig11(outdir):
    s = SVG(1360, 840, "评估协议与已有结果：RD 扫描 · BD-rate · bit-exact",
            "统一协议下与 HAC++ 官方实现逐 λ 对比；每个点走完整 编码→解码→评估，解码模型必须 bit-exact")

    # 协议
    s.panel(24, 88, 1312, 96, "统一协议（所有表格/曲线共用）", "blue")
    s.chips(40, 118, [
        ("场景 1-78 · 2-06 · 4-10（大规模 UAV）", "blue"), ("30,000 迭代", "blue"),
        ("seed 42", "blue"), ("fp32 体积口径", "blue"),
        ("bit-exact 解码", "blue"), ("HAC++ 官方实现同协议对照", "blue"),
    ])

    # RD sweep
    s.panel(24, 204, 700, 300, "RD 扫描流程（scripts/rd_sweep.py）", "blue")
    r1 = s.box(40, 244, 640, "先存未压缩参考（全精度锚点）", [], "blue", tsize=13)
    r2 = s.box(40, 306, 640, "for λ ∈ {0.0005, 0.002, 0.004}：", [], "plain", tsize=13)
    r3 = s.box(40, 348, 640, "训练（该 λ 的 λ_rate）→ encode → decode（同 q_scale）→ evaluate", [], "blue", tsize=13)
    r4 = s.box(40, 410, 640, "编码旋钮：q_scale_{feat,scaling,offsets} 与 mask_keep_ratio（编码时再丢低分锚）", [], "plain", tsize=12.5)
    s.arrow(360, r1[1]+r1[3], 360, r2[1]-3, color=C["blue"])
    s.arrow(360, r2[1]+r2[3], 360, r3[1]-3, color=C["blue"])
    s.arrow(360, r3[1]+r3[3], 360, r4[1]-3, color=C["blue"])

    # 指标
    s.panel(748, 204, 588, 300, "指标", "blue")
    s.box(764, 244, 556, "RD 曲线", [
        "PSNR / SSIM / LPIPS vs total_MB",
        "size 轴 log 与 linear 双版本（低码率段用 log 才看得清差距）"], "blue", tsize=13)
    s.box(764, 344, 556, "BD-rate（vs HAC++，共享质量区）", [
        "旧值 −11.1% 因静默门控作废（→ fig12），重验证中",
        "目标：负 BD-rate（同质量更小体积）"], "blue", tsize=13)

    # 结果
    s.panel(24, 528, 1312, 240, "已有结果（1-78 · 2-06 · 4-10，30k，seed 42）", "green")
    rows = [
        ("主对比", "平均 58% 的 HAC++ 体积（匹配 λ），PSNR −0.27 ~ −0.61 dB"),
        ("复杂度乘子消融", "去掉乘子 −0.41 dB（1-78，λ=0.002）——创新①的价值下界"),
        ("融合剪枝修复", "w=0.5 无帽塌缩（27.51→25.95 dB / 低码率 1.62 dB）被 软帽+覆盖约束 消除"),
        ("bit-exact", "全部上报点 codec_roundtrip mismatch=0"),
    ]
    ry = 566
    for k, v in rows:
        s.text(44, ry, k, size=13, color=C["green"], weight="bold")
        s.text(200, ry, v, size=13, color=C["ink"])
        ry += 34
    s.text(44, ry + 6, "进行中：BD-rate 重验证矩阵（gen-matrix / 60k）· 30k/60k/110k 迭代研究 · 存储与编码时间分解 · 比特分配可视化", size=12, color=C["sub"])

    s.footer("scripts/rd_sweep.py:112–215 · scripts/eval_decoded.py · docs/05-paper/icra2027/main.tex（实验计划）",
             note="负 BD-rate 数字待重验证矩阵完成后回填")
    s.save(f"{outdir}/fig11-evaluation.svg")


# =====================================================================
def fig12(outdir):
    s = SVG(1380, 830, "过程资产：静默门控审计 + 负结果清单（论文贡献③：记录塑造设计的负结果）",
            "“融合剪枝无效”曾是错误结论——机制根本没有启用；修复后所有机制改动都必须带 engaged 日志；负结果同样构成方法的一部分")

    # 左：审计
    s.panel(24, 92, 660, 640, "静默门控审计（fusion silent-gate）", "red")
    a1 = s.box(40, 132, 628, "现象", [
        "早期 RD 图：融合剪枝 vs 基线几乎无差异",
        "→ 一度得出“融合分数无效”的错误结论"], "red", tsize=13)
    a2 = s.box(40, 232, 628, "根因链", [
        "① 覆盖 provider 只在 mini_splat_enabled 时由 trainer 接线，",
        "   实验配置未启用 → 融合分支从未执行",
        "② 融合分数曾被 mini_splat_full_selected 静默门控",
        "   （commit d3b4f49 修复）",
        "③ 当时无 engaged 日志 → 投影静默走普通 a+u，曲线看起来“正常”"], "red", tsize=12.5)
    a3 = s.box(40, 400, 628, "修复", [
        "· 显式接线（fusion_prune && mini_splat 时注册 provider）",
        "· 每次投影打印 [FusionPrune] engaged=1 eff=… cov/sens 均值",
        "· engaged=0 的运行一律视为无效；RD 图与跨场景报告附撤回声明",
        "· 旧 BD-rate −11.1% 作废 → 重验证矩阵（gen-matrix / 60k）"], "green", tsize=12.5)
    a4 = s.box(40, 560, 628, "教训（写进实验纪律）", [
        "“无效果的实验”必须先证明机制真的 engaged，再谈结论；",
        "对照实验的每一档都保留 engaged 日志与配置快照"], "plain", tsize=12.5)
    for a, b in ((a1, a2), (a2, a3), (a3, a4)):
        s.arrow(354, a[1]+a[3], 354, b[1]-3, color=C["red"])

    # 右：负结果清单
    s.panel(704, 92, 652, 640, "负结果 / 改道清单（判定 + 一句话原因）", "grey")
    rows = [
        ("语义先验门控（DINO，方向一）", "判负", "语义特征对 RD 无稳定增益，训练/缓存开销大", "red"),
        ("重要性加权重建损失", "判负", "逐像素加权未带来稳定增益", "red"),
        ("MLP 权重量化", "判负", "精度损失 > 体积收益，保留 32bit/参数口径", "red"),
        ("敏感度方差 z-score 归一", "改道", "α=0.99 下方差 EMA 不收敛、信号压平 → 改相对归一化（保留）", "amber"),
        ("融合 w=0.5 无帽", "改道", "1.62 dB 低码率塌缩 → w≤0.3 硬帽 + 覆盖约束（保留）", "amber"),
        ("mlp_complexity 8 维输入", "改道", "4 个图像统计恒为零 → 删除，8 维旧检查点切片兼容", "amber"),
        ("feat_dim=50（默认）", "改道", "扫描定 32 最优（隐藏层 16）", "amber"),
    ]
    ry = 132
    for name, verdict, why, ck in rows:
        s.rrect(720, ry, 620, 64, rx=8, fill="#ffffff", stroke=C["grey"], sw=1.2)
        s.text(734, ry + 26, name, size=12.5, color=C["ink"], weight="bold")
        s.text(734, ry + 48, why, size=11.5, color=C["sub"])
        s.chip(1236, ry + 10, verdict, ck, size=11.5)
        ry += 76
    s.text(720, ry + 8, "判负 = 移出方法主线；改道 = 保留修正后的版本（修正即当前代码默认行为）", size=11.5, color=C["sub"])

    # 底部
    s.panel(24, 752, 1332, 46, "", "plain")
    s.text(40, 780, "这些内容写进论文实验节的“negative results”与附录：读者复现时能避开同样的坑，也让“为什么是这四个统计量 / 为什么是 0.3 / 为什么有覆盖约束”有出处。", size=12.5, color=C["ink"])

    s.footer("commit d3b4f49（silent-gate 修复） · f4f389e（审计报告+撤回声明） · docs/03-reports/ · docs/02-design/（融合剪枝锚点体积设计提案）",
             note="DCCA-GS 过程资产图 · 2026-09-08")
    s.save(f"{outdir}/fig12-audit-negative.svg")
