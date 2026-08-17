# 创新点 R4：条件熵调整（attr_ctx）集成报告

日期：2026-08-17；状态：**可选路径已接入并实测净省 0.41% total_MB，默认关闭**

## 1. 结论

阶段 A 判 R 方向关闭，但 R4（P0-1 式“已解码 feat → 调整 scaling/offsets 的高斯
mean/log-scale”）是唯一有正增益的变体。本次按“蚊子腿也算肉”把它做成**默认关闭的
可选 codec 路径**：不重训模型，仅在压缩时用后训练预测器调整熵参数，重建质量不变，
只改码流体积。

## 2. 实测（4-28 90k h32 λ0.002，338,130 anchors，1600 宽口径）

| 项 | 数值 |
| --- | --- |
| baseline compress total_MB | 8.3069 |
| R4（scaling-only，8-bit payload）total_MB | 8.2730 |
| 净省 | **0.0339 MB（0.41%）** |
| 其中 scaling bits | 13,293,920 → 12,902,184（省 391,736 bits） |
| offsets bits | 不变（9,373,600） |
| 预测器 payload | 13,381 bytes（107,048 bits） |
| 拟合验证集增益 | scaling +3.15%（val 20%） |
| PSNR/SSIM/LPIPS | 与 baseline 一致（decode/eval 验证中，应逐位相同） |

## 3. 为什么这样设计

- **只保留 scaling**：offsets 拟合在验证集有 +24% 增益，但全量编码时 offsets bits
  反而 +2,376 bits（拟合过度偏向验证集），不值得付 payload；
- **8-bit 量化**：16-bit + 静态算术编码的符号表开销太大（29,128 参数 → 187KB
  payload），8-bit 后 payload 仅 13KB；
- **bit-exact 保证**：encode 端先把预测器权重做 8-bit per-channel 量化并反量化，
  再用于调整熵参数；decode 端从 payload 重建同一组权重，两侧逐位一致。

## 4. 代码与用法

- 预测器/载荷：`scaffold_gs/attr_ctx.py`（`AttrCtxPredictor`、8-bit 算术编码 payload）
- 拟合：`scripts/fit_attr_ctx.py`（`--fields scaling`）
- 编解码接入：`scaffold_gs/hacpp.py` `encode_attributes/decode_attributes`
  （`attr_ctx_enabled` 写进 codec header，decode 自动加载）
- 压缩命令：

```bash
python scripts/fit_attr_ctx.py --ckpt <ckpt> --out runs/attr_ctx.pt --fields scaling
python train.py compress --cfg.ckpt <ckpt> --cfg.out-dir <out>/bitstreams \
  --cfg.codec hac_pp --cfg.attr-ctx runs/attr_ctx.pt
```

不带 `--cfg.attr-ctx` 时行为与原来完全一致（默认关闭，零回归风险）。

## 5. 已知边界

- 预测器是“后训练”拟合，训练期 `_estimate_rate_terms` 未同步改，码率损失与真实
  码流存在轻微口径差；若要进一步榨取，阶段 C 需端到端联合训练；
- 8-bit 量化会让拟合增益略降（实测仍为正）；
- 预测器权重计入 total_MB（按实际 payload 字节），不占模型 checkpoint。

## 6. 产物

- 5090：`runs/attr_ctx_4_28_h32_l0p002_s.pt`、
  `runs/attr_ctx_cmp_base/bitstreams`、`runs/attr_ctx_cmp_r4b/bitstreams`
- 结果 JSON：`runs/attr_ctx_r4b.log`（compress meta）
