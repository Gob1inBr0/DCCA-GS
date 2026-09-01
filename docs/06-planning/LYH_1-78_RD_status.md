# LYH 1-78 RD 实验（2026-08-29 启动）

> 只记录当前正在跑的 1-78 批量实验；结果出来后由 `collect_lyh_rd.py` 汇总，
> 再用 `plot_lyh_rd.py` 出图。

## 运行路径

- 远程工作区：`/home/project2/DCCA-GS-minifull`
- 训练/结果根：`/dev/shm/dcca_runs/1-78`
- 临时数据：`/dev/shm/dcca_data/1-78`
- 远程机器：`36.138.63.143:65022`（仅 SSH 密钥）
- 归档：`/mnt/newproject2/results/1-78_results.tar.zst`

## 调度器

1. `scripts/lyh_178_rd_queue.sh`：第一波 HAC++ high/low + no-SPA 高码率；
2. `scripts/lyh_178_rd_continue.sh`：HAC++ 完成后继续 no-SPA λ=0.002/0.0005；
   no-SPA 高码率完成后继续 λ=0.001；
3. `scripts/lyh_178_finalize.sh`：等全部任务完成，压到 `/mnt` 后清理 `/dev/shm`。

## 已启动点

| 类型 | tag | GPU | λ / 配置 |
| --- | --- | ---: | --- |
| HAC++ high | `hacpp_1-78_high_l0004` | 3 | 30k, λ=0.004 |
| HAC++ low | `hacpp_1-78_low_l0005` | 6 | 30k, λ=0.0005 |
| DCCA no-SPA | `dcca_1-78_nospa_l0004` | 7 | 110k, λ=0.004 |
| DCCA no-SPA | `dcca_1-78_nospa_l0002` | 3 | 第二轮, λ=0.002 |
| DCCA no-SPA | `dcca_1-78_nospa_l0005` | 6 | 第二轮, λ=0.0005 |
| DCCA no-SPA | `dcca_1-78_nospa_l0001` | 7 | 第三轮, λ=0.001 |
| DCCA depth | `lyh_178_depth_110k_l0002` | 4 | depth-reinit+SPA, λ=0.002 |
| DCCA depth | `lyh_178_depth_110k_l0001` | 5 | depth-reinit+SPA, λ=0.001 |
| DCCA depth | `lyh_178_depth_110k_l0005` | 4→q | depth-reinit+SPA, λ=0.0005 |

当前还有原来三路 SPA 110k（GPU0/1/2）：
`lyh_178_baseline_110k`、`lyh_178_depth_110k`、`lyh_178_full_110k`。

## HAC++ 已完成（8-30）

| 结果 | 实际语义 | PSNR | SSIM | LPIPS | 编码后 total_MB |
| --- | --- | ---: | ---: | ---: | ---: |
| `hacpp_1-78_high_l0004`（λ=0.004） | **lowrate** | 27.8406 | 0.8829 | 0.1597 | 20.0265 |
| `hacpp_1-78_low_l0005`（λ=0.0005） | **highrate** | 28.6780 | 0.9001 | 0.1396 | 40.5589 |

注意：HAC++ 的 λ 越小、码率越高，因此 survey 的 highrate/lowrate 与 tag 名相反；
最终 RD 图必须按体积语义标注，不能按 lambda 字面标注。

## 已踩的坑

- HAC++ 读取 1200 张图会触发 `Too many open files`，启动前必须 `ulimit -n 65536`；
- HAC++ 官方 `capture()` 引用未初始化的 `denom`，不要加 `--checkpoint_iterations`；
- 旧 watcher 只等最初三路训练，完成后会 `rm -rf` 整个运行目录，已停掉，换成统一 finalizer。

## 当前状态（8-30）

- HAC++ 两点已完成，见上表；
- 1-78 三路 SPA 110k 尚未完成：baseline 已到约 48.8k（锚点约 848k）；
  depth/full 前三次都在约 15k 的 MiniSplat reinit 阶段失败，当前第 4 次到约 10.5k；
- no-SPA 第一波 110k 正在跑，HAC++ 完成后已自动启动 λ=0.002/0.0005 两路；λ=0.001 会在第一波 no-SPA 完成后启动；
- `diag/depth_tail.log`、`diag/full_tail.log` 已经接入，若再次失败可保留 traceback。

## 最新已完成结果（8-31 检查）

| run | PSNR | SSIM | LPIPS | total_MB |
| --- | ---: | ---: | ---: | ---: |
| HAC++ lowrate（λ=0.004） | 27.8406 | 0.8829 | 0.1597 | 20.0265 |
| HAC++ highrate（λ=0.0005） | 28.6780 | 0.9001 | 0.1396 | 40.5589 |
| `lyh_178_baseline_110k`（SPA, MLP quant） | 27.5047 | 0.8645 | 0.1882 | 13.6696 |
| `dcca_1-78_nospa_l0002`（110k, MLP quant） | 28.8084 | 0.8956 | 0.1449 | 22.9820 |

四路 depth（λ=0.004/0.002/0.001/0.0005）仍在训练中，`depth_rd_done.txt` 尚未出现。

## 16:37 修复与重跑

从 sidecar 拿到真实失败栈：

- depth/full 的 MiniSplat reinit 后，`offset_gradient_accum` / `offset_denom`
  没有随锚点同步增长；
- full 直接 `prune_anchor` 时同样没有裁剪 `offset_*`、`opacity_accum`、
  `anchor_demon`、`max_radii2D`；
- 下一轮 `adjust_anchor`/渲染因 N*K 长度不匹配触发 device-side assert / IndexError。

已给远端 `hacplus/scene/gaussian_model.py` 打补丁，停掉旧重试，并用修复后的代码
在 GPU1/2/4/5 同时重启四路 depth λ（0.004/0.002/0.001/0.0005）。

## 30k 新版协议（用户确认，2026-09-01）

当前已停掉 1-78 110k 重跑队列和旧 finalizer；保留已完成的
no-SPA 110k / baseline / HAC++ 结果。

新启动（叠加在 SHARC 卡上，`WAIT_VRAM_MB=35000`）：

- `lyh_178_30k_s42_l0004` / `_l0002` / `_l0005`
- `lyh_178_30k_s2026_l0004` / `_l0002` / `_l0005`
- `lyh_178_30k_nospa_l0004`
- `hacpp_1-78_mid_l0002`

全部 30k、1600 宽、test-every 8；主路径为 depth-reinit + SPA 0.85 + I2/I6。
2-06、4-10 之后按同样协议推进。
