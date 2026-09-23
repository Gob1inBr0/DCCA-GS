---
name: check-status
description: 检查 LYH 服务器上 DCCA-GS GPU 训练的完整状态。每次触发都执行完整检查：GPU 占用归属、每个训练任务的进度和预计完成时间、新完成的待登记结果、总队列剩余、磁盘与异常，并顺手把已完成的 run 登记进 experiments.csv。凡是用户问"检查状态"、"状态"、"运行情况"、"进度"、"跑到哪了"、"结果出来了没"、"GPU 状态"，或要求收结果、登记数据时都用。
---

# LYH 训练状态检查

对 LYH 服务器（`ssh LYH`，别名在 `~/.ssh/config`）上的 DCCA-GS 训练做一次完整检查，
并把手头已完成的结果登记进 `docs/data/experiments.csv`。原则：**只读优先**——
巡检本身绝不写入、不杀进程；只有登记 CSV 和 git 推送是写操作，且只写本地仓库。

## 1. 一键巡检

```bash
bash .agents/skills/check-status/scripts/status_sweep.sh
```

脚本只读，一次 SSH 拉回：服务器时间、GPU 占用、我们的 train.py 进程、
每个在跑任务的最后一行进度、最近 3 天新完成的解码评测、队列事件尾部、
队列锁、根盘和 /dev/shm 用量。

注意：训练日志是 tqdm 进度条，整段 tail 会刷出几十 KB。
要看更多日志时必须用 `tail -c 1200 <log> | tr '\r' '\n' | tail -5`，禁止直接 `tail`。

## 2. 判读规则

**GPU 归属**：我们的单卡任务约占 6 GB 显存；别人占的卡显存 35 GB 上下。
**卡位政策（2026-09-23 起）**：卡 2、3、4、5、6、7 归我们使用，空闲即可
大胆提交任务；卡 0、1 不碰（不提交、不操作）。别人的进程只读报告，
不做任何操作。

**进度与 ETA**：进度行形如
`Scaffold-GS training: 74%|...| 22300/30000 [5:04:55<1:27:36, 1.46it/s, anchors=..., loss=...]`
方括号第二段是剩余时间估计，直接引用为 ETA。30k 一个 run 全程约 10–12 小时。

**run 在哪**：当前主存储是 NFS 上的 `/mnt/newproject2/dcca_runs`，历史 run 在
`/dev/shm/dcca_runs`。日志和锁是 run 目录的同级文件：`<tag>.log`（训练）、
`<tag>.launch.log`（runner 输出，ALL_DONE 在这里）。队列事件日志：
`/mnt/newproject2/dcca_runs/jr2_total_queue.log`。

**什么才算"跑完"**：只有 run 目录下存在 `decoded_eval/metrics.jsonl`
（压缩 + 解码 + 150 视角评测）才算完整跑完，这个数字才能登记。
run 目录顶层的 `metrics.jsonl` 只是训练时评测（模型未过熵编码），**不能**拿去登记。
runner 完成的标志是 `<tag>.launch.log` 里出现 `ALL_DONE`。

**一个易踩的坑**：训练评测出了数字（顶层 metrics.jsonl）但解码评测缺失，多半是
压缩阶段死了（历史案例：/dev/shm 写满压垮 compress）。看到这种"训练完但没解码"
的 run，去队列日志确认是否已被重发，不要自己另起炉灶。

## 3. 收结果并登记（巡检后的写操作）

1. `git pull --rebase` 拿最新 CSV——另一个会话的定时任务也在登记，先拉再查重，
   用 `grep <run_id> docs/data/experiments.csv` 判断是否已有行，**别登记重复行**。
2. 对每个未登记且已有 `decoded_eval/metrics.jsonl` 的 run，从服务器取数：
   ```bash
   ssh LYH 'cat <ROOT>/<tag>/decoded_eval/metrics.jsonl; grep -o "total_MB.: [0-9.]*" <ROOT>/<tag>/compress.log | tail -1'
   ```
   compress.log 的 `[Compress] {...}` 行里有 total_MB、num_anchors（编码数）、
   num_anchors_total（训练数）。
3. 按现有行格式追加一行（对照相邻行确认列序）：
   `group,scene,run_id,variant,iteration,lambda,feat_dim,mlp_quant,psnr,ssim,lpips,total_mb,anchors_trained,anchors_coded,metric_type,metric_value,notes,source`
   notes 用白话写清和谁的对比、差多少、判定门结论是否变化；commit 并 push。
4. push 走 GitHub SSH 443 端口（`~/.ssh/config` 已配 `Host github.com → ssh.github.com:443`；
   22 端口会被网络劫持断连，报 "Connection closed by 198.18.x.x" 就是这个原因）。

## 4. 对比基准（notes 里引用，别凭记忆）

| 参照 | 数字 |
| --- | --- |
| 新代码 base @λ0.002（r2_base_newcode） | 27.695 @ 13.904 MB |
| 旧协议 base 30k（λ .004/.002/.0005） | 27.511 / 27.750 / 27.953 |
| 复跑噪声带 | ±0.058 dB（同配置对）；判定门一律 ±0.1 dB |
| 判定门数值源 | `docs/03-reports/结果异常与位置性负增益_报告及排查任务书.md` 的预注册表 |

ph0c 系（r_post=0.85 咬合预算）base 普遍比旧协议低 0.3–0.5 dB，对比时认准同协议基准。

## 5. 红线

- 巡检只读。用户没让动，就不杀进程、不重启、不清理、不发新任务；
  队列操作归另一个会话的定时任务管，别重复操作。
- `pkill`/`pgrep` 模式必须避免匹配到自己（用 `[x]` 方括号技巧或写成脚本文件）。
- 别人的进程和显存只在报告里出现，永远不操作。
- 根盘用到 90% 以上要在大门报出（历史事故：盘满导致任务丢失）。
