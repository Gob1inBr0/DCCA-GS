# GPU 占卡器使用说明（gpu_keeper v2）

> 位置：`/mnt/newproject2/competitors/`（LYH 服务器）
> 目的：**让空闲卡的利用率读数不为零**——组里有横向检查 GPU 利用率，
> 我们不用卡的时候用占卡负载把卡"暖"着，谁要用谁直接跑任务（占卡器自动让位）。

---

## 0. 一句话原理

`gpu_keeper.py` 是常驻 supervisor：每 5 秒扫一遍 8 张卡，**有计算进程的卡
一律不碰，空卡自动放一个占卡 worker**（吃 93% 显存 + 35% 占空比的 memset
负载，利用率读数 20–60%）。你的训练/分析任务直接往任何卡上发，不用关
占卡器——supervisor 发现卡上有你的进程就不再占，你的任务结束后它会自动
把卡重新占上。

---

## 1. 日常使用（最常用的三个操作）

### 查看当前状态（只读，随时可执行）

```bash
ssh LYH 'cat /mnt/newproject2/competitors/keeper_status.json'
```

看三个字段：`workers`（哪些卡被占卡器占着，值是 worker PID）、
`idle_gpus`（当前无任何负载的卡，正常情况应为空）、`gpus`（每张卡的
used/busy 快照）。

### 发任务（不用管占卡器）

照常把训练往任意卡上发（runner 或手动 CUDA_VISIBLE_DEVICES）。占卡器
**不会阻止你**：你的进程一起，supervisor 在下个周期就不再占那张卡；你的
进程退出后它自动把卡占回去。

> 唯一注意：占卡 worker 吃 93% 显存。往一张"正被占着"的卡上发任务前，
> 先按 §2 释放，等显存退掉再发（直接发会 OOM——占卡负载会顶到
> `WAIT_VRAM_MB` 排队逻辑，训练 runner 自带等待，所以 runner 方式不用
> 手动释放）。

### 释放某张卡给自己用（手动发非 runner 任务时）

```bash
ssh LYH 'touch /mnt/newproject2/competitors/RELEASE-3'
# → GPU3 的占卡 worker 在 5 秒内退出并清掉标记，显存全退
# 用完后不需要任何操作，supervisor 会自动重新占上
```

---

## 2. 启动 / 停止 / 重启

```bash
# 启动（通常不需要——它应该一直在跑）
ssh LYH 'setsid /mnt/newproject2/miniconda3/bin/python /mnt/newproject2/competitors/gpu_keeper.py > /mnt/newproject2/competitors/keeper.log 2>&1 < /dev/null &'

# 确认在跑（应有 1 个 supervisor 进程 + N 个 worker）
ssh LYH 'pgrep -af gpu_keeper'

# 停止（杀 supervisor + 全部 worker）
ssh LYH 'pkill -f gpu_keeper.py; pkill -f gpu_keeper_worker'
```

supervisor 用锁文件（`keeper_monitor.lock`）防双开，重复启动会直接退出。

---

## 3. 行为细则（改代码前必读）

| 规则 | 说明 |
| --- | --- |
| 占用判定 | **通用规则，无任何名单特判**：`nvidia-smi` 查到该卡有 compute 进程，或显存 >512 MiB，即视为"在用"，跳过 |
| 占卡负载 | `cuMemAlloc` 吃掉总显存的 93%（留 2 GB 余量），循环 memset（105 ms 工作 / 300 ms 周期 ≈ 35% duty） |
| 让位 | 只看事实（有没有进程/显存），不看进程名、不看路径——**任何人的任务都能直接上** |
| 显式释放 | `RELEASE-<卡号>` 文件，supervisor 每周期检查，释放后删除标记 |
| 失败冷却 | worker 异常退出的卡冷却 60 秒后再试，避免坏卡死循环 |
| 防双开 | supervisor 持 `keeper_monitor.lock`，第二个实例启动即退出 |
| 状态落盘 | `keeper_status.json`（每 5 秒）、`keeper_gpu-<n>.json`（每个 worker）、`keeper_gpu-<n>.log`（worker 输出） |

---

## 4. 与旧版（solarwm training_reservations）的关系

- 旧版在 `/tmp/solarwm_training_reservations/`，**已被 v2 替换并停用**
  （旧 supervisor 已杀）。v2 不读旧版的任何文件；
- 旧版有"受保护 GPU 名单"（按进程参数名匹配 run_revision.py 等），v2
  刻意去掉：规则只有"有进程就让"，行为对所有人一致；
- 旧版的 `RELEASE-n` 机制在 v2 里保留同名文件（路径在
  `/mnt/newproject2/competitors/`），习惯兼容。

---

## 5. 故障排查

| 症状 | 排查 |
| --- | --- |
| 空卡没被占 | 先看 `keeper_status.json` 的 `workers`；再看 supervisor 进程在不在（§2 的 pgrep）；看 `keeper_last_error.json` |
| 某卡显存没退 | worker 是 memset 大块，杀 worker 即退：`pkill -f gpu_keeper_worker`（supervisor 会重占）|
| 卡被占但利用率 0% | 正常瞬间（35% duty 的空档）；连续 0% 才是异常，看 worker log |
| worker 反复重启 | `keeper_gpu-<n>.log` 看异常；常见是卡上残留显存 >512 MiB 判定为"在用" |
| 想让占卡器完全不干扰某个临时实验 | 临时 `pkill -f gpu_keeper`，用完重启（§2） |

---

## 6. 设计取舍记录（为什么这么做）

- **不特判名单**：任何基于进程名/参数的"保护"都会在换代码后失效（旧版
  匹配 run_revision.py 就是例子），且对组外人不透明。事实判定（有没有
  进程）永远成立；
- **memset 而非真训练**：不产生数据、不碰文件系统、显存占满但可瞬间释放，
  对真任务零风险；
- **93% 不是 100%**：给真任务的 OOM 排查留 2 GB + 驱动余量；
- **文件放 /mnt/newproject2/competitors/**：随项目迁移，不依赖 /tmp（重启
  会丢）。

—— 2026-09-29 初版。维护人：DCCA-GS 组。
