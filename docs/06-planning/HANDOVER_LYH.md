---
title: "LYH 环境交接手册（DCCA-GS / SHARC / 40T NFS）"
type: handover
tags: [dcca-gs, sharc, lyn, nfs, handover]
created: "2026-08-27"
updated: "2026-08-27"
---

# LYH 环境交接手册

> 面向下一个接手 Agent。本文覆盖：LYH 机器访问、DCCA/SHARC 两套环境、
> 40T NFS（/mnt）的**卡死问题与绕过方案**、当前目录/数据布局、数据集下载状态、
> 全套操作惯例与坑。**所有关键决策已与用户确认，直接按其约定执行。**

---

## 1. 机器与访问

| 项 | 值 |
| --- | --- |
| 接入名 | `LYH`（在本地 `~/.ssh/config` 已配） |
| HostName / Port / User | `36.138.63.143` / `65022` / `root` |
| 主机名 | `ecs-89478139-001` |
| 系统 | Ubuntu 22.04.5（Linux 5.15） |
| 硬件 | 8× NVIDIA A100-PCIE-40GB，CUDA 12.8（驱动 570.133.20），80 逻辑核，629GB 内存 |

- **登录只用密钥，不要用密码**（用户已配好，密码禁止明文存储/写入脚本）：
  ```bash
  ssh -o BatchMode=yes LYH '<command>'
  ```
- 本机（Mac）公钥已在 LYH；SSH 前缀已获批准。

  `~/.ssh/config` 可直接使用：
  ```text
  Host LYH
      HostName 36.138.63.143
      Port 65022
      User root
      IdentityFile ~/.ssh/id_ed25519
  ```

  GitHub 当前环境使用：
  ```text
  git@github.com:Gob1inBr0/DCCA-GS.git
  ```
  `github.com:22` 在本环境被拦截，推送时使用：
  ```bash
  git push ssh://git@ssh.github.com:443/Gob1inBr0/DCCA-GS.git \
    codex/color-asg:codex/color-asg
  ```
- 共享目录约定（用户明确指定）：
  - 项目/代码/环境 → `/home/T0ng`
  - 数据集 → `/home/data`（当前为空目录，按需重建）
  - **运行日志、shell 脚本 → `/home/T0ng/run_shell`**（用户要求，后续所有日志/脚本默认放这里）
- 别名说明：用户口中的“我的目录”=`/home/T0ng` + `/home/data`。

---

## 2. conda 环境（在 `/home/T0ng/miniconda3`，conda 26.5.3）

### 2.1 `DCCA`（DCCA-GS 主环境）
- Python 3.10.21；`torch 2.7.1+cu128`、`torchvision 0.22.1+cu128`；
  `gsplat 1.5.3`（源码安装，`/home/T0ng/gsplat-main`）；
  CUDA 扩展 `_gridencoder`、`simple_knn`、`arithmetic`、`torch_scatter 2.1.2`；
  `tmc3`（GPCC，已放入 env bin）；`numpy 1.23.5`（已手动固定，防止被解析器升到 2.2.6）。
- 激活：
  ```bash
  conda activate DCCA
  source /home/T0ng/DCCA-GS/scripts/env_dcca.sh
  ```
- 验证：`import torch, gsplat` 正常；CUDA 可用（A100）。
- 测试：`pytest tests/ -q` → **34 passed / 3 failed**（3 个是代码级，非环境问题：
  `test_mini_splat_default_off` 断言默认关、`test_attr_ctx` 的 roundtrip/shape）。
- 注意：`env_dcca.sh` 已写入 `/home/T0ng/DCCA-GS/scripts/`（不是仓库原版
  `env_5090.sh`，原版硬编码 env 名）。

### 2.2 `SHARC`（SHARC-GS / HAC++ / PKU-GS 环境）
- Python 3.10；`torch 2.7.1+cu128`、`torchvision 0.22.1+cu128`、`torchaudio 2.7.1+cu128`；
  `torch_scatter 2.1.2`；CUDA 扩展按 README：
  `diff-gaussian-rasterization`、`simple-knn`、`gridencoder`、`arithmetic`
  （A100 只编 `TORCH_CUDA_ARCH_LIST=8.0`；
  已打补丁：gridencoder `-std=c++17`、simple_knn 补 `<cfloat>`/`FLT_MAX`）；
  `tmc3`；`numpy 1.23.5`；`plyfile/tqdm/einops/wandb/lpips/scipy`。
- 激活：
  ```bash
  conda activate SHARC
  source /home/T0ng/SHARC-GS/scripts/env_sharc.sh
  ```
- 验证（README 校验片段已跑通）：
  `simple_knn._C.distCUDA2`、`_gridencoder`、`arithmetic`、`GaussianRasterizer` 均可导入。

### 2.3 环境重建要点（如果必须重建）
- 用阿里云 pip 镜像装通用依赖（快）：`--index-url https://mirrors.aliyun.com/pypi/simple/`
- torch+cu128 必须用 `--index-url https://download.pytorch.org/whl/cu128`
  （注意：pytorch 索引上 numpy 下载易卡，先本地 wheel 装 `numpy==1.23.5` 再装 torch）。
- `torch_scatter` 从 `https://data.pyg.org/whl/torch-2.7.1+cu128.html`。
- CUDA 扩展需 `CUDA_HOME=/usr/local/cuda`、`PATH=/usr/local/cuda/bin:$PATH`、
  `CXXFLAGS=-std=c++17`（nvcc 12.8.93 已确认存在）。
- 长任务必须放 **tmux**（SSH 会话断开会杀前台 pip；expect 也有 180s 超时坑）。

---

## 3. ⚠️ 40T NFS（`/mnt`）——最关键的问题

- 挂载：`/mnt` = `10.17.28.46:/8ea3a0f3-8cee-415e-80cd-1758fb786829`
  （40T，已用约 615G，剩余约 39T；与已摘除的旧 `/opt` NFS `10.17.28.15` 不是同一台，**不要用 /opt**）。
- **单文件/大文件读写正常**：实测 `dd` 118MB/s、tar 单文件传入 **156MB/s**。
- **海量小文件的任何操作都会卡死**（进程进入 `D+ rpc_wait_bit_killable`，不恢复）：
  - `rsync -a`（每文件 stat+比较+写，约 4 万文件后卡）；
  - `rm -rf`（每文件 unlink，卡）；
  - **`tar -xf` 解包**（每文件 open/write/close，约 2500 文件后卡）；
  - 对 /mnt 递归 `du` / `df` 也会触发（**禁止对 /mnt 跑 du/df**）。
- 原因：共享 NFS 服务器对海量小文件的高频 RPC 扛不住；只有单文件顺序流没问题。
- 卡死恢复：`umount -l /mnt` 后重新 `mount`（数据不丢，之前写的文件仍在）；
  之后再操作。**不要持续往 /mnt 做小文件操作，会连带影响共享盘其他用户。**

---

## 4. 数据与目录布局（2026-08-27 状态）

### 4.1 `/mnt/newproject2/`（40T 上，作为“单文件对象库”）
| 文件/目录 | 大小 | 说明 |
| --- | --- | --- |
| `T0ng.tar` | 20G | `/home/T0ng` 完整备份（仓库 + miniconda3 + runs + run_shell） |
| `data.tar` | 76G | `/home/data` 完整备份（**已验证完整**：15602 条目=本地 15601+`./`，`tar -tf` rc=0 无损坏） |
| `data/_archives/` | 增长中 | 16 个场景 tar.zst 正在下载（见 §5） |
| `DCCA-GS/…`、`SHARC-GS/…`、`miniconda3/…` 等 | 不完整 | 这是**之前 rsync/解包失败留下的半成品，不可信**；要清理时同样会卡 NFS（用重挂恢复），或用 tar 覆盖 |

### 4.2 本地（保留可用副本）
- `/home/T0ng/`（**保留**，这是当前可用工作区）：`DCCA-GS`、`SHARC-GS`、
  `gsplat-main`、`HAC-plus`、`tmc13`、`runs`、`run_shell`、`miniconda3`。
- `/home/data/`：**已删除**（腾出本地约 89G；数据在 `/mnt/newproject2/data.tar`）。
  按需重建，见 §5。
- 本地盘 `/dev/sda2`：148G，约 53G 已用 / 89G 空闲（38%）。

### 4.3 下载/任务进程
- tmux `dlstore`：16 场景 tar 下载（`bash /tmp/dl_store_archives.sh`）。
- 日志：`/mnt/newproject2/run_shell/dl_store_archives.log`。
- 脚本本地副本：`/private/tmp/dl_store_archives.sh`（仅下载、不解包、不删除）。

---

## 5. 核心绕过方案（已实测通过）

把 40T 当“**单文件对象库**”，小文件操作全部放本地：

```bash
# 按需从 /mnt 单文件归档解一个场景到本地（NFS 单文件读 + 本地写，不卡）
tar -xf /mnt/newproject2/data.tar -C /home/data ./4-28
# 实测：4-28 用时 2m15s，2415 文件、14G，完整成功，无卡死。

# 16 场景下载完成后，从 /mnt 的 tar.zst 单文件读+本地解包：
tar --zstd -xf "/mnt/newproject2/data/_archives/<路径>/<scene>.tar.zst" -C /home/data

# 结果/模型写回 /mnt：以单文件（tar 或单个文件）形式，不要写小文件目录。
```

**铁律**：
- 只在 /mnt 做**单文件读写**；任何“很多小文件”的操作都先在本地做。
- 本地空间预算：1 个场景约 15–25G，本地 89G 可同时放约 2–4 个；用完可删或重新打 tar 存回 /mnt。

---

## 6. 数据集：20 个场景 + 下载状态

### 6.1 目标场景（用户要求）
`1-23, 1-42, 1-71, 1-73, 1-78, 1-100, 2-6, 2-8, 3-7, 3-11, 4-10, 4-25, 4-28, 5-6, 5-13, 6-7, 6-22, 6-26, 6-31, 6-44`

### 6.2 云端位置（aliyunpan 账号 `tb382575915`，资源库；CLI 配置在 `/etc/aliyunpan`）
- 16 个 tar.zst（零填充命名，注意 `2-6→2-06`、`3-7→3-07`、`5-6→5-06`、`6-7→6-07`）：
  - `/1-Standalone Buildings/`：`1-71`、`3-07`、`3-11`
  - `/2-Architecture Complex/`：`1-73`、`1-78`、`1-100`、`2-06`、`2-08`、`6-22`、`6-31`
  - `/3-Sports Field/`：`5-13`、`6-07`、`6-26`、`6-44`
  - `/4-Natural Landscape/`：`4-10`（`tar(1)`）、`4-25`
  - `/5-Sculpture/`：`5-06`
  - `/6-Waterfront Bridges/`：`4-28`（`tar(1)`）
- 下载命令示例：
  ```bash
  /usr/local/bin/aliyunpan download --saveto /mnt/newproject2/data/_archives "/2-Architecture Complex/1-100.tar.zst"
  ```
- **1-23、1-42 只有目录**（`/dat/Standalone Buildings/1-23/`、`/1-42/`），没有 tar；
  直接下到 /mnt 会触发小文件卡死。处理方式：下载到本地 → `tar` 成单文件 → 传 /mnt
  （本地现约 89G，可分批做）。

### 6.3 状态
- `4-28`、`3-07`：已处理（归档在 `data.tar`，可本地按需解出）。
- 16 个 tar 下载中（`dlstore`）：速度约 **1.5MB/s**（普通账号限速），每个约 3h，
  全部约 **2.5–3 天**。
- 下载脚本已上传 `LYH:/tmp/dl_store_archives.sh`，内容=顺序 `aliyunpan download` 16 个包，
  **不解包不删除**。

---

## 7. 常用命令速查

```bash
# 机器
ssh -o BatchMode=yes LYH '...'

# 环境
source /home/T0ng/miniconda3/etc/profile.d/conda.sh
conda activate DCCA      # 或 SHARC

# 查看下载进度（不要对 /mnt 跑 du/df）
tail -c 1500 /mnt/newproject2/run_shell/dl_store_archives.log | tr '\r' '\n' | tail -5
ps -eo pid,stat,cmd | grep '[a]liyunpan download'   # 正常应为 S/Sl，若 D 即卡

# 校验归档完整性（单文件读，安全）
tar -tf /mnt/newproject2/data.tar > /tmp/list.txt && echo "rc=$?"

# NFS 卡死恢复（数据不丢）
umount -l /mnt && mount -t nfs -o rw,vers=3,rsize=1048576,wsize=1048576,timeo=600,retrans=2,hard,nolock \
  10.17.28.46:/8ea3a0f3-8cee-415e-80cd-1758fb786829 /mnt
```

---

## 8. 坑与禁忌（重要）

1. **不要用密码**：SSH 密钥已配好；密码禁止明文写文件。
2. **不要在 /mnt 做任何多小文件操作**（rsync / rm -rf / tar -xf / 递归 du / df）——
   必卡死（D 状态），且影响共享盘他人。恢复用 lazy umount+remount。
3. **不要对 /mnt 跑 `du -sh` / `df -h /mnt`**（递归 stat 风暴会触发卡死）。
4. **长任务放 tmux**；前台 SSH 命令被断开会杀进程（此前 pip/torch 被此坑过）。
5. **`/opt` 是坏的旧 NFS（10.17.28.15），已 lazy 卸载，不要用。**
6. **日志/脚本统一放 `/home/T0ng/run_shell`**（用户要求）；重要结果（checkpoints/数据集）
   放 `/home/data` 与项目目录，不要混进 run_shell。
7. **numpy 版本**：两环境均为 1.23.5，pip 解析器容易把它升到 2.2.6，需用本地 wheel 固定。
8. gridencoder 默认 `-std=c++14` 不兼容 torch 2.7.1（需 C++17）；simple_knn 缺 `FLT_MAX`
   （补 `<cfloat>`）——这两处补丁已打在 `/home/T0ng/HAC-plus` 与 `/home/T0ng/SHARC-GS` 源码里。
9. 本机工作区 = `/home/T0ng`；数据集工作区 = `/home/data`（按需重建）；40T 只放单文件归档。

---

## 9. 下一步（给下一位 Agent）

1. **监控 16 场景下载**（`dlstore` / 日志）；每个包完成确认后继续。
2. 下载完成后，按 §5 公式**本地按需解包**跑实验（4-28/3-07 直接从 `data.tar` 解）。
3. 处理 **1-23/1-42**（本地→tar→单文件传 /mnt）。
4. 如 /mnt 共享盘被修复/更换，可改回“先解包再上传正式项目目录”，并与用户确认。
5. 跑 `pytest tests/ -q`（DCCA 34 passed / 3 前置代码级失败），确认环境无恙后再开训练。
6. 参考旧交接：[环境配置与交接.md](../04-guides/环境配置与交接.md)、
   [HANDOVER.md](./HANDOVER.md)（DCCA-GS 原始流程）。

> 有任何“与本文矛盾”的情况，先问用户，不要擅自对 /mnt 做小文件操作或删本地数据。
