---
name: submit-task
description: 在远程 GPU 服务器上提交新的训练任务（完整流程：确认环境、同步代码、选卡、冒烟、提交、登记）。凡是用户说"发任务""提交任务""跑实验""把这个配置跑一下""launch"或要求排队、批量提交训练时都用。路径不写死：每次先从现有脚本或交接手册确认服务器的具体路径，再代入模板。
---

# 提交新训练任务

把一批训练任务安全地提交到远程 GPU 服务器。核心纪律：**路径每次现查现填，
不凭记忆写死；一次批次只改一个因素；功能真正生效要有日志证据；别人的卡和
进程绝对不碰。**

## 1. 先确认五项环境信息（不要假设，不要写死）

提交前从服务器现状确认这五项，填进本批次的变量里：

| 变量 | 含义 | 怎么确认 |
| --- | --- | --- |
| `HOST` | SSH 别名 | 看本地 `~/.ssh/config`；用户说过用哪台就用哪台 |
| `CODE_ROOT` | 服务器上的代码目录（git 检出） | 看最近一次队列脚本里的 `RUNROOT`，或交接手册 |
| `RUNS_ROOT` | 结果写入目录 | 同上；注意历史上换过结果目录，以最近的为准 |
| `DATA_DIR` | 数据集目录 | 看最近队列脚本（按场景各一份） |
| `PY_BIN` | Python 环境的可执行文件目录 | 看最近队列脚本里的 `CONDA_ENV_BIN` |

确认方法：`ssh $HOST 'ls -t <run_shell 目录>/*.sh | head -3'` 找最近用过的
队列脚本，读它头部的变量。找不到就问用户，不要猜。

## 2. 同步代码并确认版本

1. 本地改动先提交并推送到 GitHub（如果连 GitHub 的 22 端口被断，
   走 ssh.github.com 的 443 端口）。
2. 服务器上 `git -C $CODE_ROOT pull`，然后 **`git status --porcelain`
   必须为空**。有未提交的本地改动时停下向用户报告——上次事故就是
   运行代码和仓库不一致，事后查不清。
3. 记下 `git -C $CODE_ROOT rev-parse HEAD`，写进提交说明。
4. 确认 runner 会写来源记录：服务器代码里的 `scripts/runner_phg_cell.sh`
   应包含写 `provenance.txt` 的段落（分支 `feat/run-provenance` 起）。
   没有的话先更新 runner 再提交任务。

## 3. 选卡规则

- `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` 看占用。
- 我们的训练单卡约占 6 GB；显存 30 GB 以上且进程不是我们的，是别人的卡，
  **绝对不用、不杀、不报告成空闲**。
- 用户明确说过不许碰的卡（即使看起来空闲）遵守用户的话。
- 用锁文件（`.qlock_<卡号>` 或 `.pglock_<卡号>`，写在 `RUNS_ROOT`，
  内容为 tag）声明占用；提交前检查已有锁。
- 批前检查磁盘：结果目录所在文件系统余量要够（历史事故：内存盘写满，
  压缩阶段死掉，训练白跑）。

## 4. 单变量规则与命名

- 一批任务只改一个因素；批次内不改代码；改了代码就换新的 tag 前缀。
- tag 格式沿用现有惯例：`<批次>_<臂名>_r<预算比>_lam<λ 去点>_s<种子>`，
  例 `r3_qoff_r085_lam0002_s42`。先 `grep <tag> docs/data/experiments.csv`
  确认没有重名。

## 5. 冒烟（先证明功能真正生效，再放全量）

新代码路径必须先跑约 400 步的短训练，在日志里找到**该功能的生效证据**
（设计文档第 8 节列了每个机制的对应字段，例如次模要看到
`[Submod] ... sens_weighted=...`、贪心没有回退后备方案）。冒烟不过不放全量。
复用已验证代码路径的批次可以免冒烟，但要在提交说明里写明依据。

## 6. 提交模板（变量现填，路径不写死）

```bash
ssh $HOST 'bash -s' <<'EOF'
set -u; ulimit -n 65536
CODE_ROOT=<填>      # git 检出的代码目录
PY_BIN=<填>         # python 环境目录
RUNS_ROOT=<填>      # 结果目录
DATA=<填>           # 本批场景的数据目录
GPU=<填>            # 按第 3 节规则选定的卡
TAG=<填>; LAM=<填>; SEED=<填>
EXTRA="<本臂独有旗标>"     # 单变量：与对照臂只差这里
echo "START tag=$TAG gpu=$GPU $(date)" > "$RUNS_ROOT/${TAG}.launch.log"
touch "$RUNS_ROOT/.qlock_$GPU" && echo "$TAG" > "$RUNS_ROOT/.qlock_$GPU"
RUNS_ROOT="$RUNS_ROOT" RUNROOT="$CODE_ROOT" CONDA_ENV_BIN="$PY_BIN" \
WAIT_VRAM_MB=30000 \
setsid bash "$CODE_ROOT/scripts/runner_phg_cell.sh" \
  "$GPU" <场景名> "$DATA" "$LAM" "$TAG" 30000 15000 \
  --cfg.model.spa-enabled --cfg.model.spa-ratio 0.85 \
  --cfg.model.mini-splat-enabled --cfg.model.mini-splat-reinit-iter 15000 \
  --cfg.model.mini-splat-max-new 4000 --cfg.model.mini-splat-views 8 \
  --cfg.model.mini-splat-voxel 0.0 --cfg.seed "$SEED" \
  $EXTRA \
  >> "$RUNS_ROOT/${TAG}.launch.log" 2>&1 < /dev/null &
echo "SUBMITTED gpu=$GPU tag=$TAG"
EOF
```

多个任务：每卡一个，循环提交并 `sleep 90`（等显存爬升再提交下一个）；
超过空闲卡数的任务写成轮询等待脚本（参考服务器上现存的队列脚本写法），
或排进现有队列。

## 7. 提交后

1. 几分钟后回看 `launch.log` 头部：有 `VRAM_OK`、`ATTEMPT 1`、
   `provenance.txt` 已生成，才算提交成功。
2. 进度与收数用 check-status skill（如果装了）或按同样的方式巡检：
   训练日志是进度条刷屏，取尾行要用 `tail -c 1200 <日志> | tr '\r' '\n' | tail -1`。
3. **结果目录里出现 `decoded_eval/metrics.jsonl` 才算完整跑完**；
   顶层的 metrics.jsonl 只是训练中的评测，不能拿去登记。
4. 登记：`git pull --rebase` 后对照 `docs/data/experiments.csv` 表头追加
   （数字从 decoded_eval 和 compress.log 的总 MB 取），提交推送。

## 8. 红线回顾

- 别人的进程与显存只在报告里出现，永远不操作。
- `pkill`/`pgrep` 的匹配串要避免匹配到自己的会话（用 `[x]` 括号技巧或
  写成脚本文件再执行）。
- 队列若由另一个会话管理，先沟通避免重复提交。
- 用户没让动就不要动在跑的任务。
