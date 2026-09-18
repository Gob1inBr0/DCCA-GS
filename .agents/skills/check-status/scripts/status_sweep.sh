#!/bin/bash
# LYH 服务器只读巡检：GPU 归属 / 我们的任务进度 / 完成情况 / 队列 / 磁盘。
# 不写入、不杀进程、不动任何人的任务。输出紧凑文本，供判读。
set -u
ssh -o ConnectTimeout=20 LYH 'bash -s' <<'REMOTE'
ROOT=/mnt/newproject2/dcca_runs
SHM=/dev/shm/dcca_runs
last_line() { tail -c 1200 "$1" 2>/dev/null | tr "\r" "\n" | grep -v "^$" | tail -1; }

echo "=== 服务器时间 ==="; date

echo "=== GPU（显存 MiB / 利用率%）==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo "=== 我们的计算进程（PID 运行时长 结果目录）==="
ps -eo pid,etime,args | grep "train.py" | grep -v grep | while read -r pid etime rest; do
  tag=$(printf "%s" "$rest" | grep -o "result-dir [^ ]*" | awk "{print \$2}")
  echo "$pid $etime ${tag:-UNKNOWN}"
done

echo "=== 在跑任务进度（尾行）==="
ps -eo args | grep "train.py" | grep -v grep | grep -o "result-dir [^ ]*" | awk "{print \$2}" | sort -u | while read -r d; do
  tag=$(basename "$d")
  log=""
  for c in "$(dirname "$d")" "$ROOT" "$SHM"; do
    [ -f "$c/$tag.log" ] && log="$c/$tag.log" && break
  done
  echo "[$tag]"
  [ -n "$log" ] && last_line "$log" || echo "  (找不到 .log)"
  # 完成判定：解码评测才算跑完；顶层 metrics.jsonl 只是训练时评测
  [ -f "$d/decoded_eval/metrics.jsonl" ] && echo "  DECODED_EVAL: $(cat "$d/decoded_eval/metrics.jsonl" | tail -1)"
  for c in "$(dirname "$d")" "$ROOT" "$SHM"; do
    [ -f "$c/$tag.launch.log" ] && grep -qa "ALL_DONE" "$c/$tag.launch.log" && echo "  RUNNER: ALL_DONE"
  done
done

echo "=== 最近 3 天新完成的解码评测（新→旧，前 10）==="
find "$ROOT" "$SHM" -maxdepth 1 -type d -newermt "3 days ago" 2>/dev/null | while read -r d; do
  f="$d/decoded_eval/metrics.jsonl"
  [ -f "$f" ] && printf "%s  %-42s  %s\n" "$(stat -c %y "$f" | cut -d. -f1)" "$(basename "$d")" "$(grep -o "\"psnr\": [0-9.]*" "$f" | tail -1)"
done | sort -r | head -10

echo "=== 队列事件（尾部 8 行）==="
tail -8 "$ROOT/jr2_total_queue.log" 2>/dev/null || echo "(无队列日志)"
echo "--- 队列锁（在跑臂）---"
ls "$ROOT"/.qlock_* 2>/dev/null || echo "(无锁文件)"
for lk in "$ROOT"/.qlock_*; do [ -f "$lk" ] && echo "$(basename "$lk") -> $(cat "$lk" 2>/dev/null)"; done

echo "=== 磁盘 ==="
df -h / /mnt 2>/dev/null | tail -2
echo "--- /dev/shm 已用 ---"
df -h /dev/shm | tail -1
REMOTE
