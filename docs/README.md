# DCCA-GS 文档中心

> 本目录是 DCCA-GS 的全部项目文档。命名约定：`<主题>_<类型>.md`，类型 ∈
> {设计, 报告, 说明, 指南, 扩展, 综述}；按主题分文件夹，文件名不再带历史前缀（如 `PHG_` 已清理）。
> 所有数字以 `data/experiments.csv` 为唯一事实源，论文/报告引用需与之一致。

## 分类索引

| 目录 | 内容 | 文件 |
| --- | --- | --- |
| [01-architecture](01-architecture/) | 架构、API、模块设计 | [DCCA-GS_架构说明](01-architecture/DCCA-GS_架构说明.md)、[DCCA-GS_创新点说明](01-architecture/DCCA-GS_创新点说明.md)、[API参考](01-architecture/API参考.md)、[modules-design](01-architecture/modules-design.md)、[DEV](01-architecture/DEV.md)、[gsplat批量渲染说明](01-architecture/gsplat批量渲染说明.md) |
| [02-design](02-design/) | 实验设计（做什么、怎么验） | [语义先验实验设计](02-design/语义先验实验设计.md)、[SPA训练侧实验设计](02-design/SPA训练侧实验设计.md)、[CompGS残差编码实验设计](02-design/CompGS残差编码实验设计.md)、[创新点P0设计文档](02-design/创新点P0设计文档.md) |
| [03-reports](03-reports/) | 实验报告（阶段 A/B、方向结论） | [MiniSplat×SPA_实验报告](03-reports/MiniSplat×SPA_实验报告.md)、[消融实验汇总](03-reports/消融实验汇总.md)、[SPA_阶段A报告](03-reports/SPA_阶段A报告.md)、[语义先验_阶段A报告](03-reports/语义先验_阶段A报告.md)、[P0_阶段A报告](03-reports/P0_阶段A报告.md)、[R_阶段A报告](03-reports/R_阶段A报告.md)、[R4_attr上下文_报告](03-reports/R4_attr上下文_报告.md)、[feat_dim扫描报告](03-reports/feat_dim扫描报告.md)、[分块解码最小实验_3.5报告](03-reports/分块解码最小实验_3.5报告.md)、[Octree层级离线熵实验_6.4报告](03-reports/Octree层级离线熵实验_6.4报告.md) |
| [04-guides](04-guides/) | 环境、安装、上手 | [上手指南](04-guides/上手指南.md)、[环境说明](04-guides/环境说明.md)、[Windows安装说明](04-guides/Windows安装说明.md)、[Makefile](04-guides/Makefile)、[requirements.txt](04-guides/requirements.txt) |
| [05-paper](05-paper/) | 论文/提案 | [3366-…提案](05-paper/3366-DCCA-GS：Decoder-Reproducible%20Content-Adaptive%20Compression%20for%20Anchor-Based%203D%20Gaussian%20Splatting.docx)、[摘要引言改写](05-paper/3366_摘要引言改写.md)、[实验改写](05-paper/3366_实验改写.md)、[提案汇报讲稿](05-paper/DCCA-GS_提案汇报讲稿.md) |
| [06-planning](06-planning/) | 规划/交接/历史 | [HANDOVER](06-planning/HANDOVER.md)、[DCCA-GS_项目变更文档](06-planning/DCCA-GS_项目变更文档.md)、[DCCA-GS_改动计划](06-planning/DCCA-GS_改动计划.md)、[大场景与无人机场景扩展方向](06-planning/大场景与无人机场景扩展方向.md)、[未来改动方向](06-planning/未来改动方向.md) |
| [07-literature](07-literature/) | 文献调研 | [神经压缩文献调研](07-literature/神经压缩文献调研.md) |
| [data](data/) | 实验数据/结果 | [experiments.csv](data/experiments.csv)（统一实验数字）、[DeepBlending_survey.csv](data/DeepBlending_survey.csv) |
| [figures](figures/) | 图 | [db_rd_curve.png](figures/db_rd_curve.png) |

## 命名与维护规则

1. 新增实验设计 → `02-design/<主题>实验设计.md`；阶段/方向报告 → `03-reports/<主题>_报告.md`。
2. 新结果**先写** `data/experiments.csv`（18 列，易追加），再写报告/README。
3. 报告中的数字必须与 CSV 或 5090 `runs/*/metrics.jsonl` + `hac_meta.json` 一致；猜测值需标注「未实测」。
4. 跨目录链接：从 `docs/<分类>/` 出发用 `../`（例：`../03-reports/SPA_阶段A报告.md`），不写绝对路径。
5. 历史遗留的 `PHG_*` 文件名/绝对路径引用应随迁移统一清理（本目录已清理一次）。
