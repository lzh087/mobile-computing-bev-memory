# 移动计算导论：BEV 时序缓存预算原型

课程研究原型的源码仓库，包含简化时序建图模型、FP16 二进制缓存编解码、预算对照评测与合成场景生成代码。

## 快速检查

需要 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-rerun.txt
.\.venv\Scripts\python.exe code/budget_state_codec.py
```

编解码检查不需要数据或训练权重。预期输出 `checks: passed`，并打印完整状态和稀疏状态的字节数。

## 完整评测

数据、模型权重和运行日志保留在本地作业包，不随本仓库发布。运行完整评测前，将已有本地作业包的下列目录放到仓库根目录：

```text
data/synthetic/val/                 # 合成验证序列及标签
data/bev/crf/val/                   # 对应冻结 BEV 特征与不确定性代理
runs/map_crf_gru_q/best.pt          # 已有冻结建图权重
```

然后运行：

```powershell
.\run.ps1 -PythonExecutable .\.venv\Scripts\python.exe
```

已有 Python 环境也可直接执行：

```bash
python code/evaluate_budget_packet.py --out rerun_evidence --threads 8 --limit 30
```

新生成的环境信息、输入哈希、结果表和状态样例写入 `rerun_evidence/`，该目录不纳入 Git 版本控制。

## 源码说明

- `code/budget_state_codec.py`：FP16 状态编码、解码与往返检查，计入协议头、背景、索引、显著网格、查询及折线参考点。
- `code/evaluate_budget_packet.py`：冻结 BEV 后的推理评测，比较 BEV 保留率、历史查询数和选点评分。
- `code/src/edgebev/models/`：简化深度、视图变换、ConvGRU、时序融合及地图解码器。
- `code/src/edgebev/synthetic/`：合成道路场景、渲染与序列数据结构。
- `code/train_map.py`、`code/train_heads.py`：既有研究原型的训练与数据加载代码。

## 实验边界

本地原型参考 BEVNeXt 的深度/时序融合与 StreamMapNet 的查询传播思想，与官方模型、数据和评价器不等价。当前评测复用既有合成验证缓存，未执行 nuScenes/Occ3D 官方复现。

旧缓存生成时未保存完整编码器参数，因此当前完整评测依赖已有缓存，不能承诺从原图生成同一批特征。验证集曾用于历史选模，分数不构成独立测试集泛化证据。编解码器的状态字节数也不等于进程峰值内存、GPU 显存或功耗。

代码使用 AI 辅助整理和实现。本次实际重新运行的是冻结特征后的推理和状态编解码，历史训练与本次评测分别记录在本地作业报告中。

## 论文与公开资源

- [BEVFormer，ECCV 2022](https://arxiv.org/abs/2203.17270)，[官方代码](https://github.com/fundamentalvision/BEVFormer)
- [GDFusion，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Rethinking_Temporal_Fusion_with_a_Unified_Gradient_Descent_View_for_CVPR_2025_paper.html)，[官方代码](https://github.com/cdb342/GDFusion)
- [StreamMapNet，WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Yuan_StreamMapNet_Streaming_Mapping_Network_for_Vectorized_Online_HD_Map_Construction_WACV_2024_paper.html)，[官方代码](https://github.com/yuantianyuan01/StreamMapNet)
- [BEVNeXt 官方代码](https://github.com/woxihuanjiangguo/BEVNeXt)
- [nuScenes](https://www.nuscenes.org/nuscenes)、[nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)、[Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)
