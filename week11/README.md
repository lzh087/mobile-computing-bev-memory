# 第 11 周：未来可见性加权蒸馏

这个课程原型检验一个问题：把更多教师监督分给当前缺失、未来可见的远方网格，能否改善单帧 BEV 建图？程序以合成观测训练小型卷积网络，输出分隔线、道路边界和人行横道的栅格概率。研究参考 [AMap（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Li_AMap_Distilling_Future_Priors_for_Ahead-Aware_Online_HD_Map_Construction_CVPR_2026_paper.html) 的训练期未来教师思路。

## 运行

实测环境为 Windows 11、Python 3.12.8、PyTorch 2.11.0 CPU、NumPy 2.4.4、SciPy 1.17.1。数据由 `bev_data.py` 按固定场景种子生成。安装依赖后可以离线执行。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -u distill.py --out runs_week11
```

如已装有依赖，可直接运行 `python -u distill.py --out runs_week11`。快速检查可用 `--train 24 --validation 12 --test 12 --epochs 1 --seeds 11 --out smoke_week11`。`--kd-weight 0.2` 可复核低权重补充实验，请指定新的输出目录。

## 训练与评估口径

1. 训练、阈值选择、测试分别使用 384、96、192 个不同种子的合成场景。每个场景模拟当前、前移 4 m、前移 8 m 三次观测。
2. 教师将未来两帧对齐至当前坐标，使用 12 通道输入。学生推理只输入当前帧的 4 通道，所有学生使用相同网络。
3. 对比纯监督、均匀蒸馏、仅远区加权、远区可见性加权。每种配置训练三次，种子为 11、23、47。
4. 在验证集选定每个模型的阈值，再在测试集分别计算 0–24 m 和 24–48 m 的三类宏 F1。额外测试当前远区条带遮挡，以及教师配准额外偏移 +2 m。
5. `summary.csv` 汇总跨种子均值与样本标准差，`per_seed.csv` 保留逐种子分数，`results.json` 包含逐类分数，`protocol.json` 记录数据划分和检查。末个种子的学生权重、延迟和预测示例输出到本地。

这些实验用于分析蒸馏权重与可见性的作用。栅格宏 F1 和论文的矢量 A-mAP 使用不同的任务和评价口径。实测日志与报告随课程提交包提供。

## 论文资源

- [AMap 论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_AMap_Distilling_Future_Priors_for_Ahead-Aware_Online_HD_Map_Construction_CVPR_2026_paper.pdf)
- [AMap 作者项目仓库](https://github.com/BUAA-RickyLi/AMap)·截至 2026-11-18，仓库提供项目页面与公开素材，README 将训练代码列为待发布。
- [MapTracker 公开实现](https://github.com/woodfrog/maptracker)
- [nuScenes 数据](https://www.nuscenes.org/nuscenes) · [Argoverse 2 Sensor 数据](https://argoverse.github.io/user-guide/datasets/sensor.html)
