# 广告投放归因分析

基于 Olist marketing funnel 公开数据集的营销漏斗与广告归因分析项目。

面向广告投放/数据运营/商业分析实习岗位：真实漏斗分析 + 4 种归因模型实现对比 + ROI 预算分配建议。

## 数据来源

Kaggle 公开数据集 [Marketing Funnel by Olist](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist)（8,000 条营销线索）。原始 CSV 不入库（.gitignore 排除）。

触点路径数据为模拟生成（真实用户路径数据涉及隐私不公开），渠道分布、路径长度、转化率全部对齐真实漏斗数据，归因模型实现与结论逻辑可原样迁移到真实投放数据。

## 目录结构

```
attribution-analysis/
├── data/                     # 原始 CSV（不入 git）
├── analysis/
│   ├── 00_funnel_analysis.py # 真实漏斗：渠道线索量 vs 转化质量
│   ├── attribution_models.py # 4 种归因模型实现 + 模拟触点数据 + 对比图
│   └── 02_roi_analysis.py    # 渠道成本 vs 归因价值 → ROI → 预算建议
└── report/
    ├── 归因分析报告.md
    └── figures/
```

## 快速开始

```bash
cd analysis
python 00_funnel_analysis.py     # 1. 真实漏斗分析（需先下载数据到 data/）
python attribution_models.py     # 2. 4 种归因模型对比
python 02_roi_analysis.py        # 3. ROI 与预算分配建议
```

## 核心结论速览

- 真实漏斗：8,000 条 MQL → 842 成交，转化率 10.5%
- social 线索量大但转化率仅 5.6%，paid_search 量少质优（12.3%）
- 同一批转化数据：social 渠道价值在首次归因下 7,404，末次归因下仅 1,252（差 6 倍）
- 末次归因（行业默认）系统性低估种草渠道、高估收割渠道
- email 末次 ROI=0 会被误砍，实际拉新 ROI 5.97
- 预算分配建议：时间衰减口径做基准，首次/末次做上下界

## 学习知识点清单

Python：pandas 数据处理、numpy 随机模拟（固定种子）、matplotlib 双轴图/子图、模块导入与复用

分析方法：销售漏斗、归因模型 4 种（首次/末次/线性/时间衰减）、ROI 计算、渠道质量评估
