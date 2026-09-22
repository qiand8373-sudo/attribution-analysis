# -*- coding: utf-8 -*-
"""
渠道 ROI 分析：归因价值 vs 渠道成本 → 预算分配建议
用法：python 02_roi_analysis.py   （在 analysis 目录下运行）

知识点：
- ROI（投资回报率）= 收益 / 成本。广告投放语境：
  渠道 ROI = 该渠道带来的转化价值 / 该渠道投入成本
- 归因模型直接决定「该渠道带来多少价值」→ 直接决定预算怎么分。
  用错模型，钱就投错地方。
- 面试考点：
  ① ROI > 1 才赚钱；ROI 排序 = 预算优先序
  ② 但 ROI 不是唯一指标：拉新渠道（social）ROI 可能难看，
     却是在给未来的收割渠道（search）供流量——所以要看「组合」而非单渠道
  ③ 实战中常见做法：末次归因给绩效（短视），首次归因辅助看拉新贡献

数据说明：渠道成本为假设值（真实投放成本数据不公开），
每条线索成本 CPL 参考行业量级设定，结论逻辑可迁移。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))  # 让 import 同目录的 01 生效

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attribution_models import (
    simulate_paths, assign_value,
    first_touch, last_touch, linear, time_decay, CHANNELS,
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')

# 渠道成本假设：每条线索成本 CPL（雷亚尔/条），
# 按「获客成本量级」设定：付费搜索最贵、邮件最便宜
CPL = {
    'organic_search':  2.0,   # 自然搜索：SEO 人力成本分摊，单线索便宜
    'paid_search':    15.0,   # 付费搜索：竞价点击，贵
    'social':          8.0,   # 社交媒体：内容+投放，中等
    'unknown':         5.0,   # 未知来源：按均值计
    'direct_traffic':  0.5,   # 直接访问：几乎零成本
    'email':           1.0,   # 邮件营销：几乎零成本
    'referral':        3.0,   # 引荐：联盟佣金
    'display':        10.0,   # 展示广告：CPM 摊下来不便宜
}

def main():
    paths_df = assign_value(simulate_paths())
    n_users = len(paths_df)

    # 渠道总成本 = 单线索成本 × 触点出现次数（每条触达都花钱）
    total_cost = {}
    for ch in CHANNELS:
        touches = paths_df['path'].apply(lambda p: p.count(ch)).sum()
        total_cost[ch] = touches * CPL[ch]

    # 4 种归因模型下的价值
    model_credits = {
        '首次触点': first_touch(paths_df),
        '末次触点': last_touch(paths_df),
        '线性':     linear(paths_df),
        '时间衰减': time_decay(paths_df),
    }

    # ROI 对比表
    rows = []
    for ch in CHANNELS:
        row = {'渠道': ch, '成本': total_cost[ch]}
        for mname, credit in model_credits.items():
            row[mname + ' ROI'] = credit[ch] / total_cost[ch]
        rows.append(row)
    roi_df = pd.DataFrame(rows).set_index('渠道')
    pd.set_option('display.float_format', '{:.2f}'.format)
    print('===== 各渠道 ROI 对比（4 种归因模型）=====')
    print(roi_df.round(2).to_string())
    print('\nROI > 1 表示渠道赚钱；ROI < 1 表示投放亏损（按该归因口径）')

    # 关键洞察输出
    print('\n===== 关键洞察 =====')
    last = roi_df['末次触点 ROI'].sort_values(ascending=False)
    print(f'按末次触点（最常用口径）：ROI 最高的渠道是 {last.index[0]}（{last.iloc[0]:.2f}），'
          f'ROI 最低的是 {last.index[-1]}（{last.iloc[-1]:.2f}）')
    ft = roi_df['首次触点 ROI'].sort_values(ascending=False)
    print(f'按首次触点：ROI 最高的渠道是 {ft.index[0]}（{ft.iloc[0]:.2f}）')
    # 找出两种口径下变化最大的渠道
    diff = (roi_df['首次触点 ROI'] - roi_df['末次触点 ROI']).abs().sort_values(ascending=False)
    print(f'口径差异最大的渠道：{diff.index[0]}——预算分配要重点讨论的对象')

    # 图：末次 vs 首次 ROI 对比
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(roi_df))
    w = 0.38
    ax.bar([i - w/2 for i in x], roi_df['首次触点 ROI'], w, label='首次触点 ROI', color='#2B579A')
    ax.bar([i + w/2 for i in x], roi_df['末次触点 ROI'], w, label='末次触点 ROI', color='#E07B39')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='盈亏平衡线 ROI=1')
    ax.set_xticks(list(x))
    ax.set_xticklabels(roi_df.index, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('ROI（归因价值 / 成本）')
    ax.set_title('同一批数据：首次 vs 末次归因口径下的渠道 ROI', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '02_roi_compare.png'), dpi=150)
    plt.close()
    print('\n✅ ROI 对比图已保存到 report/figures/02_roi_compare.png')


if __name__ == '__main__':
    main()
