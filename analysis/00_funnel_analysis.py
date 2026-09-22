# -*- coding: utf-8 -*-
"""
销售漏斗分析：Olist marketing funnel 真实数据
用法：python 00_funnel_analysis.py   （在 analysis 目录下运行）

知识点：
- 漏斗分析：MQL（营销合格线索）→ SQL（销售合格线索）→ 成交，
  每一层转化率相乘 = 总转化率
- 渠道质量：不同来源渠道的线索转化率差异 = 渠道质量差异
  （线索量 ≠ 线索质量，social 量多但转化差）
- 面试考点：
  ① 漏斗每一层掉了多少人、掉在哪一层 → 优化点就在那一层
  ② 「线索质量」比「线索数量」重要：1 万条垃圾线索不如 100 条精准线索
  ③ 数据里 unknown 渠道转化率最高——未知来源可能是线下/合作伙伴推荐，
     实际工作中要去查埋点，把 unknown 拆开，别让预算投给"看不清的渠道"
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

mql = pd.read_csv(os.path.join(DATA_DIR, 'olist_marketing_qualified_leads_dataset.csv'))
closed = pd.read_csv(os.path.join(DATA_DIR, 'olist_closed_deals_dataset.csv'))

# ---------- 1. 整体漏斗 ----------
print('===== 1. 整体销售漏斗 =====')
n_mql = len(mql)
n_won = len(closed)
# 数据集没有中间层（SQL 商机）数据，漏斗简化为两层
print(f'MQL 营销线索：{n_mql}')
print(f'成交：{n_won}')
print(f'总转化率：{n_won / n_mql * 100:.1f}%\n')

# ---------- 2. 渠道线索量 vs 转化率 ----------
print('===== 2. 渠道线索质量对比 =====')
merged = mql.merge(closed[['mql_id']].assign(成交=1), on='mql_id', how='left').fillna({'成交': 0})
by_origin = merged.groupby('origin').agg(
    线索数=('mql_id', 'count'),
    成交数=('成交', 'sum'),
)
by_origin['转化率%'] = (by_origin['成交数'] / by_origin['线索数'] * 100).round(1)
by_origin = by_origin.sort_values('线索数', ascending=False)
print(by_origin.to_string())

# ---------- 3. 落地页质量 ----------
print('\n===== 3. 落地页（landing_page_id）线索量 TOP5 =====')
lp = merged.groupby('landing_page_id').agg(线索数=('mql_id', 'count'), 成交数=('成交', 'sum'))
lp['转化率%'] = (lp['成交数'] / lp['线索数'] * 100).round(1)
print(lp.sort_values('线索数', ascending=False).head(5).to_string())

# ---------- 4. 成交结构：业务类型 ----------
print('\n===== 4. 成交卖家的业务类型分布 =====')
print(closed['business_segment'].value_counts().head(5).to_string())
print('\n线索类型（lead_type）:')
print(closed['lead_type'].value_counts().head(5).to_string())

# ---------- 图 1：漏斗图 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# 左：两层漏斗
ax = axes[0]
stages = ['MQL\n营销线索', '成交\n（入驻卖家）']
values = [n_mql, n_won]
colors = ['#2B579A', '#E07B39']
bars = ax.bar(stages, values, color=colors, width=0.5)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 100, f'{v:,}\n({v/n_mql*100:.1f}%)',
            ha='center', fontsize=11)
ax.set_title('营销漏斗：MQL → 成交', fontsize=14)
ax.set_ylabel('人数')
ax.grid(axis='y', alpha=0.3)
# 右：渠道线索量 vs 转化率（双重视角）
ax = axes[1]
top = by_origin.head(8)
x = range(len(top))
ax.bar(x, top['线索数'], color='#2B579A', label='线索数', width=0.6)
ax.set_xticks(list(x))
ax.set_xticklabels(top.index, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('线索数')
ax2 = ax.twinx()
ax2.plot(x, top['转化率%'], color='#E07B39', marker='o', label='转化率%')
ax2.set_ylabel('转化率 %')
ax.set_title('渠道线索量 vs 转化率（量多≠质好）', fontsize=14)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '00_funnel.png'), dpi=150)
plt.close()
print('\n✅ 漏斗图已保存到 report/figures/00_funnel.png')
