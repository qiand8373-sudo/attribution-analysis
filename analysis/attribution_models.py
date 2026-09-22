# -*- coding: utf-8 -*-
"""
广告投放归因分析：4 种归因模型对比
用法：python 01_attribution_models.py   （在 analysis 目录下运行）

知识点：
- 归因问题：用户从「第一次看到广告」到「下单」会经过多个触点，
  转化价值该算在哪个渠道头上？不同模型给出不同答案。
- 4 种模型：
  ① 首次触点 First Touch：全部归给第一个触点（简单，偏重拉新）
  ② 末次触点 Last Touch：全部归给最后一个触点（Google Analytics 默认，
     偏重转化，低估了种草渠道）
  ③ 线性 Linear：平均分给每个触点（公平但无区分）
  ④ 时间衰减 Time Decay：越靠近转化的触点分得越多（半衰期 7 天，
     介于首次和末次之间，最接近真实决策过程）
- 面试考点：
  ① 为什么不同模型下渠道价值排序完全不同 → 预算分配结果天差地别
  ② Google/Facebook 默认末次归因，所以大家都觉得「搜索广告」最值钱，
     实际上它只是收割了别人种草的成果
  ③ 归因没有唯一正确答案，选模型 = 选业务假设

数据说明：触点路径数据是模拟的（真实用户路径数据涉及隐私，平台不会公开），
但渠道分布、路径长度分布、转化率全部对齐 Olist marketing funnel 的真实数据
（见 00_funnel_analysis.py），模型实现和结论逻辑可以原样迁移到真实数据。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

rng = np.random.default_rng(20260922)  # 固定随机种子，结果可复现

# ------------------------------------------------------------
# 第 1 步：生成模拟触点路径（对齐 Olist marketing funnel 真实分布）
# ------------------------------------------------------------
# 渠道及其真实线索占比（来自 00_funnel_analysis.py 的分析结果）
CHANNELS = {
    'organic_search':  0.287,  # 自然搜索 2,296/8,000
    'paid_search':     0.198,  # 付费搜索
    'social':          0.169,  # 社交媒体
    'unknown':         0.137,  # 未知来源
    'direct_traffic':  0.062,  # 直接访问
    'email':           0.062,  # 邮件营销
    'referral':        0.036,  # 外部引荐
    'display':         0.015,  # 展示广告
}
# 各渠道线索质量（用真实转化率，social/email 偏低）
CHANNEL_QUALITY = {
    'organic_search':  11.8,
    'paid_search':     12.3,
    'social':           5.6,
    'unknown':         16.3,
    'direct_traffic':  11.2,
    'email':            3.0,
    'referral':         8.5,
    'display':          5.1,
}

def simulate_paths(n_users=5000):
    """模拟 n_users 条用户触点路径
    - 路径长度：1~5 个触点（短路径占多数，贴近真实用户行为）
    - 转化与否：整体约 10.5%，与真实漏斗一致；
      每个触点的渠道质量影响最终转化概率（高质量渠道越多越容易转化）
    - 🔑 位置效应（归因模型冲突的来源）：
      种草型渠道（social/display/referral/email）多出现在路径前端，
      收割型渠道（paid_search/organic_search/direct_traffic）多出现在末端。
      于是「首次触点」模型会把价值判给种草渠道，
      「末次触点」模型会把价值判给收割渠道——排名打架。
    """
    channels = list(CHANNELS.keys())
    probs = np.array(list(CHANNELS.values()))
    probs = probs / probs.sum()  # 归一化：以上是 8 个主渠道占比，合计约 96.6%
    front_channels = ['social', 'display', 'referral', 'email']      # 种草型
    back_channels = ['paid_search', 'organic_search', 'direct_traffic']  # 收割型
    paths = []
    for _ in range(n_users):
        # 路径长度：1(40%) 2(30%) 3(17%) 4(9%) 5(4%)
        length = int(rng.choice([1, 2, 3, 4, 5], p=[0.40, 0.30, 0.17, 0.09, 0.04]))
        seq = list(rng.choice(channels, size=length, p=probs))
        # 注入位置效应：首触点 80% 概率换成种草型渠道，末触点 80% 换成收割型
        if length > 1 and rng.random() < 0.8:
            seq[0] = str(rng.choice(front_channels))
        if rng.random() < 0.8:
            seq[-1] = str(rng.choice(back_channels))
        # 转化概率 = 平均渠道质量换算到 0-1，再乘一个随机因子
        avg_quality = np.mean([CHANNEL_QUALITY[c] for c in seq])
        base_prob = avg_quality / 16.3 * 0.105  # 归一化：质量满分时转化率 10.5%
        # 路径越长触点越多，转化概率略升（多次触达加深印象）
        convert = rng.random() < base_prob * (1 + 0.05 * length)
        paths.append({'path': seq, 'converted': convert, 'value': 0.0})
    return pd.DataFrame(paths)

def assign_value(df, avg_order_value=137.0):
    """转化用户的消费价值：均值 137（对齐 Olist 客单价）"""
    mask = df['converted']
    df.loc[mask, 'value'] = rng.normal(avg_order_value, 40, size=mask.sum()).clip(20, 600)
    return df

# ------------------------------------------------------------
# 第 2 步：4 种归因模型的实现
# ------------------------------------------------------------
def first_touch(df):
    """首次触点：价值全归第一个渠道"""
    credit = {c: 0.0 for c in CHANNELS}
    for _, row in df[df['converted']].iterrows():
        credit[row['path'][0]] += row['value']
    return credit

def last_touch(df):
    """末次触点：价值全归最后一个渠道"""
    credit = {c: 0.0 for c in CHANNELS}
    for _, row in df[df['converted']].iterrows():
        credit[row['path'][-1]] += row['value']
    return credit

def linear(df):
    """线性：价值平均分给路径上每个触点"""
    credit = {c: 0.0 for c in CHANNELS}
    for _, row in df[df['converted']].iterrows():
        share = row['value'] / len(row['path'])
        for ch in row['path']:
            credit[ch] += share
    return credit

def time_decay(df, halflife=3):
    """时间衰减：离转化越近的触点权重越大
    权重公式：w_i = 2^(-t_i / 半衰期)，t_i = 距转化的触点数
    例：路径 [A, B, C] 转化，半衰期 3：
        C 距转化 0 个触点 → 权重 1
        B 距转化 1 个触点 → 权重 2^(-1/3) = 0.79
        A 距转化 2 个触点 → 权重 2^(-2/3) = 0.63
    """
    credit = {c: 0.0 for c in CHANNELS}
    for _, row in df[df['converted']].iterrows():
        path = row['path']
        # 距转化天数：假设触点间间隔 2 天，最后一个触点到转化 0 天
        distances = [2 * (len(path) - 1 - i) for i in range(len(path))]
        weights = np.array([2 ** (-d / halflife) for d in distances])
        weights = weights / weights.sum()  # 归一化，总权重 = 1
        for ch, w in zip(path, weights):
            credit[ch] += row['value'] * w
    return credit

def main():
    # ------------------------------------------------------------
    # 第 3 步：4 种模型结果对比
    # ------------------------------------------------------------
    paths_df = assign_value(simulate_paths())
    print(f"模拟 {len(paths_df)} 条路径，转化 {paths_df['converted'].sum()} 条"
          f"（转化率 {paths_df['converted'].mean()*100:.1f}%），"
          f"总转化价值 {paths_df['value'].sum():,.0f}")

    models = {
        '首次触点': first_touch(paths_df),
        '末次触点': last_touch(paths_df),
        '线性':     linear(paths_df),
        '时间衰减': time_decay(paths_df),
    }
    result = pd.DataFrame(models)
    result['首次触点排名'] = result['首次触点'].rank(ascending=False).astype(int)
    result['末次触点排名'] = result['末次触点'].rank(ascending=False).astype(int)
    print('\n===== 各渠道归因价值对比（4 种模型）=====')
    print(result.round(0).to_string())

    # ------------------------------------------------------------
    # 第 4 步：可视化对比
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (name, credit) in zip(axes.flat, models.items()):
        s = pd.Series(credit).sort_values(ascending=True)
        ax.barh(s.index, s.values, color='#2B579A')
        ax.set_title(name, fontsize=13)
        ax.set_xlabel('归因价值（雷亚尔）')
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle('同一批转化数据，4 种归因模型给出的渠道价值', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '01_attribution_compare.png'), dpi=150)
    plt.close()
    print('\n✅ 对比图已保存到 report/figures/01_attribution_compare.png')


if __name__ == '__main__':
    main()
