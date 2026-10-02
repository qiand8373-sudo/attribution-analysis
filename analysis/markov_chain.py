# -*- coding: utf-8 -*-
"""
马尔可夫链归因模型（第 5 种归因方法，方法论升级）
用法：python markov_chain.py   （在 analysis 目录下运行）

知识点：
- 前面 4 种模型都是「启发式规则」（人为规定价值怎么分），
  马尔可夫链是「数据驱动」：从用户路径数据里统计渠道转移概率，
  用「移除效应」（Removal Effect）衡量每个渠道对整体转化的贡献。
- 核心思想：把用户旅程看成一条马尔可夫链——状态是「开始→渠道A→渠道B
  →……→转化/流失」，每步的转移概率由数据统计得到。
  移除某个渠道 = 把它从所有路径里删掉，重新计算整体转化概率；
  转化概率掉得越多，说明这个渠道越不可替代，归因价值就越大。
- 面试考点：
  ① 启发式模型需要人为拍「首次/末次/衰减」规则，马尔可夫链让数据自己说话
  ② Removal Effect 的商业解释：「如果没有这个渠道，多少转化会消失」
  ③ 关键洞察：马尔可夫链能捕捉「渠道协同」——social 自身转化率很低，
     但它把用户推给 paid_search 的路径依赖，让它对整体转化有真实贡献；
     末次归因看不到这种协同，会把 social 的价值错算为 ≈0
  ④ 局限：一阶马尔可夫假设下一步只依赖当前状态（不看更早的历史）

数据说明：本模块使用「带渠道转移依赖」的马尔可夫模拟数据（扩展自 4 模型
同分布的独立抽样数据）——真实用户路径存在强顺序依赖（先种草后收割），
马尔可夫归因正是为了捕捉这种依赖而设计，数据假设与模型假设自洽。
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import attribution_models as am  # 复用 4 种启发式模型函数

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')

START, CONV, NULL = 'START', 'CONV', 'NULL'
MAX_LEN = 5          # 路径最长 5 个触点（与 4 模型口径一致）
AVG_VALUE = 137.0    # 客单价对齐 Olist

# ------------------------------------------------------------
# 第 1 步：马尔可夫数据生成器（带渠道转移依赖）
# ------------------------------------------------------------
# 渠道占比对齐 Olist marketing funnel 真实分布
CHANNEL_SHARE = {
    'organic_search': 0.287, 'paid_search': 0.198, 'social': 0.169,
    'unknown': 0.137, 'direct_traffic': 0.062, 'email': 0.062,
    'referral': 0.036, 'display': 0.015,
}
# 渠道 → 转化 的基准概率（渠道「直接收割力」）：
# 收割型渠道（paid/organic/direct）转化率高；纯种草渠道（social/display）
# 自身转化率极低——它们的价值藏在「把流量导给收割渠道」的转移结构里
CHANNEL_CONV_BASE = {
    'paid_search': 0.09, 'organic_search': 0.075, 'direct_traffic': 0.06,
    'unknown': 0.045, 'referral': 0.022, 'email': 0.011,
    'social': 0.004, 'display': 0.004,
}
# 渠道 → 流失 基准概率
CHANNEL_NULL_BASE = {
    'paid_search': 0.35, 'organic_search': 0.40, 'unknown': 0.40,
    'direct_traffic': 0.45, 'referral': 0.40, 'social': 0.30,
    'email': 0.45, 'display': 0.40,
}
# 渠道间转移偏好（种草型 → 收割型的协同路径）：
# social/display 是「流量入口」——自身不转化，但把大部分流量导给
# paid_search / organic_search 完成收割。这是归因冲突的结构性来源。
TRANSITION_PREF = {
    'social': {'paid_search': 0.55, 'organic_search': 0.20, 'display': 0.10},
    'display': {'paid_search': 0.50, 'organic_search': 0.20, 'social': 0.10},
    'referral': {'paid_search': 0.40, 'organic_search': 0.25},
    'email': {'organic_search': 0.35, 'paid_search': 0.25},
    'paid_search': {'direct_traffic': 0.20},
    'organic_search': {'direct_traffic': 0.20},
    'unknown': {'paid_search': 0.25, 'organic_search': 0.15},
    'direct_traffic': {},
}


def markov_simulate(n_users=5000, seed=20260929):
    """按一阶马尔可夫过程生成用户路径
    - START → 渠道（按真实占比）
    - 渠道 → CONV / NULL / 下一渠道（下一渠道按转移偏好，剩余概率
      均匀散到所有渠道模拟随机探索）
    - 到达 MAX_LEN 触点仍没转化 → 按基准概率直接判定 CONV/NULL
    """
    rng = np.random.default_rng(seed)
    channels = list(CHANNEL_SHARE.keys())
    share = np.array(list(CHANNEL_SHARE.values()))
    share = share / share.sum()

    rows = []
    for _ in range(n_users):
        path = [channels[int(rng.choice(len(channels), p=share))]]
        while True:
            cur = path[-1]
            conv_p = CHANNEL_CONV_BASE[cur]
            null_p = CHANNEL_NULL_BASE[cur]
            # 位置加成：越靠后触点离转化越近（时间衰减思想），收割力+30%
            bonus = 1 + 0.3 * (len(path) / MAX_LEN)
            if len(path) >= MAX_LEN:
                converted = rng.random() < min(conv_p * bonus * 1.5, 0.9)
                rows.append({'path': path, 'converted': converted,
                             'value': 0.0})
                break
            r = rng.random()
            if r < conv_p * bonus:                       # 转化
                rows.append({'path': path, 'converted': True, 'value': 0.0})
                break
            if r < (conv_p + null_p) * bonus:            # 流失
                rows.append({'path': path, 'converted': False, 'value': 0.0})
                break
            # 继续走下一触点：按转移偏好 + 随机探索
            nxt = np.random.default_rng(int(rng.integers(1e9)))
            pref = TRANSITION_PREF.get(cur, {})
            pref_keys = list(pref.keys())
            if pref_keys and nxt.random() < 0.5:         # 50% 走偏好转移
                w = np.array([pref[k] for k in pref_keys])
                path.append(pref_keys[int(nxt.choice(len(pref_keys), p=w / w.sum()))])
            else:                                        # 50% 随机探索
                path.append(channels[int(nxt.choice(len(channels), p=share))])
    df = pd.DataFrame(rows)
    mask = df['converted']
    rng_v = np.random.default_rng(seed + 1)
    df.loc[mask, 'value'] = rng_v.normal(AVG_VALUE, 40, size=mask.sum()).clip(20, 600)
    return df


# ------------------------------------------------------------
# 第 2 步：转移矩阵统计与吸收概率求解
# ------------------------------------------------------------
def build_transition(df):
    """从路径数据统计一阶转移概率矩阵"""
    channels = list(CHANNEL_SHARE.keys())
    states = [START] + channels + [CONV, NULL]
    trans = pd.DataFrame(0.0, index=states, columns=states)
    for _, row in df.iterrows():
        end = CONV if row['converted'] else NULL
        full = [START] + list(row['path']) + [end]
        for i in range(len(full) - 1):
            trans.loc[full[i], full[i + 1]] += 1
    prob = trans.div(trans.sum(axis=1), axis=0).fillna(0.0)
    return trans, prob


def overall_conversion(prob):
    """解马尔可夫链吸收概率：P(START → CONV) = ((I-Q)^-1 · R)[START, CONV]"""
    channels = list(CHANNEL_SHARE.keys())
    non_abs = [START] + channels
    Q = prob.loc[non_abs, non_abs].values
    R = prob.loc[non_abs, [CONV, NULL]].values
    N = np.linalg.inv(np.eye(len(non_abs)) - Q)
    absorb = N @ R
    return absorb[0, 0]


def removal_effect(df):
    """移除效应：删掉渠道 c 后整体转化概率下降比例"""
    _, prob = build_transition(df)
    baseline = overall_conversion(prob)
    effects = {}
    for ch in am.CHANNELS.keys():
        df_r = df.copy()
        df_r['path'] = df_r['path'].apply(lambda p: [c for c in p if c != ch])
        df_r = df_r[df_r['path'].apply(len) > 0]
        p_conv = overall_conversion(build_transition(df_r)[1])
        effects[ch] = max(1 - p_conv / baseline, 0.0)
    return effects, baseline


# ------------------------------------------------------------
# 第 3 步：Shapley 值归因（合作博弈论，捕捉渠道协同）
# ------------------------------------------------------------
def shapley_value(df):
    """Shapley 值：每个渠道按「边际贡献」分配转化价值
    联盟特征函数 v(S) = 只保留联盟 S 内渠道的触点后，路径产生的转化价值
    （把非 S 渠道从路径中删掉，模拟「没有这些渠道的世界」）
    渠道 i 的 Shapley 值 = 对所有不含 i 的联盟 S 加权平均：
    φ_i = Σ [v(S∪{i}) - v(S)] × |S|!(n-|S|-1)!/n!
    """
    channels = list(CHANNEL_SHARE.keys())
    n = len(channels)
    paths = df['path'].tolist()
    values = df['value'].values
    converted = df['converted'].values

    # 每条路径的渠道位图（加速联盟价值计算）
    path_masks = []
    for p in paths:
        mask = 0
        for c in p:
            mask |= 1 << channels.index(c)
        path_masks.append(mask)
    path_masks = np.array(path_masks)
    conv_value = np.where(converted, values, 0.0)

    # 预计算所有联盟的价值 v(S)：S 的位图 = 联盟成员渠道
    n_subsets = 1 << n
    v = np.zeros(n_subsets)
    for s in range(1, n_subsets):
        # 路径的渠道集 ∩ S ≠ ∅ 且转化 → 计入联盟价值
        hit = (path_masks & s) > 0
        v[s] = conv_value[hit].sum()
    v_full = v[n_subsets - 1]

    # 阶乘预计算
    from math import factorial
    fact = [factorial(i) for i in range(n + 1)]
    shapley = {}
    for i, ch in enumerate(channels):
        bit = 1 << i
        phi = 0.0
        for s in range(n_subsets):
            if s & bit:
                continue            # 只遍历不含 i 的联盟
            marginal = v[s | bit] - v[s]
            size = bin(s).count('1')
            phi += marginal * fact[size] * fact[n - size - 1] / fact[n]
        shapley[ch] = phi
    print(f"Shapley 值总和 {sum(shapley.values()):,.0f} vs 总转化价值 "
          f"{v_full:,.0f}（应相等，验证分配完备性）")
    return shapley


def main():
    df = markov_simulate()
    conv_rate = df['converted'].mean()
    print(f"马尔可夫模拟样本: {len(df)} 条路径, 转化 {df['converted'].sum()} 条"
          f"（转化率 {conv_rate * 100:.1f}%）, 总价值 {df['value'].sum():,.0f} 雷亚尔")

    effects, baseline = removal_effect(df)
    print(f"\n基准整体转化概率（模型推算）: {baseline:.4f}")
    print("移除效应（按效应从大到小）：")
    for ch, e in sorted(effects.items(), key=lambda x: -x[1]):
        print(f"  {ch:<15s} {e * 100:6.2f}%")

    # 归因价值 = 移除效应归一化 × 总转化价值
    total_effect = sum(effects.values())
    total_value = df['value'].sum()
    markov_credit = {c: e / total_effect * total_value for c, e in effects.items()}

    shapley = shapley_value(df)

    # 6 模型同批数据公平对比
    compare = pd.DataFrame({
        '首次触点': am.first_touch(df),
        '末次触点': am.last_touch(df),
        '线性': am.linear(df),
        '时间衰减': am.time_decay(df),
        '马尔可夫': markov_credit,
        'Shapley': shapley,
    })
    print('\n===== 6 种归因模型渠道价值对比（雷亚尔）=====')
    print(compare.round(0).to_string())
    print('\n种草型渠道 social：末次归因 '
          f"{compare.loc['social', '末次触点']:,.0f} vs 马尔可夫 "
          f"{compare.loc['social', '马尔可夫']:,.0f} vs Shapley "
          f"{compare.loc['social', 'Shapley']:,.0f}")

    # 可视化：末次 vs 马尔可夫 vs Shapley
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, ['末次触点', '马尔可夫', 'Shapley']):
        s = compare[col].sort_values()
        ax.barh(s.index, s.values, color='#2B579A')
        ax.set_title(f'{col}归因', fontsize=13)
        ax.set_xlabel('归因价值（雷亚尔）')
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle('行业默认 vs 马尔可夫移除效应 vs Shapley 博弈论归因', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '02_markov_compare.png'), dpi=150)
    plt.close()
    print('\n✅ 对比图已保存到 report/figures/02_markov_compare.png')


if __name__ == '__main__':
    main()
