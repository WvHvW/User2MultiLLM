import os
from typing import Dict, List, Tuple, Any

import pandas as pd

import Entity
import User2LLM


def parse_sheet_name(sheet_name: str) -> Tuple[str, str]:
    parts = sheet_name.split('-', 1)
    if len(parts) != 2:
        raise ValueError(f"非法 sheet 名称格式: {sheet_name}")
    return parts[0], parts[1]


def get_first_user_id(user_distribution: str) -> int:
    users = Entity.load_user_info(user_distribution)
    sorted_users = sorted(users.items(),
                          key=lambda item: item[1].bw,
                          reverse=True)
    assert sorted_users, "用户集合为空，无法确定第一个用户"
    return sorted_users[0][0]


def extract_algorithm_totals(df: pd.DataFrame) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    if 'algorithm' not in df.columns or 'total_cost' not in df.columns:
        return totals
    mask = df['algorithm'].astype(str).str.endswith('-Total')
    for _, row in df[mask].iterrows():
        alg_total = str(row['algorithm'])
        base_alg = alg_total[:-len('-Total')]
        totals[base_alg] = float(row['total_cost'])
    return totals


def normalize_algorithm_for_link(alg_origin: str) -> str:
    if alg_origin == 'no-split':
        return 'no-split'
    parts = alg_origin.split('-')
    if len(parts) >= 2 and parts[1] == 'split':
        k_str = parts[0]
        if len(parts) >= 3 and parts[2] == 'augment':
            return f'k-split-augment-{k_str}'
        return f'k-split-{k_str}'
    return alg_origin


def collect_anomaly_data(origin_path: str,
                         link_util_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(origin_path):
        raise FileNotFoundError(f"未找到 OriginResults 文件: {origin_path}")
    if not os.path.exists(link_util_path):
        raise FileNotFoundError(f"未找到 LinkUtilization 文件: {link_util_path}")

    origin_xls = pd.ExcelFile(origin_path)
    link_xls = pd.ExcelFile(link_util_path)

    cost_rows: List[Dict[str, Any]] = []
    link_rows: List[Dict[str, Any]] = []
    first_user_rows: List[Dict[str, Any]] = []

    for sheet_name in origin_xls.sheet_names:
        if sheet_name not in link_xls.sheet_names:
            continue

        origin_df = origin_xls.parse(sheet_name)
        link_df = link_xls.parse(sheet_name)

        totals = extract_algorithm_totals(origin_df)
        if not totals:
            continue
        no_split_cost = totals.get('no-split')
        if no_split_cost is None:
            continue

        user_distribution, llm_distribution = parse_sheet_name(sheet_name)
        first_user_id = get_first_user_id(user_distribution)

        ks = set()
        for alg_name in totals.keys():
            parts = alg_name.split('-')
            if len(parts) >= 2 and parts[1] == 'split':
                ks.add(parts[0])

        for k_str in sorted(ks, key=lambda x: float(x)):
            k_split_name = f'{k_str}-split'
            k_split_aug_name = f'{k_str}-split-augment'
            k_split_cost = totals.get(k_split_name)
            k_split_aug_cost = totals.get(k_split_aug_name)

            if k_split_cost is None and k_split_aug_cost is None:
                continue

            is_k_split_gt_no = (k_split_cost is not None
                                and k_split_cost > no_split_cost)
            is_k_split_aug_gt_no = (k_split_aug_cost is not None
                                    and k_split_aug_cost > no_split_cost)
            is_k_split_aug_gt_k = (k_split_cost is not None
                                   and k_split_aug_cost is not None
                                   and k_split_aug_cost > k_split_cost)

            if not (is_k_split_gt_no or is_k_split_aug_gt_no
                    or is_k_split_aug_gt_k):
                continue

            cost_rows.append({
                'sheet_name': sheet_name,
                'user_distribution': user_distribution,
                'llm_distribution': llm_distribution,
                'k': float(k_str),
                'no_split_cost': no_split_cost,
                'k_split_cost': k_split_cost,
                'k_split_augment_cost': k_split_aug_cost,
                'is_k_split_gt_no_split': is_k_split_gt_no,
                'is_k_split_augment_gt_no_split': is_k_split_aug_gt_no,
                'is_k_split_augment_gt_k_split': is_k_split_aug_gt_k,
            })

            # 链路利用率异常汇总：对三个算法同时记录关键链路利用率
            for alg_origin in ('no-split', k_split_name, k_split_aug_name):
                if alg_origin not in totals:
                    continue
                link_alg = normalize_algorithm_for_link(alg_origin)
                link_row_candidates = link_df[link_df['algorithm'] == link_alg]
                if link_row_candidates.empty:
                    continue
                link_row = link_row_candidates.iloc[0]

                lr: Dict[str, Any] = {
                    'sheet_name': sheet_name,
                    'user_distribution': user_distribution,
                    'llm_distribution': llm_distribution,
                    'k': float(k_str),
                    'algorithm': alg_origin,
                    'no_split_cost': no_split_cost,
                    'k_split_cost': k_split_cost,
                    'k_split_augment_cost': k_split_aug_cost,
                    'is_k_split_gt_no_split': is_k_split_gt_no,
                    'is_k_split_augment_gt_no_split': is_k_split_aug_gt_no,
                    'is_k_split_augment_gt_k_split': is_k_split_aug_gt_k,
                }

                for col in link_df.columns:
                    if col.startswith('link_'):
                        lr[col] = link_row[col]
                link_rows.append(lr)

            # 第一个用户在三个算法下的分配情况
            for alg_origin in ('no-split', k_split_name, k_split_aug_name):
                if alg_origin not in totals:
                    continue
                mask = (
                    (origin_df['algorithm'] == alg_origin)
                    & (origin_df['user_id'] == first_user_id)
                    & (origin_df['path'] != 'Total')
                )
                cols = [
                    'algorithm', 'user_id', 'llm_id', 'path', 'total_flow',
                    'total_cost'
                ]
                if not all(c in origin_df.columns for c in cols):
                    continue
                sub = origin_df.loc[mask, cols]
                for _, row in sub.iterrows():
                    first_user_rows.append({
                        'sheet_name': sheet_name,
                        'user_distribution': user_distribution,
                        'llm_distribution': llm_distribution,
                        'k': float(k_str),
                        'first_user_id': first_user_id,
                        'algorithm': row['algorithm'],
                        'llm_id': row['llm_id'],
                        'path': row['path'],
                        'total_flow': row['total_flow'],
                        'total_cost': row['total_cost'],
                    })

    cost_df = pd.DataFrame(cost_rows)
    link_df = pd.DataFrame(link_rows)
    first_user_df = pd.DataFrame(first_user_rows)
    return cost_df, link_df, first_user_df


def run_and_save_anomaly_report() -> str:
    origin_path = User2LLM.origin_file
    link_util_path = User2LLM.link_util_file
    cost_df, link_df, first_user_df = collect_anomaly_data(
        origin_path, link_util_path)

    result_dir = os.path.dirname(origin_path)
    out_path = os.path.join(result_dir, 'AnomalyAnalysis.xlsx')

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        cost_df.to_excel(writer, sheet_name='cost_anomalies', index=False)
        link_df.to_excel(writer, sheet_name='link_util_anomalies', index=False)
        first_user_df.to_excel(writer,
                               sheet_name='first_user_allocations',
                               index=False)

    return out_path


if __name__ == "__main__":
    out_file = run_and_save_anomaly_report()
    print(f"异常分析结果已写入: {out_file}")

