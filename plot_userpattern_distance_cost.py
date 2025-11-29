import os
from typing import Dict, List, Tuple

import pandas as pd

import Entity
from analyze_bandwidth_llm_combined import (
    compute_llm_user_distance_indicator,
    plot_dual_axis,
    ALGORITHMS,
    _USER_PATTERN_DIR,
)


def _load_userpattern_all() -> pd.DataFrame:
    excel_path = os.path.join(_USER_PATTERN_DIR, "userpattern_all.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"未找到聚合数据文件: {excel_path}")

    excel_file = pd.ExcelFile(excel_path)
    frames: List[pd.DataFrame] = []

    for sheet_name in excel_file.sheet_names:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
        df_sheet["distribution"] = sheet_name
        frames.append(df_sheet)

    if not frames:
        raise ValueError("userpattern_all.xlsx 中没有任何数据 sheet")

    return pd.concat(frames, ignore_index=True)


def _build_distance_indicator_cache(distributions: List[str]) -> Dict[str, float]:
    cache: Dict[str, float] = {}

    for dist_name in distributions:
        if dist_name in cache:
            continue

        # 解析 user/llm 分布名称
        parts = dist_name.split("-", 1)
        if len(parts) != 2:
            continue
        user_distribution, llm_distribution = parts[0], parts[1]

        json_obj = Entity.load_network_from_sheets()
        network = json_obj["network"]
        llms = Entity.load_llm_info(user_distribution, llm_distribution)
        users = Entity.load_user_info(user_distribution)

        indicator = compute_llm_user_distance_indicator(network, users, llms)
        cache[dist_name] = indicator

    return cache


def plot_cost_vs_distance_from_excel() -> None:
    """
    基于 userpattern_all.xlsx 的聚合数据，按“LLM-User 距离指标”绘制 cost / optimization / service_rate：
    - X 轴：每个 distribution 的距离指标（sigmoid 归一化后的标量）；
    - Y 轴：四种算法在固定 (pattern, bandwidth, llm_computation) 下的 cost；
    - 右侧叠加 optimization 与 service_rate 折线，复用 plot_dual_axis 的逻辑。
    """
    df_all = _load_userpattern_all()

    if "view" in df_all.columns:
        df_bw = df_all[df_all["view"] == "bandwidth"].copy()
    else:
        df_bw = df_all.copy()

    if df_bw.empty:
        print("userpattern_all.xlsx 中缺少 bandwidth 视角的数据，无法绘制距离指标图。")
        return

    # 为每个 distribution 计算距离指标
    distributions = sorted(df_bw["distribution"].unique())
    distance_indicator_cache = _build_distance_indicator_cache(distributions)

    # 按 pattern 分图
    pattern_indices = sorted(df_bw["pattern_index"].unique())

    for pattern_index in pattern_indices:
        df_p = df_bw[df_bw["pattern_index"] == pattern_index]
        if df_p.empty:
            continue

        base_dir = os.path.join(_USER_PATTERN_DIR,
                                f"pattern{int(pattern_index)}",
                                "perdistance")
        os.makedirs(base_dir, exist_ok=True)

        # 每个 (bandwidth, llm_computation) 一张图
        for (bandwidth, llm_comp), group in df_p.groupby(
                ["bandwidth", "llm_computation"]):
            # 按距离指标排序 distribution，并构造“分布名(指标值)”标签
            dists_in_group = sorted(
                group["distribution"].unique(),
                key=lambda name: distance_indicator_cache.get(name, 0.0))
            x_values: List[str] = []
            for name in dists_in_group:
                indicator_val = distance_indicator_cache.get(name, 0.0)
                x_values.append(f"{name}({indicator_val:.3f})")

            data_by_algorithm: Dict[str, List[Tuple[float, float, float]]] = {}

            for alg in ['no-split', '1-split', 'task-offloading',
                        'bottleneck-augment']:
                values: List[Tuple[float, float, float]] = []
                for dist_name in dists_in_group:
                    rows = group[(group["distribution"] == dist_name)
                                 & (group["algorithm"] == alg)]
                    if rows.empty:
                        continue
                    row = rows.iloc[0]
                    opt = float(row.get("optimization", 0.0))
                    sr = float(row.get("service_rate", 0.0))
                    cost = float(row.get("total_cost", 0.0))
                    values.append((opt, sr, cost))
                if values:
                    data_by_algorithm[alg] = values

            if not data_by_algorithm:
                continue

            title = (f"pattern={int(pattern_index)} - "
                     f"Bandwidth={int(bandwidth)}, LLM computation={int(llm_comp)}")
            filename = os.path.join(
                base_dir,
                f"pattern{int(pattern_index)}_bw{int(bandwidth)}_comp{int(llm_comp)}.png"
            )

            plot_dual_axis(x_values, data_by_algorithm,
                           x_label="LLM-User distance indicator",
                           title=title,
                           filename=filename)
            print(f"已生成距离指标图: {filename}")


def main() -> None:
    plot_cost_vs_distance_from_excel()


if __name__ == "__main__":
    main()
