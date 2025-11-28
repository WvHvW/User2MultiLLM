import os
from typing import Dict, List, Tuple

import pandas as pd

from analyze_bandwidth_llm_combined import (
    BANDWIDTH_VALUES,
    COMPUTATION_VALUES,
    plot_dual_axis,
    plot_user_distance_flow_ratio_bars,
    _USER_PATTERN_DIR,
)
from plot_userpattern_distance_cost import plot_cost_vs_distance_from_excel


ALG_BW_LLM = ['no-split', '1-split', '100-split-augment', 'bottleneck-augment']
ALG_USER = ['no-split', '1-split', '1-split-augment', 'bottleneck-augment']


def _load_all() -> pd.DataFrame:
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


def _plot_bandwidth_view(df_all: pd.DataFrame) -> None:
    if "view" in df_all.columns:
        df_bw = df_all[df_all["view"] == "bandwidth"].copy()
    else:
        df_bw = df_all.copy()

    if df_bw.empty:
        return

    pattern_indices = sorted(df_bw["pattern_index"].unique())

    for pattern_index in pattern_indices:
        df_p = df_bw[df_bw["pattern_index"] == pattern_index]
        if df_p.empty:
            continue

        pattern_dir = os.path.join(_USER_PATTERN_DIR,
                                   f"pattern{int(pattern_index)}")
        for distribution in sorted(df_p["distribution"].unique()):
            df_d = df_p[df_p["distribution"] == distribution]
            if df_d.empty:
                continue

            bandwidth_base_dir = os.path.join(pattern_dir, "bandwidth",
                                              distribution)
            os.makedirs(bandwidth_base_dir, exist_ok=True)

            for bandwidth in sorted(df_d["bandwidth"].unique()):
                df_b = df_d[df_d["bandwidth"] == bandwidth]
                if df_b.empty:
                    continue

                x_values = sorted(df_b["llm_computation"].unique())
                data_by_algorithm: Dict[str, List[Tuple[float, float,
                                                        float]]] = {}

                for alg in ALG_BW_LLM:
                    triples: List[Tuple[float, float, float]] = []
                    for comp in x_values:
                        rows = df_b[(df_b["llm_computation"] == comp)
                                    & (df_b["algorithm"] == alg)]
                        if rows.empty:
                            continue
                        row = rows.iloc[0]
                        opt = float(row.get("optimization", 0.0))
                        sr = float(row.get("service_rate", 0.0))
                        cost = float(row.get("total_cost", 0.0))
                        triples.append((opt, sr, cost))
                    if triples:
                        data_by_algorithm[alg] = triples

                if not data_by_algorithm:
                    continue

                filename = os.path.join(bandwidth_base_dir,
                                        f"bw{int(bandwidth)}.png")
                title = f"{distribution} - Bandwidth={int(bandwidth)}"

                plot_dual_axis(x_values, data_by_algorithm, "LLM Computation",
                               title, filename)


def _plot_llm_view(df_all: pd.DataFrame) -> None:
    if "view" in df_all.columns:
        df_llm = df_all[df_all["view"] == "bandwidth"].copy()
    else:
        df_llm = df_all.copy()

    if df_llm.empty:
        return

    pattern_indices = sorted(df_llm["pattern_index"].unique())

    for pattern_index in pattern_indices:
        df_p = df_llm[df_llm["pattern_index"] == pattern_index]
        if df_p.empty:
            continue

        pattern_dir = os.path.join(_USER_PATTERN_DIR,
                                   f"pattern{int(pattern_index)}")
        for distribution in sorted(df_p["distribution"].unique()):
            df_d = df_p[df_p["distribution"] == distribution]
            if df_d.empty:
                continue

            llm_base_dir = os.path.join(pattern_dir, "llm", distribution)
            os.makedirs(llm_base_dir, exist_ok=True)

            for comp in sorted(df_d["llm_computation"].unique()):
                df_c = df_d[df_d["llm_computation"] == comp]
                if df_c.empty:
                    continue

                x_values = sorted(df_c["bandwidth"].unique())
                data_by_algorithm: Dict[str, List[Tuple[float, float,
                                                        float]]] = {}

                for alg in ALG_BW_LLM:
                    triples: List[Tuple[float, float, float]] = []
                    for bw in x_values:
                        rows = df_c[(df_c["bandwidth"] == bw)
                                    & (df_c["algorithm"] == alg)]
                        if rows.empty:
                            continue
                        row = rows.iloc[0]
                        opt = float(row.get("optimization", 0.0))
                        sr = float(row.get("service_rate", 0.0))
                        cost = float(row.get("total_cost", 0.0))
                        triples.append((opt, sr, cost))
                    if triples:
                        data_by_algorithm[alg] = triples

                if not data_by_algorithm:
                    continue

                filename = os.path.join(llm_base_dir,
                                        f"comp{int(comp)}.png")
                title = f"{distribution} - LLM Computation={int(comp)}"

                plot_dual_axis(x_values, data_by_algorithm, "Bandwidth",
                               title, filename)


def _plot_user_level_view(df_all: pd.DataFrame) -> None:
    if "view" in df_all.columns:
        df_user = df_all[df_all["view"] == "user"].copy()
    else:
        df_user = df_all.copy()

    if df_user.empty:
        return

    pattern_indices = sorted(df_user["pattern_index"].unique())

    for pattern_index in pattern_indices:
        df_p = df_user[df_user["pattern_index"] == pattern_index]
        if df_p.empty:
            continue

        pattern_dir = os.path.join(_USER_PATTERN_DIR,
                                   f"pattern{int(pattern_index)}")
        for distribution in sorted(df_p["distribution"].unique()):
            df_d = df_p[df_p["distribution"] == distribution]
            if df_d.empty:
                continue

            user_level_dir = os.path.join(pattern_dir, "user_level",
                                          distribution)
            os.makedirs(user_level_dir, exist_ok=True)

            for (bandwidth, llm_comp), group in df_d.groupby(
                    ["bandwidth", "llm_computation"]):
                if group.empty:
                    continue

                # 构造用户标签：按 user_index 升序，并带带宽
                user_indices = sorted(group["user_index"].unique())
                user_labels: List[str] = []
                for idx in user_indices:
                    sub = group[group["user_index"] == idx]
                    if sub.empty:
                        continue
                    bw_val = int(sub.iloc[0]["user_bandwidth"])
                    user_labels.append(f"{int(idx)}({bw_val})")

                ratios_by_algorithm: Dict[str, List[float]] = {}
                service_rates_by_algorithm: Dict[str, float] = {}
                served_flows_by_algorithm: Dict[str, List[float]] = {}

                for alg in ALG_USER:
                    g_alg = group[group["algorithm"] == alg]
                    if g_alg.empty:
                        continue

                    ratios: List[float] = []
                    flows: List[float] = []
                    for idx in user_indices:
                        row = g_alg[g_alg["user_index"] == idx]
                        if row.empty:
                            ratios.append(0.0)
                            flows.append(0.0)
                        else:
                            r = float(row.iloc[0]["distance_per_unit_flow"])
                            f = float(row.iloc[0]["served_flow"])
                            ratios.append(r)
                            flows.append(f)

                    ratios_by_algorithm[alg] = ratios
                    served_flows_by_algorithm[alg] = flows
                    service_rates_by_algorithm[alg] = float(
                        g_alg.iloc[0]["service_rate"])

                if not ratios_by_algorithm:
                    continue

                title = (f"Bandwidth={int(bandwidth)}, "
                         f"LLM computation={int(llm_comp)}")
                filename = os.path.join(
                    user_level_dir,
                    f"bw{int(bandwidth)}_comp{int(llm_comp)}.png")

                plot_user_distance_flow_ratio_bars(
                    user_labels,
                    ratios_by_algorithm,
                    service_rates_by_algorithm,
                    title,
                    filename,
                )


def plot_all_from_userpattern_all() -> None:
    """
    仅依赖 userpattern_all.xlsx，重新生成：
    - 带宽视角双轴图（bandwidth）
    - LLM 容量视角双轴图（llm）
    - user-level 柱状图
    - 距离指标视角图（调用 plot_cost_vs_distance_from_excel）
    """
    df_all = _load_all()

    _plot_bandwidth_view(df_all)
    _plot_llm_view(df_all)
    _plot_user_level_view(df_all)

    # 距离指标视角复用已有实现
    plot_cost_vs_distance_from_excel()


def main() -> None:
    plot_all_from_userpattern_all()


if __name__ == "__main__":
    main()

