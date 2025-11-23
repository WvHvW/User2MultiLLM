"""分析SPFA在大规模网络上的性能瓶颈"""

# 理论分析：1000节点网络
V = 1002  # 1000物理节点 + 超源 + 超汇
avg_degree = 10
E_physical = V * avg_degree / 2  # 无向图
E_total = E_physical * 2  # 加上反向边
E_super = V  # 超源/超汇连接

print("=" * 70)
print("SPFA性能分析：1000节点网络")
print("=" * 70)

print(f"\n网络规模：")
print(f"  节点数V: {V}")
print(f"  物理边数: {int(E_physical)}")
print(f"  总边数E (含反向边): {int(E_total + E_super)}")

print(f"\nSPFA复杂度：")
print(f"  最好情况: O(E) = {int(E_total + E_super):,}")
print(f"  平均情况: O(VE) = {int(V * (E_total + E_super)):,}")
print(f"  最坏情况: 指数级（有负环或大量负权边时）")

print(f"\nSSP总复杂度（k=1）：")
total_flow_estimate = 1000 * 1000  # 1000用户，每个1000带宽
iterations = total_flow_estimate / 1  # k=1
avg_ops_per_spfa = V * (E_total + E_super)
total_ops = iterations * avg_ops_per_spfa

print(f"  估计流量: {total_flow_estimate:,}")
print(f"  迭代次数: {int(iterations):,}")
print(f"  单次SPFA: {int(avg_ops_per_spfa):,} 操作")
print(f"  总操作数: {total_ops:.2e} ← 天文数字！")

print(f"\nSSP总复杂度（k=100）：")
iterations_k100 = total_flow_estimate / 100
total_ops_k100 = iterations_k100 * avg_ops_per_spfa
print(f"  迭代次数: {int(iterations_k100):,}")
print(f"  总操作数: {total_ops_k100:.2e}")

print(f"\n内存消耗估算：")
dict_size = 8 * 8 * V  # 8个dict，每个V个entry，每个entry约8字节
print(f"  每次SPFA: {dict_size / 1024 / 1024:.2f} MB")
print(f"  峰值内存: 可能更高（取决于队列大小）")

print("\n" + "=" * 70)
print("结论：")
print("=" * 70)
print("1. k=1时操作数是天文数字，必然被killed")
print("2. k=100时可能可行，但仍然很慢")
print("3. 需要优化策略：")
print("   - 限制最小k值（如k >= 10）")
print("   - 添加SPFA超时检测")
print("   - 考虑使用更高效的算法（如Dijkstra + Capacity Scaling）")
