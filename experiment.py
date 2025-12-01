import os
import csv
import time
import numpy as np
from config import Config
from data_generator import DataGenerator
from random_sharding import RandomSharding
from static_clustering import StaticClusteringSharding
from dynamic_sharding import DynamicSharding

# 实验结果保存路径（自动创建文件夹）
RESULT_DIR = Config.PLOT_SAVE_PATH
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_CSV = os.path.join(RESULT_DIR, "experiment_results.csv")

# 要对比的算法列表（覆盖你论文4.4节所有对比需求）
ALGORITHMS = [
    {"name": "随机分片", "type": "random"},
    {"name": "静态聚类分片", "type": "static_clustering"},
    {"name": "动态分片（SA求解）", "type": "dynamic_sa"},
    {"name": "动态分片（贪婪求解）", "type": "dynamic_greedy"}
]

def run_single_experiment(num_nodes, num_shards, seed, initial_shard_type="random"):
    """
    运行单次实验（一组规模+一个随机种子）
    :param num_nodes: 节点数
    :param num_shards: 分片数
    :param seed: 随机种子（保证可复现）
    :param initial_shard_type: 统一初始分片类型（所有动态算法用同一个初始分片）
    :return: 单次实验的所有算法结果（列表）
    """
    # 1. 生成统一的实验数据（所有算法用同一组数据）
    np.random.seed(seed)
    generator = DataGenerator(num_nodes=num_nodes)
    data = generator.get_data()

    # 2. 生成统一的初始分片（所有动态算法基于此优化，保证对比公平）
    initial_sharding = None
    if initial_shard_type == "random":
        initial_sharding = RandomSharding(data=data, num_shards=num_shards)
    elif initial_shard_type == "clustering":
        initial_sharding = StaticClusteringSharding(data=data, num_shards=num_shards)
    else:
        raise ValueError("initial_shard_type must be 'random' or 'clustering'")

    initial_sharding.sharding()
    # 保存初始分片结果（供动态算法复用）
    initial_shards = {k: list(v) for k, v in initial_sharding.shards.items()}

    experiment_results = []
    start_time_total = time.time()

    # 3. 逐个运行所有算法
    for algo in ALGORITHMS:
        algo_name = algo["name"]
        algo_type = algo["type"]
        print(f"\n=== 运行实验：{num_nodes}节点×{num_shards}分片×种子{seed} → {algo_name} ===")

        # 初始化算法实例
        sharding_algo = None
        run_time = None
        iterations = 0

        if algo_type == "random":
            # 随机分片（基准）
            sharding_algo = RandomSharding(data=data, num_shards=num_shards)
            start_time = time.time()
            sharding_algo.sharding()
            run_time = time.time() - start_time
            iterations = 0  # 随机分片无迭代

        elif algo_type == "static_clustering":
            # 静态聚类分片（基准）
            sharding_algo = StaticClusteringSharding(data=data, num_shards=num_shards)
            start_time = time.time()
            sharding_algo.sharding()
            run_time = time.time() - start_time
            iterations = 0  # 静态聚类无迭代

        elif algo_type == "dynamic_sa":
            # 动态分片（SA求解器）- 复用统一初始分片（修正版）
            sharding_algo = DynamicSharding(
                data=data, num_shards=num_shards,
                init_sharding_type=initial_shard_type, seed=seed
            )
            # 强制设置统一初始分片（避免 DynamicSharding.sharding() 覆盖）
            sharding_algo.shards = {k: list(v) for k, v in initial_shards.items()}
            sharding_algo._update_shard_mapping()
            sharding_algo.reset_metrics_cache()

            # 直接构造并调用 SASolver（不要再调用 sharding_algo.sharding()，因为那会重新生成初始分片）
            from sa_solver import SASolver
            sa_solver = SASolver(sharding_algorithm=sharding_algo, seed=seed)
            # 使用求解器返回的 total_time（内部计时）作为运行时间，避免两次计时
            best_shards, total_iters, total_time_sa, final_utility_sa = sa_solver.solve(verbose=False)
            run_time = total_time_sa
            iterations = total_iters

            # 把最优结果写回 sharding_algo（方便后续统一指标读取）
            sharding_algo.shards = best_shards
            sharding_algo._update_shard_mapping()
            sharding_algo.reset_metrics_cache()

        elif algo_type == "dynamic_greedy":
            # 动态分片（贪婪求解器）- 复用统一初始分片
            sharding_algo = DynamicSharding(
                data=data, num_shards=num_shards,
                init_sharding_type=initial_shard_type, seed=seed
            )
            # 强制设置统一初始分片
            sharding_algo.shards = {k: list(v) for k, v in initial_shards.items()}
            sharding_algo._update_shard_mapping()
            sharding_algo.reset_metrics_cache()

            # 替换为贪婪求解器
            from greedy_solver import GreedySolver
            greedy_solver = GreedySolver(sharding_algorithm=sharding_algo)
            # 运行优化（GreedySolver.solve 已返回 total_time）
            final_shards, iter_num, total_time_greedy, _ = greedy_solver.solve(verbose=False)
            run_time = total_time_greedy
            iterations = iter_num

            # 更新分片结果
            sharding_algo.shards = final_shards
            sharding_algo._update_shard_mapping()
            sharding_algo.reset_metrics_cache()

        else:
            raise ValueError(f"未知算法类型：{algo_type}")

        # 4. 计算所有核心指标（确保缓存已刷新）
        intra_rate = sharding_algo.calculate_intra_shard_rate()
        total_loss = sharding_algo.calculate_transmission_loss()
        cross_loss = sharding_algo.metrics_cache.get("transmission_loss_cross", 0.0)
        total_utility = sharding_algo.calculate_system_total_utility()
        gini_shard = sharding_algo.calculate_gini_coefficient(level="shard")
        gini_node = sharding_algo.calculate_gini_coefficient(level="node")

        # 5. 保存单次算法结果
        result = {
            "节点数": num_nodes,
            "分片数": num_shards,
            "随机种子": seed,
            "算法名称": algo_name,
            "片内交易率": round(intra_rate, 6),
            "总传输损耗": round(total_loss, 2),
            "跨片传输损耗": round(cross_loss, 2),
            "系统总效用": round(total_utility, 6),
            "分片基尼系数": round(gini_shard, 6),
            "节点基尼系数": round(gini_node, 6),
            "运行时间(秒)": round(run_time if run_time is not None else 0.0, 4),
            "迭代次数": iterations
        }
        experiment_results.append(result)
        print(f"✅ {algo_name} 完成：效用={total_utility:.4f}，耗时={result['运行时间(秒)']:.2f}秒")

    total_time = time.time() - start_time_total
    print(f"\n📊 单次实验完成（{num_nodes}节点×{num_shards}分片×种子{seed}），总耗时：{total_time:.2f}秒")
    return experiment_results

def main():
    """
    主实验流程：遍历所有规模+重复次数，批量运行实验
    """
    print("="*80)
    print("🎯 微电网动态分片算法对比实验（论文4.4节）")
    print(f"实验规模组合：{Config.SCALES}")
    print(f"每种规模重复次数：{Config.EXPERIMENT_TIMES}")
    print(f"对比算法：{[algo['name'] for algo in ALGORITHMS]}")
    print(f"结果保存路径：{RESULT_CSV}")
    print("="*80)

    # 初始化CSV文件（写入表头）
    fieldnames = [
        "节点数", "分片数", "随机种子", "算法名称",
        "片内交易率", "总传输损耗", "跨片传输损耗",
        "系统总效用", "分片基尼系数", "节点基尼系数",
        "运行时间(秒)", "迭代次数"
    ]
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # 遍历所有实验组合（规模×重复次数）
    total_experiments = len(Config.SCALES) * Config.EXPERIMENT_TIMES
    current_experiment = 0
    for (num_nodes, num_shards) in Config.SCALES:
        for seed_offset in range(Config.EXPERIMENT_TIMES):
            current_experiment += 1
            seed = Config.RANDOM_SEED + seed_offset  # 不同种子保证随机性
            print(f"\n" + "="*80)
            print(f"📌 正在运行实验 {current_experiment}/{total_experiments}：{num_nodes}节点×{num_shards}分片×第{seed_offset+1}次重复")
            print("="*80)

            # 运行单次实验（统一初始分片为随机分片，保证对比公平）
            single_results = run_single_experiment(
                num_nodes=num_nodes, num_shards=num_shards,
                seed=seed, initial_shard_type="random"
            )

            # 将单次实验结果写入CSV
            with open(RESULT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(single_results)

    print("\n" + "="*80)
    print("🎉 所有实验运行完成！")
    print(f"📁 实验数据已保存至：{RESULT_CSV}")
    print("👉 下一步：运行 plot_results.py 生成论文所需图表（收敛曲线、分组柱状图等）")
    print("="*80)

if __name__ == "__main__":
    main()
