# dynamic_sharding.py
import numpy as np
from config import Config
from base_algorithm import BaseShardingAlgorithm
from static_clustering import StaticClusteringSharding
from random_sharding import RandomSharding
from sa_solver import SASolver
from exchange_matcher import ExchangeMatcher


def percent_change(new, old):
    """安全计算百分比变化：如果 old ~ 0 返回 None"""
    if abs(old) < getattr(Config, "EPS", 1e-12):
        return None
    return (new - old) / old * 100.0


class DynamicSharding(BaseShardingAlgorithm):
    """
    基于交换匹配的动态分片算法（论文第四章实现）
    初始分片（聚类/随机） -> 模拟退火求解（ExchangeMatcher）-> 输出最终分片
    """

    def __init__(self, data, num_shards, init_sharding_type="clustering", seed=None):
        super().__init__(data, num_shards)

        self.seed = Config.RANDOM_SEED if seed is None else seed
        self.init_sharding_type = init_sharding_type
        self.initial_sharding = None
        self.sa_solver = None

    def _get_initial_sharding(self):
        """获取初始分片（μ₀），并确保随机性可复现（设置 np 随机种子）"""
        # 保证初始分片可复现（最小改动）
        np.random.seed(self.seed)

        if self.init_sharding_type == "clustering":
            self.initial_sharding = StaticClusteringSharding(data=self.data, num_shards=self.num_shards)
        elif self.init_sharding_type == "random":
            self.initial_sharding = RandomSharding(data=self.data, num_shards=self.num_shards)
        else:
            raise ValueError("init_sharding_type只能是'clustering'或'random'")

        initial_shards = self.initial_sharding.sharding()
        if initial_shards is None:
            raise RuntimeError("初始分片返回 None，请检查 initial_sharding.sharding() 实现")
        return initial_shards

    def sharding(self, verbose: bool = True):
        """
        执行动态分片流程
        :param verbose: 是否在内部打印求解器日志（会传递给 SASolver）
        """
        print(f"\n=== 动态分片算法开始执行 ===")
        print(f"初始分片类型：{self.init_sharding_type}，分片数量：{self.num_shards}")
        print(f"优化目标：最大化系统总效用（含分片效用+节点效用-公平性损失）")

        # 1. 初始分片
        self.shards = self._get_initial_sharding()
        self._update_shard_mapping()
        print(f"初始分片完成，各分片节点数：{[len(nodes) for nodes in self.shards.values()]}")

        # 2. 初始化 SA（注意：SASolver 要求 self.shards 和 self.shard_id_of_node 已就绪）
        self.sa_solver = SASolver(sharding_algorithm=self, seed=self.seed)

        # 3. 执行模拟退火优化（传递 verbose）
        final_shards, total_iters, final_utility = self.sa_solver.solve(verbose=verbose)

        # 4. 更新最终分片结果并重置缓存
        self.shards = final_shards
        self._update_shard_mapping()
        self.reset_metrics_cache()

        # 5. 打印最终结果
        print(f"\n=== 动态分片算法执行结束 ===")
        print(f"总迭代次数：{total_iters}")
        print(f"最终系统总效用（归一化）：{final_utility:.4f}")
        print(f"最终分片节点数：{[len(nodes) for nodes in self.shards.values()]}")

        return self.shards

    @property
    def data(self):
        """适配初始分片算法需要的数据格式"""
        return {
            "node_ids": self.node_ids,
            "trust_values": self.trust_values,
            "energy_matrix": self.energy_matrix,
            "electrical_distance": self.electrical_distance
        }


# 测试代码：验证动态分片算法是否正常运行，对比初始/优化后指标
if __name__ == "__main__":
    # 1. 生成测试数据（注意：DataGenerator 不接收 seed 参数）
    from data_generator import DataGenerator

    generator = DataGenerator(num_nodes=50)  # 不传 seed 参数（DataGenerator __init__ 无此参数）
    test_data = generator.get_data()

    # 2. 初始化并执行动态分片
    dynamic_sharding = DynamicSharding(
        data=test_data,
        num_shards=5,
        init_sharding_type="clustering",
        seed=Config.RANDOM_SEED
    )

    dynamic_sharding.sharding(verbose=True)

    # 3. 比较初始 vs 优化后指标（安全打印百分比，已修正参数顺序）
    initial_intra_rate = dynamic_sharding.initial_sharding.calculate_intra_shard_rate()
    initial_loss = dynamic_sharding.initial_sharding.calculate_transmission_loss()
    initial_utility = dynamic_sharding.initial_sharding.calculate_system_total_utility()

    final_intra_rate = dynamic_sharding.calculate_intra_shard_rate()
    final_loss = dynamic_sharding.calculate_transmission_loss()
    final_utility = dynamic_sharding.calculate_system_total_utility()
    final_cross_loss = dynamic_sharding.metrics_cache.get('transmission_loss_cross', 0.0)

    # 片内交易率提升百分比（原逻辑正确，保留）
    pct = percent_change(final_intra_rate, initial_intra_rate)
    if pct is None:
        intra_str = "N/A (初始为0)"
    else:
        intra_str = f"{pct:.2f}%"
    print("\n" + "=" * 60)
    print("动态分片算法 - 初始vs优化后指标对比：")
    print(f"初始片内交易率：{initial_intra_rate:.4f} → 优化后：{final_intra_rate:.4f}（提升：{intra_str}）")

    # 【修正1】总传输损耗降低百分比（参数顺序：new=final_loss，old=initial_loss，取绝对值）
    loss_pct = percent_change(final_loss, initial_loss)
    if loss_pct is None:
        loss_str = "N/A"
    else:
        loss_str = f"{abs(loss_pct):.2f}%"  # 取绝对值，确保显示“降低x%”
    print(f"初始总传输损耗：{initial_loss:.2f} → 优化后：{final_loss:.2f}（降低：{loss_str}）")

    # 【修正2】跨片传输损耗降低百分比（参数顺序+取绝对值）
    cross_loss_initial = dynamic_sharding.initial_sharding.metrics_cache.get('transmission_loss_cross', 0.0)
    cross_loss_pct = percent_change(final_cross_loss, cross_loss_initial)
    if cross_loss_pct is None:
        cross_loss_str = "N/A"
    else:
        cross_loss_str = f"{abs(cross_loss_pct):.2f}%"  # 取绝对值，显示“降低x%”
    print(f"初始跨片传输损耗：{cross_loss_initial:.2f} → 优化后：{final_cross_loss:.2f}（降低：{cross_loss_str}）")

    # 系统总效用提升百分比（原逻辑正确，保留）
    util_pct = percent_change(final_utility, initial_utility)
    if util_pct is None:
        util_str = "N/A"
    else:
        util_str = f"{util_pct:.2f}%"
    print(f"初始系统总效用：{initial_utility:.4f} → 优化后：{final_utility:.4f}（提升：{util_str}）")
    print("=" * 60)
