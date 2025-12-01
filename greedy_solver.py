# greedy_solver.py
import numpy as np
import time
from config import Config


class GreedySolver:
    """
    贪婪求解器：用于动态分片的局部最优调整
    核心逻辑：
    1. 遍历每个节点，尝试迁移到其他所有分片
    2. 计算迁移后系统总效用的变化（ΔU）
    3. 只保留“ΔU>0（效用提升）+ 分片规模合规”的迁移
    4. 重复迭代，直到没有可提升的迁移或达到最大步数
    特点：速度快、逻辑简单，但容易陷入局部最优
    """

    def __init__(self, sharding_algorithm):
        """
        初始化求解器
        :param sharding_algorithm: 分片算法实例（继承BaseShardingAlgorithm的类，如DynamicSharding）
        """
        # 从分片算法实例中获取核心数据（避免重复传递）
        self.sharding_algo = sharding_algorithm  # 分片算法实例（含指标计算方法）
        self.num_nodes = sharding_algorithm.num_nodes
        self.num_shards = sharding_algorithm.num_shards
        self.shards = sharding_algorithm.shards  # 初始分片结果
        self.shard_id_of_node = sharding_algorithm.shard_id_of_node  # 节点→分片映射

        # 约束参数（从Config读取）
        self.max_steps = Config.GREEDY_STEPS  # 最大迭代步数
        self.shard_min_size = Config.SHARD_SIZE_MIN
        self.shard_max_size = Config.SHARD_SIZE_MAX

    def _calculate_utility_gain(self, node, current_shard, target_shard):
        """
        计算“节点node从current_shard迁移到target_shard”的效用增益（ΔU）
        :return: gain = 迁移后总效用 - 迁移前总效用（>0表示增益）
        """
        # 1. 保存当前分片状态（用于后续恢复）
        current_shard_nodes = self.shards[current_shard].copy()
        target_shard_nodes = self.shards[target_shard].copy()

        # 2. 模拟迁移（临时修改分片状态）
        self.shards[current_shard].remove(node)  # 从当前分片移除
        self.shards[target_shard].append(node)  # 加入目标分片
        self.shard_id_of_node[node] = target_shard  # 更新映射
        self.sharding_algo.reset_metrics_cache()  # 重置指标缓存（重新计算效用）

        # 3. 计算迁移后的总效用
        new_utility = self.sharding_algo.calculate_system_total_utility()

        # 4. 恢复原始分片状态（避免影响后续计算）
        self.shards[current_shard] = current_shard_nodes
        self.shards[target_shard] = target_shard_nodes
        self.shard_id_of_node[node] = current_shard
        self.sharding_algo.reset_metrics_cache()

        # 5. 计算效用增益（新效用 - 原效用）
        original_utility = self.sharding_algo.calculate_system_total_utility()
        gain = new_utility - original_utility
        return gain

    def _is_migration_valid(self, node, current_shard, target_shard):
        """
        验证迁移是否合规：
        1. 目标分片迁移后不超过最大规模
        2. 当前分片迁移后不低于最小规模
        3. 目标分片≠当前分片
        """
        if target_shard == current_shard:
            return False  # 不能迁移到自身所在分片
        # 迁移后当前分片的节点数
        current_size_after = len(self.shards[current_shard]) - 1
        # 迁移后目标分片的节点数
        target_size_after = len(self.shards[target_shard]) + 1
        # 同时满足规模约束
        return (current_size_after >= self.shard_min_size) and (target_size_after <= self.shard_max_size)

    def solve(self,verbose: bool = True):
        """
        核心求解逻辑：迭代寻找最优迁移，直到收敛
        :return: 优化后的分片结果、迭代次数、最终系统总效用
        """
        if verbose:
            print(f"\n=== 贪婪求解器开始优化（最大{self.max_steps}步）===")
        start_time = time.time()  # 补充1：统计总耗时（用于效率对比）
        iteration = 0  # 迭代次数
        best_utility = self.sharding_algo.calculate_system_total_utility()  # 初始最优效用
        no_improve_count = 0  # 记录“无提升”的迭代次数（用于提前收敛）

        while iteration < self.max_steps:
            iteration += 1
            max_gain = -np.inf  # 最大效用增益（初始为负无穷）
            best_migration = None  # 最优迁移方案：(node, current_shard, target_shard)

            # 1. 遍历所有节点，寻找最优迁移
            for node in range(self.num_nodes):
                current_shard = self.shard_id_of_node[node]
                # 尝试迁移到所有其他分片
                for target_shard in range(self.num_shards):
                    # 验证迁移是否合规
                    if not self._is_migration_valid(node, current_shard, target_shard):
                        continue
                    # 计算迁移的效用增益
                    gain = self._calculate_utility_gain(node, current_shard, target_shard)
                    # 更新最大增益和最优迁移
                    if gain > max_gain:
                        max_gain = gain
                        best_migration = (node, current_shard, target_shard)

            # 2. 执行最优迁移（如果有增益）
            if best_migration is not None and max_gain > 0:
                node, current_shard, target_shard = best_migration
                # 执行实际迁移
                self.shards[current_shard].remove(node)
                self.shards[target_shard].append(node)
                self.shard_id_of_node[node] = target_shard
                # 更新最优效用
                best_utility = self.sharding_algo.calculate_system_total_utility()
                if verbose:
                    print(
                        f"迭代{iteration}：迁移节点{node}（从分片{current_shard}→{target_shard}），效用增益{max_gain:.4f}，当前最优效用{best_utility:.4f}")
                no_improve_count = 0  # 重置无提升计数
            else:
                # 无有效迁移或无增益，计数+1
                no_improve_count += 1
                if verbose:
                    print(f"迭代{iteration}：无有效迁移或效用无提升（已连续{no_improve_count}次）")
                # 连续3次无提升，提前收敛（可选，加快速度）
                if no_improve_count >= 3:
                    if verbose:
                        print(f"提前收敛：连续3次无效用提升")
                    break

        # 3. 优化完成，更新分片算法实例的分片结果
        self.sharding_algo.shards = self.shards
        self.sharding_algo._update_shard_mapping()
        self.sharding_algo.reset_metrics_cache()

        total_time = time.time() - start_time  # 补充2：计算总耗时
        if verbose:
            print(f"\n=== 贪婪求解器优化结束 ===")
            print(f"总迭代次数：{iteration}")
            print(f"总耗时：{total_time:.2f}秒")  # 打印耗时
            print(f"最终系统总效用：{best_utility:.4f}")
        return self.shards, iteration, total_time, best_utility


# greedy_solver.py 中的测试代码（修改后）
if __name__ == "__main__":
    # 1. 生成测试数据（50节点）
    from data_generator import DataGenerator

    generator = DataGenerator(num_nodes=50)
    test_data = generator.get_data()

    # 2. 用随机分片作为初始分片（贪婪求解器基于初始分片优化）
    from random_sharding import RandomSharding

    random_sharding = RandomSharding(data=test_data, num_shards=5)
    random_sharding.sharding()  # 初始随机分片

    # --------------------------
    # 关键修改：提前保存优化前的所有指标（避免被就地修改覆盖）
    # --------------------------
    original_utility = random_sharding.calculate_system_total_utility()
    original_intra_rate = random_sharding.calculate_intra_shard_rate()
    original_loss = random_sharding.calculate_transmission_loss()

    # 3. 初始化贪婪求解器（传入随机分片实例）
    greedy_solver = GreedySolver(sharding_algorithm=random_sharding)

    # 4. 执行优化（会就地修改random_sharding的分片结果）
    optimized_shards, iterations, total_time, final_utility = greedy_solver.solve()

    # 5. 对比优化前后的核心指标（使用提前保存的初始值）
    print("\n=== 优化前后指标对比 ===")
    print(f"优化前系统总效用：{original_utility:.4f}")
    print(f"优化后系统总效用：{final_utility:.4f}")
    print(f"优化前片内交易率：{original_intra_rate:.4f}")
    print(f"优化后片内交易率：{random_sharding.calculate_intra_shard_rate():.4f}")
    print(f"优化前传输损耗：{original_loss:.2f}")
    print(f"优化后传输损耗：{random_sharding.calculate_transmission_loss():.2f}")
