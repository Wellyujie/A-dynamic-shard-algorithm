# sa_solver.py（修正版）
import time
import numpy as np
from config import Config
from exchange_matcher import ExchangeMatcher  # 假定为我们之前改进的版本


class SASolver:
    """
    模拟退火求解器（修正版，兼容 ExchangeMatcher 的逐步应用批量匹配策略）
    - 与 ExchangeMatcher 协作：match_batch_nodes() 会逐步应用 moves（已在匹配时修改状态）
    - solve() 在调用 match_batch_nodes() 之前对状态做快照；若拒绝则完整恢复
    """

    def __init__(self, sharding_algorithm, seed=None):
        if getattr(sharding_algorithm, "shards", None) is None or getattr(
                sharding_algorithm, "shard_id_of_node", None) is None:
            raise ValueError("sharding_algorithm 必须先执行 sharding()，确保 shards 与 shard_id_of_node 已初始化")

        self.sharding_algo = sharding_algorithm
        self.num_nodes = sharding_algorithm.num_nodes
        self.num_shards = sharding_algorithm.num_shards

        # 绑定分片状态引用（注意快照/恢复需要同步所有持有引用的对象）
        self.shards = sharding_algorithm.shards
        self.shard_id_of_node = np.array(sharding_algorithm.shard_id_of_node, copy=False)

        # ExchangeMatcher（内部使用独立 RNG，建议传入 seed 保持一致）
        self.exchange_matcher = ExchangeMatcher(sharding_algorithm)

        # SA 参数（可在 Config 中设置）
        self.init_temp = float(getattr(Config, "SA_TEMPERATURE", 100.0))
        self.cool_rate = float(getattr(Config, "SA_COOL_RATE", 0.95))
        self.min_temp = float(getattr(Config, "SA_MIN_TEMPERATURE", 1e-3))
        self.max_iter_per_temp = int(getattr(Config, "SA_MAX_ITER_PER_TEMP", 20))

        # 可选最大总迭代防护
        self.max_total_iters = int(getattr(Config, "SA_MAX_TOTAL_ITERS", 100000))

        # 数值容差
        self.EPS = 1e-9

        # RNG（与 ExchangeMatcher 的 seed 保持一致）
        self.seed = Config.RANDOM_SEED if seed is None else seed
        self.rng = np.random.RandomState(self.seed)

    def _accept_probability(self, gain, temperature):
        """劣解接受概率：若 gain>0 则 1；否则 exp(gain/temperature) （temperature 越高越易接受）"""
        if gain > self.EPS:
            return 1.0
        if temperature <= self.EPS:
            return 0.0
        # gain 可能为负，exp(neg/temperature) ∈ (0,1)
        return float(np.exp(gain / temperature))

    def solve(self, verbose: bool = True):
        """
        执行模拟退火优化（修正版）
        返回： (best_shards, total_iterations, best_utility)
        """
        start_time = time.time()  # 新增1：统计总耗时
        if verbose:
            print("\n=== 模拟退火求解器开始优化 ===")
            print(f"初始温度：{self.init_temp:.2f}，降温速率：{self.cool_rate:.4f}，最低温度：{self.min_temp:.4f}")
            print(f"K（批量交换最大节点数）：{self.exchange_matcher.K}，每温度迭代：{self.max_iter_per_temp}")

        temperature = self.init_temp
        total_iterations = 0
        # 基准效用（当前状态）
        base_utility = float(self.sharding_algo.calculate_system_total_utility())
        best_utility = base_utility
        best_shards = {k: list(v) for k, v in self.shards.items()}

        # 恢复函数：把快照状态恢复到 sharding_algo 与 exchange_matcher 等所有相关引用
        def _restore_from_snapshot(shards_snapshot, shard_id_snapshot):
            # 恢复 SASolver 视图
            self.shards = {k: list(v) for k, v in shards_snapshot.items()}
            self.shard_id_of_node = shard_id_snapshot.copy()
            # 恢复 parent sharding_algo
            self.sharding_algo.shards = self.shards
            # 更新映射并重置缓存（父类方法）
            self.sharding_algo._update_shard_mapping()
            self.sharding_algo.reset_metrics_cache()
            # 同步 SASolver 本地 shard id array，确保所有视图一致
            self.shard_id_of_node = self.sharding_algo.shard_id_of_node
            # 同步 exchange_matcher 的引用和内部视图
            self.exchange_matcher.shards = self.sharding_algo.shards
            self.exchange_matcher.shard_id_of_node = self.sharding_algo.shard_id_of_node

        # 主循环：温度衰减直至 min_temp 或达到总迭代上限
        while temperature > self.min_temp and total_iterations < self.max_total_iters:
            for _ in range(self.max_iter_per_temp):
                total_iterations += 1

                # 在调用 match_batch_nodes() 之前做完整快照（因为 matcher 会在匹配时直接应用 moves）
                shards_snapshot = {k: list(v) for k, v in self.shards.items()}
                shard_id_snapshot = self.shard_id_of_node.copy()

                # 生成并应用moves
                try:
                    batch_moves = self.exchange_matcher.match_batch_nodes(base_utility=base_utility)
                except TypeError:
                    batch_moves = self.exchange_matcher.match_batch_nodes()

                if not batch_moves:
                    if verbose and total_iterations % 100 == 0:
                        print(f"[Iter {total_iterations}] 无有效批量交换方案")
                    continue

                self.exchange_matcher.execute_batch_moves(batch_moves)
                # 现在 state 应该已更新，可以计算 new_utility
                new_utility = float(self.sharding_algo.calculate_system_total_utility())
                gain = new_utility - base_utility

                # 决定是否接受（SA 规则）
                accept_prob = self._accept_probability(gain, temperature)
                if self.rng.rand() < accept_prob:
                    # 接受：更新 base_utility；若更优则记录 best
                    base_utility = new_utility
                    if base_utility > best_utility + self.EPS:
                        best_utility = base_utility
                        best_shards = {k: list(v) for k, v in self.shards.items()}
                        if verbose:
                            # 打印本次被接受的 moves 概览
                            move_details = []
                            for mv in batch_moves:
                                if mv[0] == "pair":
                                    _, a, sa, b, sb = mv
                                    move_details.append(f"{a}<->{b}({sa}↔{sb})")
                                else:
                                    _, n, s_old, s_new = mv
                                    move_details.append(f"{n}({s_old}->{s_new})")

                            print(f"[Iter {total_iterations}] 温度{temperature:.4f}，接受：{'; '.join(move_details)}，增益 {gain:.6f}，最优 {best_utility:.6f}")
                else:
                    # 拒绝：恢复快照（完整恢复到调用 match_batch_nodes 之前）
                    _restore_from_snapshot(shards_snapshot, shard_id_snapshot)
                    if verbose and total_iterations % 500 == 0:
                        print(f"[Iter {total_iterations}] 温度{temperature:.4f}，拒绝交换（增益 {gain:.6f}，接受概率 {accept_prob:.6f}）")

            # 降温
            temperature *= self.cool_rate

        # 结束：把 best 写回父分片算法实例并更新映射
        self.sharding_algo.shards = best_shards
        self.sharding_algo._update_shard_mapping()
        self.sharding_algo.reset_metrics_cache()

        total_time = time.time() - start_time  # 新增2：计算总耗时
        if verbose:
            print("\n=== 模拟退火求解器优化结束 ===")
            print(f"总迭代次数：{total_iterations}，总耗时：{total_time:.2f}秒")
            print(f"最终最优系统总效用：{best_utility:.6f}")

        return best_shards, total_iterations, total_time, best_utility


# -------------------------
# 测试段（直接运行可验证基本流程）
# -------------------------
if __name__ == "__main__":
    # 1. 生成测试数据
    from data_generator import DataGenerator
    generator = DataGenerator(num_nodes=50)
    test_data = generator.get_data()

    # 2. 初始分片（随机分片作为起点）
    from random_sharding import RandomSharding
    initial_sharding = RandomSharding(data=test_data, num_shards=5)
    initial_sharding.sharding()

    # 3. 保存优化前指标
    original_utility = initial_sharding.calculate_system_total_utility()
    original_intra_rate = initial_sharding.calculate_intra_shard_rate()
    original_loss = initial_sharding.calculate_transmission_loss()

    # 4. 初始化并运行 SA 求解器
    sa_solver = SASolver(sharding_algorithm=initial_sharding, seed=Config.RANDOM_SEED)
    best_shards, total_iters, total_time, final_utility = sa_solver.solve(verbose=True)  # 适配新返回值

    # 5. 打印对比
    print("\n" + "=" * 60)
    print("优化前后核心指标对比：")
    print(f"优化前系统总效用：{original_utility:.6f}")
    print(f"优化后系统总效用：{final_utility:.6f}，提升幅度：{(final_utility - original_utility) / (original_utility + 1e-12) * 100:.2f}%")
    print(f"优化前片内交易率：{original_intra_rate:.6f}")
    print(f"优化后片内交易率：{initial_sharding.calculate_intra_shard_rate():.6f}")
    print(f"优化前总传输损耗：{original_loss:.2f}")
    print(f"优化后总传输损耗：{initial_sharding.calculate_transmission_loss():.2f}")
    print(f"总迭代次数：{total_iters}，总耗时：{total_time:.2f}秒")  # 打印耗时
    print("=" * 60)
