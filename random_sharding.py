# random_sharding.py
import numpy as np
from base_algorithm import BaseShardingAlgorithm
from config import Config


class RandomSharding(BaseShardingAlgorithm):
    """
    随机分片算法：将节点完全随机分配到指定数量的分片
    核心特点：不考虑任何微电网场景特征（交易电量、电气距离、信任值），仅随机分配
    作用：作为最基础的对比基准，验证其他算法（如你的动态分片）的优化效果
    """

    def __init__(self, data, num_shards):
        # 调用父类（BaseShardingAlgorithm）的初始化方法，自动获取节点数据和分片数量
        super().__init__(data, num_shards)

        # 额外记录分片规模约束（从Config读取，确保分配符合规则）
        self.shard_min_size = Config.SHARD_SIZE_MIN
        self.shard_max_size = Config.SHARD_SIZE_MAX

    def sharding(self):
        """
        核心分片逻辑：随机分配节点，同时满足分片规模上下限
        步骤：1. 初始化分片 → 2. 确保每个分片达到最小规模 → 3. 随机分配剩余节点 → 4. 更新映射
        """
        # 1. 初始化分片字典（key：分片ID，value：节点列表）
        self.shards = {shard_id: [] for shard_id in range(self.num_shards)}

        # 2. 先给每个分片分配最小数量的节点（确保不违反SHARD_SIZE_MIN）
        remaining_nodes = self.node_ids.copy()  # 待分配的节点列表
        for shard_id in range(self.num_shards):
            # 每个分片先取min_size个节点（如果剩余节点足够）
            take_num = min(self.shard_min_size, len(remaining_nodes))
            # 随机从剩余节点中选take_num个
            selected_nodes = np.random.choice(remaining_nodes, size=take_num, replace=False)
            # 转换为列表（numpy数组转普通列表）
            selected_nodes = selected_nodes.tolist()
            # 分配到当前分片
            self.shards[shard_id] = selected_nodes
            # 从剩余节点中移除已分配的节点
            remaining_nodes = [node for node in remaining_nodes if node not in selected_nodes]

        # 3. 随机分配剩余节点（每个分片不超过最大规模SHARD_SIZE_MAX）
        for node in remaining_nodes:
            # 找到所有还没到最大规模的分片（候选分片）
            valid_shards = [
                shard_id for shard_id, nodes in self.shards.items()
                if len(nodes) < self.shard_max_size
            ]
            if not valid_shards:
                # 理论上不会走到这（Config中参数已确保节点数和分片数匹配）
                raise ValueError("没有可用的分片，节点分配失败")
            # 从候选分片中随机选一个
            target_shard = np.random.choice(valid_shards)
            # 将节点加入目标分片
            self.shards[target_shard].append(node)

        # 4. 更新节点→分片的映射（调用父类的辅助方法）
        self._update_shard_mapping()

        # 可选：打印分片分配结果（调试用）
        print(f"随机分片完成：共{self.num_shards}个分片，各分片节点数：")
        for shard_id, nodes in self.shards.items():
            print(f"  分片{shard_id}：{len(nodes)}个节点")

        return self.shards


# 测试代码：验证随机分片是否正常运行
if __name__ == "__main__":
    # 1. 生成测试数据（50节点）
    from data_generator import DataGenerator

    generator = DataGenerator(num_nodes=50)
    test_data = generator.get_data()

    # 2. 初始化随机分片算法（50节点→5个分片）
    random_sharding = RandomSharding(data=test_data, num_shards=5)

    # 3. 执行分片
    random_sharding.sharding()

    # 4. 计算并打印核心指标（验证是否能正常调用父类的指标方法）
    print("\n=== 随机分片核心指标 ===")
    print(f"片内交易率：{random_sharding.calculate_intra_shard_rate():.4f}（预期≈0.2，5个分片随机分配）")
    print(f"总传输损耗：{random_sharding.calculate_transmission_loss():.2f}")
    print(f"系统总效用：{random_sharding.calculate_system_total_utility():.4f}")
    print(f"分片效用基尼系数：{random_sharding.calculate_gini_coefficient(level='shard'):.4f}")
