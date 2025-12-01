# static_clustering.py
import numpy as np
from sklearn.cluster import KMeans  # 用于按电气距离聚类（需安装：pip install scikit-learn）
from base_algorithm import BaseShardingAlgorithm
from config import Config


class StaticClusteringSharding(BaseShardingAlgorithm):
    """
    静态聚类分片算法：模拟微电网物理分片，按电气距离聚类
    核心逻辑：
    1. 把每个节点的"电气距离向量"作为特征（反映节点与所有其他节点的物理关联）
    2. 用KMeans聚类（聚类数=分片数），将电气距离近的节点分到同一分片
    3. 调整分片规模至合规（满足SHARD_SIZE_MIN/MAX），一旦分配永久不变
    作用：作为"物理分片"的基准，对比动态分片的动态优化优势
    """

    def __init__(self, data, num_shards):
        # 调用父类初始化，获取节点数据和分片数量
        super().__init__(data, num_shards)

        # 分片规模约束（从Config读取）
        self.shard_min_size = Config.SHARD_SIZE_MIN
        self.shard_max_size = Config.SHARD_SIZE_MAX

        # 聚类特征：每个节点的电气距离向量（N维，N=节点数）
        # 例如：节点i的特征向量 = [d(i,0), d(i,1), ..., d(i,N-1)]，反映i与所有节点的物理距离
        self.cluster_features = self.electrical_distance  # 直接用电气距离矩阵作为特征（每行是一个节点的特征）

    def _adjust_shard_size(self, clusters):
        """
        调整聚类结果，确保每个分片的节点数符合规模约束
        :param clusters: KMeans输出的聚类标签（数组，clusters[i] = 节点i的聚类ID）
        :return: 调整后的分片字典 {分片ID: [节点列表]}
        """
        # 1. 先按聚类标签分组，得到初始分片
        initial_shards = {}
        for node_id in self.node_ids:
            cluster_id = clusters[node_id]
            if cluster_id not in initial_shards:
                initial_shards[cluster_id] = []
            initial_shards[cluster_id].append(node_id)

        # 2. 转换为列表，方便调整（初始聚类数可能等于分片数，直接用）
        shard_list = list(initial_shards.values())

        # 3. 调整分片规模：处理过小/过大的分片
        # 3.1 先收集"节点溢出"的分片（超过max_size）和"节点不足"的分片（低于min_size）
        overflow_shards = []  # 存储需要转出节点的分片（节点列表）
        underflow_shards = []  # 存储需要转入节点的分片（分片索引）

        for idx, shard_nodes in enumerate(shard_list):
            shard_size = len(shard_nodes)
            if shard_size > self.shard_max_size:
                # 分片过大，计算需要转出的节点数
                overflow_count = shard_size - self.shard_max_size
                # 取出溢出的节点（随机选，不影响物理聚类特性）
                overflow_nodes = np.random.choice(shard_nodes, size=overflow_count, replace=False).tolist()
                # 保留核心节点（不溢出的部分）
                shard_list[idx] = [node for node in shard_nodes if node not in overflow_nodes]
                # 记录溢出节点，后续分配给不足的分片
                overflow_shards.extend(overflow_nodes)
            elif shard_size < self.shard_min_size:
                # 分片过小，加入待补充列表
                underflow_shards.append(idx)

        # 3.2 将溢出节点分配给不足的分片
        for underflow_idx in underflow_shards:
            if not overflow_shards:
                break  # 没有溢出节点可分配，退出
            # 计算需要补充的节点数
            need_count = self.shard_min_size - len(shard_list[underflow_idx])
            # 从溢出节点中取需要的数量（优先选电气距离近的节点，增强聚类特性）
            target_shard_nodes = shard_list[underflow_idx]
            selected_overflow = []
            for overflow_node in overflow_shards:
                if len(selected_overflow) >= need_count:
                    break
                # 计算溢出节点与目标分片的平均电气距离，选最近的
                avg_dist = np.mean([self.electrical_distance[overflow_node][n] for n in target_shard_nodes])
                selected_overflow.append((avg_dist, overflow_node))
            # 按距离排序，选最近的need_count个
            selected_overflow.sort(key=lambda x: x[0])
            selected_nodes = [node for (dist, node) in selected_overflow[:need_count]]
            # 分配给目标分片
            shard_list[underflow_idx].extend(selected_nodes)
            # 从溢出列表中移除已分配的节点
            for node in selected_nodes:
                overflow_shards.remove(node)

        # 3.3 处理剩余的溢出节点（如果还有）
        if overflow_shards:
            for node in overflow_shards:
                # 找到节点数最少的合规分片，加入
                min_size_shard_idx = np.argmin([len(shard) for shard in shard_list])
                if len(shard_list[min_size_shard_idx]) < self.shard_max_size:
                    shard_list[min_size_shard_idx].append(node)

        # 4. 转换为分片字典（分片ID从0开始）
        final_shards = {idx: shard for idx, shard in enumerate(shard_list)}
        return final_shards

    def sharding(self):
        """
        核心分片逻辑：KMeans聚类→规模调整→生成映射
        """
        # 1. 执行KMeans聚类（按电气距离）
        # n_clusters=分片数，random_state=固定种子确保可复现
        kmeans = KMeans(n_clusters=self.num_shards, random_state=Config.RANDOM_SEED, n_init=10)
        # 拟合特征数据，得到每个节点的聚类标签
        clusters = kmeans.fit_predict(self.cluster_features)

        # 2. 调整分片规模，确保符合约束
        self.shards = self._adjust_shard_size(clusters)

        # 3. 更新节点→分片的映射（调用父类辅助方法）
        self._update_shard_mapping()

        # 可选：打印分片结果（调试用）
        print(f"静态聚类分片完成：共{self.num_shards}个分片，各分片节点数：")
        for shard_id, nodes in self.shards.items():
            print(f"  分片{shard_id}：{len(nodes)}个节点")

        return self.shards


# 测试代码：验证静态聚类分片是否正常运行
if __name__ == "__main__":
    # 1. 生成测试数据（50节点）
    from data_generator import DataGenerator

    generator = DataGenerator(num_nodes=50)
    test_data = generator.get_data()

    # 2. 初始化静态聚类分片算法（50节点→5个分片）
    static_sharding = StaticClusteringSharding(data=test_data, num_shards=5)

    # 3. 执行分片
    static_sharding.sharding()

    # 4. 计算并打印核心指标（对比随机分片，片内交易率应更高）
    print("\n=== 静态聚类分片核心指标 ===")
    print(f"片内交易率：{static_sharding.calculate_intra_shard_rate():.4f}（预期≈0.4~0.5，高于随机分片）")
    print(f"总传输损耗：{static_sharding.calculate_transmission_loss():.2f}（预期低于随机分片）")
    print(f"系统总效用：{static_sharding.calculate_system_total_utility():.4f}（预期高于随机分片）")
    print(f"分片效用基尼系数：{static_sharding.calculate_gini_coefficient(level='shard'):.4f}")
