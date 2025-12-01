# base_algorithm.py
import numpy as np
from abc import ABC, abstractmethod  # 用于定义抽象基类
from config import Config


class BaseShardingAlgorithm(ABC):
    """
    所有分片算法的基础类：
    - 统一接收节点数据（信任值、电气距离、交易电量）
    - 定义所有分片算法必须实现的接口（sharding方法）
    - 封装通用指标计算逻辑（片内交易率、系统总效用等），避免重复代码
    """

    def __init__(self, data, num_shards):
        """
        初始化基础参数
        :param data: 从DataGenerator获取的数据集（含trust_values, energy_matrix等）
        :param num_shards: 分片数量
        """
        # 节点基础数据（从data中提取，方便后续调用）
        self.node_ids = data["node_ids"]  # 节点ID列表
        self.num_nodes = len(self.node_ids)  # 节点总数
        self.trust_values = data["trust_values"]  # 节点信任值数组
        self.energy_matrix = data["energy_matrix"]  # 交易电量矩阵（N×N）
        self.electrical_distance = data["electrical_distance"]  # 电气距离矩阵（N×N）

        # 分片参数
        self.num_shards = num_shards  # 分片数量
        self.shards = None  # 分片结果（子类实现sharding后赋值，格式：{分片ID: [节点ID列表]}）
        self.shard_id_of_node = None  # 节点→分片的映射（快速查询：节点i属于哪个分片）

        # 缓存指标计算结果（避免重复计算）
        self.total_energy = self.energy_matrix.sum()  # 系统总交易量（固定值）
        self.metrics_cache = {}  # 缓存指标结果，如{"intra_shard_rate": 0.8, ...}

    @abstractmethod
    def sharding(self):
        """
        抽象方法：执行分片算法
        子类（如动态分片、随机分片）必须实现此方法，返回分片结果
        执行后需更新：self.shards 和 self.shard_id_of_node
        """
        pass

    def _update_shard_mapping(self):
        """
        辅助方法：根据self.shards生成self.shard_id_of_node（节点→分片的映射）
        例如：shard_id_of_node[i] = s 表示节点i属于分片s
        """
        if self.shards is None:
            raise ValueError("请先执行sharding()生成分片结果")

        self.shard_id_of_node = np.zeros(self.num_nodes, dtype=int)  # 初始化映射数组
        for shard_id, nodes in self.shards.items():
            for node in nodes:
                self.shard_id_of_node[node] = shard_id

    # ------------------------------
    # 核心指标计算方法（实验需要的所有指标）
    # ------------------------------
    # 节点效用依赖的辅助计算
    def _calculate_node_local_metrics(self, node, shard_nodes):
        # L: 片内交易量（保持）
        L = sum(self.energy_matrix[node][k] for k in shard_nodes if k != node)
        total_node_energy = self.energy_matrix[node].sum()
        LTC = L / total_node_energy if total_node_energy > getattr(Config, "EPS", 1e-12) else 0.0
        # D：节点与所有对端的物理传输损耗贡献（包括片内和跨片）
        loss_coef = getattr(Config, "LOSS_COEF", 0.1)
        D = 0.0
        for k in range(self.num_nodes):
            if k == node:
                continue
            e = self.energy_matrix[node][k]
            d = self.electrical_distance[node][k]
            D += e * d * loss_coef  # 这里按单向口径累加（与节点效用的尺度一致）
        return L, LTC, D

    def calculate_intra_shard_rate(self):
        """
        计算片内交易率：片内总交易量 / 系统总交易量
        片内交易：交易双方属于同一分片的交易
        """
        if "intra_shard_rate" in self.metrics_cache:
            return self.metrics_cache["intra_shard_rate"]

        if self.shards is None:
            raise ValueError("请先执行sharding()生成分片结果")

        intra_energy = 0.0  # 片内总交易量
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):  # 只算i<j，避免重复
                if self.shard_id_of_node[i] == self.shard_id_of_node[j]:
                    # 双方同分片，累加交易电量（i→j和j→i在矩阵中是同一值，这里只加一次）
                    intra_energy += self.energy_matrix[i][j] * 2  # 因为矩阵是对称的，i→j和j→i都算

        # 片内交易率 = 片内总交易量 / 系统总交易量（总交易量已包含双向）
        rate = intra_energy / self.total_energy if self.total_energy != 0 else 0.0
        self.metrics_cache["intra_shard_rate"] = rate
        return rate

    def calculate_transmission_loss(self):
        """
        计算物理传输损耗（所有交易的物理损耗），并同时缓存跨片的物理损耗。
        采用口径说明：这里对 i<j 累加 energy*distance，然后乘以2（双向计），
        与代码其余处保持“总量为矩阵元素之和”的风格一致。
        """
        if "transmission_loss" in self.metrics_cache:
            return self.metrics_cache["transmission_loss"]

        if self.shards is None:
            raise ValueError("请先执行sharding()生成分片结果")

        loss_coef = getattr(Config, "LOSS_COEF", 0.1)
        total_loss = 0.0
        cross_loss = 0.0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                energy = self.energy_matrix[i][j]
                distance = self.electrical_distance[i][j]
                loss = energy * distance * loss_coef * 2  # 双向口径
                total_loss += loss
                if self.shard_id_of_node[i] != self.shard_id_of_node[j]:
                    cross_loss += loss

        self.metrics_cache["transmission_loss"] = total_loss
        self.metrics_cache["transmission_loss_cross"] = cross_loss
        return total_loss

    def calculate_shard_utilities(self):
        if "shard_utilities" in self.metrics_cache:
            return self.metrics_cache["shard_utilities"]
        if self.shards is None:
            raise ValueError("请先执行sharding()生成分片结果")

        shard_utils = {}
        for shard_id, nodes in self.shards.items():
            # 1. 计算分片内/跨片交易电量（用于手续费和成本）
            intra_energy = 0.0  # 片内交易总电量
            cross_energy = 0.0  # 跨片交易总电量
            node_set = set(nodes)  # **最小改动**：把列表转集合以加速 "in" 检查
            for i in nodes:
                for j in range(self.num_nodes):
                    if i == j:
                        continue
                    if j in node_set:
                        intra_energy += self.energy_matrix[i][j]
                    else:
                        cross_energy += self.energy_matrix[i][j]

            # 2. 分片收益：手续费收益 + 平均信任值（论文公式4.1.2）
            # 在原 calculate_shard_utilities 的相关段落将 H_in/H_cross 与 C_in/C_cross 修改
            H_in = 0.5 * intra_energy * Config.R_IN
            H_cross = 0.5 * cross_energy * Config.R_CROSS
            avg_trust = np.mean([self.trust_values[node] for node in nodes])
            shard_reward = Config.ALPHA1 * (H_in + H_cross) + Config.ALPHA2 * avg_trust

            # 3. 分片成本：交易处理成本（论文公式4.1.2）
            C_in = 0.5 * intra_energy * Config.C_IN
            C_cross = 0.5 * cross_energy * Config.C_CROSS
            shard_cost = C_in + C_cross

            # 4. 分片效用 = 收益 - 成本
            shard_util = shard_reward - shard_cost
            shard_utils[shard_id] = shard_util

        self.metrics_cache["shard_utilities"] = shard_utils
        return shard_utils

    # --------------------------
    # 节点效用计算
    # --------------------------
    def calculate_node_utilities(self):
        if "node_utilities" in self.metrics_cache:
            return self.metrics_cache["node_utilities"]
        if self.shards is None:
            raise ValueError("请先执行sharding()生成分片结果")

        node_utils = {}
        for node in self.node_ids:
            shard_id = self.shard_id_of_node[node]
            shard_nodes = self.shards[shard_id]
            # 获取L（本地交易量）、LTC（本地交易率）、D（传输损耗）
            L, LTC, D = self._calculate_node_local_metrics(node, shard_nodes)
            # 节点效用 = β1*L + β2*LTC - β3*D（论文公式4.1.3）
            node_util = (
                Config.BETA1 * L
                + Config.BETA2 * LTC
                - Config.BETA3 * D
            )
            node_utils[node] = node_util

        self.metrics_cache["node_utilities"] = node_utils
        return node_utils

    def calculate_system_total_utility(self):
        if "system_total_utility" in self.metrics_cache:
            return self.metrics_cache["system_total_utility"]

        # 1. 原始分片与节点效用
        shard_utils = self.calculate_shard_utilities()
        node_utils = self.calculate_node_utilities()

        shard_values = np.array(list(shard_utils.values()))
        node_values = np.array(list(node_utils.values()))

        # 存原始总和以便调试/记录
        total_shard_util = shard_values.sum()
        total_node_util = node_values.sum()
        self.metrics_cache["total_shard_util_raw"] = total_shard_util
        self.metrics_cache["total_node_util_raw"] = total_node_util

        eps = getattr(Config, "EPS", 1e-12)

        # 2. 对分片效用做 min-max 归一化（得到 avg_shard_norm ∈ [0,1]）
        if shard_values.size == 0:
            avg_shard_norm = 0.0
        else:
            min_s = shard_values.min()
            max_s = shard_values.max()
            if max_s - min_s < eps:
                # 所有分片效用相等时，视为完全平衡（取 1 或 0 均可；这里我们取 1 表示没有可改进空间）
                avg_shard_norm = 1.0
            else:
                avg_shard = shard_values.mean()
                avg_shard_norm = (avg_shard - min_s) / (max_s - min_s)

        # 3. 对节点效用做 min-max 归一化（得到 avg_node_norm ∈ [0,1]）
        if node_values.size == 0:
            avg_node_norm = 0.0
        else:
            min_n = node_values.min()
            max_n = node_values.max()
            if max_n - min_n < eps:
                avg_node_norm = 1.0
            else:
                avg_node = node_values.mean()
                avg_node_norm = (avg_node - min_n) / (max_n - min_n)

        # 4. 公平性损失（保持 0~1）
        gini_shard = self.calculate_gini_coefficient(level="shard")
        gini_node = self.calculate_gini_coefficient(level="node")
        total_fair_loss = Config.OMEGA_G * gini_shard + (1 - Config.OMEGA_G) * gini_node
        self.metrics_cache["gini_shard"] = gini_shard
        self.metrics_cache["gini_node"] = gini_node
        self.metrics_cache["total_fair_loss"] = total_fair_loss

        # 5. 合成系统总效用（各项现在可比较）
        total_util = (
                Config.OMEGA_SHARD * avg_shard_norm
                + (1 - Config.OMEGA_SHARD) * avg_node_norm
                - Config.OMEGA_FAIR * total_fair_loss
        )

        # 缓存并返回
        self.metrics_cache["avg_shard_norm"] = avg_shard_norm
        self.metrics_cache["avg_node_norm"] = avg_node_norm
        self.metrics_cache["system_total_utility"] = total_util
        return total_util

    def calculate_gini_coefficient(self, level="shard"):
        """按论文公式计算基尼系数：G = [2Σ(i×u_i)]/[nΣu_i] - (n+1)/n"""
        cache_key = f"gini_{level}"
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]

        # 获取待计算的效用数组
        if level == "shard":
            shard_utils = self.calculate_shard_utilities()
            values = np.array(list(shard_utils.values()))
        elif level == "node":
            node_utils = self.calculate_node_utilities()
            values = np.array(list(node_utils.values()))
        else:
            raise ValueError("level必须是'shard'或'node'")

        # 边界处理：无有效效用值时返回0（完全公平）
        n = len(values)
        total = values.sum()
        if n <= 1 or total <= Config.EPS:
            self.metrics_cache[cache_key] = 0.0
            return 0.0

        # 按论文公式计算（先排序，再计算加权和）
        values = np.sort(values)  # 升序排列
        weighted_sum = np.sum(np.arange(1, n+1) * values)  # Σ(i×u_i)
        gini = (2 * weighted_sum) / (n * total) - (n + 1) / n  # 论文核心公式
        gini = max(0.0, gini)  # 避免数值误差导致的负基尼系数

        self.metrics_cache[cache_key] = gini
        return gini

    def reset_metrics_cache(self):
        """重置指标缓存（分片结果更新后需调用）"""
        self.metrics_cache = {}


# 测试代码：验证基础类的指标计算是否正确
if __name__ == "__main__":
    # 1. 生成测试数据（30节点）
    from data_generator import DataGenerator

    generator = DataGenerator(num_nodes=30)
    test_data = generator.get_data()


    # 2. 定义一个简单的测试分片算法（继承BaseShardingAlgorithm）
    class TestSharding(BaseShardingAlgorithm):
        def sharding(self):
            """简单分片：按节点ID平均分配到3个分片"""
            shards = {0: [], 1: [], 2: []}
            for i in range(self.num_nodes):
                shard_id = i % 3  # 0→0,1→1,2→2,3→0...
                shards[shard_id].append(i)
            self.shards = shards
            self._update_shard_mapping()  # 更新节点→分片映射
            return shards


    # 3. 执行测试分片并计算指标
    test_algorithm = TestSharding(data=test_data, num_shards=3)
    test_algorithm.sharding()  # 执行分片

    # 4. 打印指标结果（验证计算逻辑）
    print(f"片内交易率：{test_algorithm.calculate_intra_shard_rate():.4f}（预期≈0.3~0.4，随机分配）")
    print(f"总传输损耗：{test_algorithm.calculate_transmission_loss():.2f}（预期为正值）")
    print(f"系统总效用（归一化后，理论范围约[-OMEGA_FAIR, 1]）：{test_algorithm.calculate_system_total_utility():.4f}")
    print(f"分片效用基尼系数：{test_algorithm.calculate_gini_coefficient(level='shard'):.4f}（预期0~0.5）")
    # 在打印系统总效用之前/之后，打印中间量
    print(f"分片效用原始总和：{test_algorithm.metrics_cache.get('total_shard_util_raw'):.4f}")
    print(f"节点效用原始总和：{test_algorithm.metrics_cache.get('total_node_util_raw'):.4f}")
    print(f"归一化分片平均（avg_shard_norm）：{test_algorithm.metrics_cache.get('avg_shard_norm'):.4f}")
    print(f"归一化节点平均（avg_node_norm）：{test_algorithm.metrics_cache.get('avg_node_norm'):.4f}")
    print(
        f"基尼（shard/node）：{test_algorithm.metrics_cache.get('gini_shard'):.4f} / {test_algorithm.metrics_cache.get('gini_node'):.4f}")
    print(f"公平性合成惩罚（total_fair_loss）：{test_algorithm.metrics_cache.get('total_fair_loss'):.4f}")
