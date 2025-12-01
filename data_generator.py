# data_generator.py
import numpy as np
from config import Config

np.random.seed(Config.RANDOM_SEED)

class DataGenerator:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.node_ids = list(range(num_nodes))

    def generate_trust_values(self):
        trust_values = np.zeros(self.num_nodes)
        high_trust_count = int(self.num_nodes * Config.HIGH_TRUST_RATIO)
        low_trust_count = self.num_nodes - high_trust_count
        # 高信任值（0.6~1.0）
        trust_values[:high_trust_count] = np.random.uniform(0.6, 1.0, high_trust_count)
        # 低信任值（0.1~0.3）
        trust_values[high_trust_count:] = np.random.uniform(0.1, 0.3, low_trust_count)
        np.random.shuffle(trust_values)
        return trust_values

    def generate_electrical_distance(self):
        cluster_assign = np.random.randint(0, Config.CLUSTER_NUM, self.num_nodes)
        dist_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if cluster_assign[i] == cluster_assign[j]:
                    dist = np.random.normal(0.5, 0.2)  # 同聚类近距离
                    dist = max(0.1, dist)
                else:
                    dist = np.random.uniform(3.0, 5.0)  # 跨聚类远距离
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        return dist_matrix

    # 替换 generate_energy_matrix 与 adjust_energy_by_trust
    import numpy as np
    from config import Config

    def generate_energy_matrix(self, electrical_distance, trust_values):
        rng = np.random.RandomState(Config.RANDOM_SEED)
        energy_matrix = np.zeros((self.num_nodes, self.num_nodes))
        is_high_trust = trust_values >= 0.6

        # 收集候选对（和你现在很像）
        priority1_pairs, priority2_pairs, priority3_pairs = [], [], []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                is_close = electrical_distance[i][j] < Config.CLOSE_DISTANCE_THRESHOLD
                both_high = is_high_trust[i] and is_high_trust[j]
                one_high = (is_high_trust[i] ^ is_high_trust[j])
                if both_high and is_close:
                    priority1_pairs.append((i, j))
                elif one_high and is_close:
                    priority2_pairs.append((i, j))
                else:
                    priority3_pairs.append((i, j))

        total_pairs = self.num_nodes * (self.num_nodes - 1) // 2
        core_pairs_needed = max(1, int(total_pairs * Config.CORE_PAIR_RATIO))
        max_high_trust_core = int(core_pairs_needed * 0.6)

        core_pairs = []
        take1 = min(len(priority1_pairs), max_high_trust_core)
        core_pairs += priority1_pairs[:take1]
        remaining = core_pairs_needed - take1

        if remaining > 0 and len(priority2_pairs) > 0:
            take2 = min(len(priority2_pairs), remaining)
            core_pairs += priority2_pairs[:take2]
            remaining -= take2

        if remaining > 0:
            take3 = min(len(priority3_pairs), remaining)
            core_pairs += priority3_pairs[:take3]

        # 总能量基准（可调）
        total_energy_base = self.num_nodes * 10 * 50
        core_energy_total = total_energy_base * Config.CORE_ENERGY_RATIO
        non_core_total = total_energy_base - core_energy_total

        # 使用更温和的重尾（Pareto shape 改为 3.0），并 +1 防止极端小值
        if len(core_pairs) > 0:
            pareto_shape = 3.0
            core_weights = rng.pareto(pareto_shape, len(core_pairs)) + 1.0
            core_weights = core_weights / core_weights.sum() * core_energy_total
            for idx, (i, j) in enumerate(core_pairs):
                energy_matrix[i, j] = energy_matrix[j, i] = core_weights[idx]

        # 非核心对使用指数分布或均匀（更温和）
        non_core_pairs = [(i, j) for i in range(self.num_nodes) for j in range(i + 1, self.num_nodes) if
                          (i, j) not in core_pairs]
        if non_core_pairs:
            non_weights = rng.exponential(scale=1.0, size=len(non_core_pairs))
            non_weights = non_weights / non_weights.sum() * non_core_total
            for idx, (i, j) in enumerate(non_core_pairs):
                energy_matrix[i, j] = energy_matrix[j, i] = non_weights[idx]

        # 明确对角为0
        np.fill_diagonal(energy_matrix, 0.0)
        return energy_matrix

    def adjust_energy_by_trust(self, energy_matrix, trust_values):
        """
        更稳定的按节点比例放大：
          - 映射 trust -> node_scale（线性，范围 [1.0, max_scale]）
          - 一次性用外积 scale_matrix 放大 energy_matrix
          - 归一化回原始总能量（保持 core/non-core 比例）
        """
        N = self.num_nodes
        orig_total = energy_matrix.sum()
        if orig_total <= 0:
            return energy_matrix

        # 线性映射：trust>=0.6 映射到 [1.0, max_scale]
        min_trust = 0.6
        max_scale = 1.3  # 保守：1.1~1.5 可选，避免 2+ 的极端放大
        node_scale = np.ones(N)
        for i in range(N):
            if trust_values[i] >= min_trust:
                node_scale[i] = 1.0 + (trust_values[i] - min_trust) / (1.0 - min_trust) * (max_scale - 1.0)
                node_scale[i] = min(node_scale[i], max_scale)

        # 一次性对称放大
        scale_matrix = np.outer(node_scale, node_scale)
        energy_scaled = energy_matrix * scale_matrix

        # 可选：对高-high对做微调（非常小，例如 1.05），但一般不需要
        # 归一化回原始总量（防止总能量发生放大）
        new_total = energy_scaled.sum()
        if new_total > 0:
            energy_scaled *= (orig_total / new_total)

        # 保证对角为0
        np.fill_diagonal(energy_scaled, 0.0)
        return energy_scaled

    def get_data(self):
        trust_values = self.generate_trust_values()
        electrical_distance = self.generate_electrical_distance()
        # 生成电量时传入信任值，优先高信任对
        energy_matrix = self.generate_energy_matrix(electrical_distance, trust_values)
        energy_matrix = self.adjust_energy_by_trust(energy_matrix, trust_values)
        return {
            "node_ids": self.node_ids,
            "trust_values": trust_values,
            "electrical_distance": electrical_distance,
            "energy_matrix": energy_matrix
        }


# 本地运行验证
if __name__ == "__main__":
    generator = DataGenerator(50)
    data = generator.get_data()

    # 1. 高信任节点占比
    high_trust = sum(1 for t in data["trust_values"] if t >= 0.6) / 50
    print(f"高信任节点占比：{high_trust:.2f}（预期0.8）")

    # 2. 近距离节点对占比
    close_pairs = 0
    total_pairs = 0
    for i in range(50):
        for j in range(i+1, 50):
            total_pairs += 1
            if data["electrical_distance"][i][j] < 1.5:
                close_pairs += 1
    print(f"近距离对占比：{close_pairs/total_pairs:.2f}（预期0.3~0.4）")

    # 3. Top10%节点对电量占比
    all_energy = []
    for i in range(50):
        for j in range(50):
            if i != j:
                all_energy.append(data["energy_matrix"][i][j])
    all_energy.sort(reverse=True)
    top10 = sum(all_energy[:int(len(all_energy)*0.1)]) / sum(all_energy)
    print(f"Top10%节点对电量占比：{top10:.2f}（预期0.6~0.7）")

    # 4. 高信任节点总交易量占比
    high_trust_nodes = [i for i, t in enumerate(data["trust_values"]) if t >= 0.6]
    high_energy = 0
    total_energy = data["energy_matrix"].sum()
    for i in high_trust_nodes:
        high_energy += data["energy_matrix"][i].sum()  # 该节点的所有交易
    print(f"高信任节点总交易量占比：{high_energy/total_energy:.2f}（预期0.5~0.6）")
