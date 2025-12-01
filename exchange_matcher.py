# exchange_matcher.py (改进版)
import numpy as np
from config import Config

class ExchangeMatcher:
    def __init__(self, sharding_algorithm, seed=None):
        """
        sharding_algorithm: 继承 BaseShardingAlgorithm 的实例，
            要求已经有初始 self.shards 和 self.shard_id_of_node
        seed: 可选随机种子，用于所有内部随机操作（确保可复现）
        """
        self.sharding_algo = sharding_algorithm
        if getattr(sharding_algorithm, "shards", None) is None or getattr(sharding_algorithm, "shard_id_of_node", None) is None:
            raise ValueError("sharding_algorithm 必须已执行 sharding() 并初始化 shards 与 shard_id_of_node")

        self.num_nodes = sharding_algorithm.num_nodes
        self.num_shards = sharding_algorithm.num_shards
        # 这里持用引用（注意：所有操作要小心恢复）
        self.shards = sharding_algorithm.shards
        # 确保 shard_id_of_node 可索引为 numpy array
        self.shard_id_of_node = np.array(sharding_algorithm.shard_id_of_node, copy=False)

        # K 的兼容读取（向后兼容）
        self.K = getattr(Config, "K", getattr(Config, "K_BATCH_SWAP", 3))
        self.shard_min_size = int(getattr(Config, "SHARD_SIZE_MIN", 1))
        self.shard_max_size = int(getattr(Config, "SHARD_SIZE_MAX", self.num_nodes))

        # 随机数生成器（统一使用）
        seed = Config.RANDOM_SEED if seed is None else seed
        self.rng = np.random.RandomState(seed)

    # -------------------------
    # 合法性检查函数
    # -------------------------
    def _is_single_node_move_valid(self, node, current_shard, target_shard):
        if target_shard == current_shard:
            return False
        current_size_after = len(self.shards[current_shard]) - 1
        target_size_after = len(self.shards[target_shard]) + 1
        return (current_size_after >= self.shard_min_size) and (target_size_after <= self.shard_max_size)

    def _is_pair_swap_valid(self, node_a, node_b):
        shard_a = int(self.shard_id_of_node[node_a])
        shard_b = int(self.shard_id_of_node[node_b])
        if shard_a == shard_b:
            return False
        # 交换不改变分片大小，因此只需检查现存分片是否满足最小/最大约束（防御性）
        if len(self.shards[shard_a]) < self.shard_min_size or len(self.shards[shard_b]) < self.shard_min_size:
            return False
        if len(self.shards[shard_a]) > self.shard_max_size or len(self.shards[shard_b]) > self.shard_max_size:
            return False
        return True

    # -------------------------
    # 增益计算（模拟-恢复，异常安全）
    # -------------------------
    def _simulate_move_and_compute_gain(self, apply_func, restore_func, base_utility):
        """
        apply_func: 执行模拟变动（修改 self.shards/self.shard_id_of_node）
        restore_func: 恢复函数（在 finally 中调用）
        base_utility: 迁移前基准效用（避免重复求原效用）
        返回 new_utility - base_utility
        """
        try:
            apply_func()
            self.sharding_algo.reset_metrics_cache()
            new_util = self.sharding_algo.calculate_system_total_utility()
        finally:
            restore_func()
            self.sharding_algo.reset_metrics_cache()
        return new_util - base_utility

    def _calculate_move_gain(self, node, current_shard, target_shard, base_utility):
        # 构造 apply/restore
        def apply_op():
            self.shards[current_shard].remove(node)
            self.shards[target_shard].append(node)
            self.shard_id_of_node[node] = target_shard

        orig_current = list(self.shards[current_shard])
        orig_target = list(self.shards[target_shard])
        orig_shard_id = int(self.shard_id_of_node[node])

        def restore_op():
            self.shards[current_shard] = orig_current
            self.shards[target_shard] = orig_target
            self.shard_id_of_node[node] = orig_shard_id

        return self._simulate_move_and_compute_gain(apply_op, restore_op, base_utility)

    def _calculate_pair_gain(self, node_a, node_b, base_utility):
        shard_a = int(self.shard_id_of_node[node_a])
        shard_b = int(self.shard_id_of_node[node_b])
        # prepare original snapshots
        orig_a = list(self.shards[shard_a])
        orig_b = list(self.shards[shard_b])
        orig_a_id = int(self.shard_id_of_node[node_a])
        orig_b_id = int(self.shard_id_of_node[node_b])

        def apply_op():
            # swap
            self.shards[shard_a].remove(node_a)
            self.shards[shard_b].append(node_a)
            self.shards[shard_b].remove(node_b)
            self.shards[shard_a].append(node_b)
            self.shard_id_of_node[node_a] = shard_b
            self.shard_id_of_node[node_b] = shard_a

        def restore_op():
            self.shards[shard_a] = orig_a
            self.shards[shard_b] = orig_b
            self.shard_id_of_node[node_a] = orig_a_id
            self.shard_id_of_node[node_b] = orig_b_id

        return self._simulate_move_and_compute_gain(apply_op, restore_op, base_utility)

    # -------------------------
    # 匹配器：单节点、节点对、批量
    # -------------------------
    def match_single_node(self, node, base_utility=None):
        """
        为单个节点找收益最高的目标分片（返回 best_target_shard, best_gain）
        base_utility: 若外部已计算好当前效用，可传入以减少重复计算
        """
        if base_utility is None:
            base_utility = self.sharding_algo.calculate_system_total_utility()

        current_shard = int(self.shard_id_of_node[node])
        candidate_shards = [s for s in range(self.num_shards) if self._is_single_node_move_valid(node, current_shard, s)]
        if not candidate_shards:
            return current_shard, 0.0

        best_gain = -np.inf
        best_target = current_shard
        for tgt in candidate_shards:
            gain = self._calculate_move_gain(node, current_shard, tgt, base_utility)
            if gain > best_gain:
                best_gain = gain
                best_target = tgt
        return best_target, best_gain

    def match_node_pair(self, num_samples=50, base_utility=None):
        """
        随机采样若干节点对并返回收益最好的对（node_a, node_b, gain)。
        num_samples: 采样次数（越大越可能找到好的pair，但更慢）
        """
        if base_utility is None:
            base_utility = self.sharding_algo.calculate_system_total_utility()

        best_gain = -np.inf
        best_pair = None
        N = self.num_nodes
        # 如果组合数较小，扫描全部对；否则随机采样
        max_pairs = N * (N - 1) // 2
        if max_pairs <= num_samples:
            # 枚举所有无序对
            for i in range(N):
                for j in range(i + 1, N):
                    if not self._is_pair_swap_valid(i, j):
                        continue
                    gain = self._calculate_pair_gain(i, j, base_utility)
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, j)
        else:
            # 随机采样 num_samples 对（不放回）
            for _ in range(num_samples):
                a, b = self.rng.choice(N, size=2, replace=False)
                if not self._is_pair_swap_valid(a, b):
                    continue
                gain = self._calculate_pair_gain(a, b, base_utility)
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)
        if best_pair is None:
            return None, 0.0
        return (best_pair[0], best_pair[1], best_gain)

    def match_batch_nodes(self, K=None, base_utility=None, num_pair_samples=50):
        """
        生成一组非冲突的批量交换 moves，返回 move 列表：
          - ("pair", a, shard_a, b, shard_b)
          - ("single", node, cur_shard, tgt_shard)
        保证：返回的 moves 中节点互不重复、且合规
        """
        if K is None:
            K = self.K
        if base_utility is None:
            base_utility = self.sharding_algo.calculate_system_total_utility()

        moves = []
        used_nodes = set()
        remaining_k = K

        # 优先采样pair（每次消耗2）
        while remaining_k >= 2:
            a, b, gain = self.match_node_pair(num_samples=num_pair_samples, base_utility=base_utility)
            if a is None or gain <= 0:
                break
            # 如果 a 或 b 已被占用，跳过并尝试下一样本（避免无限循环）
            if a in used_nodes or b in used_nodes:
                # mark as tried but continue to next sample
                # To avoid infinite loop, decrement remaining_k attempt count?
                # Here we just break to avoid expensive repeated trials
                break
            # record move
            shard_a = int(self.shard_id_of_node[a])
            shard_b = int(self.shard_id_of_node[b])
            moves.append(("pair", int(a), shard_a, int(b), shard_b))
            used_nodes.update([a, b])
            remaining_k -= 2
            # update base_utility by applying the move temporarily so subsequent gains are evaluated on updated state
            # apply move
            self.shards[shard_a].remove(a); self.shards[shard_b].append(a)
            self.shards[shard_b].remove(b); self.shards[shard_a].append(b)
            self.shard_id_of_node[a] = shard_b; self.shard_id_of_node[b] = shard_a
            self.sharding_algo.reset_metrics_cache()
            base_utility = self.sharding_algo.calculate_system_total_utility()

        # 再尝试 single moves（每次消耗1）
        # 为避免重复/冲突，从未使用节点中随机抽样
        candidate_nodes = [n for n in range(self.num_nodes) if n not in used_nodes]
        self.rng.shuffle(candidate_nodes)
        for node in candidate_nodes:
            if remaining_k <= 0:
                break
            cur = int(self.shard_id_of_node[node])
            tgt, gain = self.match_single_node(node, base_utility=base_utility)
            if tgt != cur and gain > 0 and node not in used_nodes:
                moves.append(("single", int(node), cur, int(tgt)))
                used_nodes.add(node)
                remaining_k -= 1
                # apply move to update base_utility for subsequent picks
                self.shards[cur].remove(node); self.shards[tgt].append(node)
                self.shard_id_of_node[node] = tgt
                self.sharding_algo.reset_metrics_cache()
                base_utility = self.sharding_algo.calculate_system_total_utility()

        # 最终 verify moves 不冲突 且 合规
        nodes_in_moves = []
        for mv in moves:
            if mv[0] == "pair":
                _, a, sa, b, sb = mv
                nodes_in_moves.extend([a, b])
            else:
                _, n, _, _ = mv
                nodes_in_moves.append(n)
        if len(set(nodes_in_moves)) != len(nodes_in_moves):
            raise RuntimeError("批量moves中存在冲突节点（不应发生）")

        return moves

    # 4. 执行批量交换（兼容两种交换类型）
    def execute_batch_moves(self, batch_moves):
        """
        执行批量交换：支持 ("pair", node_a, shard_a, node_b, shard_b)
                         与 ("single", node, current_shard, target_shard)
        注意：执行后需同步 parent sharding_algo 的映射并 reset metrics cache。
        """
        if not batch_moves:
            return self.shards

        for move in batch_moves:
            if move[0] == "pair":  # 执行节点-节点双边交换
                _, node_a, shard_a, node_b, shard_b = move
                # 移动（按记录的原分片执行）
                # 为安全起见，先检查存在性再操作，避免重复/冲突
                if node_a in self.shards[shard_a] and node_b in self.shards[shard_b]:
                    self.shards[shard_a].remove(node_a)
                    self.shards[shard_b].append(node_a)
                    self.shards[shard_b].remove(node_b)
                    self.shards[shard_a].append(node_b)
                    # 直接同步映射（也会在最后做一次整体更新）
                    self.shard_id_of_node[node_a] = shard_b
                    self.shard_id_of_node[node_b] = shard_a
            elif move[0] == "single":  # 执行节点-空位交换
                _, node, current_shard, target_shard = move
                # 再次检查合规后执行，避免重复添加/删除
                if node in self.shards[current_shard] and len(self.shards[target_shard]) < self.shard_max_size:
                    self.shards[current_shard].remove(node)
                    self.shards[target_shard].append(node)
                    self.shard_id_of_node[node] = target_shard
            else:
                # 未知 move 类型，跳过（或抛出）
                raise ValueError(f"未知的 move 类型: {move[0]}")

        # 最后统一更新 parent 的映射并清缓存，确保所有视图一致
        try:
            # 将执行结果写回 parent sharding_algo
            self.sharding_algo.shards = self.shards
            # 重新生成映射并清空缓存，保证后续指标按最新分片计算
            self.sharding_algo._update_shard_mapping()
            self.sharding_algo.reset_metrics_cache()
            # 同步 matcher 的映射引用（保证本对象与 parent 视图一致）
            self.shard_id_of_node = self.sharding_algo.shard_id_of_node
        except Exception:
            # 若 parent 没有这些方法/属性，仍保证本地一致性
            pass

        return self.shards

