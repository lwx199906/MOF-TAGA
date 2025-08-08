import math
import random
import numpy as np
import pandas as pd

class GeneticOptimizer:
    """改进的自适应遗传算法
    """

    def __init__(
        self,
        init_pop,
        bounds,
        eval_func,
        selection_rate=0.95,
        max_cross=0.65,
        min_cross=0.4,
        max_mutate=0.075,
        min_mutate=0.05,
        a_coef=0.002,
        min_fitness=8.0,
    ):
        # 参数保存
        self.eval_func = eval_func
        self.selection_rate = selection_rate
        self.max_cross = max_cross
        self.min_cross = min_cross
        self.max_mutate = max_mutate
        self.min_mutate = min_mutate
        self.a_coef = a_coef
        self.min_fitness = min_fitness

        # 数据与边界
        self.population = np.array(init_pop, dtype=int)
        self.bounds = np.array(bounds, dtype=int)
        self.feature_size = self.bounds.shape[0]

        if self.population.shape[1] != self.feature_size:
            raise ValueError(
                f"变量数量不匹配！\n"
                f"- 种群维度: {self.population.shape[1]}\n"
                f"- 边界数量: {self.feature_size}"
            )

        # 边界合法性检查
        for i, (min_val, max_val) in enumerate(self.bounds):
            if min_val > max_val:
                raise ValueError(f"边界{i}范围错误：{min_val} > {max_val}")
            col_data = self.population[:, i]
            if np.min(col_data) < min_val or np.max(col_data) > max_val:
                raise ValueError(f"初始种群第{i}列越界[{min_val}, {max_val}]")

        # 状态
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.generation = 0

    def _dynamic_rate(self, current_fitness, max_rate, min_rate, avg_fitness):
        """动态调整概率核心方法"""
        delta = self.a_coef * (avg_fitness - current_fitness)
        try:
            return min_rate + (max_rate - min_rate) / (1 + math.exp(delta))
        except OverflowError:
            return min_rate

    def select(self):
        """改进版：从多个候选中选择最优或次优父代"""
        selected = []
        pop_size = len(self.population)

        for _ in range(2):  # 选择两个父代
            pool_size = min(4, pop_size)
            replace = pop_size < 4

            # 从种群中随机抽取候选个体
            indices = np.random.choice(pop_size, size=pool_size, replace=replace)
            candidates = [self.population[i] for i in indices]

            # 对候选个体进行适应度排序（从高到低）
            candidates.sort(key=self.eval_func, reverse=True)

            # 以概率选择最优或次优
            if random.random() < self.selection_rate:
                winner = candidates[0]  # 最优
            else:
                winner = candidates[1] if len(candidates) > 1 else candidates[0]  # 次优

            selected.append(winner)

        return selected

    def crossover(self, parent1, parent2, avg_fitness):
        """自适应交叉操作"""
        f1 = self.eval_func(parent1)
        f2 = self.eval_func(parent2)
        cross_rate = self._dynamic_rate(
            max(f1, f2),
            self.max_cross,
            self.min_cross,
            avg_fitness
        )

        if random.random() < cross_rate:
            pt = random.randint(1, self.feature_size - 1)
            child = np.concatenate([parent1[:pt], parent2[pt:]])
        else:
            child = parent1.copy() if f1 >= f2 else parent2.copy()
        return child

    def mutate(self, individual, avg_fitness):
        """通用变异操作"""
        fitness = self.eval_func(individual)

        mutate_rate = self._dynamic_rate(
            fitness,
            self.max_mutate,
            self.min_mutate,
            avg_fitness
        )

        if random.random() < mutate_rate:
            pos = random.randint(0, self.feature_size - 1)
            min_val, max_val = self.bounds[pos]
            old_val = individual[pos]

            possible_values = list(range(min_val, max_val + 1))
            if len(possible_values) > 1:
                possible_values.remove(int(old_val))
                individual[pos] = random.choice(possible_values)

        return individual

    def evolve(self, generations):
        """执行进化流程，每代记录所有子代的基因型与适应度"""
        for _ in range(generations):
            new_pop = []
            gen_records = []  # 当前代的基因型 + 适应度记录

            avg_fitness = np.mean([self.eval_func(ind) for ind in self.population])

            while len(new_pop) < len(self.population):
                # 选择
                parents = self.select()

                # 交叉
                child = self.crossover(parents[0], parents[1], avg_fitness)

                # 变异
                child = self.mutate(child, avg_fitness)

                fitness = self.eval_func(child)

                if fitness > self.min_fitness:
                    new_pop.append(child)

                    # 记录子代基因型和适应度
                    gen_records.append({
                        "generation": self.generation + 1,
                        "genotype": list(map(int, child)),  # 转换为普通列表
                        "fitness": float(fitness)
                    })

            # 更新种群
            self.population = np.array(new_pop, dtype=int)
            self.generation += 1

            # 更新全局最优适应度
            current_best = max(self.eval_func(ind) for ind in self.population)
            self.best_fitness = max(self.best_fitness, current_best)

            # 添加当前代的所有记录
            self.fitness_history.extend(gen_records)

            # 打印进度
            print(
                f"Generation {self.generation} current_best = {current_best:.2f} best_fitness = {self.best_fitness:.2f}"
            )

    def export_fitness_history(self, filename):
        """将历史基因型和适应度导出为CSV，并额外导出基因型去重后的CSV"""
        # 兼容 Path 对象
        filename = str(filename)

        flat_records = []
        for record in self.fitness_history:
            flat = {
                "generation": record["generation"],
                "fitness": record["fitness"]
            }
            for i, val in enumerate(record["genotype"]):
                flat[f"gene_{i + 1}"] = val
            flat_records.append(flat)

        df = pd.DataFrame(flat_records)
        df.to_csv(filename, index=False)
        print(f"Fitness history saved to {filename}")

        # 基因型去重（保留每个唯一基因型的第一条记录）
        gene_cols = [col for col in df.columns if col.startswith("gene_")]
        df_unique = df.drop_duplicates(subset=gene_cols, keep="first")

        # 构造去重文件名
        if filename.lower().endswith(".csv"):
            unique_filename = filename[:-4] + "_unique.csv"
        else:
            unique_filename = filename + "_unique"

        df_unique.to_csv(unique_filename, index=False)
        print(f"Unique genotypes saved to {unique_filename}")
