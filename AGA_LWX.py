import time
import math
import joblib
import random
import numpy as np
import pandas as pd
from pathlib import Path

# ================== 全局配置 ==================
A = 0.002  # 自适应系数
FEATURE_SIZE = 6  # 特征数量
DEFAULT_BOUNDS = [  # 变量边界定义
    (1, 6),
    (1, 50),
    (0, 28),
    (1, 50),
    (0, 28),
    (1, 158)
]
MIN_FITNESS = 8  # 最小适应度阈值

# ================== 路径配置 ==================


RESULTS_DIR = Path("results")

DATA_PATH = r"database\GA100.csv"  # 数据文件路径
MODEL_PATH = "model_output/xgb_model.pkl"  # 模型文件路径

# # 读取合法组合 CSV（列名应为：metal, linker1, linker2, topology）
# valid_df = pd.read_csv("combination_counts.csv")
#
# # 将所有合法组合存入集合，便于高效查找
# # 每行转为元组：(metal, linker1, linker2, topology)
# valid_combinations = set(
#     tuple(row) for row in valid_df[['metal', 'linker1', 'linker2', 'topology']].values
# )

# 自动创建目录

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ================== 算法核心类 ==================
class GeneticOptimizer:
    """改进的自适应遗传算法"""

    def __init__(self, init_pop, bounds, eval_func,
                 selection_rate=0.95,
                 max_cross=0.65,
                 min_cross=0.4,
                 max_mutate=0.075,
                 min_mutate=0.05):

        # 输入验证
        self.population = np.array(init_pop)
        self.bounds = np.array(bounds)

        if self.population.shape[1] != self.bounds.shape[0]:
            raise ValueError(
                f"变量数量不匹配！\n"
                f"- 种群维度: {self.population.shape[1]}\n"
                f"- 边界数量: {self.bounds.shape[0]}"
            )

        # 边界合法性检查
        for i, (min_val, max_val) in enumerate(self.bounds):
            if min_val > max_val:
                raise ValueError(f"边界{i}范围错误：{min_val} > {max_val}")
            col_data = self.population[:, i]
            if np.min(col_data) < min_val or np.max(col_data) > max_val:
                raise ValueError(f"初始种群第{i}列越界[{min_val}, {max_val}]")

        # 算法参数
        self.eval_func = eval_func
        self.selection_rate = selection_rate
        self.max_cross = max_cross
        self.min_cross = min_cross
        self.max_mutate = max_mutate
        self.min_mutate = min_mutate

        # 状态跟踪
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.generation = 0

    def _dynamic_rate(self, current_fitness, max_rate, min_rate, avg_fitness):
        """动态调整概率核心方法"""
        delta = A * (avg_fitness - current_fitness)
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
        # 计算交叉率
        f1 = self.eval_func(parent1)
        f2 = self.eval_func(parent2)
        cross_rate = self._dynamic_rate(
            max(f1, f2),
            self.max_cross,
            self.min_cross,
            avg_fitness
        )

        # 执行交叉
        if random.random() < cross_rate:
            pt = random.randint(1, FEATURE_SIZE - 1)
            child = np.concatenate([parent1[:pt], parent2[pt:]])
        else:
            child = parent1.copy() if f1 >= f2 else parent2.copy()
        return child

    def mutate(self, individual, avg_fitness):
        """通用变异操作（无互穿约束，确保变异发生）"""
        fitness = self.eval_func(individual)

        mutate_rate = self._dynamic_rate(
            fitness,
            self.max_mutate,
            self.min_mutate,
            avg_fitness
        )

        if random.random() < mutate_rate:
            pos = random.randint(0, FEATURE_SIZE - 1)
            min_val, max_val = self.bounds[pos]
            old_val = individual[pos]

            # 获取除当前值以外的可能变异值
            possible_values = list(range(min_val, max_val + 1))
            if len(possible_values) > 1:
                possible_values.remove(old_val)
                individual[pos] = random.choice(possible_values)
            # 如果 min_val == max_val（无法变异），直接跳过

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

                # combo_key = (child[0], child[1], child[3], child[5])
                # 精英保留
                # 若组合不合法，则强制设适应度为0
                # if combo_key not in valid_combinations:
                #     fitness = 0
                # else:
                #     fitness = self.eval_func(child)

                # if child[1] == 34 and child[2] != 0 :
                #     fitness = 0
                fitness = self.eval_func(child)

                if fitness > MIN_FITNESS:
                    new_pop.append(child)

                    # 记录子代基因型和适应度
                    gen_records.append({
                        "generation": self.generation + 1,
                        "genotype": list(child),  # 转换为普通列表
                        "fitness": fitness
                    })

            # 更新种群
            self.population = np.array(new_pop)
            self.generation += 1

            # 更新全局最优适应度
            current_best = max(self.eval_func(ind) for ind in self.population)
            self.best_fitness = max(self.best_fitness, current_best)

            # 添加当前代的所有记录
            self.fitness_history.extend(gen_records)

            # 打印进度
            print(
                f"Generation {self.generation} current_best = {current_best:.2f} best_fitness = {self.best_fitness:.2f}")


    def export_fitness_history_to_csv(self, filename):
        """将历史基因型和适应度导出为CSV"""
        # 拍平 genotype 中的列表为独立列
        flat_records = []
        for record in self.fitness_history:
            flat = {
                "generation": record["generation"],
                "fitness": record["fitness"]
            }
            for i, val in enumerate(record["genotype"]):
                flat[f"gene_{i+1}"] = val
            flat_records.append(flat)

        df = pd.DataFrame(flat_records)
        df.to_csv(filename, index=False)
        print(f"Fitness history saved to {filename}")

# ================== 主流程 ==================
def main(max_generations, pop_size, output_dir=RESULTS_DIR):
    """主控制流程"""
    try:
        start_time = time.time()


        model = joblib.load(MODEL_PATH)
        data = pd.read_csv(DATA_PATH, header=None)

        init_pop = data.sample(
            n=pop_size,
            replace=pop_size > len(data),
            random_state=42
        ).values[:, :FEATURE_SIZE]

        # 4. 创建优化器
        optimizer = GeneticOptimizer(
            init_pop=init_pop,
            bounds=DEFAULT_BOUNDS,
            eval_func=lambda x: model.predict(x.reshape(1, -1))[0]
        )

        # 5. 执行进化
        print("\n启动进化流程...")
        optimizer.evolve(max_generations)

        result_file = output_dir / f"result_g{max_generations}_p{pop_size}.csv"
        optimizer.export_fitness_history_to_csv(result_file)


        # 7. 最终报告
        print(f"\n优化完成！耗时 {time.time() - start_time:.2f} 秒")
        print(f"最佳适应度: {optimizer.best_fitness:.2f}")
        print(f"结果保存至: {result_file}")

    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 参数配置
    main(
        max_generations=200,  # 进化代数
        pop_size=100  # 种群规模
    )