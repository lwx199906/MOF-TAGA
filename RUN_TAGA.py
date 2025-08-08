import time
import joblib
import pandas as pd
from pathlib import Path
from TAGA4MOF import GeneticOptimizer

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
RESULTS_DIR = Path("results/test")
DATA_PATH = r"data/S_CH4_N2/initial_top_100.csv"  # 数据文件路径
MODEL_PATH = "models/xgb_model_S_CH4_N2.pkl"      # 模型文件路径
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

        # 创建优化器（把 a_coef / min_fitness 显式传给类，以与原始常量一致）
        optimizer = GeneticOptimizer(
            init_pop=init_pop,
            bounds=DEFAULT_BOUNDS,
            eval_func=lambda x: model.predict(x.reshape(1, -1))[0],
            a_coef=A,
            min_fitness=MIN_FITNESS,
        )

        print("\n启动进化流程...")
        optimizer.evolve(max_generations)

        result_file = output_dir / f"result_g{max_generations}_p{pop_size}.csv"
        optimizer.export_fitness_history(result_file)

        print(f"\n优化完成！耗时 {time.time() - start_time:.2f} 秒")
        print(f"最佳适应度: {optimizer.best_fitness:.2f}")
        print(f"结果保存至: {result_file}")

    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main(
        max_generations=200,  # 进化代数
        pop_size=100          # 种群规模
    )
