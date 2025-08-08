import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
import joblib
import os

# 读取CSV数据
df = pd.read_csv(r"data\S_CH4_N2\Training_data.csv")

# 选取特征与目标
features = ['metal', 'linker1', 'linker1_F', 'linker2', 'linker2_F', 'topology']
target = 'S_CH4_N2'

# 丢弃含缺失的行
df = df.dropna(subset=features + [target])

X = df[features].to_numpy()
y = df[target].to_numpy()

# -----------------------------
# 5折交叉验证
# -----------------------------
print("===== 5折交叉验证评估 =====")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_list = []
cv_mae_list = []
cv_srcc_list = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_cv_train, X_cv_test = X[train_index], X[test_index]
    y_cv_train, y_cv_test = y[train_index], y[test_index]

    model = XGBRegressor(random_state=42)
    model.fit(X_cv_train, y_cv_train)

    y_cv_pred = model.predict(X_cv_test)

    r2 = r2_score(y_cv_test, y_cv_pred)
    mae = mean_absolute_error(y_cv_test, y_cv_pred)
    srcc, _ = spearmanr(y_cv_test, y_cv_pred)

    cv_r2_list.append(r2)
    cv_mae_list.append(mae)
    cv_srcc_list.append(srcc)

    print(f"Fold {fold + 1}: R²={r2:.4f}, MAE={mae:.4f}, SRCC={srcc:.4f}")

print("\n平均交叉验证结果：")
print(f"Mean R²:   {np.mean(cv_r2_list):.4f}")
print(f"Mean MAE:  {np.mean(cv_mae_list):.4f}")
print(f"Mean SRCC: {np.mean(cv_srcc_list):.4f}\n")

# -----------------------------
# 常规训练测试划分评估
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
srcc_train, _ = spearmanr(y_train, y_train_pred)

y_test_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
srcc_test, _ = spearmanr(y_test, y_test_pred)

print("训练集评估:")
print(f"  R²:   {r2_train:.4f}")
print(f"  MAE:  {mae_train:.4f}")
print(f"  SRCC: {srcc_train:.4f}")

print("\n测试集评估:")
print(f"  R²:   {r2_test:.4f}")
print(f"  MAE:  {mae_test:.4f}")
print(f"  SRCC: {srcc_test:.4f}")

# -----------------------------
# 可视化预测性能图（含分布曲线）
# -----------------------------
def plot_performance(true, pred, title, color):
    sns.set(style="white", color_codes=True)
    g = sns.jointplot(x=true, y=pred, kind="reg", color=color,
                      marginal_kws=dict(bins=25, fill=True),
                      line_kws={"color": "black", "lw": 1})
    g.fig.suptitle(title, fontsize=14)
    g.set_axis_labels("True", "Predicted", fontsize=12)
    plt.subplots_adjust(top=0.9)
    plt.show()

plot_performance(y_train, y_train_pred,
                 f"Train Set (R²={r2_train:.2f}, MAE={mae_train:.2f}, SRCC={srcc_train:.2f})",
                 color="blue")

plot_performance(y_test, y_test_pred,
                 f"Test Set (R²={r2_test:.2f}, MAE={mae_test:.2f}, SRCC={srcc_test:.2f})",
                 color="green")

# -----------------------------
# 保存结果和模型
# -----------------------------
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# 保存训练集预测结果
train_df = pd.DataFrame({
    'True': y_train,
    'Predicted': y_train_pred
})
train_df.to_csv(os.path.join(output_dir, f"{target}_train_predictions.csv"), index=False)

# 保存测试集预测结果
test_df = pd.DataFrame({
    'True': y_test,
    'Predicted': y_test_pred
})
test_df.to_csv(os.path.join(output_dir, f"{target}_test_predictions.csv"), index=False)

# 保存模型
model_path = os.path.join(output_dir, f"xgb_model_{target}.pkl")
joblib.dump(model, model_path)

print(f"\n✅ 已保存模型到：{model_path}")