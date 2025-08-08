import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------
# 输入的特征值（保持顺序一致）
input_features = [1, 32, 18, 18, 11, 2]

# 模型路径
model_path = "models/xgb_model_S_CH4_N2.pkl"
model = joblib.load(model_path)
feature_names = ['Node', 'Linker 1', 'FG 1', 'Linker 2', 'FG 2', 'Topology']
X_sample = pd.DataFrame([input_features], columns=feature_names)
# 计算 SHAP 值
explainer = shap.Explainer(model)
shap_values = explainer(X_sample)
plt.figure(figsize=(6, 8))
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.show()
