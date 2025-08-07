import sys
import importlib.util
import os

# Point to the actual app.py file inside 'load-predictor'
module_path = os.path.join("load-predictor", "app.py")

# Load it as a module manually
spec = importlib.util.spec_from_file_location("app", module_path)
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)

# Now access FCNModelPredictor
FCNModelPredictor = app_module.FCNModelPredictor

# Proceed with plotting
import matplotlib.pyplot as plt

predictor = FCNModelPredictor()
df = predictor.training_data

plt.scatter(df['model_a_tokens'], df['cpu_usage'], alpha=0.5)
plt.xlabel('Model A Tokens')
plt.ylabel('CPU Usage')
plt.title('Model A Tokens vs CPU Usage')
plt.grid(True)
plt.show()

