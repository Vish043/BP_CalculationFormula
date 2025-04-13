import scipy.io
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Define the list of dataset file paths
file_paths = [
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_1.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_2.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_3.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_4.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_5.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_6.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_7.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_8.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_9.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_10.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_11.mat',
    r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_12.mat',
    # Add paths to other 8 files similarly
]

# Initialize lists to hold features and target values for all datasets
features_list = []
targets_list = []

# Loop over each file path and process the data
for file_path in file_paths:
    # Load .mat file
    mat_data = scipy.io.loadmat(file_path)
    
    # Extract the data in 'p' variable
    p = mat_data['p']
    
    # Extract ABP signal (assuming it's the second signal)
    abp_signal = np.array(p[0][1])
    
    # Calculate systolic and diastolic BP values
    systolic_value = np.max(abp_signal)
    diastolic_value = np.min(abp_signal)
    
    # Extract PPG signal (assuming it's the first signal)
    ppg_signal = np.array(p[0][0]).ravel()  # Flatten to ensure it's 1D
    
    # Feature extraction from PPG
    pulse_amplitude = np.max(ppg_signal) - np.min(ppg_signal)
    peaks, _ = find_peaks(ppg_signal, distance=50)
    heart_rate = len(peaks) * 60 / len(ppg_signal) * 125  # in beats per minute
    
    # Append the features and targets to the lists
    features_list.append([pulse_amplitude, heart_rate])
    targets_list.append([systolic_value, diastolic_value])

# Convert features and targets to numpy arrays
X = np.array(features_list)
y = np.array(targets_list)

# Train Linear Regression models
model_sys = LinearRegression()
model_dia = LinearRegression()

model_sys.fit(X, y[:, 0])  # Systolic BP model
model_dia.fit(X, y[:, 1])  # Diastolic BP model

y_pred_sys = model_sys.predict(X)
y_pred_dia = model_dia.predict(X)

# Calculate accuracy metrics
mae_sys = mean_absolute_error(y[:, 0], y_pred_sys)
mae_dia = mean_absolute_error(y[:, 1], y_pred_dia)

mape_sys = mean_absolute_percentage_error(y[:, 0], y_pred_sys)
mape_dia = mean_absolute_percentage_error(y[:, 1], y_pred_dia)

r2_sys = r2_score(y[:, 0], y_pred_sys)
r2_dia = r2_score(y[:, 1], y_pred_dia)

# Output learned formulas for systolic and diastolic BP
systolic_formula = f"systolic = {model_sys.coef_[0]:.3f} * pulse_amplitude + {model_sys.coef_[1]:.3f} * heart_rate + {model_sys.intercept_:.3f}"
diastolic_formula = f"diastolic = {model_dia.coef_[0]:.3f} * pulse_amplitude + {model_dia.coef_[1]:.3f} * heart_rate + {model_dia.intercept_:.3f}"

print("\nSystolic BP Formula:", systolic_formula)
print("Diastolic BP Formula:", diastolic_formula)

print("\n--- Accuracy Metrics ---")
print(f"Mean Absolute Error (Systolic): {mae_sys:.2f}")
print(f"Mean Absolute Error (Diastolic): {mae_dia:.2f}")

print(f"MAPE (Systolic): {mape_sys*100:.2f}%")
print(f"MAPE (Diastolic): {mape_dia*100:.2f}%")

print(f"R² Score (Systolic): {r2_sys:.3f}")
print(f"R² Score (Diastolic): {r2_dia:.3f}")
