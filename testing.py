import scipy.io
import numpy as np
from scipy.signal import find_peaks

# Load data
data = scipy.io.loadmat(r'C:\Users\Vishal\OneDrive\Desktop\testing\archive\part_12.mat')
p = data['p']

# Process the first sample (you can loop over more)
ppg = p[0][0].flatten()  # Row 0: PPG
abp = p[0][1].flatten()  # Row 1: BP

# Pulse Amplitude = max - min of PPG
pulse_amplitude = np.max(ppg) - np.min(ppg)

# Estimate Heart Rate from PPG peaks
peaks, _ = find_peaks(ppg, distance=50)  # adjust distance as needed
heart_rate = len(peaks) * 60 / len(ppg) * 125  # 125 Hz sampling rate assumed

# Estimate true BP from ABP
systolic_true = np.max(abp)
diastolic_true = np.min(abp)

# Calculate your formula-based BP
# Improved Formula based on model coefficients
systolic_est = 0.984* pulse_amplitude + (-.356) * heart_rate + 27.575
diastolic_est = -0.001 * pulse_amplitude + (-0.009) * heart_rate + 0.935

# Print all values
print(f"PulseAmplitude: {pulse_amplitude:.2f}")
print(f"HeartRate: {heart_rate:.2f} bpm")
print(f"True Systolic: {systolic_true:.2f} | Estimated: {systolic_est:.2f}")
print(f"True Diastolic: {diastolic_true:.2f} | Estimated: {diastolic_est:.2f}")
