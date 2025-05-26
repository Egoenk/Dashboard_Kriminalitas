import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#ganti disini
data = [None, 1195816, 1212681, 1244428, 1227980, 1260586, 1276269, 1332184, 1378916, 1393618, 1411036, 1428398, 1445796
]

# Create indices for all points
indices = list(range(len(data)))

# Separate known points from missing points
known_indices = [i for i, val in enumerate(data) if val is not None]
known_values = [data[i] for i in known_indices]
missing_indices = [i for i, val in enumerate(data) if val is None]

# Linear extrapolation using numpy's polyfit with degree=1
# We'll fit a linear function (ax + b) to the known data
poly_coeffs = np.polyfit(known_indices, known_values, 1)
linear_model = np.poly1d(poly_coeffs)

# Extrapolate missing values at the beginning
for idx in missing_indices:
    data[idx] = linear_model(idx)

# Extrapolate future values (next 5 points)
future_indices = range(len(data), len(data) + 5)
future_values = [linear_model(idx) for idx in future_indices]

# Display the results
print("Extrapolated missing values at the beginning (linear):")
for idx in missing_indices:
    print(f"Position {idx+1}: {data[idx]:.3f}")

print("\nExtrapolated future values (linear):")
for i, idx in enumerate(future_indices):
    print(f"Position {idx+1}: {future_values[i]:.3f}")

print("\nComplete time series with extrapolated values:")
full_data = data + future_values
full_indices = indices + list(future_indices)
for i, val in enumerate(full_data):
    print(f"Position {i+1}: {val:.3f}")


# Calculate the slope and explain the trend
slope = poly_coeffs[0]
if slope > 0:
    trend = "increasing"
else:
    trend = "decreasing"
    
print(f"\nLinear trend analysis:")
print(f"Slope: {slope:.2f}")
print(f"The time series shows a {trend} trend with approximately {abs(slope):.2f} units per time step.")