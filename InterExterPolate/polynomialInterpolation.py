import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

# Given time series data with missing values (marked as x)
data = [160618, 160464, 180744, 230686, None, 301792, 331847, 287253, 300155, 319538, 312880, 343707, 336570]


# Create indices for all points
indices = list(range(len(data)))

# Separate known points from missing points
known_indices = [i for i, val in enumerate(data) if val is not None]
known_values = [data[i] for i in known_indices]
missing_indices = [i for i, val in enumerate(data) if val is None]

# Perform polynomial interpolation of degree 4
poly = lagrange(known_indices, known_values)

# Calculate the missing values
missing_values = [poly(i) for i in missing_indices]

# Fill in the missing values
for idx, val in zip(missing_indices, missing_values):
    data[idx] = val

# Verify with a higher degree polynomial for comparison
poly_higher = np.polyfit(known_indices, known_values, 4)
polynomial = np.poly1d(poly_higher)

# Calculate values using the polynomial
interp_values = [polynomial(i) for i in missing_indices]
print("\nVerification with numpy's polyfit (degree 4):")
for idx, val in zip(missing_indices, interp_values):
    print(f"Position {idx+1}: {val:.3f}")