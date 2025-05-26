import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

data = [None, None, None, 509826, 557688, 584188, 591495, 581300, 577400, 546444, 557822, 581044, 649127
]

# Create indices for all points
indices = list(range(len(data)))

# Separate known points from missing points
known_indices = [i for i, val in enumerate(data) if val is not None]
known_values = [data[i] for i in known_indices]
missing_indices = [i for i, val in enumerate(data) if val is None]

# For extrapolation, we'll use a polynomial fit
# Let's try different degrees to find a good balance
degrees = [3, 4, 5]
extrapolations = {}
polynomials = {}

for degree in degrees:
    if degree < len(known_indices):  # Ensure we have enough points for the degree
        # Fit polynomial of the specified degree
        poly_coeffs = np.polyfit(known_indices, known_values, degree)
        poly = np.poly1d(poly_coeffs)
        polynomials[degree] = poly
        
        # Find the values for missing indices (beginning)
        for idx in missing_indices:
            if idx not in extrapolations:
                extrapolations[idx] = {}
            extrapolations[idx][degree] = poly(idx)
        
        # Extrapolate future values (next 5 points)
        future_indices = range(len(data), len(data) + 5)
        for idx in future_indices:
            if idx not in extrapolations:
                extrapolations[idx] = {}
            extrapolations[idx][degree] = poly(idx)

# Let's use degree 4 as our primary model
primary_degree = 4

# Fill in the missing values at the beginning using the primary model
for idx in missing_indices:
    data[idx] = extrapolations[idx][primary_degree]

# Prepare the extrapolated future values
future_indices = range(len(data), len(data) + 5)
future_values = [extrapolations[idx][primary_degree] for idx in future_indices]

# Display the results
print("Extrapolated missing values at the beginning:")
for idx in missing_indices:
    print(f"Position {idx+1}: {extrapolations[idx][primary_degree]:.3f}")

print("\nExtrapolated future values:")
for i, idx in enumerate(future_indices):
    print(f"Position {idx+1}: {extrapolations[idx][primary_degree]:.3f}")

print("\nComplete time series with extrapolated values:")
full_data = data + future_values
full_indices = indices + list(future_indices)
for i, val in enumerate(full_data):
    print(f"Position {i+1}: {val:.3f}")


# Compare different polynomial degrees for extrapolation
print("\nComparison of different polynomial degrees for extrapolation:")
print("\nMissing beginning values:")
for idx in missing_indices:
    print(f"Position {idx+1}:", end=" ")
    for degree in degrees:
        if degree in polynomials:
            print(f"Degree {degree}: {extrapolations[idx][degree]:.3f}", end=" | ")
    print()