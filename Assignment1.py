import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: LOAD DATA FROM FILE
# =============================================================================
try:
    # Attempts to open and reads the data file
    with open('banknote_data.csv', 'r') as f:
        # Read header row and convert to numpy array
        headers = np.array(f.readline().strip().split(','))

        # Load numerical data using numpy's built-in parser
        data = np.genfromtxt(f, delimiter=',')
except FileNotFoundError:
    # Error handling for missing file
    print("Error: Required file 'banknote_data.csv' not found.")
    print("Please ensure the file is in the same directory as this script.")
    exit(1)

# =============================================================================
# SECTION 2: SPLIT DATA INTO COMPONENTS
# =============================================================================
# Separates features (measurements) and labels (classifications)
# Features: First 4 columns as floating point numbers
# Labels: Last column as whole numbers (0=real, 1=fake)
features = data[:, :-1].astype(np.float64)
labels = data[:, -1].astype(np.int64)

# =============================================================================
# SECTION 3: CALCULATE STATISTICS
# =============================================================================
# Computes statistics using vectorized numpy operations (no loops!)
minimums = np.min(features, axis=0)
maximums = np.max(features, axis=0)
averages = np.mean(features, axis=0)
medians = np.median(features, axis=0)

# Prints formatted table of results
print("\n" + "="*65)
print("BANKNOTE ANALYSIS REPORT")
print("="*65)
print(f"{'Feature':<12} | {'Minimum':>10} | {'Maximum':>10} | {'Average':>10} | {'Median':>10}")
print("-"*65)
for i in range(features.shape[1]):
    print(f"{headers[i]:<12} | "
          f"{minimums[i]:>10.2f} | "
          f"{maximums[i]:>10.2f} | "
          f"{averages[i]:>10.2f} | "
          f"{medians[i]:>10.2f}")
print("="*65)

# =============================================================================
# SECTION 4: CREATE VISUALIZATIONS
# =============================================================================
# -------------------------
# Plot 1: Scatter Plot
# -------------------------
plt.figure(figsize=(10, 6))

# Creates separates plots for real vs fake banknotes
for class_value, plot_color in [(0, 'blue'), (1, 'orange')]:
    # Filter data for current class
    class_mask = labels == class_value

    # Creates scatter plot points
    plt.scatter(features[class_mask, 0],  # X-axis: First feature
                features[class_mask, 3],  # Y-axis: Fourth feature
                c=plot_color,
                label=f'Class {class_value}',
                alpha=0.7,
                edgecolor='white')

# Adds labels and formatting
plt.title(f'Feature Comparison: {headers[0]} vs {headers[3]}')
plt.xlabel(headers[0])
plt.ylabel(headers[3])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------
# Plot 2: Bar Chart
# -------------------------
plt.figure(figsize=(10, 6))

# Creates bar positions and labels
x_positions = np.arange(len(headers)-1)
plt.bar(x_positions, averages,
        tick_label=headers[:-1],
        color=['blue', 'orange', 'green', 'red'])

# Add labels and formatting
plt.title('Average Feature Values Comparison')
plt.xlabel('Measurement Features')
plt.ylabel('Average Value')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()