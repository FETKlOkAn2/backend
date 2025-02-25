import vectorbt as vbt
import numpy as np
import pandas as pd

# Define the hyperparameter ranges
rsi_windows = np.arange(5, 20, step=5)       # e.g. [5, 10, 15]
buy_thresholds = np.arange(20, 40, step=5)     # e.g. [20, 25, 30, 35]
sell_thresholds = np.arange(60, 80, step=5)    # e.g. [60, 65, 70, 75]

# For demonstration, create dummy performance data
# Assume the performance metric is total return (in %), so we build a 3D array:
#   Dimension 0: rsi_window, Dimension 1: buy_threshold, Dimension 2: sell_threshold
performance = np.random.uniform(0, 100, 
                size=(len(rsi_windows), len(buy_thresholds), len(sell_thresholds)))

# Prepare string labels for each axis
x_labels = [str(x) for x in rsi_windows]       # x-axis: RSI Window
y_labels = [str(y) for y in buy_thresholds]      # y-axis: Buy Threshold
z_labels = [str(z) for z in sell_thresholds]     # z-axis: Sell Threshold

# Create the 3D volume plot
volume = vbt.plotting.Volume(
    data=performance,
    x_labels=x_labels,
    y_labels=y_labels,
    z_labels=z_labels
)

# Optionally, update layout details (titles, axis labels, etc.)
volume.fig.update_layout(
    title='RSI Hyperoptimization Performance',
    scene=dict(
        xaxis_title='RSI Window',
        yaxis_title='Buy Threshold',
        zaxis_title='Sell Threshold'
    )
)

# Display the figure
volume.fig.show()
