# Python code to compute growth and plot the chart for the user's scenario.
# It will display a table and save a PNG plot to /mnt/data for download.
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters from the user
pv = 4000.0          # initial investment in EUR
r = 0.10             # interest per period (6 months) = 10%
periods_per_year = 2
years = 40

# Compute balances for each half-year period
total_periods = years * periods_per_year
periods = list(range(total_periods + 1))  # include period 0
balances = [pv * ((1 + r) ** n) for n in periods]
time_years = [n / periods_per_year for n in periods]

# Create DataFrame
df = pd.DataFrame({
    "Period (half-years)": periods,
    "Time (years)": time_years,
    "Balance (€)": balances
})

# Also create a yearly summary (end of each year)
yearly_periods = [p * periods_per_year for p in range(years + 1)]
df_yearly = df[df["Period (half-years)"].isin(yearly_periods)].reset_index(drop=True)

# Display the detailed table to the user (interactive)

# Plot the growth (single plot, matplotlib, no explicit color)
plt.figure(figsize=(8,5))
plt.plot(df["Time (years)"], df["Balance (€)"], marker='o')
plt.title(f"Rast investície: 4 000 € pri 10% za 6 mesiacov (zložené), {years} rokov")
plt.xlabel("Čas (roky)")
plt.ylabel("Hodnota portfólia (€)")
plt.grid(True)
plt.xticks(range(0, years+1))
plt.tight_layout()

# Save the figure for download
output_path = Path.cwd() / "results/investment_growth_plot.png"
output_path = output_path.resolve()
plt.savefig(output_path, dpi=150)

# Also show the yearly summary table
plt.show()

# Print path to the saved figure so the notebook output includes it
output_path

