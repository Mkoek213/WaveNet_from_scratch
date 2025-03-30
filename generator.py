import numpy as np
import pandas as pd

# Set the number of samples
num_samples = 350000

# Generate time values (incrementing by 0.0001)
time = np.round(np.arange(0.0001, (num_samples + 1) * 0.0001, 0.0001), 4)

# Generate signal values (random floating points within a reasonable range)
signal = np.round(np.random.uniform(-3.5, 3.5, num_samples), 4)

# Generate open_channels values (random integers from 0 to 10)
open_channels = np.random.choice([0] * 90 + list(range(1, 11)), num_samples)  # Mostly 0, some values up to 10

# Create DataFrame
df = pd.DataFrame({'time': time, 'signal': signal, 'open_channels': open_channels})

# Save to CSV
df.to_csv("synthetic_signal_data.csv", index=False)

print("Generated 350,000 samples and saved to synthetic_signal_data.csv")