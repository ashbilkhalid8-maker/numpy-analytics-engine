import numpy as np

# 1. Create a fake "image" (Height=100, Width=100, Color Channels=3 for RGB)
# Filled with zeros (pure black image)
image = np.zeros((100, 100, 3), dtype=np.uint8)
print("Initial Image Shape:", image.shape)

# 2. Slice a region and make it Red 
# (Row 20 to 50, Column 30 to 70, Channel 0 which is Red = 255)
image[20:50, 30:70, 0] = 255

# 3. Apply Broadcasting: Make the whole image brighter by adding 50 
# (We convert to int16 first so values don't overflow past 255)
brighter_image = image.astype(np.int16) + 50
brighter_image = np.clip(brighter_image, 0, 255).astype(np.uint8)

print("Brighter image max value:", np.max(brighter_image))



# --- TASK 2: Financial Market Simulator ---

# 1. Create a random number generator using the modern NumPy API
rng = np.random.default_rng(seed=42)

# 2. Simulate daily stock returns for 5 different stocks over 250 trading days
# Shape: (250 days, 5 stocks). Values represent daily percentage changes.
stock_returns = rng.normal(loc=0.001, scale=0.02, size=(250, 5))
print("\nStock Returns Shape:", stock_returns.shape)

# 3. Calculate aggregate statistics across axes
# axis=0 means operations down the rows (per stock across all days)
# axis=1 means operations across the columns (per day across all stocks)
mean_daily_return_per_stock = np.mean(stock_returns, axis=0)
volatility_per_stock = np.std(stock_returns, axis=0)

print("Mean Daily Returns per Stock:", mean_daily_return_per_stock)
print("Volatility (Std Dev) per Stock:", volatility_per_stock)

# 4. Use boolean masking to find "extreme market days" (returns > 5% or < -5%)
extreme_days = np.abs(stock_returns) > 0.05
print("Total extreme movement events across all stocks:", np.sum(extreme_days))



# --- TASK 3: Linear Algebra & Matrix Operations ---

# 1. Transpose the stock returns matrix using .T
# This flips the shape from (250 days, 5 stocks) to (5 stocks, 250 days)
transposed_returns = stock_returns.T
print("\nTransposed Returns Shape:", transposed_returns.shape)

# 2. Calculate the Covariance Matrix using np.cov()
# This measures how the 5 different stocks move together
cov_matrix = np.cov(stock_returns, rowvar=False)
print("Covariance Matrix Shape:", cov_matrix.shape)

# 3. Matrix Multiplication using the @ operator
# Calculate a simple dot product matrix of the transposed returns with itself
# (Dimensions: (5, 250) @ (250, 5) -> (5, 5))
dot_product_matrix = transposed_returns @ stock_returns
print("Dot Product Matrix Shape:", dot_product_matrix.shape)

# 4. Find the minimum and maximum values and their indices
max_return_value = np.max(stock_returns)
max_return_index = np.argmax(stock_returns)
print(f"Maximum single-day return ({max_return_value:.4f}) is at flattened index: {max_return_index}")