import numpy as np
import matplotlib.pyplot as plt

# Define the linear equation parameters
intercept = 49.95
slope = -0.129



# Generate x values
x = np.linspace(0, 60, 100)  # Adjust range as needed
y = intercept + slope * x

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='y = 49.95 + -0.129x', color='blue')
plt.title('Simple Linear Regression Chart')
plt.xlabel('Screentime')
plt.ylabel('Wellbeing')
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()
