import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#question 1 
py_array=[1,2,3,4,5,6,7]
print(type(py_array))
np_array=np.array(py_array)
print(type(np_array))
pd_series=pd.Series(py_array)
(type(pd_series))


# #question 2
# nd_arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
# nd_series=pd.Series([[1,2,3],[4,5,6],[7,8,9]])
# flattened_Array1=nd_arr.flatten()
# print(flattened_Array1)
# flattened_Array2=nd_series.explode()
# print(flattened_Array2)

#question 3

import numpy as np
import matplotlib.pyplot as plt

# Given points (x, y)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Best fit line
w, b = np.polyfit(x, y, 1)
y_pred = w * x + b

# Mean Absolute Error
mae = np.mean(np.abs(y_pred - y))
print(f"Mean Absolute Error: {mae}")
print(f"Best Fit Line Values: {y_pred}")

# Plot actual points
plt.scatter(x, y, marker='x', label="Actual Points")

# Plot predicted line
plt.plot(x, y_pred, label="Predicted Line")

# 🔹 ADD THIS: dotted perpendiculars (residuals)
for xi, yi, ypi in zip(x, y, y_pred):
    plt.plot([xi, xi], [yi, ypi], linestyle='dotted')

# Labels
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Actual vs Predicted Line with Residuals")
plt.legend()

plt.show()
