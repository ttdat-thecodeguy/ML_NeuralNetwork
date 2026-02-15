from sklearn.linear_model import LinearRegression
from load_dataset import load_dataset
from matplotlib import pyplot as plt
import numpy as np

# y=b0​+b1​x1​+b2​x2​+b3​x3
# Y = b0 + b1X 1+ b2X2
	
# Với 
	
# Dataset 1: B1 = 2, b2 = 3, y = 5
# Ta có: 
	# 5 = b0 + 2b 1+ 3b2 (1)
	# Dataset 2: B1 = 1, b2 = 2, y = 7
	# 7 = b0 + 1b 1+ 2b2 (2)
	# Dataset 3: B1 = 2, b2 = 5, y= 9
	# (1-2) 
	# (5−7)=(b0​+2b1​+3b2​)−(b0​+b1​+2b2​)
	# -2 = b1 + b2
	# B1 = b2 + 2
	
	# Thế vào 2 => b0 = 9 - b2
	# ​b0 = 7, b1 = -4, b2 = -2

df = load_dataset()
X =df.drop('ielts_score', axis=1)
y = df['ielts_score']

model = LinearRegression()
model.fit(X, y)

# Predicting IELTS scores based on study hours per day
X_pred = df['study_hours_per_day']
Y_pred = model.predict(X)

# Visualize the results
plt.figure(figsize=(8,6)) 
plt.scatter(X_pred, y, color='blue', label='Total Study Hours') 
plt.plot(X_pred, Y_pred, color='red', linewidth=2, label='IELTS Score Prediction') 
plt.title('Linear Regression on Random Dataset')
plt.xlabel('Study Hours per Day')
plt.ylabel('IELTS Score')
plt.legend()
plt.grid(True)
plt.show()