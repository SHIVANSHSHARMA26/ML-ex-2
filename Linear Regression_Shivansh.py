import numpy as np

ages = np.array([1, 42, 13, 25, 63, 15])  
areas = np.array([50.73, 41.83, 46.54, 58.27, 72.53, 51.47])  
prices = np.array([523902.67, 325104.45, 434919.86, 575719.18, 629274.54, 390576.98])

X = np.column_stack((ages, areas))
print(X)

y = prices
print(y)

w = np.linalg.inv(X.T @ X) @ X.T @ y

age_user = float(input("Enter the age of the new house (in years): "))
area_user = float(input("Enter the area of the new house (in square meters): "))
x_user = np.array([age_user, area_user])

y_pred = x_user @ w
print(y_pred)

# Now for Loss Function

y_real = float(input("Enter the real price of the new house for comparison (in euros): ")) 


l2_loss = (y_pred - y_real) ** 2
print("L2 Loss (Least-Squares):", l2_loss)

l1_loss = abs(y_pred - y_real)
print("L1 Loss:", l1_loss)