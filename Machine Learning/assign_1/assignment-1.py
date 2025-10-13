#!/usr/bin/env python3

# Name: Sayan Mondal
# Student ID: 24377372
# Strand: Future Networked Systems
# Course: MSc. Computer Science

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("week2.csv", skiprows=1, header=None)
df.columns = ["X1", "X2", "y"]

# Extract features and target
X1 = df["X1"].values
X2 = df["X2"].values
y = df["y"].values
X = np.column_stack((X1, X2))

# (a)(i) Data visualization
plt.figure(figsize=(10, 6))
plt.scatter(X1[y==1], X2[y==1], marker='+', color='blue', s=50, label="Class +1")
plt.scatter(X1[y==-1], X2[y==-1], marker='o', color='green', s=50, label="Class -1")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("(a)(i) Data Visualization")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (a)(ii) Train logistic regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Extract parameters
w1, w2 = logistic_model.coef_[0]
b = logistic_model.intercept_[0]

print(f"Logistic Regression Model Parameters:")
print(f"Model: h(x) = sign({b:.4f} + {w1:.4f}*x1 + {w2:.4f}*x2)")
print(f"Weights: w1={w1:.4f}, w2={w2:.4f}")
print(f"Bias: {b:.4f}")

# (a)(iii) Predictions on training data
y_pred_train = logistic_model.predict(X_train)

plt.figure(figsize=(10, 6))
# Plot actual training data
plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], 
           marker='+', color='blue', s=50, label="Actual +1")
plt.scatter(X_train[y_train==-1][:,0], X_train[y_train==-1][:,1], 
           marker='o', color='green', s=50, label="Actual -1")

# Plot predictions
plt.scatter(X_train[y_pred_train==1][:,0], X_train[y_pred_train==1][:,1], 
           marker='x', color='cyan', s=50, label="Predicted +1")
plt.scatter(X_train[y_pred_train==-1][:,0], X_train[y_pred_train==-1][:,1], 
           marker='s', color='orange', s=50, label="Predicted -1")

# Decision boundary: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
x_min, x_max = X_train[:,0].min(), X_train[:,0].max()
y_min_boundary = -(w1*x_min + b)/w2
y_max_boundary = -(w1*x_max + b)/w2

plt.plot([x_min, x_max], [y_min_boundary, y_max_boundary], 
         color='red', linewidth=2, label="Decision Boundary")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("(a)(iii) Logistic Regression: Training Data vs Predictions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# (a)(iv) Training accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Misclassified points: {np.sum(y_train != y_pred_train)}/{len(y_train)}")

# (b)(i) Train SVM with different C values
from sklearn.svm import LinearSVC
C_values = [0.001, 1, 100]
svm_models = {}

print("\n(b)(i) SVM Models with Different C Values:")
for C in C_values:
    model = LinearSVC(C=C, max_iter=10000)
    model.fit(X_train, y_train)
    svm_models[C] = model
    
    w1_svm, w2_svm = model.coef_[0]
    b_svm = model.intercept_[0]
    
    print(f"\nC = {C}:")
    print(f"  Model: h(x) = sign({b_svm:.4f} + {w1_svm:.4f}*x1 + {w2_svm:.4f}*x2)")
    print(f"  Weights: w1={w1_svm:.4f}, w2={w2_svm:.4f}")
    print(f"  Bias: {b_svm:.4f}")

# (b)(ii) Plot SVM results
for C, model in svm_models.items():
    y_pred_svm = model.predict(X_train)
    w1_svm, w2_svm = model.coef_[0]
    b_svm = model.intercept_[0]
    
    plt.figure(figsize=(10, 6))
    
    # Plot actual data
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], 
               marker='+', color='blue', s=50, label="Actual +1")
    plt.scatter(X_train[y_train==-1][:,0], X_train[y_train==-1][:,1], 
               marker='o', color='green', s=50, label="Actual -1")
    
    # Plot predictions
    plt.scatter(X_train[y_pred_svm==1][:,0], X_train[y_pred_svm==1][:,1], 
               marker='x', color='cyan', s=50, label="Predicted +1")
    plt.scatter(X_train[y_pred_svm==-1][:,0], X_train[y_pred_svm==-1][:,1], 
               marker='s', color='orange', s=50, label="Predicted -1")
    
    # Decision boundary
    x_min, x_max = X_train[:,0].min(), X_train[:,0].max()
    y_min_boundary = -(w1_svm*x_min + b_svm)/w2_svm
    y_max_boundary = -(w1_svm*x_max + b_svm)/w2_svm
    
    plt.plot([x_min, x_max], [y_min_boundary, y_max_boundary], 
             color='red', linewidth=2, label="Decision Boundary")
    
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"(b)(ii) SVM with C={C}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# (c)(i) Create polynomial features
X1_squared = X1**2
X2_squared = X2**2
X_extended = np.column_stack((X1, X2, X1_squared, X2_squared))

print("\n(c)(i) Extended Feature Set:")
print("Original features: X1, X2")
print("New features: X1², X2²")
print("Total features: 4")
print(f"Extended feature matrix shape: {X_extended.shape}")

# Train logistic regression with extended features
poly_model = LogisticRegression()
poly_model.fit(X_extended, y)

# Extract parameters
w1_poly, w2_poly, w3_poly, w4_poly = poly_model.coef_[0]
b_poly = poly_model.intercept_[0]

print(f"\nPolynomial Logistic Regression Model:")
print(f"h(x) = sign({b_poly:.4f} + {w1_poly:.4f}*x1 + {w2_poly:.4f}*x2 + {w3_poly:.4f}*x1² + {w4_poly:.4f}*x2²)")

# (c)(ii) Predictions and comparison
y_pred_poly = poly_model.predict(X_extended)
poly_accuracy = accuracy_score(y, y_pred_poly)

plt.figure(figsize=(10, 6))
# Plot actual data
plt.scatter(X1[y==1], X2[y==1], marker='+', color='blue', s=50, label="Actual +1")
plt.scatter(X1[y==-1], X2[y==-1], marker='o', color='green', s=50, label="Actual -1")

# Plot predictions
plt.scatter(X1[y_pred_poly==1], X2[y_pred_poly==1], marker='x', color='cyan', s=50, label="Predicted +1")
plt.scatter(X1[y_pred_poly==-1], X2[y_pred_poly==-1], marker='s', color='orange', s=50, label="Predicted -1")

# Quadratic decision boundary
x_vals = np.linspace(-1, 1, 300)
y_boundary_pos = []
y_boundary_neg = []

for x in x_vals:
    # Quadratic equation: w2*x2 + w4*x2² + w1*x1 + w3*x1² + b = 0
    A = w4_poly
    B = w2_poly
    C = w1_poly*x + w3_poly*x**2 + b_poly
    
    discriminant = B**2 - 4*A*C
    if discriminant >= 0:
        sqrt_disc = np.sqrt(discriminant)
        y_boundary_pos.append((-B + sqrt_disc)/(2*A))
        y_boundary_neg.append((-B - sqrt_disc)/(2*A))
    else:
        y_boundary_pos.append(np.nan)
        y_boundary_neg.append(np.nan)

# Filter boundaries to be within the plot range
y_boundary_pos = [y if -1 <= y <= 1 else np.nan for y in y_boundary_pos]
y_boundary_neg = [y if -1 <= y <= 1 else np.nan for y in y_boundary_neg]

# Plot decision boundary
plt.plot(x_vals, y_boundary_pos, color='red', linewidth=2, label="Decision Boundary")
plt.plot(x_vals, y_boundary_neg, color='red', linewidth=2)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("(c)(iv) Polynomial Logistic Regression")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nTraining Accuracy: {poly_accuracy:.4f}")

# (c)(iii) Baseline comparison
most_common_class = np.bincount(y + 1).argmax() - 1  # Convert -1,1 to 0,2 then back
baseline_accuracy = np.mean(y == most_common_class)

print(f"\n(c)(iii) Baseline Comparison:")
print(f"Baseline (always predict most common class): {baseline_accuracy:.4f}")
print(f"Polynomial Logistic Regression: {poly_accuracy:.4f}")
print(f"Improvement over baseline: {poly_accuracy - baseline_accuracy:.4f} ({((poly_accuracy - baseline_accuracy)/baseline_accuracy)*100:.1f}%)")
