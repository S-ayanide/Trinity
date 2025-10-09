import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("week2.csv", skiprows=1, header=None)
df.columns = ["X1", "X2", "y"]
X1 = df["X1"].values; X2 = df["X2"].values; y = df["y"].values
X = np.column_stack((X1, X2))

plt.figure(figsize=(10, 6))
plt.scatter(X1[y==1], X2[y==1], marker='+', color='blue', s=50, label="Class +1")
plt.scatter(X1[y==-1], X2[y==-1], marker='o', color='green', s=50, label="Class -1")
plt.xlabel("X1"); plt.ylabel("X2"); plt.legend(); plt.grid(True, alpha=0.3); plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression(); logistic_model.fit(X_train, y_train)
w1, w2 = logistic_model.coef_[0]; b = logistic_model.intercept_[0]

y_pred_train = logistic_model.predict(X_train)
plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], marker='+', color='blue', s=50, label="Actual +1")
plt.scatter(X_train[y_train==-1][:,0], X_train[y_train==-1][:,1], marker='o', color='green', s=50, label="Actual -1")
plt.scatter(X_train[y_pred_train==1][:,0], X_train[y_pred_train==1][:,1], marker='x', color='cyan', s=50, label="Predicted +1")
plt.scatter(X_train[y_pred_train==-1][:,0], X_train[y_pred_train==-1][:,1], marker='s', color='orange', s=50, label="Predicted -1")
x_min, x_max = X_train[:,0].min(), X_train[:,0].max()
y_min_boundary = -(w1*x_min + b)/w2; y_max_boundary = -(w1*x_max + b)/w2
plt.plot([x_min, x_max], [y_min_boundary, y_max_boundary], color='red', linewidth=2, label="Decision Boundary")
plt.xlabel("X1"); plt.ylabel("X2"); plt.legend(); plt.grid(True, alpha=0.3); plt.show()
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {train_accuracy:.4f}")


# SVM across C values (training, predictions, and boundaries)
from sklearn.svm import LinearSVC
C_values = [0.001, 1, 100]
svm_models = {}
for C in C_values:
    model = LinearSVC(C=C, max_iter=10000)
    model.fit(X_train, y_train)
    svm_models[C] = model
    w1_svm, w2_svm = model.coef_[0]; b_svm = model.intercept_[0]
    print(f"C={C}: h(x) = sign({b_svm:.4f} + {w1_svm:.4f}*x1 + {w2_svm:.4f}*x2)")
    y_pred_svm = model.predict(X_train)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], marker='+', color='blue', s=50, label="Actual +1")
    plt.scatter(X_train[y_train==-1][:,0], X_train[y_train==-1][:,1], marker='o', color='green', s=50, label="Actual -1")
    plt.scatter(X_train[y_pred_svm==1][:,0], X_train[y_pred_svm==1][:,1], marker='x', color='cyan', s=50, label="Predicted +1")
    plt.scatter(X_train[y_pred_svm==-1][:,0], X_train[y_pred_svm==-1][:,1], marker='s', color='orange', s=50, label="Predicted -1")
    x_min, x_max = X_train[:,0].min(), X_train[:,0].max()
    y_min_boundary = -(w1_svm*x_min + b_svm)/w2_svm
    y_max_boundary = -(w1_svm*x_max + b_svm)/w2_svm
    plt.plot([x_min, x_max], [y_min_boundary, y_max_boundary], color='red', linewidth=2, label="Decision Boundary")
    plt.xlabel("X1"); plt.ylabel("X2"); plt.legend(); plt.grid(True, alpha=0.3); plt.show()


# Polynomial logistic regression and quadratic boundary
from sklearn.linear_model import LogisticRegression as LR
X1_sq, X2_sq = X1**2, X2**2
X_ext = np.column_stack((X1, X2, X1_sq, X2_sq))
poly_model = LR().fit(X_ext, y)
w1p, w2p, w3p, w4p = poly_model.coef_[0]; bp = poly_model.intercept_[0]
y_pred_poly = poly_model.predict(X_ext)
plt.figure(figsize=(8, 8))
plt.scatter(X1[y==1], X2[y==1], marker='+', color='blue', s=50, label="Actual +1")
plt.scatter(X1[y==-1], X2[y==-1], marker='o', color='green', s=50, label="Actual -1")
plt.scatter(X1[y_pred_poly==1], X2[y_pred_poly==1], marker='x', color='cyan', s=50, label="Predicted +1")
plt.scatter(X1[y_pred_poly==-1], X2[y_pred_poly==-1], marker='s', color='orange', s=50, label="Predicted -1")
x_vals = np.linspace(X1.min()-0.1, X1.max()+0.1, 300)
y_pos, y_neg = [], []
for x in x_vals:
    A = w4p; B = w2p; Cc = w1p*x + w3p*x**2 + bp
    disc = B*B - 4*A*Cc
    if disc >= 0:
        r = np.sqrt(disc)
        y_pos.append((-B + r)/(2*A)); y_neg.append((-B - r)/(2*A))
    else:
        y_pos.append(np.nan); y_neg.append(np.nan)
plt.plot(x_vals, y_pos, color='red', linewidth=2, label="Decision Boundary")
plt.plot(x_vals, y_neg, color='red', linewidth=2)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(X1.min()-0.1, X1.max()+0.1); plt.ylim(X2.min()-0.1, X2.max()+0.1)
plt.xlabel("X1"); plt.ylabel("X2"); plt.legend(); plt.grid(True, alpha=0.3); plt.show()