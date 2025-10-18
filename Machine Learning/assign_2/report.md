# Machine Learning Assignment 2 - Lasso and Ridge Regression

**Dataset ID:** # id:23-23--23

---

## Introduction

In this assignment I've used Lasso and Ridge regression on a dataset with polynomial features to understand how regularization works and how to pick the best hyperparameter using cross-validation. The dataset has 199 data points with two input features (x1, x2) and one target variable (y).

## Part (i): Training Lasso and Ridge Models

### (i)(a) Looking at the Data

First I loaded the data and plotted it in 3D to see what it looks like:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt('week3.csv', delimiter=',', skiprows=1)
X = data[:, :2]
y = data[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot points colored by target value
scatter = ax.scatter(X[:,0], X[:,1], y, c=y, cmap='viridis', s=30)

ax.set_xlabel('Feature 1 (x1)')
ax.set_ylabel('Feature 2 (x2)')
ax.set_zlabel('Target (y)')
ax.set_title('3D Scatter Plot')

# add colorbar
plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Target (y)')

plt.show()
```

The points are colored by their target value using the viridis colormap - darker colors represent lower y values and brighter colors represent higher y values. This color coding helps visualize how the target changes across the feature space.

**Does the data lie on a plane or curve?**

Looking at the scatter plot, the data definitely looks curved, not flat. You can see from the color gradient that the target values follow a non-linear pattern across the feature space. If I tried fitting just a plane (linear model) to this it wouldn't work well. This tells me I need polynomial features to capture the non-linear patterns.

### (i)(b) Lasso with Polynomial Features

I created polynomial features up to degree 5, which gives 21 features total:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

poly = PolynomialFeatures(degree=5, include_bias=True)
Xpoly = poly.fit_transform(X)
```

The 21 features are: 1, x1, x2, x1², x1x2, x2², x1³, x1²x2, x1x2², x2³, x1⁴, x1³x2, x1²x2², x1x2³, x2⁴, x1⁵, x1⁴x2, x1³x2², x1²x2³, x1x2⁴, x2⁵

Then I trained Lasso models with different C values. Remember that sklearn uses alpha = 1/(2C), so:

```python
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for C in C_values:
    model = Lasso(alpha=1/(2*C), max_iter=10000)
    model.fit(Xpoly, y)
```

**Results:**

| C     | Non-zero Coefficients | Training MSE |
|-------|----------------------|--------------|
| 0.001 | 0                    | 0.4574       |
| 0.01  | 0                    | 0.4574       |
| 0.1   | 0                    | 0.4574       |
| 1     | 0                    | 0.4574       |
| 10    | 2                    | 0.0772       |
| 100   | 2                    | 0.0392       |
| 1000  | 10                   | 0.0368       |

**What's happening as C changes?**

When C is small (0.001 to 1), the regularization is really strong - so strong that ALL coefficients get pushed to zero. The model just predicts a constant value, which gives bad MSE.

At C=10, finally two coefficients become non-zero: x2 and x1². As C increases further (100, 1000), more coefficients become non-zero and the training error goes down.

**Detailed coefficients for all C values:**

| Feature | C=0.001 | C=0.01 | C=0.1 | C=1 | C=10 | C=100 | C=1000 |
|---------|---------|--------|-------|-----|------|-------|--------|
| x2 | 0 | 0 | 0 | 0 | -0.845251 | -0.987955 | -1.049456 |
| x1² | 0 | 0 | 0 | 0 | 0.510417 | 1.060313 | 1.109467 |
| x1x2 | 0 | 0 | 0 | 0 | 0 | 0 | -0.179636 |
| x1³ | 0 | 0 | 0 | 0 | 0 | 0 | -0.012399 |
| x1²x2 | 0 | 0 | 0 | 0 | 0 | 0 | 0.044510 |
| x1³x2 | 0 | 0 | 0 | 0 | 0 | 0 | 0.213033 |
| x1⁴x2 | 0 | 0 | 0 | 0 | 0 | 0 | -0.094151 |
| x1³x2² | 0 | 0 | 0 | 0 | 0 | 0 | -0.008270 |
| x1x2⁴ | 0 | 0 | 0 | 0 | 0 | 0 | -0.033363 |
| x2⁵ | 0 | 0 | 0 | 0 | 0 | 0 | 0.125468 |

This is the L1 penalty in action - it forces coefficients to be exactly zero, not just small. This is called *feature selection*.

### (i)(c) Visualizing Predictions

I generated predictions on a grid that extends beyond the training data range (as required). My data goes from about -1 to 1, so I extended the grid to about -3 to 3:

```python
x1_range = X[:,0].max() - X[:,0].min()
x2_range = X[:,1].max() - X[:,1].min()
x1_min = X[:,0].min() - 1.0*x1_range
x1_max = X[:,0].max() + 1.0*x1_range
x2_min = X[:,1].min() - 1.0*x2_range
x2_max = X[:,1].max() + 1.0*x2_range

x1_grid = np.linspace(x1_min, x1_max, 60)
x2_grid = np.linspace(x2_min, x2_max, 60)
X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)

Xg = np.c_[X1_mesh.ravel(), X2_mesh.ravel()]
Xg_poly = poly.transform(Xg)
Z = model.predict(Xg_poly).reshape(X1_mesh.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1_mesh, X2_mesh, Z, alpha=0.6, cmap='viridis')
ax.scatter(X[:,0], X[:,1], y, c='red', s=30)
plt.show()
```

I plotted surfaces for C = 0.001, C = 1, and C = 1000.

**What I see:**

- **C = 0.001**: Almost completely flat surface. The model is totally underfitting - it's just predicting roughly the mean value everywhere because all coefficients are zero.

- **C = 1**: Still flat. Same problem - regularization is too strong.

- **C = 1000**: Now we see a curved surface that fits the data points pretty well. The model is complex enough to capture the patterns. Maybe a bit too complex though - could be starting to overfit.

The plots clearly show that as C increases, the prediction surface goes from flat (underfitting) to curved (better fit).

### (i)(d) Underfitting vs Overfitting

**What is underfitting?** When your model is too simple to capture the actual patterns in the data. In Lasso, this happens when C is too small and regularization is too strong. You get high bias (model is biased toward being too simple) and low variance (predictions don't change much with different training sets). Both training error and test error are high.

**What is overfitting?** When your model is too complex and starts fitting the noise in the training data instead of just the real pattern. Happens when C is too large. You get low bias but high variance (predictions change a lot with different training sets). Training error is low but test error is high.

**How C controls this trade-off:**

The C parameter controls the strength of regularization through the penalty term:

J(θ) = (1/m) Σ(h<sub>θ</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)² + (1/C) Σ|θ<sub>j</sub>|

- **Small C** → large penalty → strong regularization → fewer non-zero coefficients → simpler model → underfitting
- **Large C** → small penalty → weak regularization → more non-zero coefficients → complex model → overfitting

From my results:
- C ≤ 1: Definitely underfitting (0 non-zero coefficients, MSE = 0.4574)
- C = 10: Starting to fit (2 coefficients, MSE = 0.0772)
- C = 100: Good fit (2 coefficients, MSE = 0.0392)
- C = 1000: Possibly overfitting (10 coefficients, MSE = 0.0368)

The sweet spot is somewhere around C = 10-100 where we balance complexity and regularization.

### (i)(e) Ridge Regression Comparison

Ridge regression uses L2 penalty instead of L1:

J(θ) = (1/m) Σ(h<sub>θ</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)² + (1/C) θ<sup>T</sup>θ

```python
from sklearn.linear_model import Ridge

for C in C_values:
    model = Ridge(alpha=1/(2*C))
    model.fit(Xpoly, y)
```

**Ridge Results:**

| C     | Training MSE |
|-------|--------------|
| 0.001 | 0.3467       |
| 0.01  | 0.1211       |
| 0.1   | 0.0455       |
| 1     | 0.0370       |
| 10    | 0.0351       |
| 100   | 0.0349       |
| 1000  | 0.0349       |

**Key Differences Between Lasso and Ridge:**

1. **Sparsity**: This is the big one. Lasso sets coefficients to *exactly* zero (L1 penalty), while Ridge just makes them small (L2 penalty). You can see this at C=1:

| Feature | Lasso | Ridge  |
|---------|-------|--------|
| x1      | 0.0000| -0.0031|
| x2      | 0.0000| -1.0192|
| x1²     | 0.0000| 0.9092 |
| x1x2    | 0.0000| -0.1991|
| x2²     | 0.0000| -0.0510|
| (and so on...) | | |

Lasso: All zeros.  
Ridge: Small but non-zero values for all 21 features.

2. **Training Error**: Ridge generally achieves lower training MSE than Lasso for the same C value.

3. **Interpretability**: Lasso gives you automatic feature selection - easier to interpret because you know which features matter. Ridge keeps everything.

**When to use which?**

- Use **Lasso** when you think only a few features really matter and you want automatic feature selection
- Use **Ridge** when you think most features contribute something and you just want to shrink them

## Part (ii): Cross-Validation for Hyperparameter Selection

### (ii)(a) 5-Fold Cross-Validation for Lasso

I used 5-fold CV to find the best C value. Split the data into 5 parts, train on 4 parts and test on the 5th, repeat for all combinations:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
C_cv = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000]

cv_means = []
cv_stds = []

for C in C_cv:
    scores = []
    for train_idx, val_idx in kf.split(Xpoly):
        model = Lasso(alpha=1/(2*C), max_iter=10000)
        model.fit(Xpoly[train_idx], y[train_idx])
        pred = model.predict(Xpoly[val_idx])
        scores.append(mean_squared_error(y[val_idx], pred))
    
    cv_means.append(np.mean(scores))
    cv_stds.append(np.std(scores))

plt.errorbar(C_cv, cv_means, yerr=cv_stds, fmt='o-', capsize=5)
plt.xscale('log')
plt.xlabel('C value')
plt.ylabel('CV MSE')
plt.show()
```

**Why this range of C values?**

I started small (0.001) where regularization is super strong, and went up to 1000 where it's very weak. Used 12 different values to get a good picture of what's happening. The lecture notes suggest increasing by factors of 5 or 10, which is roughly what I did.

**CV Results:**

| C    | Mean CV MSE | Std Dev |
|------|-------------|---------|
| 0.001| 0.4670      | 0.1241  |
| 0.01 | 0.4670      | 0.1241  |
| 0.1  | 0.4670      | 0.1241  |
| 0.5  | 0.4670      | 0.1241  |
| 1    | 0.4670      | 0.1241  |
| 2    | 0.3571      | 0.1239  |
| 5    | 0.1801      | 0.0552  |
| 10   | 0.0814      | 0.0260  |
| 50   | 0.0416      | 0.0119  |
| **100** | **0.0408** | **0.0119** |
| 500  | 0.0419      | 0.0135  |
| 1000 | 0.0429      | 0.0146  |

The error bars (standard deviation) tell us how stable the model is across different folds.

### (ii)(b) Recommended C Value for Lasso

**Best C = 100**

**Why?**
- It has the lowest cross-validation MSE: 0.0408
- The standard deviation is reasonable (0.0119), so it's stable
- Looking at the plot, C=50 and C=100 are pretty close, but C=100 edges it slightly
- After C=100, the error starts going back up, which suggests we'd be overfitting

**Final Lasso Model (C=100):**
- CV MSE: 0.0408 ± 0.0119
- Non-zero coefficients: 2 out of 21
- The two features selected: x2 and x1²
- Coefficients: 
  - x2: -0.987955
  - x1²: 1.060313

This is a really nice sparse solution - out of 21 possible features, Lasso picked just 2 that actually matter. Makes sense given what we saw in the 3D plot.

### (ii)(c) Cross-Validation for Ridge

Did the same thing for Ridge:

```python
ridge_cv_means = []
ridge_cv_stds = []

for C in C_cv:
    scores = []
    for train_idx, val_idx in kf.split(Xpoly):
        model = Ridge(alpha=1/(2*C))
        model.fit(Xpoly[train_idx], y[train_idx])
        pred = model.predict(Xpoly[val_idx])
        scores.append(mean_squared_error(y[val_idx], pred))
    
    ridge_cv_means.append(np.mean(scores))
    ridge_cv_stds.append(np.std(scores))
```

**Ridge CV Results:**

| C    | Mean CV MSE | Std Dev |
|------|-------------|---------|
| 0.001| 0.3752      | 0.1071  |
| 0.01 | 0.1476      | 0.0476  |
| 0.1  | 0.0521      | 0.0164  |
| 0.5  | 0.0440      | 0.0140  |
| 1    | 0.0435      | 0.0142  |
| 2    | 0.0433      | 0.0146  |
| 5    | 0.0432      | 0.0151  |
| **10** | **0.0431** | **0.0154** |
| 50   | 0.0435      | 0.0161  |
| 100  | 0.0437      | 0.0164  |
| 500  | 0.0439      | 0.0167  |
| 1000 | 0.0440      | 0.0167  |

**Best Ridge C = 10**

Ridge behaves differently from Lasso. The CV error decreases smoothly as C increases, then levels off around C=10 and stays roughly constant. No dramatic jumps like we saw with Lasso.

## Final Comparison

**Best Lasso Model:**
- C = 100
- CV MSE = 0.0408 ± 0.0119
- Non-zero coefficients = 2/21

**Best Ridge Model:**
- C = 10
- CV MSE = 0.0431 ± 0.0154
- Non-zero coefficients = 21/21 (all features used)

**Winner: Lasso performs slightly better** with about 0.0023 lower CV MSE.

But honestly both are pretty close. The choice between them depends on what you want:
- If you want a **simpler, more interpretable model** → go with Lasso (only 2 features!)
- If you want **slightly more stable predictions** → Ridge might be better (smaller error bars at small C)

## Key Takeaways

1. **Polynomial features** let us fit curved data, but we need regularization to avoid overfitting

2. **L1 (Lasso)** gives sparse solutions - automatically picks important features and sets others to zero

3. **L2 (Ridge)** gives smoother solutions - keeps all features but shrinks coefficients

4. **Cross-validation is essential** - you can't just look at training error to pick C. My training MSE kept going down as C increased, but CV MSE started going back up after C=100. Without CV I would have picked too large a C and overfitted.

5. **The grid range matters** - plotting predictions on an extended grid (beyond the training data) really helps visualize what the model is doing

6. **There's always a trade-off** between bias and variance. Small C = high bias (underfitting), large C = high variance (overfitting). Cross-validation helps find the sweet spot.

---

## Code Appendix

The full implementation is in `assignment-2.ipynb`. Key libraries used:
- numpy: array operations and numerical computations
- matplotlib: plotting 3D surfaces and error bars
- sklearn: Lasso, Ridge, PolynomialFeatures, KFold, mean_squared_error

All experiments use random_state=42 for reproducibility.
