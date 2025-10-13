# » Classification with Two Classes

* Will Johnny like or dislike the pie?  
* Training data:

![](images/c0744574eeb14c1d0b7213f7601a17dca2cf09a57fc46782a113b002d8408aa6.jpg)

* Features: Shape (round, square, triangle), filling (white, gray, dark), crust (thick, thin, light, dark), size (big, small) etc  
* To make prediction match features of pie against previous examples from training data

# » Classification with Two Classes

# * Examples:

* Movies reviews: positive or negative?  
* Images: Does a picture contain human faces?  
* Finance: Is person applying for a loan likely to repay it?  
* Advertising: If display an ad on web page is the person likely to click on it?  
* Online transactions: fraudulent or not?  
* Tumor: malignant or benign?

* As before  $x =$  "input" variable/features e.g. text of email, location, nationality  
* Now  $y =$  "output" variable/"target" variable only takes values -1 or 1 (with linear regression  $y$  was real valued). In classification  $y$  often referred to as the label<sup>1</sup>.  
* We want to build a classifier that predicts the label of a new object e.g whether a new email is spam or not.

# > Logistic Regression: Model

![](images/e1b952526db014c6bd4741eb5550dbab58dcb7c181d8933b6998764d5d155cf9.jpg)

* As before  ${\theta }^{T}x = {\theta }_{0}{x}_{0} + {\theta }_{1}{x}_{1} + {\theta }_{2}{x}_{2} + \cdots  + {\theta }_{n}{x}_{n}$  with  ${x}_{0} = 1$  ,  ${x}_{1},\ldots ,{x}_{n}$  the input features and  ${\theta }_{0},\ldots ,{\theta }_{n}$  the (unknown) parameters.  
* Model:  $\operatorname{sign}\left( {{\theta }^{T}x}\right)$  i.e. predict output +1 when  ${\theta }^{T}x > 0$  and output -1 when  ${\theta }^{T}x < 0$  . Decision boundary is  ${\theta }^{T}x = 0$  (green line in plot above)  
$\theta^T x = 0$  defines a point in one dimension e.g.

$$
1 + 0. 5 x _ {1} = 0 \rightarrow x _ {1} = - 2 \dots
$$

* ... a line in two dimensions e.g.

$$
2 + x _ {1} + 2 x _ {2} = 0 \Rightarrow x _ {2} = - x _ {1} / 2 - 1 \dots
$$

* .. and a plane in higher dimensions

# $\gg$  Logistic Regression: Decision Boundary

* Example: suppose  $x$  is vector  $x = [1, x_1, x_2]^T$  e.g.  $x_1$  might be tumour size and  $x_2$  patient age.

![](images/d11d8e6a59ae1c940faf05e212abc8c464c41e35e6ff34f3aba03938747964b7.jpg)

![](images/c15a37a71d00467436b1eb526bd413b45a43fa6f819d937720f116c80ad97a9b.jpg)

*  ${\theta }_{0} = 0,{\theta }_{1} = {0.5},{\theta }_{2} =  - {0.5}$  .  
* sign(θTx) = +1 when 0.5x1 - 0.5x2 > 0 i.e. when x1 > x2.  
* When data can be separated in this way we say that it is linearly separable.  
* Often the 3D plot on left is sketched in 2D as shown in right (easier to draw!)

* Not all data is linearly separable e.g.

![](images/473b197d4d9e293362d60f79db5b3ad2ab362a5480e0b0a3733f33ebcd1c869a.jpg)

# $\gg$  Logistic Regression: Choice of Cost Function

* Training data:  $\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\dots ,(x^{(m)},y^{(m)})\}$  
\*  $x\in \left[ \begin{array}{c}x_0\\ x_1\\ \vdots \\ x_n \end{array} \right],x_0 = 1,y\in \{-1,1\}$  
* Model:  $h_{\theta}(x) = \text{sign}(\theta^T x)$  
* How to choose parameters  $\theta$  ?

# $\gg$  Logistic Regression: Choice of Cost Function

![](images/59f877483e9a3770d763d15f326e9292fca4b9a0a24d8b4ddc5bda06a8d31c85.jpg)

![](images/d31dcc0b3d092ca3ab1399850a05b5613162f5137732bd805054e1376ddb1d84.jpg)

* Model: sign  $(\theta^T x)$  i.e. predict output  $+1$  when  $\theta^T x > 0$  and output -1 when  $\theta^T x < 0$  
* Suppose we try square error  $\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\left( {{h}_{\theta }\left( {x}^{\left( i\right) }\right)  - {y}^{\left( i\right) }}\right) {}^{2}$  .

* Roughly speaking, minimising square error fits the same line to the data that we would fit by eye  
* Works ok on left-hand plot above:  $\operatorname{sign}\left( {{\theta }^{T}x}\right)  =  - 1$  for points to the left of green line and  $\operatorname{sign}\left( {{\theta }^{T}x}\right)  =  + 1$  for points to the right.  
* Not so good in right-hand plot - data points far to the right pull point where  $\theta^T x$  crosses zero to the right, so causing misclassification

# $\gg$  Logistic Regression: Choice of Cost Function

* We might consider the 0-1 loss function:

$$
\frac {1}{m} \sum_ {i = 1} ^ {m} \mathbb {I} \left(h _ {\theta} \left(x ^ {(i)}\right) \neq y ^ {(i)}\right)
$$

where indicator function  $\mathbb{I} = 1$  if  $h_{\theta}(x^{(i)})\neq y^{(i)}$  and  $\mathbb{I} = 0$  otherwise. But hard to work with.

* For logistic regression we use:

$$
\frac {1}{m} \sum_ {i = 1} ^ {m} \log \left(1 + e ^ {- y ^ {(i)} \theta^ {T} x ^ {(i)}}\right) / \log (2)
$$

noting that  $y = -1$  or  $y = +1$ . Scaling by  $\log(2)$  is optional, but makes the loss 1 when  $y^{(i)}\theta^T x^{(i)} = 0$ .

# $\gg$  Logistic Regression: Choice of Cost Function

Loss function:  $\log (1 + e^{-y\theta^T x}) / \log (2)$

![](images/eb807b026308a0cbf0f9fdd15beb20d34f21bbe31d103da7d48b56bef7ca4357.jpg)

![](images/8c146733561bc5af4946c66045ef29a479301b57c501069cc5d279fefc8a9f49.jpg)

* So a small penalty when  ${\theta }^{T}x \gg  0$  and  $y = 1$  ,and when  ${\theta }^{T}x \ll  0$  and  $y =  - 1$  .  
* Minimising this thus gives preference to  $\theta$  values that push  ${\theta }^{T}x$  well away from the decision boundary  ${\theta }^{T}x = 0$  .

* Model:  $h_{\theta}(x) = \text{sign}(\theta^T x)$  
* Parameters: θ  
* Cost Function:  $J\left( \theta \right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\log \left( {1 + {e}^{-{y}^{\left( i\right) }{\theta }^{T}x^{\left( i\right) }}}\right)$  
* Optimisation: Select  $\theta$  that minimises  $J(\theta)$

# $\gg$  Gradient Descent

As before, can find  $\theta$  using gradient descent. For

$$
\begin{array}{l} J (\theta) = \frac {1}{m} \sum_ {i = 1} ^ {m} \log (1 + e ^ {- y ^ {(i)} \theta^ {T} x ^ {(i)}}): \\ * \frac {\partial}{\partial \theta_ {j}} J (\theta) = \frac {1}{m} \sum_ {i = 1} ^ {m} - y ^ {(i)} x _ {j} ^ {(i)} \frac {e ^ {- y ^ {(i)} \theta T _ {x} (i)}}{1 + e ^ {- y ^ {(i)} \theta T _ {x} (i)}} \\ * \left(\text {R e m e m b e r} \frac {d \log (x)}{d x} = \frac {1}{x}, \frac {d \exp (x)}{d x} = \exp (x) \text {a n d c h a i n r u l e} \right. \\ \frac {d f (z (x))}{d x} = \frac {d f}{d z} \frac {d z}{d x}) \\ \end{array}
$$

So gradient descent algorithm is:

* Start with some  $\theta$  
\*Repeat:

$$
\begin{array}{l} f o r j = 0 \text {t o} \left\{\delta_ {j} := - \frac {\alpha}{m} \sum_ {i = 1} ^ {m} y ^ {(i)} x _ {j} ^ {(i)} \frac {e ^ {- y ^ {(i)} \theta T _ {x} (i)}}{1 + e ^ {- y ^ {(i)} \theta T _ {x} (i)}} \right\} \\ f o r j = 0 t o n \left\{\theta_ {j} := \theta_ {j} + \delta_ {j} \right\} \\ \end{array}
$$

*  $J(\theta)$  is convex, has a single minimum. Iteration moves downhill until it reaches the minimum

# Logistic regression:

```python
import numpy as np  
Xtrain = np.random.uniform(0,1,100)  
ytrain = np.sign(Xtrain - 0.5)  
Xtrain = Xtrain.reshape(-1,1)  
from sklearn.metrics_model import LogisticRegression  
model = LogisticRegression(penalty='none', solver='lbfgs')  
model.fit(Xtrain, ytrain)  
print("intercept %f, slope %f"%%(model.intercept_, model.coef()))
```

# Typical output: intercept -267.026565, slope 529.954559

* Prediction  $\widehat{y} = \operatorname{sign}\left( {-{267.026565} + {529.954559x}}\right)$  
i.e.  $y = +1$  when  $-267.026565 + 529.954559x > 0$  and  $y = -1$  when  $-267.026565 + 529.954559x < 0$  
i.e.  $y = +1$  when  $x > 267.026565 / 529.954559 = 0.50392$  and  $y = -1$  when  $x < 267.026565 / 529.954559 = 0.5039$  
* We generated data using  $y = +1$  when  $x > 0.5$  and  $y = -1$  when  $x < 0.5$ . So model learned from training data is roughly correct, but not perfect. That's normal. Why?

# Plotting predictions

import matplotlib.pyplot as plt  
plt.rcParams['figure.costrained.layout.use'] = True  
pltscatter(Xtrain, ytrain, color='red', marker='+')  
pltscatter(Xtrain, ypred, color='green', marker='+')  
plt.xlabel('input x'); pltylabel('output y')  
plt.legend(['train', 'predict'])  
plt.show()

![](images/6aa0f0d1796782a3bf0194ec8537a00d355554055563fb7427ee15db8668f3ac.jpg)

# Logistic Regression With Multiple Classes

* Examples:

* Email folder tagging: work, friends, family, hobby  
* Weather, sunny, cloudy, rain, snow  
* Given where I live in Dublin, predict which political party I'll vote for.

* Now  $y =$  "output" variable/"target" variable takes values 0,1,2,... E.g.  $y = 0$  if sunny,  $y = 1$  if cloudy,  $y = 2$  if rain etc.

![](images/7bddc6e50fa348a8dd9a96fbf9f329ea2784d4724b982346a4d48280bd5cd0a8.jpg)

![](images/c1866380bfc23ec9c8dd49c422bb39584e25f0e4629bf2015b8a5ad9475d8d27.jpg)

![](images/372fab30e6c14897e419c1c0e3c38e8639ea583320aad17f3ce95110fbb62cb2.jpg)  
↓

![](images/bcf67bb1ef03df00afdef3b601407de1c9b136362c32ca8997522fe31a5b71e0.jpg)

![](images/1f95ccdba0fd04503d985e6541b9a0a84a07106e110e3c7d44063f8bd5a23bc9.jpg)

![](images/16929a3237470f7acb18402d7652a33972b92e45867a489aedb7c6cc9513b5c8.jpg)

* Train a classifier  $\operatorname{sign}\left( {{\theta }^{T}x}\right)$  for each class  $i$  to predict the probability that  $y = i$  . Training data: re-label data as  $y =  - 1$  when  $y \neq  i$  and as  $y =  = 1$  when  $y = i$  ,so we're back to a binary classification task.

In an SVM use the "hinge" loss function  $\max (0,1 - y\theta^T x)$ :

![](images/ee8f681da9f6aca8e084a6f2948f5d56f100ced98a5d32c636f1b86004e0511b.jpg)

![](images/f8019993ccc21555a4c5ef56cf467d8fec760994955efb0fc8ccf1c5f0980a5b.jpg)

Main differences from logistic loss function:

* hinge-loss is not differentiable ("non-smooth")  
* hinge loss assigns zero penalty to all values of  $\theta$  which ensure  ${\theta }^{T}x \geq  1$  when  $y = 1$  ,and  ${\theta }^{T}x \leq   - 1$  when  $y =  - 1$

In an SVM use the "hinge" loss function  $\max (0,1 - y\theta^T x)$ :

* So long as  $y\theta^T x > 0$  then by scaling up  $\theta$  sufficiently, e.g. to  $10\theta$  or  $100\theta$ , then we can always force  $y\theta^T x > 1$  i.e.  $\max (0, 1 - y\theta^T x) = 0$  
* To get sensible behaviour we have to penalise large values of  $\theta$ . We do this by adding penalty  ${\theta }^{T}\theta  = \mathop{\sum }\limits_{{j = 1}}^{n}{\theta }_{j}^{2}$  
* Final SVM cost function is:

$$
J (\theta) = \frac {1}{m} \sum_ {i = 1} ^ {m} \max  (0, 1 - y ^ {(i)} \theta^ {T} x ^ {(i)}) + \theta^ {T} \theta / C
$$

where  $C > 0$  is a weighting parameter that we have to choose (making  $C$  bigger makes penalty less important).

* Model:  $h_{\theta}(x) = \text{sign}(\theta^T x)$  
* Parameters: θ  
* Cost Function:  $J\left( \theta \right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\max \left( {0,1 - {y}^{\left( i\right) }{\theta }^{T}{x}^{\left( i\right) }}\right)  + {\theta }^{T}\theta /C$  
* Optimisation: Select  $\theta$  that minimise  $J(\theta)$  
* Observe that only difference from Logistic Regression is in the choice of cost function.  
* SVM cost function:

* Includes penalty  $\theta^T\theta /C$ . We can also add a penalty like this to Logistic Regression though  $\rightarrow$  regularisation, we'll come back to this later  
* Terms in sum are zero for points where  $y^{(i)}\theta^T x^{(i)} \geq 1 \rightarrow$  this is the important difference.

* It means that training data points  $(x^{(i)}, y^{(i)})$  with  $y^{(i)}\theta^T x^{(i)} \geq 1$  don't contribute to the cost function  
* Only the training data points with  $y^{(i)}\theta^T x^{(i)} < 1$  are relevant → support vectors. Can make computations more efficient.

# $\gg$  Gradient Descent for SVMs

Subgradient descent algorithm for SVMs is:

* Start with some  $\theta$  
\*Repeat:

for  $j = 0$  to  $n$

$$
\left\{\delta_ {j} := - \alpha \left(2 \theta_ {j} / C - \frac {1}{m} \sum_ {i = 1} ^ {m} y ^ {(i)} x _ {j} ^ {(i)} \mathbb {1} \left(y ^ {(i)} \theta^ {T} x ^ {(i)} \leq 1\right)\right) \right.
$$

for  $j = 0$  to  $n\{\theta_j \coloneqq \theta_j + \delta_j\}$

where  $\mathbb{1}(y^{(i)}\theta^T x^{(i)}\leq 1) = 1$  when  $y^{(i)}\theta^{T}x^{(i)}\leq 1$  and zero otherwise.

$J(\theta)$  is convex, has a single minimum. Iteration moves downhill until it reaches the minimum

# Logistic regression:

from sklearn.metrics import LinearSVC  
model = LinearSVC(C=1.0).fit(Xtrain, ytrain)  
print("intercept %f, slope %f(%{model.intercept\_}, model.coef\_))

# Typical output: intercept -1.890453, slope 3.867570

* So prediction is  $y = +1$  when  $x > 1.890453 / 3.867570 = 0.4888$  and  $y = -1$  when  $x < 0.4888$  
* cf Logistic Regression:intercept -267.026565, slope 529.954559 i.e.  $y = +1$  when  $x > 0.5039$  and  $y = -1$  when  $x < 0.5039$  
* Recall penalty term encourages SVM to choose smaller  $\theta$ . Changing to use  $C = 1000$  gives intercept -19.830632, slope 40.271028 i.e. decision boundary  $x = 19.830632 / 40.271028 = 0.4924$
