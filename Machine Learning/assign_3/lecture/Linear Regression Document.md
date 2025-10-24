# Examples of Predicting A Number

* Predict price of a house given its location, floor area, #rooms etc  
* Predict income of a person given their age, gender, handset model, web pages browsed  
* Predict temperature outside tomorrow given weather forecast and past measurements at your location  
* Predict whether distance between mobile handsets given Bluetooth recieved signal strength, handset models, type of location (house, bus, office, supermarket etc), whether handsets in a pocket/bag or not

* Data https://www.statlearning.com/s/Advertising.csv  
* Data consists of the advertising budgets for three media (TV, radio and newspapers) and the overall sales in 200 different markets.

<table><tr><td>TV</td><td>Radio</td><td>Newspaper</td><td>Sales</td></tr><tr><td>230.1</td><td>37.8</td><td>69.2</td><td>22.1</td></tr><tr><td>44.5</td><td>39.3</td><td>45.1</td><td>10.4</td></tr><tr><td>17.2</td><td>45.9</td><td>69.3</td><td>9.3</td></tr><tr><td>:</td><td>:</td><td>:</td><td>:</td></tr></table>

![](images/0fd21d8d3db672773d61e1cc3f9b7fe5d9f56eaa8feb07eec4fbe55f00eb34a8.jpg)

* Suppose we want to predict sales in a new area?  
* Predict sales when the TV advertising budget is increased to 350 ?  
* ... Draw a line that fits through the data points

Training data:

<table><tr><td>TV (x)</td><td>Sales (y)</td></tr><tr><td>230.1</td><td>22.1</td></tr><tr><td>44.5</td><td>10.4</td></tr><tr><td>17.2</td><td>9.3</td></tr><tr><td>:</td><td>:</td></tr></table>

*  $m$  = number of training examples  
*  $x=$  "input" variable/features  
* y="output"
variable/"target" variable

*  $(x^{(i)}, y^{(i)})$  the  $i$ th training example  
$\begin{array}{rl} & {\star x^{(1)} = 230.1,y^{(1)} = 22.1,}\\ & {x^{(2)} = 44.5,y^{(2)} = 10.4} \end{array}$

* Prediction:  $\hat{y} = {h}_{\theta }\left( x\right)  = {\theta }_{0} + {\theta }_{1}x$  
*  ${\theta }_{0},{\theta }_{1}$  are (unknown) parameters  
* often abbreviate  $h_{\theta}(x)$  to  $h(x)$

![](images/3fb6ef45c53add676bdc4c2a5109062e24b4b4826a00cbffeba985ea680e5a83.jpg)

$$
\theta_ {0} = 1 5, \theta_ {1} = 0
$$

![](images/47a32fc5c3a34f0ee25bdb46745f9c92af62d420cb52b299feaf0a0820e60b48.jpg)

$$
\theta_ {0} = 0, \theta_ {1} = 0. 1
$$

![](images/5323526a81ad59fc0c3c87620bd40b55d17de478572febed7e579a0b89f04bdf.jpg)

$$
\theta_ {0} = 1 5, \theta_ {1} = 0. 1
$$

# $\gg$  Cost Function: How to choose model parameters  $\theta$ ?

* Prediction:  $\hat{y} = {h}_{\theta }\left( x\right)  = {\theta }_{0} + {\theta }_{1}x$  
* Idea: Choose  $\theta_0$  and  $\theta_1$  so that  $h_\theta(x^{(i)})$  is close to  $y^{(i)}$  for each of our training examples  $(x^{(i)}, y^{(i)})$ ,  $i = 1, \ldots, m$ .  
* Least squares case: select the values for  ${\theta }_{0}$  and  ${\theta }_{1}$  that minimise cost function:

$$
J (\theta_ {0}, \theta_ {1}) = \frac {1}{m} \sum_ {i = 1} ^ {m} (h _ {\theta} (x ^ {(i)}) - y ^ {(i)}) ^ {2}
$$

Note: cost function is a sum over prediction error at each training point so can also write as

$$
J (\theta_ {0}, \theta_ {1}) = \frac {1}{m} \sum_ {i = 1} ^ {m} I _ {i} (\theta_ {0}, \theta_ {1})
$$

where  $l_{i}(\theta_{0},\theta_{1}) = (h_{\theta}(x^{(i)}) - y^{(i)})^{2}$

* Suppose our training data consists of just two observations:  $(3,1)$ ,  $(2,1)$ , and to keep things simple we know that  $\theta_0 = 0$ .  
* The cost function is

$$
\frac {1}{2} \sum_ {j = 1} ^ {2} (y ^ {(j)} + \theta_ {1} x ^ {(j)}) ^ {2} = \frac {1}{2} (1 - 3 \theta_ {1}) ^ {2} + (1 - 2 \theta_ {1}) ^ {2}
$$

* What value of  ${\theta }_{1}$  minimises  ${\left( 1 - 3{\theta }_{1}\right) }^{2} + {\left( 1 - 2{\theta }_{1}\right) }^{2}$  ?

![](images/e95a29015d6662158dac1ea882e8e109e6ad7e6ada76cae564b4a19c109cabfd.jpg)

![](images/f7c37f6845a43c7e6acea26ad71115f857fda9fb1c84a638c8c3c7274ac4dcdf.jpg)

# Example: Advertising Data

* Least square linear fit  
* Residuals are the difference between the value predicted by the fit and the measured value.

* Do the residuals look “random” or do they have some “structure”? Is our model satisfactory?  
* We can use the residuals to estimate a confidence interval for the prediction made by our linear fit.

![](images/3199f0b633900edd7ee9a37f71fdfe96837ebd2b2b4ea750af37a599b83ebc4f.jpg)

![](images/5592ccafc8298b242f56502ee90c0dad57b77a5c9cc9918fd7e2017cc73c33ba.jpg)

# > Summary: Linear Regression With One Feature

* Feature: x  
* Linear Model:  $h_{\theta}(x) = \theta_0 + \theta_1 x$  
* Parameters:  ${\theta }_{0},{\theta }_{1}$  
* Cost Function:  $J\left( {{\theta }_{0},{\theta }_{1}}\right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\left( {{h}_{\theta }\left( {x}^{\left( i\right) }\right)  - {y}^{\left( i\right) }}\right)$  
* Optimisation: Select  ${\theta }_{0}$  and  ${\theta }_{1}$  that minimise  $J\left( {{\theta }_{0},{\theta }_{1}}\right)$  "least squares" - why?

# $\gg$  Gradient Descent

Need to select  $\theta_0$  and  $\theta_{1}$  that minimise  $J(\theta_0,\theta_1)$ . Brute force search over pairs of values of  $\theta_0$  and  $\theta_{1}$  is inefficient, can we be smarter?

* Start with some  ${\theta }_{0}$  and  ${\theta }_{1}$  
\*Repeat:

Update  $\theta_0$  and  $\theta_{1}$  to new value which makes  $J(\theta_0,\theta_1)$  smaller

![](images/dd598ed91e678d424cc7da079f08a696fd5d6ecb763c187dfb4f54abda4a8d13.jpg)

* When curve is "bowl shaped" or convex then this must eventually find the minimum.

# $\gg$  Gradient Descent

* Start with some  ${\theta }_{0}$  and  ${\theta }_{1}$  
\*Repeat:

Update  $\theta_0$  and  $\theta_{1}$  to new value which makes  $J(\theta_0,\theta_1)$  smaller

* When curve has several minima then we can't be sure which we will converge to.  
* Might converge to a local minimum, not the global minimum

![](images/327b9d5c848958ad0a9a409b8559fb261f0f6f5b2624c9460d8fd1e18d7fa910.jpg)

![](images/234454700998722034ee55666891e646013bebd61542aacc26f8a923f9cb22e1.jpg)

# $\gg$  Derivatives

* When  $x$  is scalar the derivative of a function  $f(\cdot)$  at point  $x$  is the slope of the line that just touches the function at  $x$  e.g.

![](images/d4575532bdfb2e56fe78928f9a9b854fe101a6833b74bb0962292f9914cbe14d.jpg)

* When  $x = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$  then the derivative is the slope of the plane that just touches the function at  $x$  e.g.

![](images/08af03fabac85b0d924b69e4bf69b719270f433861d229fa599dcc8a577fb317.jpg)

* And similarly when  $x$  has  $> 2$  elements, but can't draw it.

* Equation of a line is  $y = mx + c$ .  $m$  is slope,  $c$  is intercept (when  $x = 0$  then  $y = c$ ).

![](images/156be522fbcf210c3b5848e3a8ad8341a1e00f8bef8c370d9ba6173a57115277.jpg)

![](images/d083b2274b79023ec2c96973518f0b90afb2e8a12e30dd53b5011fc2ccfbed74.jpg)

![](images/1c6a25bd113352d1e8d8deee4e44d40ee3ef16f4a49732a3f59a021faa78f006.jpg)

# $\gg$  Derivatives

* Derivative of a function  $f\left( \cdot \right)$  at point  ${x}^{\prime }$  is the slope of the line that just touches the function at  ${x}^{\prime }$

![](images/496f975c4d72e354dce7ccc441a3dc68dbe6002d3802d57f551f5fe7d6c76226.jpg)

* Equation of a line is  ${mx} + c$  ,slope  $m$  and intercept  $c$  .  
* Line touches  $f(\cdot)$  at point  $x'$ , so  $mx' + c = f(x')$  i.e.  $c = f(x') - mx'$  
* Slope  $m$  of line is derivative, i.e.  $m = \frac{df}{dx} (x')$  (this is just notation $^1$ , the important point is that we can calculate  $\frac{df}{dx} (x')$  and so  $m$  using standard tools).  
* Putting this together, equation of line is

$$
\frac {d f}{d x} \left(x ^ {\prime}\right) \left(x - x ^ {\prime}\right) + f \left(x ^ {\prime}\right)
$$

and so

$$
f (x) \approx f \left(x ^ {\prime}\right) + \frac {d f}{d x} \left(x ^ {\prime}\right) \left(x - x ^ {\prime}\right)
$$

* Equation of a plane is  $y = {m}_{1}{x}_{1} + {m}_{2}{x}_{2} + c$  
* Note: we now have two slopes  $m_{1}$ ,  $m_{2}$  and intercept  $c$  
* Notation: For plane just touching  $f(\cdot)$  at point  $x'$ .

*  $m_{1} = \frac{\partial f}{\partial x_{1}}(x')$ ,  $m_{2} = \frac{\partial f}{\partial x_{2}}(x')$  
$\frac{\partial f}{\partial x_1}(x')$  is the partial derivative of  $f(\cdot)$  wrt  $x_1$  at point  $x'$ .  
*  $\frac{\partial f}{\partial x_2}(x')$  is the partial derivative of  $f(\cdot)$  wrt  $x_2$  at point  $x'$ .  
*  $\nabla f(x') = [\frac{\partial f}{\partial x_1}(x'), \frac{\partial f}{\partial x_2}(x'), \dots, \frac{\partial f}{\partial x_n}(x')]$ , the vector of partial derivatives. And sometimes  $\nabla_{x_1}f(x')$  is used for  $\frac{\partial f}{\partial x_1}(x')$  etc.

* This plane is an approximation to function  $f(\cdot)$  near point  $x'$  i.e.

$$
f (x) \approx f (x ^ {\prime}) + \frac {\partial f}{\partial x _ {1}} (x ^ {\prime}) (x _ {1} - x _ {1} ^ {\prime}) + \frac {\partial f}{\partial x _ {2}} (x ^ {\prime}) (x _ {2} - x _ {2} ^ {\prime})
$$

* If we choose  $x_1 = x' - \alpha \frac{\partial f}{\partial x_1}$  and  $x_2 = x' - \alpha \frac{\partial f}{\partial x_2}$  then moving from point  $x'$  to  $x$  tends to decrease function  $f(\cdot)$  i.e.  $f(x) \lesssim f(x')$

# Repeat:

$$
\delta_ {0} := - \alpha \frac {\partial}{\partial \theta_ {0}} J (\theta_ {0}, \theta_ {1}), \delta_ {1} := - \alpha \frac {\partial}{\partial \theta_ {1}} J (\theta_ {0}, \theta_ {1})
$$

$$
\theta_ {0} := \theta_ {0} + \delta_ {0}, \theta_ {1} := \theta_ {1} + \delta_ {1}
$$

$\alpha$  is called the step size or learning rate, its value needs to be selected appropriately (not too large, not too small).

For  $J(\theta_0,\theta_1) = \frac{1}{m}\sum_{i = 1}^{m}(h_\theta (x^{(i)}) - y^{(i)})^2$  with  $h_\theta (x) = \theta_0 + \theta_1x$ :

$$
* \frac {\partial}{\partial \theta_ {0}} J (\theta_ {0}, \theta_ {1}) = \frac {2}{m} \sum_ {i = 1} ^ {m} (h _ {\theta} (x ^ {(i)}) - y ^ {(i)})
$$

$$
* \frac {\partial}{\partial \theta_ {1}} J (\theta_ {0}, \theta_ {1}) = \frac {2}{m} \sum_ {i = 1} ^ {m} (h _ {\theta} (x ^ {(i)}) - y ^ {(i)}) x ^ {(i)}
$$

So gradient descent algorithm is:

\*repeat:

$$
\delta_ {0} := - \frac {2 \alpha}{m} \sum_ {i = 1} ^ {m} \left(h _ {\theta} \left(x ^ {(i)}\right) - y ^ {(i)}\right)
$$

$$
\delta_ {1} := - \frac {2 \alpha}{m} \sum_ {i = 1} ^ {m} \left(h _ {\theta} (x ^ {(i)}) - y ^ {(i)}\right) x ^ {(i)}
$$

$$
\theta_ {0} := \theta_ {0} + \delta_ {0}, \theta_ {1} := \theta_ {1} + \delta_ {1}
$$

# » Practicalities: Normalising Data

* When using gradient descent (and also more generally) its a good idea to normalise your data i.e. scale and shift the inputs and outputs so that they lie roughly between  $0 \rightarrow 1$  or  $-1 \rightarrow 1$  
* Its ok if data range spans  $1 \rightarrow 100$ , problem is when range is very large e.g.  $1 \rightarrow 10^{6} \rightarrow$  large ranges (i) mess up numerics, (ii) larger valued data tends to dominate cost function and training focusses on that data

<table><tr><td></td><td>Age</td><td>Income</td></tr><tr><td>Person 1</td><td>18</td><td>5,000 €</td></tr><tr><td>Person 2</td><td>20</td><td>25,000 €</td></tr><tr><td>Person 3</td><td>25</td><td>40,000 €</td></tr><tr><td>Person 4</td><td>30</td><td>50,000 €</td></tr><tr><td>Person 5</td><td>30</td><td>55,000 €</td></tr><tr><td>Person 6</td><td>28</td><td>60,000 €</td></tr><tr><td>Person 7</td><td>46</td><td>65,000 €</td></tr><tr><td>Person 8</td><td>58</td><td>70,000 €</td></tr><tr><td>Person 9</td><td>62</td><td>80,000 €</td></tr><tr><td>Person 10</td><td>61</td><td>100,000 €</td></tr><tr><td>Person 11</td><td>19</td><td>150,000 €</td></tr></table>

<table><tr><td>Min</td><td>18</td><td>5000</td></tr><tr><td>Max</td><td>62</td><td>150000</td></tr><tr><td>Mean</td><td>36.09</td><td>63636</td></tr><tr><td>Std Dev</td><td>16.54</td><td>36748</td></tr></table>

![](images/1b3f1e2045e4084a29504fd19399eb5f841ed8876b61dd49fafb0a9ddcb60272.jpg)

nannnnnne nnnnnnne  

<table><tr><td></td><td>Age</td><td>Income</td></tr><tr><td>Person 1</td><td>0.00</td><td>0.00</td></tr><tr><td>Person 2</td><td>0.05</td><td>0.14</td></tr><tr><td>Person 3</td><td>0.16</td><td>0.24</td></tr><tr><td>Person 4</td><td>0.27</td><td>0.31</td></tr><tr><td>Person 5</td><td>0.27</td><td>0.34</td></tr><tr><td>Person 6</td><td>0.23</td><td>0.38</td></tr><tr><td>Person 7</td><td>0.64</td><td>0.41</td></tr><tr><td>Person 8</td><td>0.91</td><td>0.45</td></tr><tr><td>Person 9</td><td>1.00</td><td>0.52</td></tr><tr><td>Person 10</td><td>0.98</td><td>0.66</td></tr><tr><td>Person 11</td><td>0.02</td><td>1.00</td></tr></table>

eaaee  

<table><tr><td></td><td>Age</td><td>Income</td></tr><tr><td>Person 1</td><td>-1.09</td><td>-1.60</td></tr><tr><td>Person 2</td><td>-0.97</td><td>-1.05</td></tr><tr><td>Person 3</td><td>-0.67</td><td>-0.64</td></tr><tr><td>Person 4</td><td>-0.37</td><td>-0.37</td></tr><tr><td>Person 5</td><td>-0.37</td><td>-0.24</td></tr><tr><td>Person 6</td><td>-0.49</td><td>-0.10</td></tr><tr><td>Person 7</td><td>0.60</td><td>0.04</td></tr><tr><td>Person 8</td><td>1.32</td><td>0.17</td></tr><tr><td>Person 9</td><td>1.57</td><td>0.45</td></tr><tr><td>Person 10</td><td>1.51</td><td>0.99</td></tr></table>

# Practicalities: Normalising Data

* Commonly replace  $x_{j}$  with  $\frac{x_{j} - \mu_{j}}{\sigma_{j}}$  where:

* Shift  ${\mu }_{j} = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{n}{x}_{j}^{\left( i\right) }$  tries to make features have approximately zero mean (do not apply to  ${x}_{0} = 1$  though)  
* Scaling factor  $\sigma_{j} = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(x_{j}^{(i)} - \mu)^{2}}$  tries to make features mostly lie between -1 and 1

or:

* Shift  ${\mu }_{j} = \min \left( {x}_{j}^{\left( i\right) }\right)$  (do not apply to  ${x}_{0} = 1$  though)  
* Scaling factor  ${\sigma }_{j} = \max \left( {x}_{j}\right)  - \min \left( {x}_{j}\right)$

* E.g. in advertising data TV budget values lie in range 0.7 to 296.4, so rescaling as  $\frac{TV - 0.7}{296 - 0.7}$  gives a feature with values in interval  $0 \leq x_{1} \leq 1$

# » Practicalities: When to stop?

![](images/bf2d41a7ccced41c3cc264fd4bac593109eef79f031775423b4e243a2e3c3486.jpg)

* "Debugging": How to make sure gradient descent is working correctly  $\rightarrow J(\theta)$  should decrease after every iteration  
* Stopping criteria: stop when decreases by less than e.g.  $10^{-2}$  or after a fixed number of iterations e.g. 200, whichever comes first.

# Practicalities: Selecting Step Size

![](images/b008226f5cb0f32cd0c74933bcbeb392bfe4a401c55b74600f1d0e2f3b5efd14.jpg)

![](images/1d883b22d85dca23951969bb0c6f9668755508c66fa74fe79d8e62f5fd51d152.jpg)

* Selecting step size  $\alpha$  too small will mean it takes a long time to converge to minimum  
* But selecting  $\alpha$  too large can lead to us overshooting the minimum  
* We need to adjust  $\alpha$  so that algorithm converges in a reasonable time.  
* There are also many automated approaches for adjusting  $\alpha$  at each iteration. E.g. using line search (at each gradient descent iteration try several values of  $\alpha$  until find one that causes descent).

* We'll use python and usually sklearn https://scikit-learn.org/stable/index.html in examples and assignments

* Sometimes you'll be asked to implement things from scratch rather than calling the sklearnn function  $\rightarrow$  to help you understand what's happening "under the hood"

* Linear regression:

import numpy as np   
Xtrain  $=$  np.arange(0,1,0.01).reshape(-1,1)   
ytrain  $=$  10+Xtrain  $+$  np.random.normal(0.0,1.0,100).reshape(-1,1)   
from sklearn.linear_model import LinearRegression   
model  $=$  LinearRegression().fit(Xtrain.reshape(-1,1),ytrain.reshape(-1,1))   
print(model.intercept_,model.coef_)

Typical output: 0.0175381 9.77234168

# Plotting model predictions:

import matplotlib.pyplot as plt

plt.rc('font', size=18)

plt.rcParams['figure.constrained.layout.use'] = True

pltscatter(Xtrain,ytrain,color  $\equiv$  black

plt.plot(Xtrain, ypred, color='blue', linewidth=3)

plt.xlabel("input x");pltilated("output y")

pltlegend([“predictions”,“training data”])

plt.show()

![](images/096f4a41af630e1ea8ea25fe93b5582f356ef3a304ad53edb27c6bbd7c7afffa.jpg)

# Notes On Presenting Plots

In your assignments and individual project reports:

* Always label axes in plots  
* Make sure text and numbers are legible and plot is easy to read (use colours, adjust line width/marker size etc). If really illegible, you should expect to lose marks.  
* Always clearly explain what data a plot shows - giving code is not enough, you must explain in english.

If you don't do this you should expect to lose marks.

# $\gg$  Linear Regression with Multiple Variables

Advertising example:

<table><tr><td>TV
x1</td><td>Radio
x2</td><td>Newspaper
x3</td><td>Sales
y</td></tr><tr><td>230.1</td><td>37.8</td><td>69.2</td><td>22.1</td></tr><tr><td>44.5</td><td>39.3</td><td>45.1</td><td>10.4</td></tr><tr><td>17.2</td><td>45.9</td><td>69.3</td><td>9.3</td></tr><tr><td>:</td><td>:</td><td>:</td><td>:</td></tr></table>

*  $n =$  number of features (3 in this example)  
*  $(x^{(i)}, y^{(i)})$  the  $i$ th training example e.g.

$$
\boldsymbol {x} ^ {(1)} = [ 2 3 0. 1, 3 7. 8, 6 9. 2 ] ^ {T} = \left[ \begin{array}{c} 2 3 0. 1 \\ 3 7. 8 \\ 6 9. 2 \end{array} \right]
$$

*  $x_{j}^{(i)}$  is feature  $j$  in the ith training example, e.g.  $x_{2}^{(1)} = 37.8$

# $\gg$  Linear Regression with Multiple Variables

Model:  $h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3$  
e.g.  $\underbrace{h_{\theta}(x)}_{\text{Sales}} = 15 + 0.1 \underbrace{x_1}_{TV} - 5 \underbrace{x_2}_{\text{Radio}} + 10 \underbrace{x_3}_{\text{Newspaper}}$

* For convenience, define  $x_0 = 1$  
* Feature vector  $x = \left[ \begin{array}{l} x_{0} \\ x_{1} \\ x_{2} \\ \vdots \\ x_{n} \end{array} \right]$

\* Parameter vector

$$
\theta = \left[ \begin{array}{c} \theta_ {0} \\ \theta_ {1} \\ \theta_ {2} \\ \vdots \\ \theta_ {n} \end{array} \right]
$$

*  $h_{\theta}(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 +$

$$
\dots + \theta_ {n} x _ {n} = \theta^ {T} x
$$

# $\gg$  Linear Regression with Multiple Variables

* Model:  $h_{\theta}(x) = \theta^{T}x$

(with  $\theta, x$  now  $n + 1$ -dimensional vectors)

* Cost Function:  $J\left( {{\theta }_{0},{\theta }_{1},\ldots ,{\theta }_{n}}\right)  = J\left( \theta \right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\left( {{h}_{\theta }\left( {x}^{\left( i\right) }\right)  - {y}^{\left( i\right) }}\right) {}^{2}$  
* Optimisation: Select  $\theta$  that minimises  $J(\theta)$  
* As before, can find  $\theta$  using e.g using gradient descent:

* Start with some  $\theta$  
\*Repeat:

$$
f o r j = 0 \text {t o} n \left\{\delta_ {j} := - \alpha \frac {\partial}{\partial \theta_ {j}} J (\theta) \right\}
$$

$$
f o r j = 0 \text {t o} n \left\{\theta_ {j} := \theta_ {j} + \delta_ {j} \right\}
$$

# $\gg$  Gradient Descent with Multiple Variables

For  $J(\theta) = \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2$  with  $h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$ :

$\frac{\partial}{\partial\theta_0} J(\theta) = \frac{2}{m}\sum_{i = 1}^{m}\bigl (h_\theta (x^{(i)}) - y^{(i)}\bigr)$  
$\frac{\partial}{\partial \theta_1} J(\theta) = \frac{2}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_1^{(i)}$  
\*  $\frac{\partial}{\partial\theta_j} J(\theta) = \frac{2}{m}\sum_{i = 1}^{m}(h_\theta (x^{(i)}) - y^{(i)})x_j^{(i)}$

So gradient descent algorithm is:

* Start with some  $\theta$  
* Repeat:

$$
\begin{array}{l} f o r j = 0 t o n \left\{\delta_ {j} := - \frac {2 \alpha}{m} \sum_ {i = 1} ^ {m} \left(h _ {\theta} \left(x ^ {(i)}\right) - y ^ {(i)}\right) x _ {j} ^ {(i)} \right\} \\ f o r j = 0 t o n \left\{\theta_ {j} := \theta_ {j} + \delta_ {j} \right\} \\ \end{array}
$$

# Example: Advertising Data

* How is the impact of the advertising spend on TV and radio related, if at all?  
* Perhaps a quadratic fit would be better? If so, what does that imply for how we allocate our advertising budget?

![](images/febf30b482bf39a4cfa1105d9398703013760bc9ee17eeaa2315fdb7efbceaca.jpg)

![](images/16f6988193bf800ed57740ee5d867034f2fe540963c5f977d74866944d8049fd.jpg)
