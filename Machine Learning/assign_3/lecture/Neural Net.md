» **NeuralNetworks**

> *∗* Linear model: *y* = *θTx* = *θ*0*x*0 + *θ*1*x*1 + *θ*2*x*2 + *...*
> *∗* Draw this schematically as:
>
> *x*0
>
> *x*1
>
> . .

*θ*0

*θ*1

> ˆ
>
> *θn*
>
> *xn*
>
> *∗* A small generalisation: *y* = *f*(*θTx*) where *f* is some
> function e.g. *sign*
>
> *x*0
>
> *x*1
>
> . .

*θ*0

*θ*1

> *f* ˆ
>
> *θn*
>
> *xn*
>
> NB:We first take the weighted sum of the inputs *x*1, *x*2 etc and
> then apply function *f* to result.

» **Multi-LayerPerceptron(MLP)**

> *∗* To get an MLP we add an extra “layer”. E.g.
>
> *x*0
>
> *x*1
>
> . .
>
> \[1\]

01 \[1\]

> 11
>
> \[1\] 02

\[1\] \[1\] *n*1 12

*f* *θ*\[2\]

> \[2\] *g* ˆ 2
>
> \[1\] *n*2
>
> *xn*

*f*

> output layer
>
> hidden layer input layer
>
> *z*1 = *f*(*θ*\[1\]*x*0 + *θ*\[1\]*x*1 + *···* + *θ*\[1\]*xn*) *z*2 =
> *f*(*θ*\[1\]*x*0 + *θ*\[1\]*x*1 + *···* + *θ*\[1\]*xn*) *y* =
> *g*(*θ*\[2\]*z*1 + *θ*\[2\]*z*2)
>
> *∗* MLP is a three layer network: (i) an *input* *layer*, (ii) a
> *hidden* *layer*, (iii) an *output* *layer*
>
> *∗* Not restricted to two nodes in hidden layer, can have many nodes.
> *∗* Not restricted to one output, can have many outputs
>
> *∗* The parameters *θ*\[2\] etc are called *weights*. It quickly gets
> messy indexing all the weights, often they’re omitted from these
> schematics
>
> *∗* The function *f* is called the *activation* function, *g* is the
> output
>
> function.

» **Multi-LayerPerceptron(MLP)**

Example

> *∗* One input, two nodes in hidden layer, activation function is
> sigmoid *f*(*x*) = *g*(*x*) = 1+*ex* .
>
> *z*1 = *f*(5*x*)*,z*2 = *f*(2*x*)*,y* = *f*(*z*1 *−* 2*z*2) =
> *f*(*f*(5*x*) *−* 2*f*(2*x*))
>
> 0.45
>
> 0.4
>
> 0.35
>
> 0.3
>
> -1 -0.5 0 0.5 1
>
> input x
>
> *∗* By varying the number of hidden nodes and the weights the MLP can
> generate a wide range of functions mapping input *x* to output *y*.

» **ChoicesofActivation&OutputFunction** {

> *∗* ReLU (Rectified Linear Unit) *f*(*x*) = 0

*x* *≥* 0

*x* *\<* 0

> *∗* Popular in *hidden* *layer*. Quick to compute, observed to work
> pretty well.
>
> *∗* But can lead to “dead” neurons where output is always zero *→*
> leaky ReLU
>
> *∗* Sigmoid *g*(*x*) = 1+*ex*
>
> *∗* Sigmoid used in *output* *layer* when output is a probability (so
> between 0 and 1). For classification problems predict +1 when 1+*ex*
> *\>* 0*.*5, *−*1 when 1+*ex* *\<* 0*.*5
>
> *∗* tanh *g*(*x*) = *ex−e−x*
>
> *∗* Used to be common for hidden layers, not so much now.
>
> <img src="./0plcuu5a.png"
> style="width:1.12359in;height:0.54714in" />2
>
> 1.5
>
> 1
>
> 0.5
>
> 0
>
> -0.5
>
> -12 -1 0 1 2 x

» **CostFunction&Regularisation** Cost function:

> *∗* Typically use logistic loss function for classification problems
>
> *∗* And square loss *m* *i*=1(*hθ*(*x*(*i*))*−y*(*i*))2 for regression
> problems *∗* In both cases the cost function is non-convex in the
> neural net
>
> weights/parameters *→* training a neural net can be tricky
>
> <img src="./w0xaem4s.png"
> style="width:1.15281in;height:0.90976in" /><img src="./bu3h50wy.png" style="width:0.13294in" /><img src="./mjhh0alp.png"
> style="width:0.18825in;height:0.13956in" /><img src="./o3mneprb.png" /><img src="./2211mjk0.png"
> style="width:0.15242in;height:0.14239in" /><img src="./cpu1vewo.png" style="width:0.1645in" /><img src="./uwglii10.png"
> style="width:0.12878in;height:0.10889in" /><img src="./bhbny11t.png"
> style="width:0.22265in;height:0.16427in" /><img src="./n3g4utex.png" style="height:0.12997in" /><img src="./sehkgijw.png" style="width:0.18216in" /><img src="./34djm0fm.png"
> style="width:0.15202in;height:0.1223in" /><img src="./qqtq2pxf.png"
> style="width:0.25147in;height:0.18751in" /><img src="./nf4dpxzg.png"
> style="width:0.22305in;height:0.10916in" /><img src="./jl55d2qg.png"
> style="width:0.27704in;height:0.20877in" /><img src="./iew4bgpg.png"
> style="width:0.17201in;height:0.13614in" /><img src="./noc1vlcf.png"
> style="width:0.24629in;height:0.11956in" /><img src="./nwltdsft.png"
> style="width:0.192in;height:0.16439in" /><img src="./ip435k4m.png"
> style="width:0.37172in;height:0.31068in" /><img src="./kgaj2cnc.png"
> style="width:0.20316in;height:0.15429in" /><img src="./ymz2srwz.png" style="height:0.2264in" /><img src="./ch3rpz3v.png" style="height:0.10577in" /><img src="./vem5m3va.png"
> style="width:0.18399in;height:0.19598in" /><img src="./vplyipr0.png" style="height:0.10101in" /><img src="./f0hj3axi.png" /><img src="./qbivpkf3.png"
> style="width:0.36523in;height:0.31306in" /><img src="./mdjzhx2c.png" style="width:0.12177in" /><img src="./rkjxxycp.png"
> style="width:0.23432in;height:0.19184in" /><img src="./aidsewti.png"
> style="width:0.30068in;height:0.15224in" /><img src="./3esohk1e.png"
> style="width:0.41729in;height:0.31664in" /><img src="./hzysyfaj.png"
> style="width:0.25431in;height:0.20591in" /><img src="./1mvkrwc5.png"
> style="width:0.15942in;height:0.11679in" /><img src="./5a1gj0nm.png"
> style="width:0.61497in;height:0.32008in" /><img src="./zwf4yhfd.png"
> style="width:0.13091in;height:0.28604in" /><img src="./y22wcrtp.png"
> style="width:0.2743in;height:0.21943in" /><img src="./mlovmw53.png"
> style="width:0.21656in;height:0.14473in" /><img src="./iiiosc2k.png"
> style="width:0.52708in;height:0.29082in" /><img src="./5h3kgqkc.png"
> style="width:0.17942in;height:0.13548in" /><img src="./bero4nz0.png"
> style="width:0.19058in;height:0.1408in" /><img src="./xxc34mds.png"
> style="width:0.19332in;height:0.1261in" /><img src="./tc34inyu.png"
> style="width:1.13049in;height:0.59292in" /><img src="./ugagdkyb.png"
> style="width:0.158in;height:0.11854in" /><img src="./bo2ogixi.png"
> style="width:0.76465in;height:0.43996in" /><img src="./nhnzslz5.png" style="width:0.13618in" /><img src="./zmuvtpuo.png"
> style="width:0.22631in;height:0.11401in" /><img src="./b2jn50ig.png" style="height:0.1658in" /><img src="./pmfsidsl.png"
> style="width:1.15281in;height:0.90976in" /><img src="./pijdggdq.png" style="width:0.13294in" /><img src="./0tgjqvck.png"
> style="width:0.18825in;height:0.13956in" /><img src="./xwzmnfpp.png" /><img src="./m3pdao1x.png"
> style="width:0.15242in;height:0.14239in" /><img src="./ifa2pttg.png" style="width:0.1645in" /><img src="./yptdm15w.png"
> style="width:0.12877in;height:0.10889in" /><img src="./vrceeg2d.png"
> style="width:0.22265in;height:0.16427in" /><img src="./gkrzkpnv.png" style="height:0.12997in" /><img src="./3ipp1oqn.png" style="width:0.18216in" /><img src="./anmspshi.png"
> style="width:0.15202in;height:0.1223in" /><img src="./ud5zkn5p.png"
> style="width:0.25147in;height:0.18751in" /><img src="./vn4np5on.png"
> style="width:0.22305in;height:0.10916in" /><img src="./nyj2hlkl.png"
> style="width:0.27704in;height:0.20877in" /><img src="./rryld14a.png"
> style="width:0.172in;height:0.13614in" /><img src="./hogngkif.png"
> style="width:0.24629in;height:0.11956in" /><img src="./fljnsdld.png"
> style="width:0.192in;height:0.16439in" /><img src="./wvf200bp.png"
> style="width:0.37172in;height:0.31068in" /><img src="./vrb14x2u.png"
> style="width:0.20316in;height:0.15429in" /><img src="./xjgtkptx.png" style="height:0.2264in" /><img src="./iam12nw1.png" style="height:0.10577in" /><img src="./vnolhv0h.png"
> style="width:0.18399in;height:0.19598in" /><img src="./jxb42esm.png" style="height:0.10101in" /><img src="./qxdivfuo.png" /><img src="./zr2dzvpj.png"
> style="width:0.36523in;height:0.31306in" /><img src="./xymbsx3p.png" style="width:0.12178in" /><img src="./vfssrwl1.png"
> style="width:0.23432in;height:0.19184in" /><img src="./3rtheug2.png"
> style="width:0.30068in;height:0.15224in" /><img src="./ict00hix.png"
> style="width:0.41729in;height:0.31664in" /><img src="./n11xyxiu.png"
> style="width:0.25431in;height:0.20591in" /><img src="./zyxtioxk.png"
> style="width:0.15942in;height:0.11679in" /><img src="./h1kit3fl.png"
> style="width:0.61497in;height:0.32008in" /><img src="./ub1cou00.png"
> style="width:0.13091in;height:0.28604in" /><img src="./dr3yy4a0.png"
> style="width:0.2743in;height:0.21943in" /><img src="./iabjc5hz.png"
> style="width:0.21656in;height:0.14473in" /><img src="./vyfsq3up.png"
> style="width:0.52708in;height:0.29082in" /><img src="./jflgmawq.png"
> style="width:0.17942in;height:0.13548in" /><img src="./uhywcral.png"
> style="width:0.19058in;height:0.1408in" /><img src="./20xlzp30.png"
> style="width:0.19332in;height:0.1261in" /><img src="./s141lgcj.png"
> style="width:1.13049in;height:0.59292in" /><img src="./1uax0oau.png"
> style="width:0.158in;height:0.11854in" /><img src="./0cnnll1q.png"
> style="width:0.76465in;height:0.43996in" /><img src="./35qpthsk.png"
> style="width:0.2263in;height:0.11401in" /><img src="./33kzsyck.png"
> style="width:0.47135in;height:0.23121in" />20 20
>
> 10 10
>
> 0 0
>
> -10 -10
>
> -20 -20 8 8
>
> 7 7
>
> 6 6
>
> 5 8 5 8 4 6 7 4 6 7
>
> 3 5 3 5 2 3 4 2 3 4
>
> 1 2 1 2 0 0 1 0 0 1

Regularisation

> *∗* *L*2 penalty i.e. the sum of the squared weights/parameters *∗*
> Nowadays more common to use *dropout* regularisation
>
> *∗* Set the outputs of randomly selected set of nodes in hidden layer
> to zero at each gradient descent step
>
> *∗* Typically remove about 50% of nodes
>
> *∗* This is similar1 to a *weighted* *L*2 penalty *i*=1 *wiθi* 1

» **MovieReviewExample**<img src="./qozfzvtc.png"
style="width:1.91338in;height:1.43503in" /><img src="./q3uj1mrk.png"
style="width:1.91338in;height:1.43503in" />

Apply MLP to movie review example. Use cross-validation to select (i)
\#hidden nodes, (ii) *L*2 penalty weight *C*.

> *∗* Performance not too sensitive to \#hidden nodes, so choose a small
> number e.g. 2
>
> *∗* Not much sign of overfitting, at least for this range of \#hidden
> nodes (*C* = 1 in plot).
>
> *∗* Performance insensitive to penalty weight *C*, so long as *C* *≥*
> 0*.*5 or thereabouts (#hidden nodes=2 in plot)

» **MovieReviewExample** MLP settings:

> *∗* hidden layer has 2 nodes, penalty weight *C* = 1, ReLU activation
> function

Confusion matrix:

> true positive true negative
>
> predicted positive predicted negative

with *m* = 160 data points (20% test split from full data set of 800
points).

<img src="./331qqtdg.png"
style="width:1.91338in;height:1.43503in" />ROC Curve:

» **PythonCodeForMLPMovieExample**

import matplotlib.pyplot as plt

plt.rc(’font’, size=18);plt.rcParams\[’figure.constrained_layout.use’\]
= True

from sklearn.neural_network import MLPClassifier from
sklearn.model_selection import cross_validate crossval=False

if crossval:

> mean_error=\[\]; std_error=\[\]; mean_error1=\[\]; std_error1=\[\]
> hidden_layer_range = \[1,2,3,4,5,10,15,25\]
>
> for n in hidden_layer_range: print(”hidden layer size %d\n”%n)
>
> model = MLPClassifier(hidden_layer_sizes=(n), alpha=1, max_iter=500)
>
> scores = cross_validate(model, X, y, cv=5, return_train_score=True,
> scoring=’roc_auc’)
> mean_error.append(np.array(scores\[’test_score’\]).mean());
> std_error.append(np.array(scores\[’test_score’\]).std())
> mean_error1.append(np.array(scores\[’train_score’\]).mean());
> std_error1.append(np.array(scores\[’train_score’\]).

std())

> plt.errorbar(hidden_layer_range,mean_error,yerr=std_error,linewidth=3)
> plt.errorbar(hidden_layer_range,mean_error1,yerr=std_error1,linewidth=3)
> plt.xlabel(’#hidden layer nodes’); plt.ylabel(’AUC’)
>
> plt.legend(\[’Test Data’, ’Training data’\]) plt.show()

mean_error=\[\]; std_error=\[\]; mean_error1=\[\]; std_error1=\[\]
C_range = \[0.001,0.01,0.1,0.5,1,2,5,10\]

for C in C_range:

> print(”C %d\n”%Ci)
>
> model = MLPClassifier(hidden_layer_sizes=(2), alpha = 1.0/C)
>
> scores = cross_validate(model, X, y, cv=5, return_train_score=True,
> scoring=’roc_auc’)
> mean_error.append(np.array(scores\[’test_score’\]).mean());
> std_error.append(np.array(scores\[’test_score’\]).std())
> mean_error1.append(np.array(scores\[’train_score’\]).mean());
> std_error1.append(np.array(scores\[’train_score’\]).std())

plt.errorbar(hidden_layer_range,mean_error,yerr=std_error,linewidth=3)
plt.errorbar(hidden_layer_range,mean_error1,yerr=std_error1,linewidth=3)
plt.xlabel(’C’); plt.ylabel(’AUC’)

plt.legend(\[’Test Data’, ’Training data’\]) plt.show()

» **PythonCodeForMLPMovieExample(cont)**

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(2), alpha=1.0).fit(Xtrain,
ytrain) preds = model.predict(Xtest)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, preds))

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy=”most_frequent”).fit(Xtrain, ytrain)
ydummy = dummy.predict(Xtest)

print(confusion_matrix(ytest, ydummy))

from sklearn.metrics import roc_curve preds = model.predict_proba(Xtest)
print(model.classes\_)

fpr, tpr, \_ = roc_curve(ytest,preds\[:,1\]) plt.plot(fpr,tpr)

from sklearn.linear_model import LogisticRegression model =
LogisticRegression(C=10000).fit(Xtrain, ytrain)

fpr, tpr, \_ = roc_curve(ytest,model.decision_function(Xtest))
plt.plot(fpr,tpr,color=’orange’)

plt.legend(\[’MLP’,’Logistic Regression’\]) plt.xlabel(’False positive
rate’) plt.ylabel(’True positive rate’)

plt.plot(\[0, 1\], \[0, 1\], color=’green’,linestyle=’−−’) plt.show()

» **TrainingNeuralNetworks:StochasticGradientDescent**

Recall gradient descent to minimise cost function *J*(*θ*): *∗* Start
with some parameter vector *θ* of size *n*

> *∗* Repeat:
>
> *θj* := *θj* *−* *α* *<u>∂J</u>* (*θj*) for *j*=0 to *n*

Cost function is a sum over prediction error at each training point,
e.g. *J*(*θ*) = *m* *i*=1(*hθ*(*x*(*i*)) *−* *y*(*i*))2. Rewrite as

> ∑ *J*(*θ*) = *li*(*θ*)
>
> *i*=1

where e.g. *li*(*θ*) = (*hθ*(*x*(*i*)) *−* *y*(*i*))2. Then

> *m*
>
> *∂θj* (*θ*) = *m* *i*=1 *∂θj* (*θ*)

When *m* is large then calculating this sum becomes slow to calculate.

» **TrainingNeuralNetworks:StochasticGradientDescent**

Stochastic gradient descent (SGD) to minimise cost function *J*(*θ*):
*∗* Pseudo-code:

> *∗* Start with some parameter vector *θ* of size *n* *∗* Repeat:
>
> Pick training data point *i*,
>
> e.g. randomly or by cycling through all data points. *θj* := *θj* *−*
> *α∂θj* (*θj*) for *j*=0 to *n*
>
> *∗* At each update we use just one point from the training data, so
> avoid sum over all points i.e. SGD update is:
>
> *θj* *←* *θj* *−* *α* *∂li* (*θj*) *j*
>
> instead of gradient descent update:
>
> *θj* *←* *θj* *−* *α* 1 *m* *∂li* (*θj*)
>
> *i*=1 *j* *∗* Each SGD update is fast to compute

*∗* But need more iterations to minimise *J*(*θ*). Now add mini-batches
and parallelise ...

» **TrainingNeuralNetworks:StochasticGradientDescent**

Stochastic gradient descent with mini-batches of size *q*: *∗* SGD
update with mini-batch size *q*:

> ∑
>
> *θj* *←* *θj* *−* *αq* *i∈Iq* *∂θj* (*θj*)
>
> where *Iq* is a set of *q* training data points.
>
> *∗* Choose *Iq* by shuffling training data and then cycling through it
> or by selecting *q* points randomly from training data
>
> *∗* Pseudo-code:
>
> *∗* Training data with *m* points, mini-batch size *q* *∗* Repeat:
>
> *∗* shuffle training data *∗* for *k* = 1 to *m* step *q*:
>
> *∗* *δj* = 0 for *j*=0 to *n*
>
> *∗* for *i* = 1 to *q*: \# *k*’th mini-batch *δj* := *δj* + *∂lk*+*i*
> (*θj*) for *j*=0 to *n*
>
> *θj* := *θj* *−* *q* *δj* for *j*=0 to *n*

» **TrainingNeuralNetworks:StochasticGradientDescent** If have *k*
processors and mini-batch size *q*:

> *∗* Divide *q* into *k* batches of size *q*/*k*.
>
> *∗* Parallelise the for *i* = 1 to *q* loop i.e. split into *k* for
> *i* = 1 to *q*/*k* loops and run each on one processor *→* *q* loop
> now runs *k* times faster.

How to choose mini-batch size *q*? *∗* Typical values of *q* are 32-256

> *∗* Computation time tends to increase when batch size *q* gets too
> small (can’t exploit parallelism as well, communication and
> synchronization costs between processors are amortised by using larger
> *q*/*k*)
>
> *∗* Small batches provide a sort of regularisation. Using large
> batches is often observed to lead to over-fitting (poor predictions
> for new data).
>
> *∗* This aspect remains poorly understood, best we have are
> heuristics2
>
> *∗* Note: choosing batch size *q* = *m* the training data size then
> mini-batch SGD = gradient descent

2[See “On Large-Batch Training For
Deep](https://openreview.net/pdf?id=H1oyRlYgg) Learning”, ICLR 2017
<https://openreview.net/pdf?id=H1oyRlYgg>

» **SomeTerminology**

When using SGD in sklearn and other packages you might see the following
terms:

> *∗* *Epoch*.
>
> *∗* SGD update with mini-batch size *q*:
>
> ∑
>
> *θj* *←* *θj* *−* *αq* *i∈Iq* *∂θj* (*θ*)
>
> where *Iq* is a set of *q* training data points.
>
> *∗* If shuffle training data and then cycle through it, then one cycle
> = an epoch i.e. one iteration of repeat loop in our pseudo-code
>
> *∗* After first epoch each training data point has been used once, in
> second epoch twice, and so on
>
> *∗* Often train for a fairly small number of epochs, e.g. 1-25
>
> *∗* *Momentum*. With SGD the gradient updates are “noisy”. So can
> average out this “noise” to try to find a good downhill direction.
>
> *∗* *Adam*. An approach for automatically choosing the step-size *α*
> plus using momentum. Currently the default in most packages, its ok to
> leave it that way for assignments in this module.

For those of you taking optimization module in semester 2 we’ll go into
these in a lot more detail.

» **SomeTerminology**

*Early* *stopping*:

> *∗* An old idea is to try to achieve regularisation by stopping SGD
> early i.e. before cost function as converged to its minimum *→* early
> stopping
>
> *∗* Repeat:
>
> *∗* Keep a hold-out test set from training data e.g. 10% of data *∗*
> Run SGD for a while, e.g. 1 epoch, on remaining training data *∗*
> Evaluate cost function on (i) held-out test data and (ii) on
>
> training data used for SGD
>
> *∗* Stop when cost function of test data stops decreasing and/or when
> these two values start to diverge
>
> *∗* Often used with SGD in combination with a penalty or dropouts for
> regularisation

» **TrainingNeuralNetworks:StochasticGradientDescent**

Calculating gradient *∂li* for neural nets *j*

> *∗* Calculate output *y* of neural network *→* *forward* *propagation*
> (the sorts of neural nets we’re considering are sometimes called
> *feedforward* *networks*
>
> *x*0
>
> *x*1
>
> . .
>
> *θ*01
>
> *θ*11 *f* *θ*1 *θ*02

*θn*1 *θ*12 *θ*2

*g* ˆ

> *θn*2 *f*
>
> *xn* output layer
>
> hidden layer input layer
>
> Apply training data input *x*(*i*) to hidden layer and calculate
> outputs of hidden layer, then apply outputs from hidden layer to
> output layer and calculate output *y*.

» **TrainingNeuralNetworks:StochasticGradientDescent**

> *∗* To calculate derivatives *∂θj* for all weights/parameters *j*
> efficiently use *backpropagation*.
>
> *∗* Calculate difference between neural network output *y* and
> training data output *y*(*i*). Adjust weights *θ*2, *θ*2 connecting
> hidden layer and output layer to reduce this error.
>
> *∗* Now calculate how hidden layer outputs should be adjusted to
> reduce error. Adjust weights *θ*01 etc connecting input layer to
> hidden layer accordingly.
>
> *x*0
>
> *x*1
>
> . .
>
> *θ*01
>
> *θ*11 *f* *θ*1 *θ*02

*θn*1 *θ*12 *θ*2

*g* ˆ

> *θn*2 *f*
>
> *xn* output layer
>
> hidden layer input layer
>
> *∗* Backpropagation = process for calculating *∂θj* for all weights
> *θj*. But often backpropagation is also used as shorthand for the
> whole process of stochastic gradient descent.
>
> *∗* Details of calc don’t matter for us, all we need to know is that
>
> output is the *∂li* ’s *j*

» **Summary**

> *∗* A neural net is just another model i.e. a function mapping from
> input to prediction. Biological analogies are generally spurious and
> just confusing hype.
>
> *∗* Hard to interpret what the weights mean *→* its a *black* *box*
> model
>
> *∗* Can be tricky/slow to train *→* cost function is non-convex in
> weights/parameters, plus often many weights/parameters that need to be
> learned
>
> *∗* Popular in 1990s, then less so. Resurgence of interest from around
> 2010 due to use in image processing *→* mainly relates to their use
> for feature engineering and especially the use of *convolutional*
> *layers* and *transformer* *blocks*.

» **SoftMaxLayer:ExpressingLogisticRegressionInNNTerminology** *∗* One
layer of a neural net is *y* = *f*(*θTx*):

> *x*0
>
> *x*1
>
> . .

*θ*0

*θ*1

> *f* ˆ
>
> *θn*
>
> *xn*
>
> *∗* Select *f*(*·*) to be sign function and we’re back to logistic
> regression/SVM model
>
> *∗* Alternatively, recall that in logistic regression with two classes
> we map from *θTx* to a confidence value using:
>
> *θTx*
>
> <img src="./bq0sizqj.png"
> style="width:0.12674in;height:0.14757in" /><img src="./3ko0r5oi.png" style="width:0.17659in" /><img src="./3er4bvip.png" style="width:0.14933in" /><img src="./viftlmcj.png"
> style="width:1.5739in;height:0.91095in" />*Prob*(*y* = 1) = *eθTx* +
> 1*,* *Prob*(*y* = 0) = 1 *−* *Prob*(*y* = 1) = *eθTx* + 1

» **SoftMaxLayer:ExpressingLogisticRegressionInNNTerminology**

> *∗* Define two outputs *y*1 = *f*(*θTx*) = *eθTx*+1, *y*2 = *θ* *x*+1.
> If *y*1 *\>* *y*2 predict class 1, else predict class 2.
>
> *∗* What about if have *K* *\>* 2 classes?
>
> *∗* Train a separate linear model for each *k* = 1*,...,K*, so have
> *zk* = *θTx* where *θk* is vector of parameters for class *k*.
>
> *∗* Probabilities have to sum to 1, so then normalise: *yk* =
> *Prob*(*y* = *k*) = ∑*Kezk* *z*
>
> *k*=1
>
> *∗* Predict class based on which output *yk* is largest
>
> *∗* This is called *softmax* function. Can draw schematically as:
>
> *x*0
>
> *x*1
>
> . .
>
> *xn*
>
> \[1\]

01 \[1\]

> 11
>
> \[1\] 02

\[1\] \[1\] *n*1 12

> \[1\] *n*2

*f* *y*1

*f* *y*2

> *∗* Called a *softmax* layer *→* its identical to a multi-class
> logistic regression model, despite NN terminology
