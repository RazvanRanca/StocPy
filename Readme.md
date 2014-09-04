StocPy
====

StocPy is an expressive [probabilistic programming language](http://probabilistic-programming.org), provided as a Python library. The language follows the "lightweight implementations" style introduced by Wingate, Stuhlmüller and Goodman ([link to pdf](http://www.mit.edu/~ast/papers/lightweight-mcmc-aistats2011.pdf)).

StocPy was developed as part of my [masters thesis](http://www.cl.cam.ac.uk/~rr463/PPL_Thesis.pdf), under the supervision of [Dan Roy](http://danroy.org/) and [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/). 

Please send any questions/suggestions/issues to [Razvan Ranca](http://www.cl.cam.ac.uk/~rr463/) - ranca.razvan@gmail.com

**Warning**: This is alpha-quality software. Expect bugs. 

Features
---

* Intuitive, succinct and flexible model specification , thanks to Python
* Can use any probabilistic primitive defined by [scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions)
* Inference engine easily extensible to new inference methods, currently supports Metropolis, Slice Sampling and weighted combinations of the two
* Automatic rewriting of user models to improve inference, currently via partitioning the priors.

Basic Usage
---
A model is specified as a normal Python function which uses the probabilistic primitives provided by the StocPy library. For instance, inferring the mean of a normal distribution of variance 1 based on a single data point would be written as:

    def guessMean():
       mean = stocPy.normal(0, 1, obs=True)
       stocPy.normal(mean, 1, cond=2)

Here, we define a prior on the mean as a Normal(0,1) and say that we wish to observe what values the mean takes in our simulation (i.e. we wish to store the sampled means). In the next line we say that sampling from a normal with variance 1 which is centred around the mean we obtained in the previous step resulted in us observing a certain data point (2 in this case).

Once this model is defined we can perform inference on it by calling:

    samples = stocPy.getSamples(guessMean, 10000, alg="met")

Where we are asking for 10,000 samples, generated with the Metropolis inference technique (which is also the default). This will return 10,000 samples for each of the variables we asked to be observed (in the above model, this would be just the mean).

Finally, to quickly visualise the distribution of our sampled mean, we can use one of the provided utility functions:

    stocPy.plotSamples(samples)

### Inference API
There are 3 methods available for performing inference. These are:

* `stocPy.getSamples(model, noSamps)` - Generate a certain number of samples
* `stocPy.getTimedSamples(model, noSecs)` - Generate samples until a certain number of seconds have ellapsed 
* `stocPy.getSamplesByLL(model, noLLs)` - Generate samples until a certain number of model simulations have been run. This is useful when benchmarking different inference techniques which may simulate the model multiple times to generate a single sample. 

All 3 of these functions take, via the optional `alg` keyword, the method by which to perform inference. The available methods are:

* `alg="met"` - Metropolis from the prior. The default option.
* `alg="sliceTD"` - Slice sampling based inference.
* `alg="sliceMet"` - A mixture of slice and Metropolis. The mixture weight can be specified by the `thresh` parameter. By default this is 0.1, corresponding to a Metropolis:Slice 1:9 mixture.
* `alg="sliceNoTrans"` - Slice which explicitly disallows trans-dimensional jumps. Useful for analysing the properties of different modes of a distribution.

Less Basic Usage
---

To complement the following descriptions, usage examples covering most of stocPy's functionality are included in the simpleModels and the anglicanModels directories. Each model is explained in its respective ".py" file.

### Observe statement

In addition to specifying that a variable should be observed upon it's definition (as in the guessMean example), we can also specifically tell stocPy to observe variables or expressions via the `observe` method. The syntax is:

    stocPy.observe(expression, name="expressionName")

Using this statement we can record the values of arbitrary expressions at any point in our model specification. The `name` parameter must always be specified when calling the `observe` method. This allows stocPy to aggregate all observations with the same name and gives the user a large degree of flexibility regarding the types of questions they wish to pose. For instance, the user could assign different names to an `observe` statement based on the value of a stochastic variable and thus see the effect that variable has on the observed expression.

### Partitioned priors

A new feature available in StocPy is that of automatically partitioning priors. Intuitively, this re-writes the prior over a variable as a sum of narrower and wider distributions and thus gives Metropolis-Hastings more choice regarding the scale at which it operates. This additional freedom should result in more efficient inference when the marginal prior of a variable is substantially wider than the same variable's marginal posterior.

To use this feature we must specify our variable via the `stocPrim` function (even if a custom wrapper exists for the variable) and pass a parameter representing a partition depth (or prior over partition depths). A big partition depth essentially means we believe there is a large difference between the width of the prior and the width of the posterior (rule of thumb: partition depth should be larger than log(priorVariance/posteriorVariance) ).

For instance, we can specify:

     mean = stocPy.stocPrim("norm", (0, 1), obs=True, part=5)

to partition the prior over the mean up to a depth of 5. Or we could write:

     mean = stocPy.stocPrim("norm", (0, 1), obs=True, part=stocPy.unifCont(2,10))

to place a prior over the partition depth.

**Note:** This feature is still experimental. Use with care. Not currently supported on the `crp` primitve.

### Explicit naming

In order for the inference engine to keep track of the stochastic values present in the current program trace, the naming method introduced by Wingate, Stuhlmüller and Goodman ([link to pdf](http://www.mit.edu/~ast/papers/lightweight-mcmc-aistats2011.pdf)) is employed. This is done via a source-to-source compilation of the file in which a StocPy model is defined (models defined over multiple files are not currently supported).

It is possible, however, to turn-off the automatic compilation and allow the user to manually specify the names of their variables. This may be useful if the user wishes to avoid the overhead introduced by the automatic compilation (eg: for time-critical models) or, more practically, if the user wishes to experiment with novel naming strategies. In order to get rid of the compilation, it is only necessary to pass `autoNames = False` to the inference engine. For instance:

    samples = stocPy.getSamples(guessMean, 10000, alg="met", autoNames=False)

**Warning:** If this parameter is passed, then it is then the user's responsibility to specify a `name` parameter for each call to a stocPy primitive. Further, if the names are not carefully chosen the result may be slowed down or even incorrect inference. See the aforementioned paper for details.

Stochastic primitives
---
In order for StocPy to be able to perform inference on a python model, the model must define its stochastic primitives via StocPy functions, so that the library can keep track of them throughout the model execution.

### Using any scipy.stats primitive
A generic `stocPrim` function is also provided through which any stochastic primitive implemented in [scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions) can be used in StocPy models.

The usage of the stocPrim is demonstrated in the "simpleModels/poisson.py" model, but in short it is:

    stocPy.stocPrim(distributionName, distributionParameters)

Here `distributionName` is a string with the exact name of the primitive in scipy.stats (eg: "beta"), and `distributionParameters` is a tuple holding an arbitrary number of parameters as taken by the scipy.stats `rvs` function corresponding to our distribution.

**Note**: When using the `stocPrim` function the parameters must be given in the same order as that defined by the corresponding scipy.stats `rvs` function. A tuple must be given even if there is a single parameter (i.e use `(parameter, )`).

### Custom wrappers

It can be convenient to add more commonly used primitives directly to StocPy rather than using the generic `stocPrim` function. In this way the primitives can also accept different parametrizations , for instance (start, end) vs (start, length) for a uniform distribution.

At the moment StocPy defines the following wrappers:

* Normal - `stocPy.normal(mean, stDev)`
* Poisson - `stocPy.poisson(shape)`
* Uniform Continuous - `stocPy.unifCont(start, end)`
* Student T - `stocPy.studentT(dof)`
* Inverse Gamma - `stocPy.invGamma(shape, scale)`
* Beta - `stocPy.beta(a, b)`
* Categorical - `stocPy.categorical(probs)`
* Chinese Restaurant Process - `stocPy.crp(a, maxClasses=None)`, where the 2nd argument is optional

### Adding custom wrappers for scipy.stats primitives
Creating new wrappers is easily done by performing the following changes in stocPy.py.

* Add the desired scipy.stats distribution to the `dists` array
* Add a (key:value) pair consisting of (distribution name : distribution index in `dists` array) to the `erps` dictionary
* Create a 3 line wrapper function which gets the distribution parameters from the user, calls `initERP` and then returns `getERP`. See the wrappers provided in stocPy (normal, poisson, etc.) to understand the pattern this wrapper follows (**Note:** the `name` attribute has to come last in the wrapper function's parameter list and must not be passed via \*args or **kwargs, otherwise the automatic naming breaks down).

### Adding new primitives to stocPy
While scipy.stats has good support for univariate distributions, we sometimes need other stochastic primitives. In this case a new class implementing the desired primitive has to be defined. In StocPy any class implementing a new primitive must have:

* `rvs(self, params)` - which returns a random sample from the defined distribution when parametrized by `params`
* `logpdf(self, val, params)` OR `logpmf(self, val, params)` - depending on whether the primitive is continuous of discrete. This returns the probability density or the probability mass associated with the value `val` under the distribution parametrized by `params`
* `ppf(self, prob, params)` - **optional, needed for partitioned priors** - the percent point function of the distribution (also known as the inverse cumulative distribution function).

Once a class with the above methods is defined, the steps are similar to those for adding a custom wrapper. Namely:

* Add an instance of the desired class to the `dists` array.
* Add a (key:value) pair consisting of (distribution's name : distribution index in `dists` array) to the `erps` dictionary
* **Optional**: define a 3 line wrapper function through which the user can access the custom distribution class without having to go through `stocPrim`. In order for the user view of the distribution names to be consistent, it is a convention in StocPy to give a distribution's wrapper function the same name as the distribution's name defined in `erps`. This may mean the internal distribution class has a different name, or different capitalization.

Two examples of custom distribution classes are given in `stocPy.py`, namely `Categorical` and `CRP`.

Installation
---
The easiest way is to clone the repository with `git clone git@github.com:RazvanRanca/StocPy.git` and install StocPy using `python setup.py install` (which might require root access).

Development
---
This repository is intended to hold a version of StocPy that is relatively easy to use. There is also a [development repository](https://github.com/RazvanRanca/StocPyDev) which contains some work-in-progress features and a lot more experimental data regarding the performance of the inference methods on different models.

Contact
---
Please send any questions/suggestions/issues to [Razvan Ranca](http://www.cl.cam.ac.uk/~rr463/) - ranca.razvan@gmail.com
