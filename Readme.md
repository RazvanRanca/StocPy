StocPy
====

StocPy is an expressive [probabilistic programming language](http://probabilistic-programming.org) written in Python. The language follows the "lightweight implementations" style introduced by Wingate, Stuhlmüller and Goodman ([link to pdf](http://www.mit.edu/~ast/papers/lightweight-mcmc-aistats2011.pdf)).

StocPy was developed as part of my [masters thesis](http://www.cl.cam.ac.uk/~rr463/PPL_Thesis.pdf)), under the supervision of [Dan Roy](http://danroy.org/) and [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/). 

Please send any questions/suggestions/issues to [Razvan Ranca](http://www.cl.cam.ac.uk/~rr463/) - ranca.razvan@gmail.com

**Warning**: This is alpha-quality software. Expect bugs. 

Features
---

* Intuitive, succinct and flexible model specification , thanks to Python
* Easily extensible. Simple to add handling of different probability distributions
* Modular inference engine architecture allowing for the implementation of different inference techniques. Currently Metropolis, Slice Sampling and combinations of the two are supported (note, slice sampling is still work in progress and may not work correctly on all models)

Basic Usage
---
A model is specified as a normal Python function which uses the probabilistic primitives provided by the StocPy library. For instance, inferring the mean of a normal distribution of variance 1 based on a single data point would be written as:

    def guessMean():
       mean = stocPy.normal(0, 1, obs=True)
       stocPy.normal(mean, 1, cond=2)

Here, we define a prior on the mean as a Normal(0,1) and also say that we wish to observe what values the mean will take in our simulation (i.e. we wish to sample the mean). In the next line we say that sampling from a normal with our mean and variance 1 gave us a certain data point. Thus we condition the model on our data point (2 in this case).

Once this model is defined we can perform inference on it by calling:

    samples = stocPy.getSamples(guessMean, 10000, alg="met")
Where we are asking for 10,000 samples, generated with the Metropolis inference technique (which is also the default). This will return 10,000 samples for each of the variables we asked to be observed (in the above model, this would be just the mean).

Finally, several utility functions are provided. For instance, to quickly visualise the distribution of our samples, we could call:

    stocPy.plotSamples(samples)

### Less Basic Usage
For more usage examples (including more advanced cases), please see the models directory. Each model is explained in its respective ".py" file.

Stochastic primitives
---
In order for StocPy to be able to perform inference on a python model, the model must define its stochastic primitives via StocPy functions, so that the library can keep track of them througout the model execution.

At the moment StocPy defines the following stochastic primitives:

* Normal - `stocPy.normal(mean, stDev)`
* Poisson - `stocPy.poisson(shape)`
* Uniform Continuous - `stocPy.unifCont(start, end)`
* Student T - `stocPy.studentT(dof)`
* Inverse Gamma - `stocPy.invGamma(shape, scale)`
* Beta - `stocPy.beta(a, b)`

### Using any scipy.stats primitive
A generic `stocPrim` function is also provided through which any stochastic primitive implemented in [scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions) can be used in StocPy models.

The usage of the stocPrim is demonstrated in the "simplePoisson" model, but in short it is:

    stocPy.stocPrim(distributionName, distributionParameters)

Here `distributionName` is a string with the exact name of the primitive in scipy.stats (eg: "beta"), and `distributionParameters` is a tuple holding an arbitrary number of parameters as taken by the scipy.stats `rvs` function corresponding to our distribution.

**Note**: When using the `stocPrim` function the parameters must be given in the same order as that defined by the corresponding scipy.stats `rvs` function. A tuple must be given even if there is a single parameter (i.e use `(parameter, )`).

### Adding new primitives to StocPy
It might be convenient to add more commonly used primitives directly to StocPy rather than using the generic `stocPrim` function. This is easily done by performing the following changes in stocPy.py.

* Add the desired distribution to the `dists` array
* Add a (key:value) pair consisting of (distribution name : distribution index in `dists` array) to the `erps` dictionary
* Create a 3 line wrapper function which gets the distribution parameters from the user, calls `initERP` and then returns `getERP`. See the wrappers provided in stocPy (normal, poisson, etc.) to understand the pattern this wrapper follows.

Installation
---
The easiest way is to clone the repository with `git clone git@github.com:RazvanRanca/StocPy.git` and install StocPy using `python setup.py install` (which might require root access).

Development
---
This repository is intended to hold a version of StocPy that is relatively easy to understand and use by others. There is also a [development repository](https://github.com/RazvanRanca/StocPyDev) which contains some work-in-progress features and more experimental data regarding the performance of the inference methods on different models.

Contact
---
Please send any questions/suggestions/issues to [Razvan Ranca](http://www.cl.cam.ac.uk/~rr463/) - ranca.razvan@gmail.com
