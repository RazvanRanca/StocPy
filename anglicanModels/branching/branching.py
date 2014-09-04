import stocPy
import cPickle

"""
This is a model borrowed from the paper: "A New Approach to 
Probabilistic Programming Inference" by Wood, Van de Meent and Mansinghka

The branching model doesn't have an intuitive description and is instead
meant to test inference performance on trans-dimensional models.

This model is also unusual in that it is conditioned on a single datapoint.
"""
def branching():
  r = stocPy.poisson(4, obs=True)
  if r > 4:
    l = 6
  else:
    p2 = stocPy.poisson(4)
    l = fib(3*r) + p2 

  stocPy.poisson(l, cond=6)

"""
Return the nth fibonacci number
"""
def fib(n):
  if n == 0:
    return 0
  if n < 3:
    return 1
  else:
    a = 1
    b = 1
    for i in range(3,n):
      c = a+b
      a = b
      b = c
    return a+b

"""
Utility function which displays the convergence rate of different sample runs.
The experiment files where removed from the StocPy repository due to their large size,
but can be obtained at: https://github.com/RazvanRanca/StocPyDev/tree/master/models/branching/experiments
"""
def displayExperiments(xlim=20000):
  paths = ["metRunsByLL", "sliceRunsByLL", "sliceMet01RunsByLL", "sliceMet05RunsByLL"]
  titles = ["Metropolis", "Slice", "Slice:Met 1:9", "Slice:Met 1:1"]

  runs = []
  for path in paths:
    with open(stocPy.getCurDir(__file__) + "/experiments/" + path,'r') as f:
      runs.append(cPickle.load(f))

  stocPy.calcKLTests(stocPy.getCurDir(__file__) + "/posterior", runs, titles, xlim=xlim, burnIn = 1000, modelName = "Branching") # show all runs
  stocPy.calcKLSumms(stocPy.getCurDir(__file__) + "/posterior", runs, titles, xlim=xlim, burnIn = 1000, modelName = "Branching") # show run quartiles

if __name__ == "__main__":
  stocPy.plotSamples(stocPy.getSamplesByLL(branching, 1000, alg="met"), xlabel = "r") # extract samples untill 10,000 trace log-likelihood calculations are performed
