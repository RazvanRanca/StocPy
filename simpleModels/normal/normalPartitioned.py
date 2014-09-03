import stocPy
import cPickle 

"""
Model descriptions:

Here we have one model which tries to infer the mean of a gaussian given 1,000 datapoints stored in "normalData".
This model is defined in 3 ways as to show StocPy's automatic prior partitioning capabilities.

The first definition ("normal") does not use the partitioning.

The second definition ("normalPartConst"), tells StocPy to partition the prior on the mean to a depth of 10.
This means the prior will be represented internally as the sum of 10 normals with exponentially decreasing variance.

The final definition ("normalPartPrior"), tells StocPy to put a prior on the depth of the partition. The depth of
the partition is now drawn uniformly from [0,20].

The performance of the 3 definitions can be seen in /experiments/normalPart.png
"""

def loadData(fn):
  with open(fn, 'r') as f:
    return cPickle.load(f)

normalData = loadData(stocPy.getCurDir(__file__) + "/normalData")

def normal():
  m = stocPy.unifCont(0, 10000, obs=True)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normalPartConst():
  m = stocPy.stocPrim("uniform", (0, 10000), obs=True, part=10)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normalPartPrior():
  m = stocPy.stocPrim("uniform", (0, 10000), obs=True, part=stocPy.stocPrim("randint", (0, 21)))
  for datum in normalData:
    stocPy.normal(m, 1, datum)

if __name__ == "__main__":
  print stocPy.getTimedSamples(normal, 10, alg="met")
  #print stocPy.getTimedSamples(normalPartConst, 10, alg="met")
  #print stocPy.getTimedSamples(normalPartPrior, 10, alg="met")
