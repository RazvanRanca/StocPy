import stocPy
import random
import math
import scipy.stats as ss
import numpy as np
import copy
import itertools
import cPickle

"""
This is a model borrowed from the paper: "A New Approach to 
Probabilistic Programming Inference" by Wood, Van de Meent and Mansinghka

In this model we have datapoints drawn from different gaussians with 
unknown meand an variances. The way the datapoints are clustered among
different gaussians is also unknown. We represent this model with a 
dirichlet process mixture of gaussians.

The dirichlet process is implemented via a custom stocPy chinese restaurant
process. This implementation is still work-in-progress and so does not currently
support slice sampling or partitioned priors (though the means and variances can
still be partitioned).

Two implementations of the model are given, one which obtains the partition of
each datapoint only when needed(Lazy) and one which does so from the beggining (Eager).
"""

obs = [1.0, 1.1, 1.2, -10.0, -15.0, -20.0, 0.01, 0.1, 0.05, 0.0]
post = {1: 0.04571063618028917, 2: 0.21363892248912586, 3: 0.32803178751362133, 4: 0.2536086916674492, 5: 0.1168560934038502, 6: 0.034586518747180765, 7: 0.006687390777635802, 8: 0.0008201140565686028, 9: 5.803774292549257e-05, 10: 1.8074213536324885e-06}

pind = None
def dpmLazy():
  crp = stocPy.crp(1.72)
  sds = {}
  ms = {}
  for i in range(len(obs)):
    c = crp(i)
    if c not in ms:
      sds[c] = math.sqrt(10 * stocPy.stocPrim("invgamma", (1, 0, 10), part=pind))
      ms[c] = stocPy.stocPrim("normal", (0, sds[c]), part=pind)
    stocPy.normal(ms[c], sds[c], obs[i])
  stocPy.observe(len(ms), name="c")

def dpmEager():
  crp = stocPy.crp(1.72, 10)
  sds = {}
  ms = {}
  cs = {}
  for ps in range(len(crp)):
    sds[ps] = math.sqrt(10 * stocPy.stocPrim("invgamma", (1, 0, 10), part=None))
    ms[ps] = stocPy.stocPrim("normal", (0, sds[ps]), part=pind)
    for p in crp[ps]:
     cs[p] = ps

  for i in range(len(obs)):
    stocPy.normal(ms[cs[i]], sds[cs[i]], obs[i])
  stocPy.observe(len(ms), name="c")

"""
Generate all set partitions of a given list.
Used when computing the analytical posterior.
"""
def allParts(data):
  def addelement(partlist, e):
    newpartlist = []
    for part in partlist:
      npart = part + [[e]]
      newpartlist += [npart]
      for i in xrange(len(part)):
        npart = copy.deepcopy(part)
        npart[i] += [e]
        newpartlist += [npart]
    return newpartlist

  if len(data) == 0: 
    return []
  partlist = [[[data[0]]]]
  for i in xrange(1, len(data)):
    partlist = addelement(partlist, data[i])
  return map(lambda ps: map(tuple, ps), partlist)

"""
Generate the set of all subsets of a given list.
Used when computing the analytical posterior.
"""
def powerSet(data):
  ps = []
  for i in range(1,len(data)+1):
    ps += itertools.combinations(data, i)
  return ps

"""
Calculate and store the log-likelihoods of all subsets of a dataset.
Allows us to avoid recomputation of these when calculating the analytical posterior.
"""
def storeLLs(data, fn):
  lls = {}
  for cl in powerSet(data):
    lls[cl] = getLL(cl)
    print cl, lls[cl]
  with open(fn, 'w') as f:
    cPickle.dump(lls, f)

"""
Return the log-likelihood of a set partition.
If possible, reads and stores the result from the file generates with "storeLLs",
which speeds up the runtime considerably.
"""
llDict = None
def getLL(cl, fn = None):
  global llDict
  if fn:
    try:
      return llDict[cl]
    except:
      with open(fn, 'r') as f:
        llDict = cPickle.load(f)
      return llDict[cl]
  else:
    ll = math.log(ss.invgamma.expect(lambda v: ss.norm.expect(lambda m: np.product([ss.norm.pdf(d, loc=m, scale=10*v) for d in cl]) ,loc=0, scale=10*v), 1, loc=0, scale=10))
  return ll

"""
Calculate the analytical posterior.
Suggested to run "storeLLs" on the dataset prior to this and pass the resulting
file as a parameter. Even with this memoization, can't handle many more than 10 observations.
"""
def getPost(ds, a, fn=None):
  crp = stocPy.CRP(a)
  parts = allParts(ds)
  post = {}

  for p in range(len(parts)):
    part = parts[p]
    ll = crp.logpmf(part)
    if p % 1000 == 0:
      print p, part, ll, 
    for cl in part:
      ll += getLL(cl, fn)
    try:
      post[len(part)].append(ll)
    except:
      post[len(part)] = [ll]
    if p % 1000 == 0:
      print ll
  post = dict([(k, sum(map(lambda x: math.e**x, v))) for (k, v) in post.items()])
  print post
  post = stocPy.norm(post)
  return post

"""
Generate specified number and type of runs for the given model.
"""
def genRuns(model, noRuns, time, fn, alg="met"):
  runs = []
  for i in range(noRuns):
    print str(model), "Run", i
    samples = stocPy.getTimedSamples(model, time, alg=alg)
    runs.append(samples["c"])
  #print map(lambda run: (min(run.values()), max(run.values())), runs)
  with open(fn, 'w') as f:
    cPickle.dump(runs, f)

if __name__ == "__main__":
  #storeLLs(obs, "dpMixLLS")
  #print getPost(obs, 1.72, "dpMixLLS")
  noRuns = 10
  runTime = 60
  term = "_" + str(noRuns) + "_" + str(runTime)

  cd = stocPy.getCurDir(__file__) + "experiments/"

  pind = None
  genRuns(dpmEager, noRuns, runTime, cd + "Eager_Met" + term, alg="met")
  pind = 2
  genRuns(dpmEager, noRuns, runTime, cd + "Eager_Met_P2v" + term, alg="met")
  pind = 5
  genRuns(dpmEager, noRuns, runTime, cd + "Eager_Met_P5v" + term, alg="met")
  stocPy.calcKLSumms(post , [cd + "Eager_Met" + term, cd + "Eager_Met_P2v" + term, cd + "Eager_Met_P5v" + term], names = ["Eager", "Eager_Part2", "Eager_Part5"], burnIn=0, modelName="DP Mixture")
