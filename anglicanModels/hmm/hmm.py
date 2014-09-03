import stocPy
import scipy.stats as ss
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import cPickle
from collections import Counter
import copy

"""
This is a model borrowed from the paper: "A New Approach to 
Probabilistic Programming Inference" by Wood, Van de Meent and Mansinghka

This model represents a latent state inference problem in a HMM which has
3 states, with known initial distribution, known transition matrix and 
gaussian emission distribution, with known mean and variance.

After each emission of the HMM is observed we wish to infer the distribution
over the 3 states.
"""

obs = (0.9, 0.8, 0.7, 0.0, -0.025, 5.0, 2.0, 0.1, 0.0, 0.13, 0.45, 6.0, 0.2, 0.3, -1.0, -1.0)

sProbs = (1.0/3, 1.0/3, 1.0/3)
 
tProbs = {
  0 : (0.1, 0.5, 0.4),
  1 : (0.2, 0.2, 0.6),
  2 : (0.15, 0.15, 0.7)
}
eMeans = (-1,1,0)
eProbs = {
  0 : lambda x: ss.norm.pdf(x, eMeans[0], 1),
  1 : lambda x: ss.norm.pdf(x, eMeans[1], 1),
  2 : lambda x: ss.norm.pdf(x, eMeans[2], 1)
}

pind = None
def hmm():
  states = []
  states.append(stocPy.stocPrim("categorical", (sProbs,), obs=True, part=pind))
  for i in range(1,17):
    states.append(stocPy.stocPrim("categorical", (tProbs[states[i-1]],), obs=True, part=pind))
    stocPy.normal(eMeans[states[i]], 1, cond=obs[i-1])

"""
Forward-Backward algorithm is used for calculation of analytical posterior.
Algorithm taken from: http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm#Python_example
"""
def fwd_bkw(x, a_0, a, e):
  L = len(x) + 1
  states = range(len(a_0))

  fwd = []
  f_prev = {}
  # forward part of the algorithm
  for i, x_i in enumerate((None,) + x):
    f_curr = {}
    for st in states:
      if i == 0:
        # base case for the forward part
        prev_f_sum = a_0[st]
      else:
        prev_f_sum = sum(f_prev[k]*a[k][st] for k in states)
 
      if i == 0:
        f_curr[st] = prev_f_sum
      else:
        f_curr[st] = e[st](x_i) * prev_f_sum
 
    fwd.append(f_curr)
    f_prev = f_curr
 
  p_fwd = sum(f_curr[k] for k in states)
 
  bkw = []
  b_prev = {}
  # backward part of the algorithm
  for i, x_i_plus in enumerate(reversed(x+(None,))):
    b_curr = {}
    for st in states:
      if i == 0:
        # base case for backward part
        b_curr[st] = 1
      else:
        b_curr[st] = sum(a[st][l]*e[l](x_i_plus)*b_prev[l] for l in states)
 
    bkw.insert(0,b_curr)
    b_prev = b_curr
 
  p_bkw = sum(a_0[l] * e[l](x[0]) * b_curr[l] for l in states)
 
  # merging the two parts
  posterior = []
  for i in range(L):
    posterior.append({st: fwd[i][st]*bkw[i][st]/p_fwd for st in states})
 
  assert abs(p_fwd - p_bkw) < 0.0001
  return posterior

"""
Plot a heatmap showing the occupation of the various HMM states as the
iteration number increases.
"""
def plotHeatMap(vals):
  plt.imshow(list(reversed(vals)), interpolation='None')
  plt.xticks(range(17))
  plt.yticks(range(3), ["2","1","0"])
  plt.show()

"""
Plot the heatmap of the HMM's analytical posterior.
"""
def hmmPost():
  rez = fwd_bkw(obs, sProbs, tProbs, eProbs)
  print '\n'.join(map(str, rez))
  vals = [[],[],[]]
  for col in rez:
    vals[0].append(col[0])
    vals[1].append(col[1])
    vals[2].append(col[2])
  plotHeatMap(vals)

"""
This method can be called on a file created with "genRuns".
It displays heat maps infered from all the runs stored in the given file.
"""
def showRuns(fn):
  with open(fn, 'r') as f:
    runs = cPickle.load(f)
  for run in runs:
    print run
    vals = [[],[],[]]
    for col in run:
      vals[0].append(col.get(0,0))
      vals[1].append(col.get(1,0))
      vals[2].append(col.get(2,0))
    plotHeatMap(vals)

"""
Generated the specified number and type of runs for the given model.
"""
def genRuns(model, noRuns, runTime, fn, alg="met"):
  runs = []
  for i in range(noRuns):
    print str(model), "Run", i
    samples = stocPy.getTimedSamples(model, runTime, alg=alg, orderNames=True) #orderNames specifies that the samples for different observed variables should respect the order in which the variables are first encountered in the model
    runs.append(samples)
  with open(fn, 'w') as f:
    cPickle.dump(runs, f)

"""
Display the sum of KL divergences for the state occupancy probabilities 
over all the iterations of the model.
"""
def showRunsConv(posts, fns, names, aggSums = True):
  runs = []
  for fn in fns:
    with open(fn, 'r') as f:
      runs.append(cPickle.load(f))
  stocPy.calcKLSumms(posts, runs, names=names, burnIn = 0, aggSums=aggSums, modelName="HMM")

if __name__ == "__main__":
  noRuns = 10
  runTime = 60
  term = "_" + str(noRuns) + "_" + str(runTime)
  cd = stocPy.getCurDir(__file__) + "experiments/"
  
  global pind
  pind = None
  #genRuns(hmm, noRuns=noRuns, runTime=runTime, fn=cd + "Met" + term, alg="met")
  #genRuns(hmm, noRuns=noRuns, runTime=runTime, fn=cd + "Slice" + term, alg="sliceTD")
  pind = 2
  #genRuns(hmm, noRuns=noRuns, runTime=runTime, fn=cd + "Met_P2" + term, alg="met")

  posts = fwd_bkw(obs, sProbs, tProbs, eProbs)
  showRunsConv(posts, [cd + "Met" + term, cd + "Slice" + term, cd + "Met_P2" + term], ["Met", "Slice", "Met_Part2"], aggSums=True)
