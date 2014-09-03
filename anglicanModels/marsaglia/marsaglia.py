import stocPy
import math
from matplotlib import pyplot as plt
import cPickle
import numpy as np
import scipy.stats as ss


"""
This is a model borrowed from the paper: "A New Approach to 
Probabilistic Programming Inference" by Wood, Van de Meent and Mansinghka

This model is similar to the simple "normal" models, in that the inference
problem concerns inferring the mean of an underlying gaussian distribution
with known variance, from which we have observed several samples.

The difference from the "normal" models is that, in this model, we specify
the gaussian prior over the mean by a user-implementation of the "Marsaglia
polar method". In this way we generate the necessary gaussian random numbers
via rejection sampling.
"""
pind = None
def marsaglia(mean, var):
  x = stocPy.stocPrim("uniform", (-1, 2), part=pind) #scipy.stats params are start and length of interval
  y = stocPy.stocPrim("uniform", (-1, 2), part=pind)
  s = x*x + y*y
  if s < 1:
    return mean + (math.sqrt(var) * (x * math.sqrt(-2 * (math.log(s) / s))))
  else:
    return marsaglia(mean, var)
 
obsMean = []
def marsagliaMean():
  global sampleInd
  mean = marsaglia(1, 5)
  stocPy.normal(mean, math.sqrt(2), cond=9)
  stocPy.normal(mean, math.sqrt(2), cond=8)
  obsMean.append(mean)

"""
Calculate the analytical posterior for this model. 
Optionally plot it or save it to a file
"""
def getPost(start, end, inc, show = True, fn=None, rfn = None):
  xs = []
  ys = []
  if rfn:
    with open(rfn,'r') as f:
      xs, ys = cPickle.load(f)
  else:
    for m in np.arange(start, end+inc, inc):
      xs.append(m)
      ys.append(ss.norm.pdf(9, m, math.sqrt(2)) * ss.norm.pdf(8, m, math.sqrt(2)) * ss.norm.pdf(m, 1, math.sqrt(5)))
    ys = stocPy.norm(ys)
  
  if show:
    plt.plot(xs,ys, linewidth=3)
    plt.ylabel("Probability", size=20)
    plt.xlabel("x", size=20)
    plt.title("True Posterior for MarsagliaMean model", size=30)
    plt.show()
  if fn:
    with open(fn,'w') as f:
      cPickle.dump((xs,ys),f)
  return dict(zip(xs, ys))

"""
Generate the specified number and type of runs for the model and 
save them to a file.
"""
def genRuns(model, noRuns, time, fn, alg="met", autoNames=True):
  global obsMean
  runs = []
  for i in range(noRuns):
    print str(model), "Run", i
    samples, traceAcc = stocPy.getTimedSamples(model, time, alg=alg, autoNames=autoNames, outTraceAcc=True)
    runs.append(stocPy.procUserSamples(obsMean, traceAcc))
    obsMean = []
  print map(lambda run: (min(run.values()), max(run.values())), runs)
  with open(fn, 'w') as f:
    cPickle.dump(runs, f)

if __name__ == "__main__":
  global pind 
  pind = None 
  cd = stocPy.getCurDir(__file__) + "experiments/"
  noRuns = 5
  time = 6
  term = "_" + str(noRuns) + "_" + str(time)
  genRuns(marsagliaMean, noRuns=noRuns, time=time, fn=cd + "MetRuns" + term, alg="met")
  #genRuns(marsagliaMean, noRuns=noRuns, time=time, fn=cd + "SliceRuns" + term, alg="sliceTD")
  pind = 5
  #genRuns(marsagliaMean, noRuns=noRuns, time=time, fn=cd + "Part" + str(pind) + "Runs" + term, alg="met")
  #genRuns(marsagliaMean, noRuns=noRuns, time=time, fn=cd + "SlicePart" + str(pind) + "Runs" + term, alg="sliceTD")
  stocPy.calcKSSumms(cd + "../marsagliaPost" , [cd + "MetRuns" + term, cd + "SliceRuns" + term, cd + "Part" + str(pind) + "Runs" + term, cd + "SlicePart" + str(pind) + "Runs" + term], names = ["Met", "Slice", "Part" + str(pind), "SlicePart" + str(pind)], burnIn=0, aggFreq=np.logspace(1,math.log(1000000,10),10), modelName="Marsaglia")
