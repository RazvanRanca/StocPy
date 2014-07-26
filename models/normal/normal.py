import stocPy
import math
import numpy as np

datum = 5

def normal1():
  m = stocPy.normal(0, 1, obs=True)
  stocPy.normal(m, 1, cond=datum)

def normal2():
  m = stocPy.normal(0, 1, obs=True)
  v = stocPy.invGamma(3, 1)
  stocPy.normal(m, math.sqrt(v), cond=datum)

def normal4():
  m = stocPy.normal(0, 1, obs=True)
  if m > 0:
    v = 1.0/3
  else:
    v = stocPy.invGamma(3, 1)
  stocPy.normal(m, math.sqrt(v), cond=datum)

def displayExperiments(modelNo, xlim = None):
  if xlim == None:
    if modelNo == 1 or modelNo == 2:
      xlim = 20000
    else:
      xlim = 200000

  mi = str(modelNo)
  paths = ["PerLLMet", "PerLLSlice", "PerLLSlicemet0.1", "PerLLSlicemet0.5"]
  titles = ["Metropolis", "Slice", "Slice:Met 1:9", "Slice:Met 1:1"]
  curDir = stocPy.getCurDir(__file__)
  paths = [curDir + "/experiments/normal" + mi + path for path in paths]

  stocPy.calcKSTests(curDir + "/normal" + mi + "Post", paths, aggFreq=np.logspace(1,math.log(xlim,10),10), burnIn=1000, xlim = xlim, names=titles) # show all runs
  stocPy.calcKSSumms(curDir + "/normal" + mi + "Post", paths, aggFreq=np.logspace(1,math.log(xlim,10),10), burnIn=1000, xlim = xlim, names=titles) # show run quartiles

if __name__ == "__main__":
  #stocPy.plotSamples(stocPy.getSamples(normal2, 10000, alg="met"))
  displayExperiments(1)
  displayExperiments(2)
  displayExperiments(4)
