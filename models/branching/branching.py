import stocPy
import cPickle

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

def branching():
  r = stocPy.poisson(4, obs=True)
  if r > 4:
    l = 6
  else:
    p2 = stocPy.poisson(4)
    l = fib(3*r) + p2 

  stocPy.poisson(l, cond=6)

def displayExperiments(xlim=20000):
  paths = ["metRunsByLL", "sliceRunsByLL", "sliceMet01RunsByLL", "sliceMet05RunsByLL"]
  titles = ["Metropolis", "Slice", "Slice:Met 1:9", "Slice:Met 1:1"]

  runs = []
  for path in paths:
    with open(stocPy.getCurDir(__file__) + "/experiments/" + path,'r') as f:
      runs.append(cPickle.load(f))

  stocPy.calcKLTests(stocPy.getCurDir(__file__) + "/posterior", runs, titles, xlim=xlim, burnIn = 1000) # show all runs
  stocPy.calcKLSumms(stocPy.getCurDir(__file__) + "/posterior", runs, titles, xlim=xlim, burnIn = 1000) # show run quartiles

if __name__ == "__main__":
  #stocPy.plotSamples(stocPy.getSamplesByLL(branching, 10000, discAll=True, alg="slice")) # extract samples untill 10,000 trace log-likelihood calculations are performed
  displayExperiments()
