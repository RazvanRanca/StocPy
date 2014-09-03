import stocPy

conds = None

"""
This is a model borrowed from the OpenBUGS repository: http://www.openbugs.net/Examples/t-df.html

Model description:

We observe 1,000 datapoints drawn from a student's T distribution and we 
want to infer the degrees of freedom of the generating distribution.
Our prior belief is simply that the degrees of freedom are equally likely to
be any number between 2 and 100.
"""
def tdf():
  dof = stocPy.unifCont(2, 100, obs=True)
  for i in range(len(conds)):
    stocPy.studentT(dof, cond=conds[i])

"""
Utility function which reads the datapoints we will condition on.
"""
def getConds(fn):
  with open(fn,'r') as f:
    data = [float(val.strip()) for val in f.read().strip().split(',')]
  return data

"""
Utility function which displays the convergence rate of different sample runs (stored in ./experiments).
Because of the slower runtime, Tdf experiments consist of single runs.
"""
def displayExperiments():
  expDir = stocPy.getCurDir(__file__) + "/experiments/"
  paths = ["metTdfSamp600", "sliceTdfSamp600", "dec5TdfSamps600"]
  titles = ["Metropolis", "Slice", "Metropolis Dec5"]
  paths = [expDir + path for path in paths]

  stocPy.calcKSTest(expDir + "../posterior", paths, names = titles) # plot performance of runs obtained by 3 inference methods

if __name__ == "__main__":
  conds = getConds(stocPy.getCurDir(__file__) + "/tdfData") # load the data we condition on

  samps = stocPy.getTimedSamples(tdf, 10, alg="sliceTD") # generate samples for 10 seconds with the slice inference engine
  stocPy.plotSamples(samps, filt=lambda x: x<8, xlabel = "Degrees of freedom") # plot samples smaller than 8
  #stocPy.saveRun(samps, expDir+"test") # save sample run

  #displayExperiments()
