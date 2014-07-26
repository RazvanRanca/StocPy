import stocPy

conds = None

def tdf():
  dof = stocPy.unifCont(2, 100, obs=True)
  for i in range(len(conds)):
    stocPy.studentT(dof, cond=conds[i])

def getConds(fn):
  with open(fn,'r') as f:
    data = [float(val.strip()) for val in f.read().strip().split(',')]
  return data

def displayExperiments(): #because of the slower runtime, Tdf experiments consist of single runs
  expDir = stocPy.getCurDir(__file__) + "/experiments/"
  paths = ["metTdfSamp600", "sliceTdfSamp600", "dec5TdfSamps600"]
  titles = ["Metropolis", "Slice", "Metropolis Dec5"]
  paths = [expDir + path for path in paths]

  stocPy.calcKSTest(expDir + "../posterior", paths, names = titles) # plot performance of runs obtained by 3 inference methods

if __name__ == "__main__":
  #conds = getConds(stocPy.getCurDir(__file__) + "/tdfData") # load the data we condition on

  #samps = stocPy.getTimedSamples(tdf, 10, alg="slice") # generate samples for 10 seconds with the slice inference engine
  #stocPy.plotSamples(samps, filt=lambda x: x<8) # plot samples smaller than 8
  #stocPy.saveRun(samps, expDir+"test") # save sample run

  displayExperiments()
