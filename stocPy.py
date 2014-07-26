import inspect
import math
import scipy.stats as ss
from matplotlib import pyplot as plt
import random
import copy
import time
import numpy as np
import cPickle
from scipy import interpolate
from scipy.interpolate import interp1d
import ast
import os

startTime = time.time()
curNames = set()
ll = 0
llFresh = 0
llStale = 0
db = {}
erps = {"unifCont":0, "studentT":1, "poisson":2, "normal":3, "invGamma":4}
dists = [ss.uniform, ss.t, ss.poisson, ss.norm, ss.invgamma]
observ = set()
condition = set()
cols = ['b','r','k','m','c','g']

def unifCont(start, end, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["unifCont"], (start,end-start))

def studentT(dof, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["studentT"], (dof,))

def poisson(shape, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["poisson"], (shape,))

def normal(mean, stDev, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["normal"], (mean, stDev))

def invGamma(shape, scale, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["invGamma"], (shape, 0, scale))

debugOn = False
def debugPrint(override, *mess):
  if debugOn or override:
    print ' '.join(map(str, mess))

def resetAll():
  global startTime
  global curNames
  global ll
  global llFresh
  global llStale
  global db
  global observ
  global condition

  startTime = time.time()
  curNames = set()
  ll = 0
  llFresh = 0
  llStale = 0
  db = {}
  observ = set()
  condition = set()

class RewriteModel(ast.NodeTransformer):

  def __init__(self, usedNames):
    tempName = "loopCount"
    count = 0
    while tempName in usedNames:
      tempName = "loopCount" + str(count)
      count += 1
    self.tempName = tempName

  def addChild(self, child, parent, loc=0):
    map(ast.increment_lineno, dict(ast.iter_fields(parent))['body'])
    dict(ast.iter_fields(parent))['body'].insert(loc, child)
    ast.fix_missing_locations(parent)
    ast.increment_lineno(child)
    #print parent.lineno
    #print map(lambda x:x.lineno, dict(ast.iter_fields(parent))['body'])
    #print ast.dump(dict(ast.iter_fields(parent))['body'][0], include_attributes=True)

  def visit_FunctionDef(self, node):
    self.addChild(ast.Assign(targets=[ast.Name(id=self.tempName, ctx=ast.Store())], value=ast.Num(n=0)), node)
    self.funcName = dict(ast.iter_fields(node))['name']
    self.generic_visit(node)
    return node

  def visit_For(self, node):
    self.addChild(ast.AugAssign(target=ast.Name(id=self.tempName, ctx=ast.Store()), op=ast.Add(), value=ast.Num(n=1)), node)
    self.generic_visit(node)
    return node

  def visit_While(self, node):
    self.addChild(ast.AugAssign(target=ast.Name(id=self.tempName, ctx=ast.Store()), op=ast.Add(), value=ast.Num(n=1)), node)
    self.generic_visit(node)
    return node

  def visit_Call(self, node):
    callType = dict(ast.iter_fields(dict(ast.iter_fields(node))['func'])).get('attr',None)
    if callType in erps.keys():
      dict(ast.iter_fields(node))['keywords'].append(ast.keyword(arg='name', value=ast.BinOp(left=ast.Str(s=self.funcName + "-" + str(node.lineno) + "-"), op=ast.Add(), right=ast.Call(func=ast.Name(id='str', ctx=ast.Load()), args=[ast.Name(id=self.tempName, ctx=ast.Load())], keywords=[], starargs=None, kwargs=None))))
      ast.fix_missing_locations(node)
      #print map(ast.dump, dict(ast.iter_fields(node))['keywords'])
    self.generic_visit(node)
    return node

class GetLocalNames(ast.NodeVisitor):

  def __init__(self):
    self.localNames = set()

  def visit_Name(self, node):
    self.localNames.add(dict(ast.iter_fields(node))['id'])
    self.generic_visit(node)

def procRawModel(model):
  lcs = {}
  tree = ast.parse(inspect.getsource(model))
  gln = GetLocalNames()
  gln.visit(tree)
  #print ast.dump(tree)
  oldGbs = inspect.stack()[2][0].f_globals
  
  RewriteModel(gln.localNames.union(set(oldGbs.keys()))).visit(tree)
  exec compile(tree, inspect.getfile(model), 'exec') in oldGbs, lcs
  assert(len(lcs) == 1)
  return lcs.values()[0]

def initModel(model):
  model()
  while math.isnan(ll):
    resetAll()
    model()

def getSamples(model, noSamps, discAll=False, alg="met", thresh=0.1):
  model = procRawModel(model)
  initModel(model)

  if alg == "met":
    sampleDict = dict([(n+1, metropolisSampleTrace(model, no = n+1, discAll = discAll)) for n in range(noSamps)])
  elif alg == "slice":
    sampleDict = dict([(n+1, sliceSampleTrace(model, no = n+1, discAll = discAll)) for n in range(noSamps)])
  elif alg == "sliceTD":
    sampleDict = dict([(n+1, sliceSampleTrace(model, no = n+1, discAll = discAll, tdCorr=True)) for n in range(noSamps)])
  elif alg == "sliceNoTrans":
    sampleDict = dict([(n+1, sliceSampleTrace(model, no = n+1, discAll = discAll, allowTransJumps = False)) for n in range(noSamps)])
  elif alg == "sliceMet":
    sampleDict = dict([(n+1, sliceMetMixSampleTrace(model, no = n+1, discAll = discAll)) for n in range(noSamps)])
  else:
    raise Exception("Unknown inference algorithm: " + str(alg))

  resetAll()
  return aggSamples(sampleDict)

def getSamplesByLL(model, noLLs, discAll=False, alg="met", thresh=0.1):
  model = procRawModel(model)
  initModel(model)

  totLLs = 0
  sampleDic = {}
  if alg == "met":
    while totLLs < noLLs:
      samp = metropolisSampleTrace(model, no = totLLs, discAll = discAll)
      totLLs += 1
      if totLLs < noLLs:
        sampleDic[totLLs] = samp
  elif alg == "slice":
    while totLLs < noLLs:
      llCount, samp = sliceSampleTrace(model, no = totLLs, discAll = discAll, countLLs = True)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDic[totLLs] = samp
  elif alg == "sliceTD":
    while totLLs < noLLs:
      llCount, samp = sliceSampleTrace(model, no = totLLs, discAll = discAll, countLLs = True, tdCorr=True)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDic[totLLs] = samp
  elif alg == "sliceNoTrans":
    while totLLs < noLLs:
      llCount, samp = sliceSampleTrace(model, no = totLLs, discAll = discAll, allowTransJumps = False, countLLs = True)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDic[totLLs] = samp
  elif alg == "sliceMet":
    while totLLs < noLLs:
      llCount, samp = sliceMetMixSampleTrace(model, no = totLLs, discAll = discAll, countLLs = True, thresh = thresh)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDic[totLLs] = samp
  else:
    raise Exception("Unknown inference algorithm: " + str(alg))

  resetAll()
  print "rejectedTransJumps", rejTransJumps
  return aggSamples(sampleDic)

def getTimedSamples(model, maxTime, discAll = False, alg = "met", thresh = 0.1):
  model = procRawModel(model)
  initModel(model)

  sampleDict = {}
  while time.time() - startTime < maxTime:
    index = len(sampleDict)+1
    if alg == "met":
      sampleDict[index] = metropolisSampleTrace(model, no = index, discAll = discAll)
    elif alg == "slice":
      sampleDict[index] = sliceSampleTrace(model, no = index, discAll = discAll)
    elif alg == "sliceTD":
      sampleDict[index] = sliceSampleTrace(model, no = index, discAll = discAll, tdCorr=True)
    elif alg == "sliceNoTrans":
      sampleDict[index], sliceSampleTrace(model, no = index, discAll = discAll, allowTransJumps = False)
    elif alg == "sliceMet":
      sampleDict[index] = sliceMetMixSampleTrace(model, no = index, discAll = discAll)
    else:
      raise Exception("Unknown inference algorithm: " + str(alg))

  resetAll()
  return aggSamples(sampleDict)

def aggSamples(samples):
  aggSamps = {}
  if isinstance(samples, list):
    for sample in samples:
      for k,v in sample.items():
        try:
          aggSamps[k].append(v)
        except:
          aggSamps[k] = [v]
  else:
    for count, sample in samples.items():
      for k,v in sample.items():
        try:
          aggSamps[k][count] = v
        except:
          aggSamps[k] = {count:v}

  return aggSamps

def sliceMetMixSampleTrace(model, no = None, discAll = False, thresh = 0.1, countLLs = False):
  if random.random() > thresh:
    return sliceSampleTrace(model, no = no, discAll = discAll, allowTransJumps = False, countLLs = countLLs)
  else:
    samp = metropolisSampleTrace(model, no = no, discAll = discAll)
    if countLLs:
      return 1, samp
    else:
      return samp

tries = {}
def metropolisSampleTrace(model, no = None, discAll = False):
  global db
  global ll
  global tries
  unCond = list(set(db.keys()).difference(condition))
  n = random.choice(unCond)
  otp, ox, ol, ops = db[n]

  x = dists[otp].rvs(*ops)
  try:
    l = dists[otp].logpdf(x, *ops)
  except:
    l = dists[otp].logpmf(x, *ops)

  odb = copy.copy(db)
  oll = ll
  recalcLL(model, n, x, l)
  
  changed = True
  acc = ll - oll + ol - l + math.log(len(unCond)) - math.log(len(set(db.keys()).difference(condition))) + llStale - llFresh
  if math.log(random.random()) < acc:
    pass
  else:
    changed = False
    db = odb
    ll = oll

  sample = {}
  for n in observ:
    sample[n] = db[n][1]

  if no % 10000 == 0:
    print no, sample, time.time() - startTime
  return sample

rejTransJumps = 0
def sliceSampleTrace(model, width = 10, no = None, discAll=False, allowTransJumps = True, countLLs = False, tdCorr=False):
  global db
  global ll
  global rejTransJumps

  llCount = 0
  unCond = list(set(db.keys()).difference(condition))
  n = random.choice(unCond)
  otp, ox, ol, ops = db[n]
  
  u = -1*ss.expon.rvs(-1*ll) #sample log(x), x~unif(0,likelihood)
  r = random.random()
  oll = ll

  xl = ox
  xr = ox
  debugPrint(False, ox, ll, u)
  curWidth = r*width

  assert(ll > u)
  llc = ll
  while llc > u:
    xl -= curWidth
    if discAll:
      xl = int(math.floor(xl))

    odb = copy.copy(db)
    recalcLL(model, n, xl)
    llCount += 1
    db = odb
    curWidth *= 2
    if allowTransJumps and tdCorr:
      llc = ll + llStale - llFresh
    else:
      llc = ll
    debugPrint(False, "l", xl, ll)

  ll = oll
  curWidth = r*width
  llc = ll
  while llc > u:
    xr += curWidth
    if discAll:
      xr = int(math.ceil(xr))

    odb = copy.copy(db)
    recalcLL(model, n, xr)
    llCount += 1
    db = odb
    curWidth *= 2
    if allowTransJumps and tdCorr:
      llc = ll + llStale - llFresh
    else:
      llc = ll
    debugPrint(False, "r", xr, ll)

  ll = oll
  first = True
  
  transJump = False
  llc = ll
  while first or llc < u or math.isnan(llc) or (transJump and (not allowTransJumps)):
    if first:
      first = False
    if discAll:
      x = random.randrange(xl,xr+1)
    else:
      x = random.uniform(xl, xr)

    odb = copy.copy(db)
    recalcLL(model, n, x)
    llCount += 1
    transJump = (llStale != 0) or (llFresh != 0) 
    if allowTransJumps and tdCorr:
      llc = ll + llStale - llFresh
    else:
      llc = ll

    debugPrint(False, "c", xl, xr, x, ll)
    if llc < u or math.isnan(llc) or (transJump and (not allowTransJumps)):
      debugPrint(False, "in")
      if transJump and (not allowTransJumps):
        rejTransJumps += 1
      db = odb
      if x > ox:
        xr = x
      else:
        xl = x

  sample = {}
  for o in observ:
    sample[o] = db[o][1]

  if no%10000 == 0:
    print no, n, xl, xr, sample[o]

  if countLLs:
    return llCount, sample
  else:
    return sample

def sliceMultSampleTrace(model, width = 100, no = None, discAll=False):
  global db
  global ll

  unCond = list(set(db.keys()).difference(condition))

  u = -1*ss.expon.rvs(-1*ll) #sample log(x), x~unif(0,likelihood)

  xls = {}
  xrs = {}
  oxs = {}
  for n in unCond:

    r = random.random()
    oll = ll 
    _, ox, _, _ = db[n]
    oxs[n] = ox
    xl = ox
    xr = ox
    debugPrint(False, ox, ll, u)
    curWidth = r*width
    assert(ll > u)
    while ll > u:
      xl -= curWidth
      if discAll:
        xl = int(math.floor(xl))

      odb = copy.copy(db)
      recalcLL(model, n, xl)
      db = odb
      curWidth *= 2
      debugPrint(False, "l", xl, ll)

    ll = oll
    curWidth = r*width
    while ll > u:
      xr += curWidth
      if discAll:
        xr = int(math.ceil(xr))

      odb = copy.copy(db)
      recalcLL(model, n, xr)
      db = odb
      curWidth *= 2
      debugPrint(False, "r", xr, ll)
    xls[n] = xl
    xrs[n] = xr
    ll = oll

  ll = oll
  first = True
  while first or ll < u or math.isnan(ll):
    if first:
      first = False
    xs = {}
    if discAll:
      for n in unCond:
        xs[n] = random.randrange(xl,xr+1)
    else:
      for n in unCond:
        xs[n] = random.uniform(xl, xr)

    odb = copy.copy(db)
    recalcMultLL(model, xs)
    debugPrint(False, "c", xl, xr, xs, ll)
    if ll < u or math.isnan(ll):
      debugPrint(False, "in")
      db = odb
      for n in unCond:
        if xs[n] > oxs[n]:
          xrs[n] = xs[n]
        else:
          xls[n] = xs[n]

  sample = {}
  for n in observ:
    sample[n] = db[n][1]

  if no%1000 == 0:
    print no, sample[n]
  return sample

def recalcLL(model, n, x, l = None):
  if l:
    ls = {n:l}
  else:
    ls = None
  recalcMultLL(model, {n:x}, ls)

def recalcMultLL(model, xs, ls = None):
  global db
  global ll
  global llStale
  global llFresh
  global curNames

  for n,x in xs.items():
    otp, ox, ol, ops = db[n]

    if not ls:
      try:
        l = dists[otp].logpdf(x, *ops)
      except:
        l = dists[otp].logpmf(x, *ops)
    else:
      l = ls[n]
    db[n] =  (otp, x, l, ops)
    if l == float("-inf"):    # TODO: Check handling of this is correct
      ll = float("-inf")
      llFresh = 0
      llStale = 0
      return

  ll = 0
  llFresh = 0
  llStale = 0
  curNames = set()
  oldLen = len(db)
  model()

  newLen = len(db)
  assert(oldLen <= newLen)

  for n in db.keys():
    if not n in curNames:
      llStale += db[n][2]
      db.pop(n)


def getName(loopInd):
  _, _, lineNo, funcName, _, _ = inspect.stack()[2]
  name = funcName + "-" + str(lineNo) + "-" + str(loopInd)
  return name

def getERP(n, c, tp, ps):
  global ll
  global llFresh
  global db
  global condition
  otp, ox, ol, ops = db.get(n, (None,None,None,None))
  if tp == otp:
    if ps == ops:
      ll += ol
      return ox
    else:
      try:
        l = dists[tp].logpdf(ox, *ps)
      except:
        l = dists[tp].logpmf(ox, *ps)

      db[n] = (tp, ox, l, ps)
      ll += l
      return ox
  else:
    if c:
      x = c
      condition.add(n)
    else:
      assert (not n in condition)
      x = dists[tp].rvs(*ps)
    try:
      l = dists[tp].logpdf(x, *ps)
    except:
      l = dists[tp].logpmf(x, *ps)

    db[n] = (tp, x, l, ps)
    ll += l
    if not c:
      llFresh += l
    return x

def initERP(name, obs):
  assert(name)
  global observ
  global curNames
  curNames.add(name)
  if obs:
    observ.add(name)

def readSamps(fn, start=0):
  with open(fn, 'r') as f:
    try:
      samps = getData(cPickle.load(f))
    except:
      data = f.read().strip().split('\n')
      if len(data) == 1: # list format
        samps = map(float, data[0][1:-1].split(','))[start:]
      else: # rows of (count, sample) format
        samps = []
        for line in data:
          count, samp = map(float, line.strip().split())
          samps.append(samp)
  return samps

def plotCumSampDist(fn, plot=True, show=True, xlim = None):
  samps = readSamps(fn)
  hSamps, ds =  np.histogram(samps, 1000)
  cSamps = []
  curSum = 0
  norm = float(sum(hSamps))
  for samp in hSamps:
    curSum += samp / norm
    cSamps.append(curSum)

  locs = []
  for d in range(len(ds)-1):
    locs.append((ds[d] + ds[d+1]) / 2.0)

  if xlim:
    cSamps = [0] + cSamps + [1]
    locs = [xlim[0]] + locs + [xlim[1]]

  if plot:
    plt.plot(locs, cSamps)
    plt.xlabel("DoFs")
    plt.ylabel("Number of samples with smaller DoF")
    if show:
      #plt.title("Venture ran for 10m - " + str(len(samps)) + " samples - mean: " + str(np.mean(samps))[:5] + ", stDev: " + str(np.std(samps))[:5])
      plt.show()
  return locs, cSamps

def plotCumPost(fn = "posterior", plot = True, show=True, zipRez=True):
  with open(fn,'r') as f:
    ds, ls = cPickle.load(f)
  cls = []
  curSum = 0
  for l in ls:
    curSum += l
    cls.append(curSum)

  if plot:
    plt.plot(ds, cls)
    plt.xlabel("DoFs")
    plt.ylabel("Number of samples with smaller DoF")

    if show:
      plt.show()
  if zipRez:
    return zip(ds,cls)
  else:
    return ds, cls

def plotCumSampDists(fns):
  for fn in fns:
    plotCumSampDist(fn, show=False)
  plt.show()

def calcKSTest(pfn, fns, names = None):
  post = plotCumPost(pfn, plot=False)
  diffs = {}
  ps = []
  ns = []
  for i in range(len(fns)):
    fn = fns[i]
    diffs[fn] = []
    xs, ys =  plotCumSampDist(fn, plot=False)
    minX = min(xs)
    maxX = max(xs)
    f = interpolate.interp1d(xs,ys)
    for x,y in post:
      if x <= minX:
        val = abs(y)
      elif x >= maxX:
        val = abs(1-y)
      else:
        val = abs(f(x) - y)
      diffs[fn].append(val)
    diffs[fn] = sorted(diffs[fn], reverse=True)

    p, = plt.plot(diffs[fn], cols[i], linewidth=3)
    ps.append(p)
    ns.append(fn[:-7])

  if not names:
    names = ns
  plt.legend(ps,names)
  plt.xscale("log")
  #plt.yscale("log")
  plt.xlabel("Nth biggest KS difference", size=20)
  plt.ylabel("KS difference from posterior", size=20)
  plt.title("Decreasing KS diff. from posterior", size=30)
  plt.show()

def calcKSRun(postFun, run, aggFreq, burnIn):
  samps = []
  xs = []
  ys = []
  curAgg = 0
  if isinstance(run, list):
    run = dict([(i+1, run[i]) for i in range(len(run))])
  for k, samp in sorted(run.items()):
    if k < burnIn:
      continue
    samps.append(samp)
    ind = k-burnIn
    if ind > aggFreq[curAgg]:
      curAgg += 1
      ksDiff = calcKSDiff(postFun, samps)
      xs.append(ind)
      ys.append(ksDiff)

  xs.append(ind)
  ys.append(calcKSDiff(postFun, samps))
  return xs, ys

def calcKSTests(pfn, fns, aggFreq, burnIn = 0, plot=True, xlim = 200000, names=None, alpha=0.25, single=False):
  postFun = interpolate.interp1d(*plotCumPost(pfn, plot=False, zipRez=False))
  ps = []
  ns = []
  np.append(aggFreq, float("inf"))
  for i in range(len(fns)):
    fn = fns[i]
    p, = plt.plot([0],[0], cols[i])
    ps.append(p)
    ns.append(fn.split('/')[1])
    runs = readSamps(fn)
    if single:
      runs = [runs]
    for r in range(len(runs)):
      print fn, r
      xs, ys = calcKSRun(postFun, runs[r], aggFreq, burnIn)
      if plot:
        plt.plot(xs,ys, cols[i], alpha=alpha)
      print zip(xs,ys)
  if not names:
    names = ns
  if plot:
    plt.legend(ps,names,loc=3)
    plt.xscale("log")
    plt.yscale("log", basey=2)
    plt.xlim([0, xlim])
    plt.xlabel("Trace Likelihood calculations", size=20)
    plt.ylabel("KS difference from posterior", size=20)
    no = fns[0].split("PerLL")[0][-1]
    if no == "4":
      no = "3"
    plt.title("Performance on NormalMean" + no + " Model", size=30)
    plt.show()

def calcKSSumms(pfn, fns, aggFreq, burnIn = 0, xlim = 200000, names=None):
  postFun = interpolate.interp1d(*plotCumPost(pfn, plot=False, zipRez=False))
  ps = []
  ns = []
  np.append(aggFreq, float("inf"))
  funcs = []
  for i in range(len(fns)):
    fn = fns[i]
    p, = plt.plot([0],[0], cols[i])
    ps.append(p)
    ns.append(fn.split('/')[1])
    fs = []
    start = float("-inf")
    end = float("inf")
    with open(fn, 'r') as f:
      runs = cPickle.load(f)
    for r in range(len(runs)):
      print fn, r
      xs, ys = calcKSRun(postFun, runs[r], aggFreq, burnIn)
      if xs[0] > start:
        start = xs[0]
      if xs[-1] < end:
        end = xs[-1]
      fs.append(interp1d(xs,ys))

    end += 1

    top = []
    med = []
    bot = []

    for x in np.arange(start, end):
      if x % 1000 == 0:
        print "x", x
      vals = []
      for f in fs:
        vals.append(f(x))

      top.append(np.percentile(vals, 25))
      med.append(np.percentile(vals, 50))
      bot.append(np.percentile(vals, 75))

    plt.plot(range(start, end), med, cols[i], linewidth=2, alpha=0.9)
    plt.plot(range(start, end), top, cols[i], linewidth=2, alpha=0.9)
    plt.plot(range(start, end), bot, cols[i], linewidth=2, alpha=0.9)

  if not names:
    names = ns
  plt.legend(ps,names,loc=3)
  plt.xscale("log")
  plt.yscale("log", basey=2)
  plt.xlim([0, xlim])
  plt.xlabel("Trace likelihood calculations", size=20)
  plt.ylabel("KS difference from posterior", size=20)
  no = fns[0].split("PerLL")[0][-1]
  if no == "4":
    no = "3"
  plt.title("Performance on NormalMean" + no + " Model", size=30)

  plt.show()

cachedPost = {}
def calcKSDiff(postFun, samps):
  global cachedPost
  cumProb = 0
  inc = 1.0/len(samps)
  maxDiff = 0
  for samp in sorted(samps):
    try:
      post = cachedPost[samps]
    except:
      post = postFun(samp)
      cachedPost[samp] = post
    preDiff = abs(post - cumProb)
    cumProb += inc
    postDiff = abs(post - cumProb)
    maxDiff = max([maxDiff, preDiff, postDiff])
  return maxDiff

def potRead(post):
  if isinstance(post, basestring):
    with open(post, 'r') as f:
      return cPickle.load(f)
  return post

def calcKLTests(post, sampsLists, names, freq = float("inf"), burnIn = 0, xlim=None):
  post = potRead(post)
  axs = []
  for i in range(len(sampsLists)):  
    ax, = plt.plot([0],[0], cols[i])
    axs.append(ax)
    for samps in sampsLists[i]:
      calcKLTest(post, samps, freq, show=False, col=cols[i], burnIn = burnIn, xlim = xlim)

  if xlim:
    plt.xlim([0,xlim])
  plt.ylabel("KL divergence from posterior", size=20)
  plt.xlabel("Trace likelihood calculations", size=20)
  plt.title("Performance on Branching model", size=30)
  plt.legend(axs, names, loc=3)
  plt.show()

def calcKLTest(post, samps, freq = float("inf"), show=True, col=None, plot=True, cutOff = float("inf"), burnIn = 0, xlim = float("inf")):
  post = potRead(post)
  sampDic = {}
  if isinstance(samps, list):
    samps = dict([(i+1,samps[i]) for i in range(len(samps))])

  xs = []
  ys = []
  prevC = 0
  sampList = []
  for c, samp in sorted(samps.items()):
    if c > xlim:
      break
    if samp > cutOff or c < burnIn:
      continue
    c = c - burnIn
    sampList.append(samp)
    try:
      sampDic[samp] += 1
    except:
      sampDic[samp] = 1.0

    kl = getKLDiv(post, norm(sampDic))
    xs.append(c)
    ys.append(kl)
    if (c+1) / freq > prevC:
      prevC += 1
      plt.hist(sampList, 100)
      plt.title(str(c) + " " + str(kl))
      #plt.show()
  if not plot:
    return xs, ys
  else:
    if col:
      ax, = plt.plot(xs,ys, col, alpha=0.15)
    else:
      ax, = plt.plot(xs,ys)
    plt.yscale("log", basey=2)
    plt.xscale("log")
    if show:
      plt.show()

    return ax

def getKLDiv(post, samp):
  post = potRead(post)
  eps = 0.0001
  assert (abs(sum(samp.values()) - 1) < eps)
  assert (abs(sum(post.values()) - 1) < eps)

  kl = 0
  for val, p in samp.items():
      kl += p * math.log(p / post[val])
  return kl

def calcKLSumms(post, sampsLists, names, burnIn = 0, xlim = None):
  post = potRead(post)
  axs = []
  for i in range(len(sampsLists)):
    axs.append(calcKLSumm(post, sampsLists[i], col=cols[i], show=False, burnIn = burnIn, xlim = xlim))

  if xlim:
    plt.xlim([0, xlim])
  plt.ylabel("KL divergence from posterior", size=20)
  plt.xlabel("Trace likelihood calculations", size=20)
  plt.title("Performance on Branching model", size=30)
  plt.legend(axs, names, loc=3)
  plt.show()

def calcKLSumm(post, sampsList, col = 'b', show=True, burnIn = 0, xlim = float("inf")):
  post = potRead(post)
  fs = []
  start = float("-inf")
  end = float("inf")

  for samps in sampsList:
    if len(fs) % 10 == 0:
      print "fs", len(fs)
    xs,ys = calcKLTest(post, samps, plot=False, burnIn = burnIn, xlim = xlim) #xs should be already sorted
    if xs[0] > start:
      start = xs[0]
    if xs[-1] < end:
      end = xs[-1]
    fs.append(interp1d(xs,ys))

  end += 1

  top = []
  med = []
  bot = []
  print start, end
  for x in range(start, end):
    if x % 1000 == 0:
      print "x", x
    vals = []
    for f in fs:
      vals.append(f(x))

    top.append(np.percentile(vals, 25))
    med.append(np.percentile(vals, 50))
    bot.append(np.percentile(vals, 75))

  plt.yscale("log", basey=2)
  plt.xscale("log")
  ax, = plt.plot(range(start, end), med, col)
  plt.plot(range(start, end), top, col)
  plt.plot(range(start, end), bot, col)
  
  if show:
    plt.show()

  return ax

def norm(vals):
  if isinstance(vals, list):
    norm = sum(vals)
    return map(lambda x: x/norm, vals)
  elif isinstance(vals, dict):
    norm = sum(vals.values())
    nvals = {}
    for k,v in vals.items():
      nvals[k] = v/norm
    return nvals
  else:
    raise Exception("Unknown datatype given to norm:" + vals)

def getData(samps, name = None):
  if isinstance(samps, list):
    data = samps
  else:
    if not name:
      if len(samps) == 1:
        name = samps.keys()[0]
        if isinstance(samps[name], list):
          data = samps[name]
        else:
          data = map(lambda (k,v): v, sorted(samps[name].items()))
      else:
        data = map(lambda (k,v): v, sorted(samps.items()))
  return data

def plotSamples(samps, name = None, filt = lambda x:True, xlabel="", title="", xlim=None):
  plt.hist(filter(filt, getData(samps, name)), 100)
  plt.ylabel("Number samples")
  plt.xlabel(xlabel)
  plt.title(title)
  if xlim:
    plt.xlim(xlim)
  plt.show()

def saveRun(run, path):
  with open(path, 'w') as f:
    f.write(str(getData(run)))

def aggDecomp(decSamps, func=lambda xs: sum(xs)):
  samples = {}
  for name,lst in decSamps.items():
    for k,v in lst.items():
      try:
        samples[k].append(v)
      except:
        samples[k] = [v]
  return dict([(k,func(v)) for (k,v) in samples.items()])

def getCurDir(f):
  return os.path.dirname(os.path.realpath(f))
