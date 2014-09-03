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
import sys

class CRP: #anglican crp formulation: http://www.robots.ox.ac.uk/~fwood/anglican/examples/dp_mixture_model/index.html
  def __init__(self, a = None):
    self.n = 0
    self.pns = []
    if a != None:
      self.a = float(a)
  
  def __call__(self, x):
    return self.getClass(x)

  def getClass(self, x):
    while self.n <= x:
      self.getNext()
    for c in range(len(self.pns)):
      if x in self.pns[c]:
        return c
    assert(False)

  def getNext(self):
    r = random.random()
    accum = 0
    for c in range(len(self.pns)):
      accum += len(self.pns[c]) / (self.n + self.a)
      if r < accum:
        return self.setClass(c)
    return self.setClass()

  def setClass(self, c=None):
    if c != None:
      self.pns[c].append(self.n)
    else:
      c = len(self.pns)
      self.pns.append([self.n])
    self.n += 1
    return c

  def reset(self):
    self.pns = []
    self.n = 0

  def rvs(self, a = None, maxElems = None):
    if a == None:
      a = self.a
    newCrp = CRP(a)
    if maxElems == None:
      return newCrp
    else:
      newCrp.getClass(maxElems - 1)
      return newCrp.pns

  def getParams(self, part = None, a = None):
    if a == None:
      a = self.a
    if part == None:
      return a, self.n, self.pns
    elif isinstance(part, CRP):
      return a, sum(map(len, part.pns)), map(len, part.pns)
    else:
      return a, sum(map(len, part)), map(len, part)

  def pmf(self, part = None, a = None, maxElem = None):
    a, n, pns = self.getParams(part, a)
    prob = (math.gamma(a) * a**(len(pns))) / math.gamma(a + n)
    for pn in pns:
      prob *= math.gamma(pn)
    return prob

  def logpmf(self, part = None, a = None, maxElem = None):
    a, n, pns = self.getParams(part, a)
    lprob = math.lgamma(a) + len(pns)*math.log(a) - math.lgamma(a + n)
    for pn in pns:
      lprob += math.lgamma(pn)
    return lprob

class Categorical():
  def __init__(self):
    self.func = lambda ps : ss.rv_discrete(name="categorical", values=(range(len(ps)), ps))

  def rvs(self, ps):
    return self.func(ps).rvs()

  def logpmf(self, x, ps):
    return self.func(ps).logpmf(x)

  def ppf(self, x, ps):
    return self.func(ps).ppf(x)

startTime = time.time()
curNames = set()
ll = 0
llFresh = 0
llStale = 0
db = {}
erps = {"unifCont":0, "studentT":1, "poisson":2, "normal":3, "invGamma":4, "beta":5, "categorical":6, "crp":7}
dists = [ss.uniform, ss.t, ss.poisson, ss.norm, ss.invgamma, ss.beta, Categorical(), CRP()]
discrete_dists = [Categorical, ss.rv_discrete]
observ = set()
condition = set()
cols = ['b','r','k','m','c','g']
partName = {}
partFunc = {}
nameOrder = {} 
seenNames = set()
traceAcc = []
metAccProbs = {} 
recAccNames = set()
autoDepth = 1

def unifCont(start, end, cond = None, obs=False, name = None): #different name than ss.uniform because parametrixation si diff: [start, end]
  initERP(name, obs)
  return getERP(name, cond, erps["unifCont"], (start,end-start))

def studentT(dof, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["studentT"], (dof,))

def poisson(shape, cond = None, obs=False, name = None):
  initERP(name, obs)
  return int(getERP(name, cond, erps["poisson"], (shape,)))

def normal(mean, stDev, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["normal"], (mean, stDev))

def invGamma(shape, scale, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["invGamma"], (shape, 0, scale))

def beta(a, b, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["beta"], (a, b))

def categorical(ps, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["categorical"], (ps,))

def crp(a, maxElem = None, cond = None, obs=False, name = None):
  initERP(name, obs)
  return getERP(name, cond, erps["crp"], (a, maxElem))

def stocPrim(distName, params, cond=None, obs=False, part=None, name=None):
  if part != None:
    global partName
    global partFunc
    global recAccNames 
    global metAccProbs
    global autoDepth

    depth = part
    if part == "auto":
      #print metAccProbs
      try:
        name = metAccProbs.keys()[0]
        accs = metAccProbs[name]
      except:
        accs = []
      if len(accs) > 10:
        mAcc = np.mean(accs[-10:])
        if mAcc < 0.75:
          autoDepth += 1
          metAccProbs[name] = []
          recAccNames = set()
          print mAcc, autoDepth
        else:
          if len(accs) > 50 and np.mean(accs[-50:]) > 0.8:
            print autoDepth, np.mean(accs[-10:]), np.mean(accs[-50:]) 
            assert(False)
        if len(accs) % 10 == 0:
          print len(accs), autoDepth, mAcc
      depth = autoDepth

    if not name in seenNames:
      seenNames.add(name)
      nameOrder[name] = len(nameOrder)

    ns = []
    for i in range(1, depth+1):
      #print i
      pName = name + "-" + str(i)
      ns.append(normal(0, math.sqrt(1.0/(2.0**i)), obs=obs, name = pName))
      partName[pName] = name
      recAccNames.add(pName)

    pName = name + "-" + str(depth) + "-r"
    recAccNames.add(pName)
    ns.append(normal(0, math.sqrt(1.0/(2.0**depth)), obs=obs, name = pName))
    partName[pName] = name
    try:
      dist = getattr(ss, distName)
    except:
      dist = dists[erps[distName]]

    if isDiscrete(dist):
      def func(xs):
        r = dist.ppf(ss.norm.cdf(sum(xs)), *params)
        try:
          r = int(r)
        except:
          pass
        return r
    else:
      func = lambda xs: dist.ppf(ss.norm.cdf(sum(xs)), *params)
    partFunc[name] = func 
    return func(ns)
  else:
    if distName not in erps:
      erps[distName] = len(erps)
      dists.append(getattr(ss, distName))
    initERP(name, obs)
    ret = getERP(name, cond, erps[distName], params)
    #print name, ret
    return ret

def isDiscrete(dist):
  return any(map(lambda x: isinstance(dist, x), discrete_dists))

def initERP(name, obs):
  assert(name)
  curNames.add(name)
  if obs:
    observ.add(name)
  if not name in seenNames:
    seenNames.add(name)
    nameOrder[name] = len(nameOrder)

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
    if c != None:
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
    if c == None:
      llFresh += l
    return x

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
  global partName
  global partFunc
  global nameOrder
  global seenNames
  global traceAcc
  global dists
  global metAccProbs
  global recAccNames
  global autoDepth

  startTime = time.time()
  curNames = set()
  ll = 0
  llFresh = 0
  llStale = 0
  db = {}
  observ = set()
  condition = set()
  partName = {}
  partFunc = {}
  nameOrder = {}
  seenNames = set()
  traceAcc = []
  dists[erps["crp"]].reset()
  metAccProbs = {} 
  recAccNames = set()
  autoDepth = 1

"""
Insert 3 stacks as globals, for functions, loopCount and lineNo-colNo pairs.
Specification follows that in Lighweight Implementations except for line-col stack.
We push and pop to this immediately before/after a call to stocPy. We keep the
additional column information to allow for multiple calls on the same line.

In order to ensure we push the function name and line count at the beggining of 
each function and pop them upon exiting the function we add the pushing at the beggining
of each function, put the body of the function in a try block and then put the popping
in a finally block.

In order to ensure we push/pop the loopStack before/after a loop we need to keep track
of the parent of a node so that we know in which context to add the stack manipulation code.

We take the existing names in the module so that we can create unique stack names.

Output of Transformer is stored in self.funcs, which contains all the functions
defined in the model and the 3 global stacks. This needs to be evaluated once in the
global context of the original module (so that globals defined by the user, such as
input data, can be seen). At least the entry point function then needs to be evaluated
again in the combined new and old global environment, so that functions calls work
correctly with the modified code.
"""
class RewriteModel(ast.NodeTransformer):

  def __init__(self, usedNames, localFuncs):
    self.usedNames = usedNames
    self.loopStack = self.genUniqueName("loopStack")
    self.funcStack = self.genUniqueName("funcStack")
    self.locStack = self.genUniqueName("locStack")
    self.calleeInfo = self.genUniqueName("calleeInfo")
    self.loopSize = self.genUniqueName("loopSize")
    self.seen = set()
    self.funcs = []
    self.localFuncs = localFuncs
    self.primsNumArgs = {}

  def genUniqueName(self, base):
    count = 0
    tempName = base
    while tempName in self.usedNames:
      tempName = base + str(count)
      count += 1
    return tempName

  def generic_visit(self, node):
    for child in ast.iter_child_nodes(node):
      self.visit(child, node)

  def visit(self, node, parent):
    funcName = "visit_" + node.__class__.__name__
    try:
      custVisit = getattr(self, funcName)
      custVisit(node, parent)
    except:
      self.generic_visit(node)

  def addChild(self, child, parent, loc=0):
    #print child, parent, loc
    map(ast.increment_lineno, dict(ast.iter_fields(parent))['body'][loc:])
    dict(ast.iter_fields(parent))['body'].insert(loc, child)
    ast.fix_missing_locations(parent)
    ast.increment_lineno(child)
    #print parent.lineno
    #print map(lambda x:x.lineno, dict(ast.iter_fields(parent))['body'])
    #print ast.dump(dict(ast.iter_fields(parent))['body'][0], include_attributes=True)
  """
  def getDescendantLoc(self, node, desc):
    block = dict(ast.iter_fields(node))['body']
    #print node, block, desc
    for i in range(len(block)):
      if self.descendantsContain(block[i], desc):
	return i

  def descendantsContain(self, node, desc):
    if node == desc:
      return True
    for child in ast.iter_child_nodes(node):
      if self.descendantsContain(child, desc):
	return True
    return False
  """
  def visit_Module(self, node, parent):
    self.funcs.append(ast.fix_missing_locations(ast.Assign(targets=[ast.Name(id=self.loopStack, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))))
    self.funcs.append(ast.fix_missing_locations(ast.Assign(targets=[ast.Name(id=self.locStack, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))))
    self.funcs.append(ast.fix_missing_locations(ast.Assign(targets=[ast.Name(id=self.funcStack, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))))
    self.generic_visit(node)
    self.funcs = ast.Module(body=self.funcs)
    return node

  def visit_FunctionDef(self, node, parent):
    if node in self.seen:
      return node
    self.seen.add(node)
    oldNode = dict(ast.iter_fields(node))
    finalBlock = []
    newNode = ast.FunctionDef(name=oldNode['name'], args=oldNode['args'], body=[ast.TryFinally(body=dict(ast.iter_fields(node))['body'], finalbody=finalBlock)], decorator_list=oldNode['decorator_list'])
    
    self.funcName = dict(ast.iter_fields(node))['name']
    if self.funcName in self.localFuncs:
      args = dict(ast.iter_fields(dict(ast.iter_fields(node))['args']))
      args['args'].append(ast.Name(id=self.calleeInfo, ctx=ast.Param()))
      args['defaults'].append(ast.Str(s=''))
      self.addChild(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.locStack, ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[ast.Name(id=self.calleeInfo, ctx=ast.Load())], keywords=[], starargs=None, kwargs=None)), newNode)

    #self.addChild(ast.Print(dest=None, values=[ast.Name(id=self.funcStack, ctx=ast.Load())], nl=True), node)
    self.addChild(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.funcStack, ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[ast.Str(s=self.funcName)], keywords=[], starargs=None, kwargs=None)), newNode)
    self.addChild(ast.Assign(targets=[ast.Name(id=self.loopSize, ctx=ast.Store())], value=ast.Call(func=ast.Name(id='len', ctx=ast.Load()), args=[ast.Name(id=self.loopStack, ctx=ast.Load())], keywords=[], starargs=None, kwargs=None)), newNode)

    finalBlock.append(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.funcStack, ctx=ast.Load()), attr='pop', ctx=ast.Load()), args=[], keywords=[], starargs=None, kwargs=None)))
    if self.funcName in self.localFuncs:
      finalBlock.append(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.locStack, ctx=ast.Load()), attr='pop', ctx=ast.Load()), args=[], keywords=[], starargs=None, kwargs=None)))
    loopCorr = ast.While(test=ast.Compare(left=ast.Call(func=ast.Name(id='len', ctx=ast.Load()), args=[ast.Name(id=self.loopStack, ctx=ast.Load())], keywords=[], starargs=None, kwargs=None), ops=[ast.Gt()], comparators=[ast.Name(id=self.loopSize, ctx=ast.Load())]), body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.loopStack, ctx=ast.Load()), attr='pop', ctx=ast.Load()), args=[], keywords=[], starargs=None, kwargs=None))], orelse=[])
    self.seen.add(loopCorr)
    finalBlock.append(loopCorr)
    #self.addChild(ast.Print(dest=None, values=[ast.Str(s=self.funcName + '_afterPop')], nl=True), node, loc=len(dict(ast.iter_fields(node))['body']))

    ast.fix_missing_locations(newNode)
    #print ast.dump(newNode) 
    self.funcs.append(newNode)
    self.generic_visit(newNode, parent)
    return newNode

  def visit_For(self, node, parent):
    if node in self.seen:
      return node
    self.seen.add(node)

    self.addChild(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.loopStack, ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[ast.Num(n=-1)], keywords=[], starargs=None, kwargs=None)), parent, loc=dict(ast.iter_fields(parent))['body'].index(node))
    self.addChild(ast.AugAssign(target=ast.Subscript(value=ast.Name(id=self.loopStack, ctx=ast.Load()), slice=ast.Index(value=ast.Num(n=-1)), ctx=ast.Store()), op=ast.Add(), value=ast.Num(n=1)), node)
    self.addChild(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.loopStack, ctx=ast.Load()), attr='pop', ctx=ast.Load()), args=[], keywords=[], starargs=None, kwargs=None)), parent, loc=dict(ast.iter_fields(parent))['body'].index(node) + 1)

    #print node.lineno, dict(ast.iter_fields(parent))['body'], dict(ast.iter_fields(parent))['body'].index(node), ast.dump(parent)
    self.generic_visit(node)
    return node

  def visit_While(self, node, parent):
    if node in seen:
      return node
    seen.add(node)

    self.addChild(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.loopStack, ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[ast.Num(n=-1)], keywords=[], starargs=None, kwargs=None)), parent, loc=dict(ast.iter_fields(parent))['body'].index(node))
    self.addChild(ast.AugAssign(target=ast.Subscript(value=ast.Name(id=self.loopStack, ctx=ast.Load()), slice=ast.Index(value=ast.Num(n=-1)), ctx=ast.Store()), op=ast.Add(), value=ast.Num(n=1)), node)
    self.addChild(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.loopStack, ctx=ast.Load()), attr='pop', ctx=ast.Load()), args=[], keywords=[], starargs=None, kwargs=None)), parent, loc=dict(ast.iter_fields(parent))['body'].index(node) + 1)

    self.generic_visit(node)
    return node

  def visit_Call(self, node, parent):
    if node in self.seen:
      return node
    self.seen.add(node)

    callName = dict(ast.iter_fields(dict(ast.iter_fields(node))['func'])).get('id', None)
    callType = dict(ast.iter_fields(dict(ast.iter_fields(node))['func'])).get('attr',None)
    #print ast.dump(dict(ast.iter_fields(node))['func']), callType, node.lineno, node.col_offset
    #print callName, self.localFuncs
    if callName in self.localFuncs:
      #print ast.dump(node)
      #print callName, node.lineno, node.col_offset
      dict(ast.iter_fields(node))['keywords'].append(ast.keyword(arg=self.calleeInfo, value=ast.Str(s=str(node.lineno) + "-" + str(node.col_offset))))
      
    if callType in erps.keys() or callType == "stocPrim":
      if callType not in self.primsNumArgs:
        self.primsNumArgs[callType] = len(inspect.getargspec(globals()[callType]).args)

      namedArgs = map(lambda x: dict(ast.iter_fields(x))['arg'], dict(ast.iter_fields(node))['keywords'])
      numArgs = len(namedArgs) + len(dict(ast.iter_fields(node))['args']) 
      #print callType, node.lineno, node.col_offset
      #print ast.dump(parent)
      if not ('name' in namedArgs or numArgs == self.primsNumArgs[callType]): #check if name already supplied
        dict(ast.iter_fields(node))['keywords'].append(ast.keyword(arg='name', value=ast.BinOp(left=ast.BinOp(left=ast.Call(func=ast.Name(id='str', ctx=ast.Load()), args=[ast.Name(id=self.funcStack, ctx=ast.Load())], keywords=[], starargs=None, kwargs=None), op=ast.Add(), right=ast.Call(func=ast.Name(id='str', ctx=ast.Load()), args=[ast.Name(id=self.locStack, ctx=ast.Load())], keywords=[], starargs=None, kwargs=None)), op=ast.Add(), right=ast.BinOp(left=ast.Str(s=str(node.lineno) + "-" + str(node.col_offset)), op=ast.Add(), right=ast.Call(func=ast.Name(id='str', ctx=ast.Load()), args=[ast.Name(id=self.loopStack, ctx=ast.Load())], keywords=[], starargs=None, kwargs=None)))))

    
    ast.fix_missing_locations(node)
    #print map(ast.dump, dict(ast.iter_fields(node))['keywords'])
    self.generic_visit(node)
    return node

class GetLocalNames(ast.NodeVisitor):

  def __init__(self):
    self.localNames = set()
    self.localFuncs = set()

  def visit_Name(self, node):
    self.localNames.add(dict(ast.iter_fields(node))['id'])
    self.generic_visit(node)

  def visit_FunctionDef(self, node):
    self.localFuncs.add(dict(ast.iter_fields(node))['name'])
    self.generic_visit(node)

def procRawModel(model):
  lcs = {}
  tree = ast.parse(inspect.getsource(inspect.getmodule(model)))
  gln = GetLocalNames()
  gln.visit(tree)
  #print ast.dump(tree)
  #print map(lambda x:type(x), dict(ast.iter_fields(dict(ast.iter_fields(tree))['body'][0]))['body'])
  oldGbs = inspect.stack()[2][0].f_globals
  
  rm = RewriteModel(gln.localNames.union(set(oldGbs.keys())), gln.localFuncs)
  rm.visit(tree, None)
  #print "\n\n", ast.dump(tree)
  #assert(False)
  #print ast.dump(rm.funcs)
  exec compile(rm.funcs, inspect.getfile(model), 'exec') in oldGbs, lcs
  for name, func in lcs.items():
    oldGbs[name] = func
  exec compile(rm.funcs, inspect.getfile(model), 'exec') in oldGbs, lcs
  entryName = dict(inspect.getmembers(model))['func_name']
  return lcs[entryName]

def initModel(model):
  model()
  while math.isnan(ll):
    resetAll()
    model()

def getSamples(model, noSamps, alg="met", thresh=0.1, autoNames = True, orderNames = False, outTraceAcc = False):
  if autoNames:
    model = procRawModel(model)
  initModel(model)
  if alg == "met":
    sampleDict = dict([(n+1, metropolisSampleTrace(model, no = n+1)) for n in range(noSamps)])
  elif alg == "sliceOld":
    sampleDict = dict([(n+1, sliceSampleTrace(model, no = n+1)) for n in range(noSamps)])
  elif alg == "sliceTD":
    sampleDict = dict([(n+1, sliceSampleTrace(model, no = n+1, tdCorr=True)) for n in range(noSamps)])
  elif alg == "sliceNoTrans":
    sampleDict = dict([(n+1, sliceSampleTrace(model, no = n+1, allowTransJumps = False)) for n in range(noSamps)])
  elif alg == "sliceMet":
    sampleDict = dict([(n+1, sliceMetMixSampleTrace(model, no = n+1)) for n in range(noSamps)])
  else:
    raise Exception("Unknown inference algorithm: " + str(alg))

  """
  #print "sampleDict", sampleDict
  totTries = 0
  for k,l in sorted(tries.items()):
    totTries += len(l)
    print k, len(l), l[0]
  print "TotTries", totTries
  """
  return aggSamples(sampleDict, orderNames, outTraceAcc)

def getSamplesByLL(model, noLLs, alg="met", thresh=0.1, autoNames = True, orderNames = False, outTraceAcc = False):
  if autoNames:
    model = procRawModel(model)

  initModel(model)
  totLLs = 0
  sampleDict = {}
  if alg == "met":
    while totLLs < noLLs:
      samp = metropolisSampleTrace(model, no = totLLs)
      totLLs += 1
      if totLLs < noLLs:
        sampleDict[totLLs] = samp
  elif alg == "sliceOld":
    while totLLs < noLLs:
      llCount, samp = sliceSampleTrace(model, no = totLLs, countLLs = True)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDict[totLLs] = samp
  elif alg == "sliceTD":
    while totLLs < noLLs:
      llCount, samp = sliceSampleTrace(model, no = totLLs, countLLs = True, tdCorr=True)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDict[totLLs] = samp
  elif alg == "sliceNoTrans":
    while totLLs < noLLs:
      llCount, samp = sliceSampleTrace(model, no = totLLs, allowTransJumps = False, countLLs = True)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDict[totLLs] = samp
  elif alg == "sliceMet":
    while totLLs < noLLs:
      llCount, samp = sliceMetMixSampleTrace(model, no = totLLs, countLLs = True, thresh = thresh)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDict[totLLs] = samp
  else:
    raise Exception("Unknown inference algorithm: " + str(alg))

  print "rejectedTransJumps", rejTransJumps
  return aggSamples(sampleDict, orderNames, outTraceAcc)

def getTimedSamples(model, maxTime, alg = "met", thresh = 0.1, autoNames=True, orderNames = False, outTraceAcc = False):
  if autoNames:
    model = procRawModel(model)

  initModel(model)
  sampleDict = {}
  while time.time() - startTime < maxTime:
    index = len(sampleDict)+1
    if alg == "met":
      sampleDict[index] = metropolisSampleTrace(model, no = index)
    elif alg == "sliceOld":
      sampleDict[index] = sliceSampleTrace(model, no = index)
    elif alg == "sliceTD":
      sampleDict[index] = sliceSampleTrace(model, no = index, tdCorr=True)
    elif alg == "sliceNoTrans":
      sampleDict[index] = sliceSampleTrace(model, no = index, allowTransJumps = False)
    elif alg == "sliceMet":
      sampleDict[index] = sliceMetMixSampleTrace(model, no = index)
    else:
      raise Exception("Unknown inference algorithm: " + str(alg))

  return aggSamples(sampleDict, orderNames, outTraceAcc)


def aggSamples(samples, orderNames = False, outTraceAcc = False):
  aggSamps = {}
  for count,vs in samples.items():
    for k,v in vs.items():
      if k in partName:
	k = partName[k]
      try:
	aggSamps[k][count].append(v)
      except:
	try:
	  aggSamps[k][count] = [v]
	except:
	  aggSamps[k] = {count:[v]}
  for k,vs in aggSamps.items():
    for count,v in vs.items():
      if k in partFunc:
	val = partFunc[k](v)
        aggSamps[k][count] = val 
      else:
	assert(len(v)==1)
	aggSamps[k][count] = v[0]

  if orderNames:
    sampList = [None for i in range(len(nameOrder))]
    #print sorted(filter(lambda (k,v):k[-1]=="]", nameOrder.items()), key = lambda (k,v):v)
    for name, dic in aggSamps.items():
      sampList[nameOrder[name]] = aggSamps[name]
    aggSamps = filter(lambda x: x != None, sampList)

  if outTraceAcc:
    aggSamps = (aggSamps, copy.copy(traceAcc))
  resetAll()
  return aggSamps

def sliceMetMixSampleTrace(model, no = None, thresh = 0.1, countLLs = False):
  if random.random() > thresh:
    return sliceSampleTrace(model, no = no, allowTransJumps = False, countLLs = countLLs)
  else:
    samp = metropolisSampleTrace(model, no = no)
    if countLLs:
      return 1, samp
    else:
      return samp

tries = {}
def metropolisSampleTrace(model, no = None):
  global db
  global ll
  global tries
  global observ
  global traceAcc
  global metAccProbs

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
  oobs = copy.copy(observ)
  recalcLL(model, n, x, l)
  #if llStale != 0:
  #  print "lls", ll, llFresh, llStale
  """
  if x == 0 and n == "branching-19-0":
    prob = math.e**(ll - oll)
    if math.isnan(prob):
      print n, x, ox, l, ll, oll, math.e**(ll - oll)
    else:
      p2 = db['branching-23-0'][1]
      try:
        tries[(ox,p2)].append((math.e**(ll - oll), ll, oll))
      except:
        tries[(ox,p2)] = [(math.e**(ll - oll), ll, oll)]
  """
    #print n, x, ox, l, ll, oll, math.e**(ll - oll),
  #print ox, x, math.e**(ll - oll)
  #print "before", db
  changed = True
  acc = ll - oll + ol - l + math.log(len(unCond)) - math.log(len(set(db.keys()).difference(condition))) + llStale - llFresh
  if n in recAccNames:
    try:
      metAccProbs[partName[n]].append(math.e**(min(0,acc)))
    except:
      metAccProbs[partName[n]] = [math.e**(min(0,acc))]

  if math.log(random.random()) < acc:
    traceAcc.append(1)
  else:
    traceAcc.append(-1)
    changed = False
    db = odb
    ll = oll
    observ = oobs

  sample = {}
  for n in observ:
    sample[n] = db[n][1]
  #print changed, n, x, sample[n]
  #if x == 0 and changed:
  #  print changed, sample
  #print changed, db
  if no % 100000 == 0:
    print no, sample, time.time() - startTime
  return sample

rejTransJumps = 0
def sliceSampleTrace(model, width = 10, no = None, allowTransJumps = True, countLLs = False, tdCorr=False):
  global db
  global ll
  global rejTransJumps
  global traceAcc

  llCount = 0
  unCond = list(set(db.keys()).difference(condition))
  n = random.choice(unCond)
  otp, ox, ol, ops = db[n]
  disc = isDiscrete(dists[otp])
  
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
    if disc:
      xl = int(math.floor(xl))

    odb = copy.copy(db)
    llCount += recalcLL(model, n, xl)
    db = odb
    curWidth *= 2
    if allowTransJumps and tdCorr:
      llc = ll + math.log(len(unCond)) - math.log(len(set(db.keys()).difference(condition)))+ llStale - llFresh
    else:
      llc = ll
    debugPrint(False, "l", xl, ll)

  ll = oll
  curWidth = r*width
  llc = ll
  while llc > u:
    xr += curWidth
    if disc:
      xr = int(math.ceil(xr))

    odb = copy.copy(db)
    llCount += recalcLL(model, n, xr)
    db = odb
    curWidth *= 2
    if allowTransJumps and tdCorr:
      llc = ll + math.log(len(unCond)) - math.log(len(set(db.keys()).difference(condition)))+ llStale - llFresh
    else:
      llc = ll
    debugPrint(False, "r", xr, ll)

  ll = oll
  first = True
  
  transJump = False
  llc = ll
  while first or llc < u or math.isnan(llc) or (transJump and (not allowTransJumps)):
    #print first, llc, u, transJump, allowTransJumps,
    if first:
      first = False
    if disc:
      x = random.randrange(xl,xr+1)
    else:
      x = random.uniform(xl, xr)

    odb = copy.copy(db)
    llCount += recalcLL(model, n, x)
    transJump = (llStale != 0) or (llFresh != 0) 
    if allowTransJumps and tdCorr:
      llc = ll + math.log(len(unCond)) - math.log(len(set(db.keys()).difference(condition))) + llStale - llFresh
    else:
      llc = ll
    #print xl, xr, x, u, ll, llc, llStale , llFresh
    #u = -1*ss.expon.rvs(-1*(oll - llStale))# +math.log(len(db)) - math.log(len(odb))))
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

  traceAcc += [0]*(llCount-1) + [1]
  sample = {}
  for o in observ:
    sample[o] = db[o][1]

  if no%10000 == 0:
    print no, n, xl, xr, sample

  if countLLs:
    return llCount, sample
  else:
    return sample

def sliceMultSampleTrace(model, width = 100, no = None):
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
    otp, ox, _, _ = db[n]
    disc = isDiscrete(dists[otp])
    oxs[n] = ox
    xl = ox
    xr = ox
    debugPrint(False, ox, ll, u)
    curWidth = r*width
    assert(ll > u)
    while ll > u:
      xl -= curWidth
      if disc:
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
      if disc:
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
    if disc:
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
  if l != None:
    ls = {n:l}
  else:
    ls = None
  return recalcMultLL(model, {n:x}, ls)

def recalcMultLL(model, xs, ls = None):
  global db
  global ll
  global llStale
  global llFresh
  global curNames
  global observ

  for n,x in xs.items():
    otp, ox, ol, ops = db[n]

    if ls == None:
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
      return 0
  #print x, db
  ll = 0
  llFresh = 0
  llStale = 0
  curNames = set()
  oldLen = len(db)
  observ = set()
  model()

  newLen = len(db)
  assert(oldLen <= newLen)

  for n in db.keys():
    if not n in curNames:
      llStale += db[n][2]
      db.pop(n)
  return 1


def getName(loopInd):
  _, _, lineNo, funcName, _, _ = inspect.stack()[2]
  name = funcName + "-" + str(lineNo) + "-" + str(loopInd)
  return name


def getExplicitName(loopInd, funcName=None, lineNo=None ):
  if funcName == None:
    return getName(loopInd)
  else:
    name = funcName + "-" + str(lineNo) + "-" + str(loopInd)
    return name

"""
def getERP1(n, c, tp, ps):
  global ll
  global llFresh
  global db
  global condition
  otp, ox, ol, ops = db.get(n, (None,None,None,None))

  if c != None:
    x = c
    condition.add(n)
  elif n in condition:
    x = ox
  else:
    assert (not n in condition)
    x = dists[tp].rvs(*ps)
  try:
    l = dists[tp].logpdf(x, *ps)
  except:
    l = dists[tp].logpmf(x, *ps)

  db[n] = (tp, x, l, ps)
  ll += l
  if not tp == otp and not c:
    llFresh += l
  return x
"""

def plotTestDist(b):
  samples = []
  for i in range(100000):
    samples.append(math.log(random.uniform(0,b)))
  print min(samples), max(samples)
  plt.hist(samples, 100)
  plt.show()

  samples = []
  for i in range(100000):
    samples.append(-1*ss.expon.rvs(-1*math.log(b)))
  plt.hist(samples, 100)
  plt.show()
  print min(samples), max(samples)

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

def plotSampDist(fn, name = "Metropolis", start=0, cutOff = float("inf")):
  samps = filter(lambda x: x<cutOff, readSamps(fn))
  plt.hist(samps, 100)
  plt.xlabel("DoFs", size=20)
  plt.ylabel("No. samples", size=20)
  plt.title(name + " ran for 10m - " + str(len(samps)) + " samples", size=30)# - mean: " + str(np.mean(samps))[:5] + ", stDev: " + str(np.std(samps))[:5])
  plt.show()

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

def plotCumPost(fn = "Posterior4", plot = True, show=True, zipRez=True, xlim=None):
  if isinstance(fn, basestring): 
    with open(fn,'r') as f:
      ds, ls = cPickle.load(f)
  else:
    ds, ls = map(list, zip(*sorted(fn.items())))
  cls = []
  curSum = 0
  for l in ls:
    curSum += l
    cls.append(curSum)

  if xlim:
    cls = [0] + cls + [1]
    ds = [xlim[0]] + ds + [xlim[1]]

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

def calcKSTest(pfn, fns, names = None, postXlim=[float("-inf"), float("inf")]):
  post = plotCumPost(pfn, plot=False, xlim=postXlim)
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
  if len(run) == 0:
    return xs, ys
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
      """
      if ind > 1000 and ksDiff > 0.9:
        print ksDiff
        print samps
        plotSamples(samps)
      """

  xs.append(ind)
  ys.append(calcKSDiff(postFun, samps))
  return xs, ys

def calcKSTests(pfn, fns, aggFreq, burnIn = 0, plot=True, xlim = 200000, names=None, alpha=0.25, single=False, modelName=None, postXlim=[float("-inf"), float("inf")]):
  postFun = interpolate.interp1d(*plotCumPost(pfn, plot=False, zipRez=False, xlim=postXlim))
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

  if not names:
    names = ns
  if plot:
    plt.legend(ps,names,loc=3)
    plt.xscale("log")
    plt.yscale("log", basey=2)
    plt.xlim([0, xlim])
    plt.xlabel("Trace Likelihood calculations", size=20)
    plt.ylabel("KS difference from posterior", size=20)
    if not modelName:
      no = fns[0].split("PerLL")[0][-1]
      if no == "4":
        no = "3"
      modelName = "NormalMean" + no
    plt.title("Performance on " + modelName + " Model", size=30)
    plt.show()

def calcKSSumms(pfn, fns, aggFreq, burnIn = 0, xlim = 200000, names=None, modelName=None, title=None, postXlim=[float("-inf"), float("inf")], ylim = None):
  postFun = interpolate.interp1d(*plotCumPost(pfn, plot=False, zipRez=False, xlim=postXlim))
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
      print map(len, runs)
    for r in range(len(runs)):
      print fn, r 
      xs, ys = calcKSRun(postFun, runs[r], aggFreq, burnIn)
      if len(xs) == 0:
        continue
      if xs[0] > start:
        start = xs[0]
      if xs[-1] < end:
        end = xs[-1]
      fs.append(interp1d(xs,ys))

    top = []
    med = []
    bot = []
    count = 1000#(end-start)/float(num)
    axis = np.logspace(math.log(start+0.001, 10), math.log(end-0.001, 10), count)
    for x in axis:
      #print x
      if x % 1000 == 0:
        print "x", x
      vals = []
      for f in fs:
        vals.append(f(x))

      top.append(np.percentile(vals, 25))
      med.append(np.percentile(vals, 50))
      bot.append(np.percentile(vals, 75))

    plt.plot(axis, med, cols[i], linewidth=2, alpha=0.9)
    plt.plot(axis, top, cols[i] + "--", linewidth=2, alpha=0.9)
    plt.plot(axis, bot, cols[i] + "--", linewidth=2, alpha=0.9)

  if not names:
    names = ns
  plt.legend(ps,names,loc=3)
  plt.xscale("log")
  plt.yscale("log", basey=2)
  plt.xlim([0, xlim])
  if ylim:
    plt.ylim(ylim)
  plt.xlabel("Trace likelihood calculations", size=20)
  plt.ylabel("KS difference from posterior", size=20)
  if not title:
    if not modelName:
      no = fns[0].split("PerLL")[0][-1]
      if no == "4":
        no = "3"
      modelName = "NormalMean" + no
    title = "Performance on " + modelName + " Model"
  plt.title(title, size=30)

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

def calcKLCondTests(posts, sampList, test, freq = float("inf")):
  names = ["Posterior for r < 5", "Posterior for r >= 5"]
  axs = [None,None]
  for samps in sampList:
    #print samps[0], ind
    ind = test(samps)
    post = posts[ind]
    ax = calcKLTest(post, samps, freq, show=False, col=cols[ind])
    axs[ind] = ax

  #plt.xlim([0,10000])
  plt.ylabel("KL divergence from posterior")
  plt.xlabel("No. Model Simulations")
  plt.title("Conv. to Posteriors for Slice Sampling w/out Trans-Dimensional Jumps")
  plt.legend(axs, names)
  plt.show()

def calcKLTests(post, sampsLists, names, freq = float("inf"), burnIn = 0, xlim=float("inf"), aggSums=False, xlabel="Trace likelihood calculations", title="Performance on Branching model"):
  post = potRead(post)
  axs = []
  for i in range(len(sampsLists)):  
    print "Run", i 
    ax, = plt.plot([0],[0], cols[i])
    axs.append(ax)
    for samps in getRuns(sampsLists[i]):
      calcKLTest(post, samps, freq, show=False, col=cols[i], burnIn = burnIn, xlim = xlim, aggSums=aggSums)

  plt.ylabel("KL divergence from posterior", size=20)
  plt.xlabel(xlabel, size=20)
  plt.title(title, size=30)
  plt.legend(axs, names, loc=3)
  plt.show()

def calcKLTest(post, samps, freq = float("inf"), show=True, col=None, plot=True, cutOff = float("inf"), burnIn = 0, xlim = float("inf"), aggSums=False):
  post = potRead(post)
  if aggSums: # have multiple posts and samps, need to do KLTest on each and sum up the results
    assert(len(post) == len(samps))
    xs = []
    ys = []
    fs = []
    start = float("-inf")
    end = float("inf")
    num = float("-inf")
    for i in range(len(post)):
      pxs, pys = calcKLTest(post[i], samps[i], plot=False, freq=freq, cutOff=cutOff, burnIn=burnIn, xlim=xlim)
      if len(pxs) == 0:
        continue
      if pxs[0] > start:
        start = pxs[0]
      if pxs[-1] < end:
        end = pxs[-1]
      if len(pxs) > num:
        num = len(pxs)
      fs.append(interp1d(pxs,pys)) 

    inc = (end - start) / float(num)
    print start, end, inc
    for x in np.arange(start+inc, end-inc, inc):
      val = 0
      for f in fs:
        val += f(x)
      xs.append(x)
      ys.append(val)
  else:
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
      ax, = plt.plot(xs,ys, col, alpha=0.5)
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

def getRuns(runs):
  if isinstance(runs, basestring):
    with open(runs, 'r') as f:
      runs = cPickle.load(f)
  return runs

def calcKLSumms(post, sampsLists, names, burnIn = 0, xlim = float("inf"), aggSums=False, xlabel="Trace likelihood calculations", modelName = None, title = None):
  post = potRead(post)
  axs = []
  for i in range(len(sampsLists)):
    axs.append(calcKLSumm(post, getRuns(sampsLists[i]), col=cols[i], show=False, burnIn = burnIn, xlim = xlim, aggSums=aggSums))

  plt.ylabel("KL divergence from posterior", size=20)
  plt.xlabel(xlabel, size=20)
  if not title:
    if not modelName:
      no = fns[0].split("PerLL")[0][-1]
      if no == "4":
        no = "3"
      modelName = "NormalMean" + no
    title = "Performance on " + modelName + " Model"
  plt.title(title, size=30)
  plt.legend(axs, names, loc=3)
  plt.show()

def calcKLSumm(post, sampsList, col = 'b', show=True, burnIn = 0, xlim = float("inf"), aggSums=False):
  post = potRead(post)
  fs = []
  start = float("-inf")
  end = float("inf")
  num = float("-inf")
  for samps in sampsList:
    print "fs", len(fs), len(samps)
    xs,ys = calcKLTest(post, samps, plot=False, burnIn = burnIn, xlim = xlim, aggSums=aggSums) #xs should be already sorted
    if len(xs) == 0:
      continue
    if xs[0] > start:
      start = xs[0]
    if xs[-1] < end:
      end = xs[-1]
    if len(xs) > num:
      num = len(xs)
    fs.append(interp1d(xs,ys))

  count = 1000#(end-start)/float(num)
  top = []
  med = []
  bot = []
  axis = np.logspace(math.log(start+0.001, 10), math.log(end-0.001, 10), count)
  for x in axis:
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
  ax, = plt.plot(axis, med, col)
  plt.plot(axis, top, col + "--")
  plt.plot(axis, bot, col + "--") 
  
  if show:
    plt.show()

  return ax

def dispVarNameTimes():
  times = []
  with open('varNameTimes', 'r') as f:
    for line in f:
      times.append(map(float, line.strip()[1:-1].split(',')))

  plt.boxplot(times)
  plt.show()

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
  return os.path.dirname(os.path.realpath(f)) + "/"

def extractDict(dic, name = None):
  if not name:
    assert(len(dic.keys()) == 1)
    name = dic.keys()[0]
  return dic[name]

def procUserSamples(samples, accs):
  #print len(samples), len(accs)
  corrSamples = {}
  prevSamp = samples[-len(accs)-1]
  samples = samples[-len(accs):]
  assert(len(samples) == len(accs))
  for i in range(len(samples)):
    if accs[i] == -1:
      corrSamples[i+1] = prevSamp
    elif accs[i] == 0:
      continue
    elif accs[i] == 1:
      corrSamples[i+1] = samples[i]
      prevSamp = samples[i]
    else:
      raise Exception("Unknown traceAcc value: " + str(accs[i]))
  return corrSamples

if __name__ == "__main__":
  #dispVarNameTimes()
  #data = [studentT(4,"p" + str(i)) for i in range(10000)]
  #print data, min(data), max(data)
  #plt.hist(data, 100)
  #plt.show()
  #plotTestDist(0.7)

  #plotSampDist("tdf/metTdfSamp600", "Slice", cutOff=6)
  #plotSampDist("tdf/tdfSamps600", "Slice", cutOff=6)
  calcKSTest(["tdf/newMetTdfSamp600", "tdf/ventureTdfSamp600", "tdf/dec5TdfSamps600"], "tdf/Posterior4", names=["Met", "Venture", "Met Decomp"])
  #calcKSTests(["normal/normal4PerLLMetDec44"], "normal/normal4Post", aggFreq=np.logspace(1,math.log(20000,10),10), burnIn=0, plot=True)
  #plotCumSampDists(["ventureTdfSamp600", "metTdfSamp600", "sliceTdfSamp600"])
  #calcKSTest(["tdf/ventureTdfSamp600", "tdf/tdfSliceTest", "tdf/newMetTdfSamp600"], "tdf/Posterior4", names=["Venture", "Slice_LI", "Metropolis_LI"])
  #calcKSTests(["normal/normal3PerLLSlicemet"], "normal/normal3Post", aggFreq=np.logspace(1,math.log(19998,10),10), burnIn=1000, plot=False)
  mi = "4"
  lim = 200000
  #calcKSTests(["normal/normal" + mi + "PerLLMet", "normal/normal" + mi + "PerLLSliceV1", "normal/normal" + mi + "PerLLSlicemet0.5"], "normal/normal" + mi + "Post", aggFreq=np.logspace(1,math.log(lim,10),10), burnIn=1000, xlim = lim, names=["Metropolis", "Slice TD1", "1:1 Metropolis:Slice"], alpha=0.75)
  #calcKSSumms(["normal/normal" + mi + "PerLLMet","normal/normal" + mi + "PerLLSlice", "normal/normal" + mi + "PerLLMetDec4", "normal/normal" + mi + "PerLLMetDec44"], "normal/normal" + mi + "Post", aggFreq=np.logspace(1,math.log(lim,10),10), burnIn=1000, xlim = lim, names=["Metropolis", "Slice", "Met Decomp4", "Met Decomp44"])



