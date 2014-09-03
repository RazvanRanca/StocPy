import stocPy

noLetters = [7,2,3,1,4,0,0] # number of letters received each day for a week

"""
Model description:

We have a letterbox which can hold up to 20 letters. 
Initially we have no idea how many letters we might receive in a day (i.e. we have a uniform prior).
We notice the number of letters we receive over the period of a week.
Assuming the number of letters received in different days are independent of one another (poisson process), we'd
like to estimate how many letters we will receive, on average, in a day.
"""
def lettersPerDay():
  expectedLetters = stocPy.unifCont(0, 20, obs=True) # prior on the number of letters, "obs" specifies we want to observe this variable
  for datum in noLetters:
    stocPy.poisson(expectedLetters, cond=datum) # condition the model on the data we have

"""
Same generative model as lettersPerDay, only now declared using generic stochastic primitve calls.
"""
def lettersPerDayGeneric():
  expectedLetters = stocPy.stocPrim("unifCont", (0, 20), obs=True) # generic uniform continuous call
  for datum in noLetters:
    stocPy.stocPrim("poisson", (expectedLetters, ), cond=datum) # generic poisson call, single parameter still needs to be wrapped in tuple

if __name__ == "__main__":
  stocPy.plotSamples(stocPy.getSamples(lettersPerDay, 10000), xlabel = "Letters per day") # perform inference, extract and plot 10,000 samples
