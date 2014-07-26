import stocPy

noLetters = [7,2,3,1,4,0,0] # number of letters received each day for a week

def lettersPerDay(): # generative model
  expectedLetters = stocPy.unifCont(0, 20, obs=True) # prior on the number of letters
  for datum in noLetters:
    stocPy.poisson(expectedLetters, cond=datum) # conditioning on the data we have

if __name__ == "__main__":
  stocPy.plotSamples(stocPy.getSamples(lettersPerDay, 10000), xlabel = "Expected letters per day") # perform inference, extract and plot 10,000 samples
