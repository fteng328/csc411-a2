from kmeans import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def mogEM(x, K, iters, minVary=0):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  randConst = 1
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
#  mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
  mu = KMeans(x, K, 5)
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.mean(logProb)

    print 'Iter %d logProb %.5f' % (i, logProbX[i])

    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(i), logProbX[:i], 'r-')
    plt.title('Log-probability of data versus # iterations of EM')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in xrange(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb



def q5(p2,p3,m2,m3, var2, var3,  inputset, targetset):
  logProb2 = mogLogProb(p2,m2,var2, inputset)
  logProb3 = mogLogProb(p3,m3,var3, inputset)
  #denominator for bayes rule
  denom = np.exp(logProb2) + np.exp(logProb3)
  
  #using bayes rule
  p2givenx = np.exp(logProb2)/denom
  p3givenx = np.exp(logProb3)/denom
  predictions = 1 - (p2givenx > p3givenx)
  return (float(1-np.sum((predictions == targetset), dtype = float)/targetset.shape[1]))

def ploterrors(error,K):
  plt.figure(2)
  plt.clf()
  plt.suptitle("Average Classification Errors")
  plots = plt.plot(K,error[0],K,error[1],K,error[2])
  plt.legend((plots[0],plots[1],plots[2]), ("Training","Validation","Test"))
  plt.ylabel("avg Error")
  plt.xlabel("K")
  plt.draw()
  return

def main():
#  K = 2
  K = [2,5,15,25]
  iters = 10
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  error = [[],[],[],[]]
  
  #Seperate the two training sets
  inputs_train2 = inputs_train[:,:300]
  inputs_train3 = inputs_train[:,300:]
  target_train2 = target_train[:,:300]
  target_train3 = target_train[:,300:]

  for k in range(len(K)):
    p2, mu2, vary2, logProbX2 = mogEM(inputs_train2, K[k], iters, minVary)
    p3, mu3, vary3, logProbX3 = mogEM(inputs_train3, K[k], iters, minVary)
    error[0].append(q5(p2,p3,mu2,mu3,vary2,vary3,inputs_train, target_train))	
    error[1].append(q5(p2,p3,mu2,mu3,vary2,vary3,inputs_valid, target_valid))
    error[2].append(q5(p2,p3,mu2,mu3,vary2,vary3,inputs_test, target_test))

 
  
    ShowMeans(mu2)
    ShowMeans(vary2)
    ShowMeans(mu3)
    ShowMeans(vary3)
  ploterrors(error,K) 
  raw_input("done")
#  raw_input("")

"""
  train_logprob = mogLogProb(p, mu, vary, inputs_train)
  valid_logprob = mogLogProb(p, mu, vary, inputs_valid)
  test_logprob = mogLogProb(p, mu, vary, inputs_test)
  print 'Logprob : Train  %f Valid %f Test %f' % (np.mean(train_logprob), np.mean(valid_logprob), np.mean(test_logprob))
  raw_input('Press Enter to exit.')
"""

if __name__ == '__main__':
  main()
