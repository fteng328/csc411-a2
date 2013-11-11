from kmeans import *
import timeit
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
  randConst = 2
  p = randConst + np.random.rand(K, 1)   #initialize random probabilities 
  #print p
  p = p / np.sum(p)   #normalize probability over clusters
  #print p
  
  #mn = np.mean(x, axis=1).reshape(-1, 1)
  #print mn
  #print "MMMMMMMMMMMMMMMMMMMM"
  vr = np.var(x, axis=1).reshape(-1, 1)
  #print vr
 
  #mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)   #dont really understand, so random?
  #print mu
  
  mu = KMeans(x, K, 5)
  vary = vr * np.ones((1, K)) * 2
  
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary  #completely lost here, just keep whichever is larger???
  #print vary
  logProbX = np.zeros((iters, 1))
  #print logProbX
  #raw_input('Press Enter plzzzzzzzzzz.')
  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    #print "respTot"
    #print respTot

    respX = np.zeros((N, K))
    #print 'respX'
    #print respX
    respDist = np.zeros((N, K))
    #print 'respDist'
    #print respDist
    logProb = np.zeros((1, T))
    #print 'logProb'
    #print logProb
    ivary = 1 / vary
    #print'vary'
    #print vary
    #print ivary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) ##what is happending here????
    #print 'logNorm'
    #print logNorm
    logPcAndx = np.zeros((K, T))
    #print ' logPcAndx'
    #print logPcAndx
    for k in xrange(K):
      # calculate the squared distance from points to the cluster centers
      dis = (x - mu[:,k].reshape(-1, 1))**2   
      #calculate the prior probabilities
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)  #look into this
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.mean(logProb)

    print 'Iter %d logProb %.5f' % (i, logProbX[i])
    ###break
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
  '''print"waiting...................."
  print p

  print logPcAndx.shape
  print logProb.shape
  print np.exp(logProb)
  print np.sum(np.exp(logProb))
  print 'now'
  #logPdgivenX= logPcAndx -np.sum(logProbX)    #using bayes rule
  print logPdgivenX
  #print np.exp(logPcAndx-logProb)
  #print np.exp(logPcAndx)/p'''
  return logProb

def main():
  start = timeit.default_timer()
  K = 5
  iters = 10
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  p, mu, vary, logProbX = mogEM(inputs_train, K, iters, minVary)
  #print"XXXXXXXXXXX"
  #print logProbX
  train_logprob = mogLogProb(p, mu, vary, inputs_train)
  #print 'printing'
  #print np.sum(np.exp(train_logprob))
  #raw_input('Press Enter to exit.')
  valid_logprob = mogLogProb(p, mu, vary, inputs_valid)
  test_logprob = mogLogProb(p, mu, vary, inputs_test)
  print 'Logprob : Train  %f Valid %f Test %f' % (np.mean(train_logprob), np.mean(valid_logprob), np.mean(test_logprob))
  #print 'mix por:'
  #print p
  stop = timeit.default_timer()
  print 'take it takes:----'
  print stop - start 
  raw_input('Press Enter to exit.')
  ShowMeans(mu)
  
  #ShowMeans(vary)
  #raw_input('Press Enter to exit.')

if __name__ == '__main__':
  main()
