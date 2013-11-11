from util import *
import scipy.linalg as lin
import matplotlib.pyplot as plt
plt.ion()

def pcaimg(X, k):
  """
  PCA matrix X to k dimensions.
  Inputs:
    X : Each column of X contains a data vector.
    k : Number of dimensions to reduce to.
  Returns:
    v : The eigenvectors. Each column of v is an eigenvector.
    mean : mean of X.
    projX : X projected down to k dimensions.
  """
  xdim, ndata = X.shape
  mean = np.sum(X, axis=1).reshape(-1, 1)
  X = X - mean
  cov = np.dot(X, X.T) / ndata

  w, v = lin.eigh(cov, eigvals=(xdim - k, xdim - 1))
  # w contains top k eigenvalues in increasing order of magnitude.
  # v contains the eigenvectors corresponding to the top k eigenvalues.

  projX = np.dot(v.T, X)
  return v, mean, projX

def ShowEigenVectors(v):
  """Displays the eigenvectors as images in decreasing order of eigen value."""
  plt.figure(1)
  plt.clf()
  for i in xrange(v.shape[1]):
    plt.subplot(1, v.shape[1], i+1)
    plt.imshow(v[:, v.shape[1] - i - 1].reshape(16, 16).T, cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')


def q6(v2,mean2,v3,mean3, var2, var3,  inputset, targetset):
  logProb2 = mogLogProb(p2,m2,var2, inputset)
  logProb3 = mogLogProb(p3,m3,var3, inputset)
  #denominator for bayes rule
  denom = np.exp(logProb2) + np.exp(logProb3)
  
  #using bayes rule
  p2givenx = np.exp(logProb2)/denom
  p3givenx = np.exp(logProb3)/denom
  predictions = 1 - (p2givenx > p3givenx)
  return (float(1-np.sum((predictions == targetset), dtype = float)/targetset.shape[1]))




def main():
  #K = 5  # Number of dimensions to PCA down to.
  K = [2,5,15,25]
  
  error = [[],[],[],[]]
  
  #Seperate the two training sets
  inputs_train2 = inputs_train[:,:300]
  inputs_train3 = inputs_train[:,300:]
  target_train2 = target_train[:,:300]
  target_train3 = target_train[:,300:]

  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')

  for k in range(len(K)):
    p2, mu2, vary2, logProbX2 = mogEM(inputs_train2, K[k], iters, minVary)
    p3, mu3, vary3, logProbX3 = mogEM(inputs_train3, K[k], iters, minVary)
    error[0].append(q5(p2,p3,mu2,mu3,vary2,vary3,inputs_train, target_train))	
    error[1].append(q5(p2,p3,mu2,mu3,vary2,vary3,inputs_valid, target_valid))
    error[2].append(q5(p2,p3,mu2,mu3,vary2,vary3,inputs_test, target_test))




"""  
  v, mean, projX = pcaimg(inputs_train, K)
  ShowEigenVectors(v)
"""
if __name__ == '__main__':
  main()
