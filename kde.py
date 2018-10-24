import keras
import keras.backend as K

import numpy as np

# Squared Euclidean distance
def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = K.expand_dims(K.sum(K.square(X), axis=1), 1)
    dists = x2 + K.transpose(x2) - 2*K.dot(X, K.transpose(X))
    return dists

def get_shape(x):
    dims = K.cast( K.shape(x)[1], K.floatx() ) 
    N    = K.cast( K.shape(x)[0], K.floatx() )
    return dims, N

# I am not sure the following calculation is correct, see equation(18) in paper, should be doublechecked.
def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
#     normconst = (dims/2.0)*K.log(2*np.pi*var)
#     lprobs = K.logsumexp(-dists2, axis=1) - K.log(N) - normconst
    lprobs = K.logsumexp(-dists2, axis=1) - K.log(N)

    h = -K.mean(lprobs)
    return dims/2 + h

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

# By looking at equation(14), the following calculation does not seem to be correct
# Paper: arXiv:1706.02419.v4, Estimating Mixture Entropy with Pairwise Distance
# in the implementation, the variance of Gaussian noise seems to be fixed, shouldn't it be adjusted according to the maximum activation value of each layer. For example, we choose a baseline sigma^2 = 1e-3, this value is scaled by the ratio of maximum activation of each layer.
def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
#     return (np.log(var**dims) + dims*np.log(2*np.pi) + dims)/2.0
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)

