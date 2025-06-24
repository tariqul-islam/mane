import numpy as np

import numba
from numba import prange

import random

import scipy.sparse
from scipy import optimize

from sklearn.decomposition import PCA

@numba.jit(nopython=True, parallel=True)
def euclidean_distances_numba(X, squared = True):
    n = X.shape[0]
    xcorr = np.zeros((n,n),dtype=X.dtype)
    for i in prange(n):
        for j in range(i,n):
            dist = np.sum( np.square(X[i,:] - X[j,:]) )
            if not squared:
                dist = np.sqrt(dist)
            xcorr[i,j] = dist
            xcorr[j,i] = dist
    
    return xcorr

#@numba.jit(nopython=True)
def get_weight_function(dists, rho, sigma):
    d = dists - rho
    #print(d)
    d[d<0] = 0
    weight = np.exp(- d / sigma )
    return weight

#@numba.jit(nopython=True)
def search_sigma(dists, rho, k, tol = 10**-5, n_iteration=200):
    sigma_min = 0
    sigma_max = 1000
    
    cur_sigma = 100
    
    logk = np.log2(k)
    #print(logk)
    
    for i in range(n_iteration):
        
        cur_sigma = (sigma_min+sigma_max)/2
        probs = get_weight_function(dists,rho,cur_sigma)
        weight = np.sum(probs)
        #print(weight)
        
        if np.abs(logk - weight) < tol:
            break
        
        if weight < logk:
            sigma_min = cur_sigma
        else:
            sigma_max = cur_sigma
        
    return cur_sigma, probs

@numba.jit(nopython=True, parallel=True)
def symmetrization_step(prob):
    n = prob.shape[0]
    P = np.zeros((n,n),dtype=np.float32)

    for i in prange(n):
        #if i%1000 == 0:
        #    print('Completed ', i, ' of ', n)
        for j in prange(i,n):
            p = prob[i,j] + prob[j,i] - prob[i,j] * prob[j,i] #t-conorm
            P[i,j] = p
            P[j,i] = p
            
    return P

def get_prob_matrix(X, n_neighbors=15):
    n = X.shape[0]
    dist = euclidean_distances_numba(X, squared = False)
    sort_idx = np.argsort(dist,axis=1)
    #sort_idx = sort_idx.astype(np.int32)
    sort_idx = sort_idx[:,1:n_neighbors+1]
    
    rho = [ dist[i, sort_idx[i,0] ] for i in range(n)]
    rho = np.array(rho)
    
    

    sigmas = []

    directed_graph = []


    #'''
    for i in range(n):
        #if (i+1)%1000 == 0:
        #    print('Processed ', i+1, ' of ', n, ' samples.')
        sigma, weights = search_sigma(dists = dist[i,sort_idx[i,:]],rho = rho[i],k = n_neighbors)

        probs = np.zeros(n)
        probs[sort_idx[i,:]] = weights
        #print(sum(weights), np.log2(n_neighbors))
        #print(sort_idx[i,:])
        #print(probs[1770:1780])

        directed_graph.append(probs)

    directed_graph = np.array(directed_graph).astype(np.float32)
    prob = directed_graph
    
    P = symmetrization_step(prob)
    
    graph = scipy.sparse.coo_matrix(P)
    
    return graph

def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.
    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.
    n_epochs: int
        The total number of epochs we want to train for.
    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    Copied from UMAP repo: https://github.com/lmcinnes/umap/
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result

def get_ab(MIN_DIST):
    x = np.linspace(0, 3, 300)

    y = (x>MIN_DIST) * np.exp(-x+MIN_DIST)
    y[x<=MIN_DIST] = 1.0

    function = lambda x, a, b: 1 / (1 + a*x**(2*b))

    p , _ = optimize.curve_fit(function, x, y) 

    a = p[0]
    b = p[1]   
    
    return a,b  
    
    
@numba.jit(nopython=True)
def clip(x,val=4.0):

    if x>val:
        return val
    elif x<-val:
        return -val
    else:
        return x
    
@numba.jit(nopython=True)
def update_attraction(x, y, a, b, dim, lr, P):
    dist = np.sum((x - y)**2)

    if dist>0.0:
        grad_coeff = 2*a*b*dist**(b-1.0) / (1 + a * dist**b)
    else:
        grad_coeff = 0.0


    for d in range(dim):
        mv = clip(grad_coeff * P * (x[0,d]-y[0,d]))  # * P[idx,idy]
        mv = mv * lr

        x[0,d] -= mv
        y[0,d] += mv
        
    return

@numba.jit(nopython=True)
def update_repulsion(x, y, a, b, dim, lr, P, repulsion_strength=1.0):
    dist = np.sum((x - y)**2)

    if dist>0.0:
        grad_coeff = 2 * repulsion_strength * b / ( (0.001+dist) * (1.0 + a * dist**b) )
    else:
        grad_coeff = 0


    for d in range(dim):
        #if grad_coeff > 0.0:
        #    grad = clip(grad_coeff  * (x[0,d]-y[0,d]))
        #    #* (1 - P[idx,idy])
        #else:
        #    grad = 0.0

        grad = clip(grad_coeff  * (x[0,d]-y[0,d]) * (1-P))
        mv = grad * lr

        x[0,d] += mv
        #y[0,d] -= mv

    
    return


@numba.jit(nopython=True)
def one_step_in_a_set(emA, emH, idx, rows, columns, a, b, dim,
                   nA, n_points,
                   epochs_per_sample,
                   epoch_of_next_sample,
                   epochs_per_negative_sample,
                   epoch_of_next_negative_sample,
                   lr, epoch,
                   repulsion_strength=1.0):
    
    if epoch_of_next_sample[idx] <= epoch:
        x_idx = rows[idx]
        y_idx = columns[idx]
        
        if x_idx < nA:
            x = emA[x_idx:x_idx+1,:]
        else:
            x = emH[x_idx-nA:x_idx-nA+1,:]
        
        if y_idx < nA:
            y = emA[y_idx:y_idx+1, :]
        else:
            y = emH[y_idx-nA:y_idx-nA+1,:]
            
        update_attraction(x, y, a, b, dim, lr, 1)
        
        epoch_of_next_sample[idx] += epochs_per_sample[idx]
        
        n_neg_samples = int(
                (epoch - epoch_of_next_negative_sample[idx]) / epochs_per_negative_sample[idx]
            )
        
        for i in range(n_neg_samples):
            y_idx = np.random.choice(n_points)
            
            if x_idx == y_idx:
                continue
            
            if y_idx < nA:
                y = emA[y_idx:y_idx+1, :]
            else:
                y = emH[y_idx-nA:y_idx-nA+1,:]
                
            update_repulsion(x, y, a, b, dim, lr, 0,repulsion_strength=repulsion_strength)
            
        epoch_of_next_negative_sample[idx] += (
                n_neg_samples * epochs_per_negative_sample[idx]
            )
            
            
            
    
    return 

@numba.jit(nopython=True,parallel=True)
def one_epoch_2sets_2(emCommon, em1, em2,
                     rows1, columns1, rows2, columns2,
                     nCommon, n_points_1, n_points_2,
                      Set, K_idx,
                     a, b, dim,
                     lr, epoch,
                     epochs_per_sample_1,
                     epoch_of_next_sample_1,
                     epochs_per_negative_sample_1,
                     epoch_of_next_negative_sample_1,
                     epochs_per_sample_2,
                     epoch_of_next_sample_2,
                     epochs_per_negative_sample_2,
                     epoch_of_next_negative_sample_2,
                     repulsion_strength=1.0):
    '''
    Set1 = 1 * np.ones(epochs_per_sample_1.shape[0])
    Set2 = 2 * np.ones(epochs_per_sample_2.shape[0])
    Set = np.random.permutation(np.concatenate((Set1,Set2)))
    '''
    
    for i in prange(len(Set)):
        if Set[i] == 1:
            one_step_in_a_set(emA=emCommon, emH=em1, idx=K_idx[i], 
                              rows=rows1, columns=columns1, a=a, b=b, dim=dim,
                              nA=nCommon, n_points=n_points_1,
                              epochs_per_sample=epochs_per_sample_1,
                              epoch_of_next_sample=epoch_of_next_sample_1,
                              epochs_per_negative_sample=epochs_per_negative_sample_1,
                              epoch_of_next_negative_sample=epoch_of_next_negative_sample_1,
                              lr=lr, epoch=epoch,
                              repulsion_strength=repulsion_strength)
        elif Set[i] == 2:
            one_step_in_a_set(emA=emCommon, emH=em2, idx=K_idx[i], 
                              rows=rows2, columns=columns2, a=a, b=b, dim=dim,
                              nA=nCommon, n_points=n_points_2,
                              epochs_per_sample=epochs_per_sample_2,
                              epoch_of_next_sample=epoch_of_next_sample_2,
                              epochs_per_negative_sample=epochs_per_negative_sample_2,
                              epoch_of_next_negative_sample=epoch_of_next_negative_sample_2,
                              lr=lr, epoch=epoch,
                              repulsion_strength=repulsion_strength)
        else:
            print('Warning: Something Wrong')
    
    return
    
def mane_2set(cdata1, cdata2=None, data1=None, data2=None, 
                n_neighbors= 15,
                n_components= 2, 
                epochs = 200,
                random_state = None,
                min_dist= 0.1,
                neg_sample_rate=5,
                repulsion_strength=1.0):
    '''
    A simple functional form for MANE algorithm.
    cdata1, cdata2 = data from domain 1 and 2 that have shared correspondance. shapes: NxD1, NxD2
    data1 = data from domain 1. shapes: N1xD1
    data2 = data from domain 2. shapes: N2xD2
    
    '''            
    if random_state is not None:
        np.random.seed(random_state)
            
    if cdata2 is None:
        cdata2 = cdata1
    
    if data1 is None:
        data1 = cdata1    
    else:
        data1 = np.concatenate((cdata1, data1))
        
    if data2 is None:
        data2 = cdata2
    else:
        data2 = np.concatenate((cdata2, data2))
    
    graph_1 = get_prob_matrix(data1,n_neighbors=n_neighbors)
    graph_1.data[graph_1.data < (graph_1.data.max() / float(epochs))] = 0.0
    graph_1.eliminate_zeros()
    epochs_per_sample_1_og = make_epochs_per_sample(graph_1.data, epochs)

    graph_2 = get_prob_matrix(data2,n_neighbors=n_neighbors)
    graph_2.data[graph_2.data < (graph_2.data.max() / float(epochs))] = 0.0
    graph_2.eliminate_zeros()
    epochs_per_sample_2_og = make_epochs_per_sample(graph_2.data, epochs)
    
    a,b = get_ab(min_dist)
    
    pca = PCA(n_components = n_components)
    init = pca.fit_transform(data1)
    expansion = 10.0 / np.abs(init).max()
    init = init*expansion
    embA = init[:len(cdata1)].astype(np.float32).copy()
    embB = init[len(cdata1):].astype(np.float32).copy()
    
    init2 = pca.fit_transform(data2)
    expansion = 10.0 / np.abs(init2).max()
    init2 = init2*expansion
    embA_2 = init2[:len(cdata2)].astype(np.float32).copy()
    embC = init2[len(cdata2):].astype(np.float32).copy()
    
    embA = (embA + embA_2)/2
    
    epochs_per_sample_1 = epochs_per_sample_1_og.copy()
    epoch_of_next_sample_1 = epochs_per_sample_1.copy()
    epochs_per_negative_sample_1 = epochs_per_sample_1 / neg_sample_rate
    epoch_of_next_negative_sample_1 = epochs_per_negative_sample_1.copy()
    
    epochs_per_sample_2 = epochs_per_sample_1_og.copy()
    epoch_of_next_sample_2 = epochs_per_sample_2.copy()
    epochs_per_negative_sample_2 = epochs_per_sample_2 / neg_sample_rate
    epoch_of_next_negative_sample_2 = epochs_per_negative_sample_2.copy()
    
    init_lr = 1.0
    
    Set1 = 1 * np.ones(epochs_per_sample_1.shape[0])
    Set2 = 2 * np.ones(epochs_per_sample_2.shape[0])
    Set = np.random.permutation(np.concatenate((Set1,Set2)))
    k_idx_1 = np.arange(epochs_per_sample_1.shape[0],dtype=np.int32)
    k_idx_2 = np.arange(epochs_per_sample_2.shape[0],dtype=np.int32)
    K_idx = np.zeros(Set.shape[0],dtype=np.int32)
    K_idx[Set==1] = k_idx_1
    K_idx[Set==2] = k_idx_2
    
    for epoch in range(epochs):
        
        lr = init_lr * (1.0 - float(epoch)/float(epochs))
        
        one_epoch_2sets_2(emCommon=embA, em1=embB, em2=embC, 
                         rows1=graph_1.row, columns1=graph_1.col, 
                         rows2=graph_2.row, columns2=graph_2.col, 
                         nCommon=len(embA), 
                         n_points_1=len(embA)+len(embB), n_points_2=len(embA)+len(embC),
                         Set = Set, K_idx=K_idx,
                         a=a, b=b, dim=n_components,
                         lr=lr, epoch=epoch,
                         epochs_per_sample_1=epochs_per_sample_1,
                         epoch_of_next_sample_1=epoch_of_next_sample_1,
                         epochs_per_negative_sample_1=epochs_per_negative_sample_1,
                         epoch_of_next_negative_sample_1=epoch_of_next_negative_sample_1,
                         epochs_per_sample_2=epochs_per_sample_2,
                         epoch_of_next_sample_2=epoch_of_next_sample_2,
                         epochs_per_negative_sample_2=epochs_per_negative_sample_2,
                         epoch_of_next_negative_sample_2=epoch_of_next_negative_sample_2,
                         repulsion_strength=repulsion_strength)
        
    return embA, embB, embC
