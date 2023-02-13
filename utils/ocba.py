# -*- coding: utf-8 -*-
import numpy as np

def get_allocations(mu, sigma, Delta, min_allocs=None, max_allocs=None, minimize = True):
    # number of concepts/designs/whatever
    n_c = len(mu)
    
    # if no min_allocs or max_allocs specified, initialise as zeros or infinity
    min_allocs = np.zeros(n_c) if not np.any(min_allocs) else min_allocs
    max_allocs = np.inf * np.ones(n_c) if not np.any(max_allocs) else max_allocs + min_allocs
            
    # convert maximization problem to minimization
    mu = -1 * mu if not minimize else mu
    
    # variable for total allocations
    tot_allocs = np.zeros(n_c)
    # get variances
    var = np.power(sigma,2)
    
    # total budget including extra delta
    T = np.sum(min_allocs) + Delta
    
    # sort unique mu values so we can keep allocating later if best is greater than max_allocs
    sorted_mu = np.unique(mu)
    
    # get best and second best concepts
    b_mu = sorted_mu[0]
    b_ids = np.flatnonzero(mu == b_mu)
    not_b_ids = np.flatnonzero(mu > b_mu)
    if sorted_mu.size > 1:
        s_mu = sorted_mu[1]
        s_ids = np.flatnonzero(mu == s_mu)
        s_id = np.random.choice(s_ids)
        rem_ids = np.flatnonzero(mu > s_mu)
    
    # variable for ratios
    ratio = np.ones(n_c)
    
    # compute ratios for not best or second best
    ratio[rem_ids] = np.power((b_mu-s_mu)/(b_mu-mu[rem_ids]),2) * var[rem_ids] / var[s_id]
    
    # compute ratios for best
    temp = np.sum(np.power(ratio[not_b_ids],2)/var[not_b_ids]) if len(not_b_ids) > 0 else 1
    ratio[b_ids] = np.sqrt(var[b_ids] * temp)
        
    # array to track allowed allocations
    do_alloc = np.array([True for _ in range(n_c)])
    
    do_alloc[ratio == np.nan] = False
    do_alloc[ratio == np.inf] = False
    do_alloc[ratio == -np.inf] = False
    
    # variable to track budget allocated
    T_1 = T
    
    cont_alloc = True
    
    while cont_alloc:
        # print(tot_allocs)
        sum_ratio = np.sum(ratio[do_alloc])
        
        # allocate to allowed concepts
        tot_allocs[do_alloc] = np.floor(T_1 / sum_ratio * ratio[do_alloc])
        
        # bump to ensure new allocs at least match previous allocs
        bump_allocs = tot_allocs < min_allocs
        tot_allocs[bump_allocs] = min_allocs[bump_allocs]
        do_alloc[bump_allocs] = False
        
        # check that nothing has gone over maximum allowed allocations
        cut_allocs = tot_allocs > max_allocs
        tot_allocs[cut_allocs] = max_allocs[cut_allocs]        
        do_alloc[cut_allocs] = False
        
        # if nothing bumped or cut then we should be okay to finish
        cont_alloc = (np.any(bump_allocs) or np.any(cut_allocs)) and np.any(do_alloc)
        
        # if something was bumped, remove from budget and continue
        if cont_alloc:
            T_1 = T - sum(tot_allocs[~do_alloc])
    
    # allocate the rest evenly among best designs with remainder arbitrarily distributed
    extra_allocs = int(T - np.nansum(tot_allocs))
    while extra_allocs > 0:
        for m in sorted_mu:
            if extra_allocs == 0: break
            m_ids = np.flatnonzero(mu == m)
            while np.all(tot_allocs[m_ids] < max_allocs[m_ids]):
                if extra_allocs == 0: break
                # randomly permute ids
                for m_id in np.random.permutation(m_ids):
                    if tot_allocs[m_id] < max_allocs[m_id]:
                        tot_allocs[m_id] = tot_allocs[m_id] + 1
                        extra_allocs = extra_allocs-1
                        if extra_allocs == 0: break
    
    # only return new allocations
    return (tot_allocs - min_allocs).astype(int)