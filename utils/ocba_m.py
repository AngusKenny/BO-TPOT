import numpy as np

def load_system(file_name, system, reps, reps_available, delim=','):
    """
    Reads system data from a txt file (assumes comma delimited by default).  
    Assumes that each column represents a system.
    Returns a numpy array.  Each row is a system (single row); each col is a replication
    
    
    @file_name = name of file containing csv data
    @system = index of system in txt file.
    @reps = replications wanted
    @reps_available = total number of replications that has been simulated.
    @delim = delimiter of file.  Default = ',' for CSV.  

    """
    
    return np.genfromtxt(file_name, delimiter=delim, usecols = system, skip_footer = (reps_available - reps))

def simulate(data, k, allocations):
    """
    Simulates the systems.  
    Each system is allocated a different budget of replications
    
    Returns list of numpy arrays
    
    @allocations = numpy array.  budget of replications for the k systems 
    """
    return [data[i][0:allocations[i]] for i in range(k)]
    
def get_ranks(array):
    """
    Returns a numpy array containing ranks of numbers within a input numpy array
    e.g. [3, 2, 1] returns [2, 1, 0]
    e.g. [3, 2, 1, 4] return [2, 1, 0, 3]
        
    @array - numpy array (only tested with 1d)
        
    """
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

def bootstrap(data, boots=1000, random_state=None):
    """
    Returns a numpy array containing the bootstrap resamples
    Useful for creating a large number of experimental datasets 
    for testing R&S routines
    
    Keyword arguments:
    data -- numpy.ndarray of systems to boostrap
    boots -- integer number of bootstraps (default = 1000)
    random_state -- int, set the numpy random number generator
    """
    
    np.random.seed(random_state)
    experiments = boots
    designs = 10
    samples = data.shape[1]
     
    datasets = np.zeros((experiments, designs, samples))
     
    for exp in range(experiments):
        
        for design in range(designs):

            for i in range(samples):

                datasets[exp][design][i] = data[design][round(np.random.uniform(0, samples)-1)]
      
    return datasets

def crn_bootstrap(data, boots=1000, random_state=None):
    """
    Returns a numpy array containing the bootstrap resamples
    Useful for creating a large number of experimental datasets 
    for testing R&S routines.  This function assumes common
    random numbers have been used to create the original data.
    
    Keyword arguments
    data -- numpy.ndarray of systems to boostrap
    boots -- integer number of bootstraps (default = 1000)
    random_state -- int, set the numpy random number generator
    """
    
    np.random.seed(random_state)

    experiments = boots
    designs = data.shape[0]
    samples = data.shape[1]
    
    datasets = np.zeros((experiments, designs, samples))
     
    for exp in range(experiments):
        
         for i in range(samples):

                row = data.T[np.random.choice(data.shape[0])]
                
                for design in range(designs):
                    datasets[exp][design][i] = row[design]  
      
    return datasets

def summary_statistics(systems, allocations):
    means = np.array([array.mean() for array in systems])
    stds = np.array([array.std() for array in systems])
    ses = np.divide(stds, np.sqrt(allocations))
    
    return means, ses

def top_m(means, m):
    """
    Returns the top m system designs
    Largest sample means
    """
    ranks = get_ranks(means)
    return np.argpartition(ranks, -m)[-m:]

def bottom_m(means, m):
    """
    Returns the bottom m system designs
    Smallest sample means
    """
    ranks = get_ranks(means)
    return np.argpartition(ranks, m)[:m]

def parameter_c(means, ses, k, m):
    order = np.argsort(means)
    s_ses = ses[order]
    s_means = means[order]

    return((s_ses[k-m+1] * s_means[k-m]) + (s_ses[k-m] * s_means[k-m+1]))/(s_ses[k-m]+s_ses[k-m+1])  

def get_allocations(means, ses, m, Delta):
    k = len(means)
    allocs = [0 for _ in range(k)]
    
    c = parameter_c(means, ses, k, m)
    deltas = means - c
        
    for i in range(Delta):
        values = np.divide(allocs, np.square(np.divide(ses, deltas)))
        ranks = get_ranks(values)
        allocs[ranks.argmin()] += 1

    return allocs

def ocba_m(dataset, k, allocations, T, delta, m):
    
    while allocations.sum() < T:
        
        #simulate systems using new allocation of budget
        reps = simulate(dataset, k, allocations) 
        
        #calculate sample means and standard errors
        means, ses = summary_statistics(reps, allocations)
        
        #calculate parameter c and deltas
        c = parameter_c(means, ses, k, m)
        deltas = means - c
        
        #allocate
        for i in range(delta):
            values = np.divide(allocations, np.square(np.divide(ses, deltas)))
            ranks = get_ranks(values)
            allocations[ranks.argmin()] += 1
            
    return means, ses, allocations

def cs(selected_top_m, true_top_m):
    """
    Returns boolean value.  
    True = correct selection of top m
    False = incorrect selection (one or more of selected top m is incorrect)
    
    Keyword arguments:
    selected_top_m - numpy.array containing the indexes of the top m means selected by the algorithm
    true_top_m - numpy.array containing the indexes of the true top m means
    
    """
    return np.array_equal(np.sort(selected_top_m), true_top_m)

def oc(true_means, selected_top_m, true_top_m):
    """
    Return the opportunity cost of the selection
    Method penalised particular bad solutions more than mildly bad
    
    @true_means - numpy.array of true means
    @selected_top_m - numpy.array containing the indexes of the top m means selected by the algorithm
    @true_top_m - numpy.array containing the indexes of the true top m means
    
    """
    return (true_means[selected_top_m] - true_means[true_top_m]).sum()

def p_cs(correct_selections):
    """
    Returns the probability of correct selection P{cs}
    Keyword arguments:
    
    correct_selections - list indicating if a given budget found the top m e.g. [False, True, True]
    """
    return np.array(correct_selections).sum() / len(correct_selections)

def p_cs2(correct_selections):
    """
    Returns the probability of correct selection P{cs}
    
    @correct_selections - numpy.array of budget versus exp. Contains True or False indicating correct selection.
    
    """
    return np.mean(correct_selections, axis=1)

def e_oc(opportunity_costs):
    """
    Return the expected opportunity cost E{oc}
    
    @opportunity costs - arrange of opportunity cost per budget
    """
    return np.array(opportunity_costs).mean()

def e_oc2(opportunity_costs):
    """
    Return the expected opportunity cost E{oc}
    
    @opportunity costs - arrange of opportunity cost per budget
    """
    return np.mean(opportunity_costs, axis=0)