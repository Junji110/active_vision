import numpy as np
import scipy.special
import scipy.stats

def f_oneway(mean, var, n, ddof=0):
    '''
    Apply one-way ANOVA to data given in the form of mean, variances, and sizes
    of the samples for individual groups.
    
    Arguments
    ---------
    mean : float array
        Means of the samples for individual groups. This can be a multi-
        dimensional array, where the 1st dimension spans across groups and the
        rest are the dimensions of each data point.
    var : float array
        Variances of the samples for individual groups. The shape of var must be
        identical to that of mean. The function assumes by default var to be
        sample variances (i.e. sum of squares divided by sample size) rather
        than unbiased estimates of population variance (i.e. sum of squares
        divided by sample size - 1). By setting the keyword argument ddof = 1,
        the function takes var as the unbiased estimates.
    n : int array
        1-dimensional array containing sizes of the samples for individual
        groups.
    ddof : int (default: 0)
        When set to 1, the function takes var as unbiased estimates of
        population variance. In concrete, when sum of squares within groups is
        calculated from the elements of the argument var, n - ddof is
        multiplied.
    
    Returns
    -------
    F : float
        F test statistics obtained by application of one-way ANOVA to the given data
    p : float
        p-value of F
    '''
    
    # check data shape and value
    if mean.shape != var.shape:
        raise ValueError("Arguments mean and var must be of the same size.")
    if n.ndim != 1:
        raise ValueError("Argument n must be a 1-dimensional array.")
    if n.shape[0] != mean.shape[0]:
        raise ValueError("The size of n must be equal to the size of the 1st dimension of mean (i.e. mean.shape[0]).")
    if n.shape[0] == 1:
        raise ValueError("There must be data for more then 1 group.")
    
    # Reshape n to allow for broadcasting
    if mean.ndim > 1:
        n = n.reshape([n.size] + [1] * (var.ndim - 1))
    
    # store the degrees of freedom of ms_within and ms_between for later use
    n_total = n.sum()
    num_group = n.size
    df_within = np.float(n_total - num_group)
    df_between = np.float(num_group - 1)
    
    # mean square within groups
    ss_within = np.sum(var * (n - ddof), axis=0)
    ms_within =  ss_within / df_within
    
    # mean square between groups    
    grand_mean = np.sum(mean * n, axis=0) / n_total
    ss_between = np.sum(np.square(mean - grand_mean) * n, axis=0)
    ms_between = ss_between / df_between
    
    f = ms_between / ms_within
    p = scipy.special.fdtrc(df_between, df_within, f)
    
    return f, p


def circ_r(alpha, w=None, d=None):
    '''
    This function is based on Circular Statistics Toolbox for Matlab by Philipp Berens
    (http://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-
    toolbox-directional-statistic).
    Below is the documentation of the original Matlab code.
    % r = circ_r(alpha, w, d)
    %   Computes mean resultant vector length for circular data.
    %
    %   Input:
    %     alpha    sample of angles in radians
    %     [w        number of incidences in case of binned angle data]
    %     [d    spacing of bin centers for binned data, if supplied 
    %           correction factor is used to correct for bias in 
    %           estimation of r, in radians (!)]
    %     [dim  compute along this dimension, default is 1]
    %
    %     If dim argument is specified, all other optional arguments can be
    %     left empty: circ_r(alpha, [], [], dim)
    %
    %   Output:
    %     r        mean resultant length
    %
    % PHB 7/6/2008
    %
    % References:
    %   Statistical analysis of circular data, N.I. Fisher
    %   Topics in circular statistics, S.R. Jammalamadaka et al. 
    %   Biostatistical Analysis, J. H. Zar
    %
    % Circular Statistics Toolbox for Matlab
    % By Philipp Berens, 2009
    % berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
    '''
    if w is None:
        w = np.ones_like(alpha)
    else:
        if w.shape != alpha.shape:
            raise ValueError("Input dimensions do not match")
    
    if d is None:
        d = 0
    
    # compute sum of cos and sin of angles
    r = np.sum(w * np.exp(1.0j * alpha))
    
    # obtain length 
    r = np.abs(r) / np.sum(w)
    
    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d != 0:
        c = d / 2 / np.sin(d / 2)
        r = c * r
    
    return r

def circ_rtest(alpha, w=None, d=None):
    '''
    This function is based on Circular Statistics Toolbox for Matlab by Philipp Berens
    (http://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-
    toolbox-directional-statistic).
    Below is the documentation of the original Matlab code.
    function [pval z] = circ_rtest(alpha, w, d)
    %
    % [pval, z] = circ_rtest(alpha,w)
    %   Computes Rayleigh test for non-uniformity of circular data.
    %   H0: the population is uniformly distributed around the circle
    %   HA: the populatoin is not distributed uniformly around the circle
    %   Assumption: the distribution has maximally one mode and the data is 
    %   sampled from a von Mises distribution!
    %
    %   Input:
    %     alpha    sample of angles in radians
    %     [w        number of incidences in case of binned angle data]
    %     [d    spacing of bin centers for binned data, if supplied 
    %           correction factor is used to correct for bias in 
    %           estimation of r, in radians (!)]
    %
    %   Output:
    %     pval  p-value of Rayleigh's test
    %     z     value of the z-statistic
    %
    % PHB 7/6/2008
    %
    % References:
    %   Statistical analysis of circular data, N. I. Fisher
    %   Topics in circular statistics, S. R. Jammalamadaka et al. 
    %   Biostatistical Analysis, J. H. Zar
    %
    % Circular Statistics Toolbox for Matlab
    % By Philipp Berens, 2009
    % berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
    '''
    r = circ_r(alpha, w, d)
    
    if w is None:
        r = circ_r(alpha)
        n = len(alpha)
    else:
        r = circ_r(alpha, w, d)
        n = np.sum(w)
    
    # compute Rayleigh's R (equ. 27.1)
    R = n * r
    
    # compute Rayleigh's z (equ. 27.2)
    z = R ** 2 / n
    
    # compute p value using an approximation in Zar, p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    # outdated version:
    # compute the p value using an approximation from Fisher, p. 70
    # pval = np.exp(-z)
    # if n < 50:
    #   pval = pval * (1 + (2*z - z**2) / (4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))
    
    return pval, z


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    '''
    Tests for f_oneway
    
    This test calculates f-values and corresponding p-values using f_oneway and
    scipy.stats.f_oneway, and check that the difference between the results is
    reasonably small (i.e. less than 1.0e-12). 
    '''
    
    
    # test for multi-dimensional arguments
    num_group = 3
    data_shape = [100, 200, 10]
    arg_shape = [num_group] + data_shape
    group_mean = np.empty(arg_shape)
    group_var = np.empty(arg_shape)
    group_size = np.empty(num_group)
    data = []
    for group_id in range(num_group):
        # generate data for scipy.stats.f_oneway
        sample_size = np.random.randint(20, 30)
        group_data_shape = [sample_size] + data_shape
        group_data = np.random.rand(*group_data_shape)
        data.append(group_data)
        
        # arguments to the local f_oneway
        group_mean[group_id] = np.mean(group_data, axis=0)
        group_var[group_id] = np.var(group_data, axis=0)        
        group_size[group_id] = sample_size
    
    f, p = f_oneway(group_mean, group_var, group_size)
    f_scipy, p_scipy = scipy.stats.f_oneway(*data)
    
    assert(np.all(f - f_scipy < 1.0e-12))
    assert(np.all(p - p_scipy < 1.0e-12))
    print("Test for multi-dimensional arguments passed.")
    print("Maximum difference: delta_f = {0}, delta_p = {1}".format((f - f_scipy).max(), (p - p_scipy).max()))
    print("")
    
    # test for one dimensional arguments
    num_group = 3
    group_mean = np.empty(num_group)
    group_var = np.empty(num_group)
    group_size = np.empty(num_group)
    data = []
    for group_id in range(num_group):
        # generate data for scipy.stats.f_oneway
        sample_size = np.random.randint(20, 30)
        group_data = np.random.rand(sample_size)
        data.append(group_data)
        
        # arguments to the local f_oneway
        group_mean[group_id] = np.mean(group_data)
        group_var[group_id] = np.var(group_data)
        group_size[group_id] = sample_size
    
    f, p = f_oneway(group_mean, group_var, group_size)
    f_scipy, p_scipy = scipy.stats.f_oneway(*data)
    
    assert(f - f_scipy < 1.0e-12)
    assert(p - p_scipy < 1.0e-12)
    print("Test for one dimensional arguments passed.")
    print("Difference: delta_f = {0}, delta_p = {1}".format(f - f_scipy, p - p_scipy))
    print("")
    
    # test for keyword argument ddof 
    num_group = 3
    group_mean = np.empty(num_group)
    group_var = np.empty(num_group)
    group_size = np.empty(num_group)
    data = []
    for group_id in range(num_group):
        # generate data for scipy.stats.f_oneway
        sample_size = np.random.randint(20, 30)
        group_data = np.random.rand(sample_size)
        data.append(group_data)
        
        # arguments to the local f_oneway
        group_mean[group_id] = np.mean(group_data)
        group_var[group_id] = np.var(group_data, ddof=1)
        group_size[group_id] = sample_size
    
    # case where ddof is properly set
    f_scipy, p_scipy = scipy.stats.f_oneway(*data)
    f, p = f_oneway(group_mean, group_var, group_size, ddof=1)
    assert(f - f_scipy < 1.0e-12)
    assert(p - p_scipy < 1.0e-12)
    
    # case where ddof is wrongly set
    f, p = f_oneway(group_mean, group_var, group_size, ddof=0)
    try:
        # These assertions should FAIL due to the wrong setting of ddof
        assert(f - f_scipy < 1.0e-12)
        assert(p - p_scipy < 1.0e-12)
        raise AssertionError("Keyword argument ddof is not used properly in the function.")
    except AssertionError:
        pass
    
    print("Test for keyword argument ddof passed.")
    print("")
    
    quit()
