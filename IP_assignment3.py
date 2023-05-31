# Author: Simon Guldager Andersen
# Date (latest update): 7/1-2023

### SETUP -----------------------------------------------------------------------------------------------------------------------------------

## Imports:
import os, sys
import iminuit
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.gridspec import GridSpec
from scipy import integrate, constants, stats
from numpy import newaxis as NA

## Change directory to current one
os.chdir('Inverse Problems')

## Set whether to read or write model parameters and log-likehood values from file
load_parameters = False
file_name_parameters = 'data\\A3_MCMC_samples3.txt'   #BEHOLD DATA0:: , data0 800_000 prior data. data2 i 800_000 uniform data.
file_name_likelihood = 'data\\A3_likelihood_values3.txt'

## Define global parameters
# Load data
grav_data = 1e-5 * np.array([-15.1, -23.9, -31.2, -36.9, -40.8, -42.8, -42.5, -40.7, -37.1, -31.5, -21.9, -12.9]) # (m / s^2)
x_measurements = np.array([535, 749, 963, 1177, 1391, 1605, 1819, 2033, 2247, 2461, 2675, 2889], dtype = 'float') #(m)

# Define constants
grav_const = constants.gravitational_constant
delta_rho = -1733 # kg/m^3

# Number of parameter and data points
N_data = len(grav_data)
N_parameters = 12

# Set boundaries and initalize rectangle edges
x0, xmax = 0, 3420
dx = (xmax - x0) / N_parameters
x_rectangles = np.linspace(x0, xmax, N_parameters + 1)

# Define covariance matrices
parameter_cov_matrix = 300 ** 2 * np.eye(N_parameters) # (m^2)
data_cov_matrix = 1e-10 * 1.2 ** 2 * np.eye(N_data)  #(Gal ^2 = m^2/s^4 )

# Estimate the expected parameter value used in the prior probability density
parameter_guess = grav_data / (2 * np.pi * grav_const * delta_rho)

### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

def model(h, x_measurement, tuning = 0):

    x = x_rectangles.astype('float')
    xj = x_measurement.astype('float')

    def antiderivative1(z):
        return - z * (np.log(z ** 2 + tuning) - 2)
    def antiderivative2(z, h):
        return z * np.log(z** 2 + h ** 2) - 2 * z + 2 * h * np.arctan(z / h)

    val1 = antiderivative1(x[-1] - xj) - antiderivative1(x[0] - xj)
    val2 = antiderivative2(x[NA, 1:] - xj[:, NA], h[NA,:]) - antiderivative2(x[NA, :-1] - xj[:, NA], h[NA, :])
    return grav_const * delta_rho * (val1 + np.sum(val2, axis = 1))

def model_1d(h, x_measurement, tuning = 0):

    x = x_rectangles.astype('float')
    xj = x_measurement.astype('float')

    def antiderivative1(z):
        return - z * (np.log(z ** 2 + tuning) - 2)
    def antiderivative2(z, h):
        return z * np.log(z** 2 + h ** 2) - 2 * z + 2 * h * np.arctan(z / h)

    val1 = antiderivative1(x[-1] - xj) - antiderivative1(x[0] - xj)
    val2 = antiderivative2(x[1:] - xj, h) - antiderivative2(x[:-1] - xj, h)
    return grav_const * delta_rho * (val1 + np.sum(val2))


def prior_prob_density(parameters, cov_matrix_diagonal = True):
    parameter_guesses = parameter_guess.astype('float')
    cov_matrix = parameter_cov_matrix.astype('float')
    if cov_matrix_diagonal:
        return np.exp(-0.5 * np.linalg.norm((parameters - parameter_guesses ) ** 2 / np.diag(cov_matrix))  )
    else:
        return np.exp(-0.5 * (parameters - parameter_guesses ).T @ np.linalg.inv(cov_matrix) @ (parameters - parameter_guesses)  )

def log_prior_prob_density(parameters, cov_matrix_diagonal = True):
    parameter_guesses = parameter_guess.astype('float')
    cov_matrix = parameter_cov_matrix.astype('float')
    if cov_matrix_diagonal:
        return - 0.5 * np.linalg.norm((parameters - parameter_guesses ) ** 2 / np.diag(cov_matrix)) 
    else:
        return -0.5 * (parameters - parameter_guesses ).T @ np.linalg.inv(cov_matrix) @ (parameters - parameter_guesses)

def likelihood_function(parameters, gravity_data, x_measurement, cov_matrix_diagonal = True):
    data = gravity_data.astype('float')
    N_data_points = len(data)
    cov_matrix = (data_cov_matrix.astype('float'))[:N_data_points]
    if cov_matrix_diagonal:
        return np.exp(-0.5 * np.linalg.norm((data - model(parameters, x_measurement) ) ** 2 / np.diag(cov_matrix))  )
    else:
        return np.exp(-0.5 * (data - model(parameters, x_measurement)).T @ np.linalg.inv(cov_matrix) @ (data - model(parameters))  )

def log_likelihood_function(parameters, gravity_data, x_measurement, cov_matrix_diagonal = True):
    data = gravity_data.astype('float')
    N_data_points = len(data)
    cov_matrix = (data_cov_matrix.astype('float'))[:N_data_points]
 
    if cov_matrix_diagonal:
        return -0.5 * np.linalg.norm((data - model(parameters, x_measurement) ) ** 2 / np.diag(cov_matrix)) 
    else:
        return -0.5 * (data - model(parameters, x_measurement)).T @ np.linalg.inv(cov_matrix) @ (data - model(parameters))  

def posterior_prop_density(parameters, gravity_data, x_measurement, cov_matrix_diagonal = True):
    if cov_matrix_diagonal:
        return likelihood_function(parameters, gravity_data, x_measurement) * prior_prob_density(parameters)
    else:
        return likelihood_function(parameters, gravity_data, x_measurement, cov_matrix_diagonal = False) * prior_prob_density(parameters, cov_matrix_diagonal = False)

def acceptance_probability(parameters_old, parameters_new, gravity_data, x_measurement):
    #misfit defined as the negative of the log of the function
    misfit_new = -log_prior_prob_density(parameters_new) - log_likelihood_function(parameters_new, gravity_data, x_measurement)
    misfit_old = -log_prior_prob_density(parameters_old) - log_likelihood_function(parameters_old, gravity_data, x_measurement)
    if misfit_new <= misfit_old:
        return 1
    else:
        return np.exp(-(misfit_new - misfit_old))

def markov_chain_MC(gravity_data, x_measurement, N_samples, N_burn_in, boundaries, scaling_factor, random_start_guess = False, save_samples = False):
    # Initialize
    N_parameters, _ = boundaries.shape
    m_samples = np.empty([N_samples, N_parameters])
    likelihood_list = []
    N_solutions = 0
    iterations = 0

    # Initialize scaling
    scaling = scaling_factor * (boundaries[:,1] - boundaries[:,0])
 
    # Define factor to multiply/divide with scaling to ensure acceptable acceptance rate
    scaling_amplification = 1.2

    # Initalize array to keep whether the last 20 iterations were accepted or rejected
    N_acceptance_history = 5
    acceptance_history = 0.5 * np.ones([N_parameters, N_acceptance_history])
    acceptance_sum = np.zeros(N_parameters)

    ## Initialize random parameter
    # generate N_parameters random numbers
    rvs = np.random
    rvs.seed(123456789)
    r = rvs.rand(N_parameters)
    if random_start_guess:
        # generate random solution by scaling r with parameter boundaries
        m_old = boundaries[:,0] + r * (boundaries[:,1] - boundaries[:,0])
    else:
        m_old = parameter_guess.astype('float')
    
    while N_solutions < N_samples:
        iterations += 1
        if iterations > N_burn_in and (iterations - N_burn_in)  % (N_samples/100) == 0:
            print('Progress: ', int(100 * ((iterations - N_burn_in) / N_samples )), " %")
        # generate random parameter index to perturb
        rand_index = int(np.floor(N_parameters * rvs.rand(1)))

        # Construct uniform perturbation
        perturbation = (rvs.rand(1)-0.5) * scaling[rand_index]

        # add perturbation with periodic boundary conditions
        m_new = m_old.astype('float')
        m_new[rand_index] =  ( m_new[rand_index] + perturbation )
   
 
        # enforce periodic boundary conditions to ensure that new solution is contained within boundaries
        if m_new[rand_index] < boundaries[rand_index, 0]:
            m_new[rand_index] = min(boundaries[rand_index, 1], boundaries[rand_index, 1] + (m_new[rand_index] - boundaries[rand_index, 0])) 

        elif m_new[rand_index] > boundaries[rand_index, 1]:
                m_new[rand_index] = max(boundaries[rand_index,0], boundaries[rand_index, 0] + (m_new[rand_index] - boundaries[rand_index, 1]))

        # calculate acceptance probability
       # print(iterations, "   ", acceptance_probability(m_old, m_new))
        if rvs.rand(1) < acceptance_probability(m_old, m_new, gravity_data, x_measurement):
        #    print(iterations, "  acceo ", acceptance_probability(m_old, m_new))
            m_old = m_new.astype('float')
 
            acceptance_history[rand_index, iterations % N_acceptance_history] = 1
            acceptance_sum[rand_index] = acceptance_sum[rand_index] + 1
        else:
            acceptance_history[rand_index, iterations % N_acceptance_history] = 0
        
      
        # change size of perturbation to ensure an acceptance rate near 50%
        if (acceptance_history[rand_index].mean() < 0.35): # & (scaling[rand_index] * np.sqrt(parameter_cov_matrix[rand_index, rand_index]) > 0.01 * (boundaries[rand_index,1] - boundaries[rand_index,0])) :                
            scaling[rand_index] = scaling[rand_index ] / scaling_amplification
        elif (acceptance_history[rand_index].mean() > 0.65): # & (scaling[rand_index] * np.sqrt(parameter_cov_matrix[rand_index, rand_index]) < 0.1 * (boundaries[rand_index,1] - boundaries[rand_index,0])):
            scaling[rand_index] = scaling[rand_index] * scaling_amplification
        if iterations >= N_burn_in:
            m_samples[N_solutions, :] = m_old
            N_solutions += 1
       # print(scaling)

        likelihood_list.append(likelihood_function(m_old, gravity_data, x_measurement))
  
    if save_samples:
        np.savetxt(file_name_parameters, m_samples)
        np.savetxt(file_name_likelihood, likelihood_list)

    print("Average acceptance rate: ", acceptance_sum / (iterations / N_parameters))
    
    return m_samples, likelihood_list

def calc_corr_matrix(data, ddof = 1):
    """assuming that each column represents a separate variable"""
    rows, cols = data.shape
    corr_matrix = np.empty([cols, cols])

    for i in range(cols):
        for j in range(i, cols):
    
            if ddof == 0:
                corr_matrix[i,j] = np.mean(data[:,i] * data[:,j]) - data[:,i].mean() * data[:,j].mean() / (data[:,i].std(ddof = 0) * data[:,j].std(ddof = 0))
            elif ddof == 1:
                corr_matrix[i,j] = 1/(rows - 1) * np.sum((data[:,i] - data[:,i].mean())*(data[:,j] - data[:,j].mean())) / (data[:,i].std(ddof = 1) * data[:,j].std(ddof = 1))
            else:
                print("The degrees of freedom must be 0 or 1")
                return None
            corr_matrix[j,i] = corr_matrix[i,j]
    return corr_matrix

def calc_harmonic_mean(distribution_function, samples, volume):
    """
    assuming that each row of samples represent a single solution
    """
    Nsamples, _ = samples.shape

    distribution_values = np.apply_along_axis(distribution_function, arr = samples, axis = 1)
    estimator = 1 / (Nsamples * volume) * np.sum(1 / distribution_values)

    return estimator

def calc_rel_entropy(parameters, bin_width, boundaries):
    m_samples = parameters.astype('float')
    Nsamples, Nparameters = m_samples.shape

    volume = np.prod(np.array([boundaries[:, 1] - boundaries[:, 0]], dtype = 'float'))

    norm1 = calc_harmonic_mean(lambda x: posterior_prop_density(x, grav_data, x_measurements), m_samples, volume)
    norm2 = calc_harmonic_mean(lambda x: posterior_prop_density(x, grav_data[np.arange(0,N_parameters,2)], x_measurements[np.arange(0,N_parameters,2)]), m_samples, volume)
    #print(norm1, norm2)

    rel_entropy = norm1 * bin_width ** Nparameters / np.log(2)



    Nbins = ((boundaries[:,1] - boundaries[:,0]) / bin_width).astype('int')
    counts = int(Nparameters/2) * [None]

  

    for i in range(int(Nparameters/2)):

        count, edges = np.histogram(m_samples[:, 2 * i + 1], bins = Nbins[2 * i + 1], \
            range = (boundaries[2 * i + 1, 0], boundaries[2 * i + 1, 1]))

        x_vals = 0.5 * (edges[1:] + edges[:-1])
      
        X_matrix = np.zeros([len(x_vals), Nparameters])
        X_matrix[:,2 * i + 1] = x_vals
      
        model_vals = np.apply_along_axis(lambda x: model_1d(x, x_measurements[2*i+1], tuning = 1e-5), arr = X_matrix, axis = 1)
    
        print("rel ent ", rel_entropy)
    
        #
        rel_entropy = rel_entropy * np.sum(count * (np.log(norm1 / norm2) -  (grav_data[2 * i + 1] - model_vals) ** 2 / data_cov_matrix[2*i+1, 2*i+1])) / np.sum(count)
        counts[i] = count
    return rel_entropy




    
### MAIN ------------------------------------------------------------------------------------------------------------------------------------

##DO: (at skelne mellem nice og need to do)

# calc Keibler Lubach distance

# autocorrelation / N_eff estimation

##FIGURE OUT / ASK:

# what to do with uestion 10
# misfit burde have gennemsnitsværdi N/2 = 6. pero no. why?

## Hvis vi bruger posterior i p_acc, så skal vi bruge uniform jump. All is well! 
 # Hvis vi bruger likelihood i p_acc, så skal vi trække jumps fra prior. Hvordan præcist?
 #  All dimensioner eller 1 ad gangen?[hvad med curse of dim?] en ting er at simulere hop ift prior, 
 # ..men hvordan tagre vi højde for m_old i det? [er m_new uaf. af m_old??
 # Hvad hvis vores p_acc bliver for lille når vi trækker fra prior? How to proceed? 
 
# Set plotting style
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster
mpl.rcParams['lines.linewidth'] = 2 
mpl.rcParams['axes.titlesize'] =  18
mpl.rcParams['axes.labelsize'] =  18
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['figure.figsize'] = (6,6)
mpl.rcParams['axes.prop_cycle'] = cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'lightblue', 'olivedrab', 'black'])


plot_misfit, plot_marginal, plot_joint = True, True, True


def main():
    # Initialize
    N_samples = 50_000
    N_burn_in = 5_000
    iterations_between_samples = 1
    height_boundary = [0, 1400] # meters

    Nbins = 140
    bin_width = (height_boundary[1] - height_boundary[0]) / Nbins
    boundaries = height_boundary * (np.ones(N_parameters)[:, NA])

    # Load or calculate sample solutions and log likelihood values
    if load_parameters:
        m_samples = np.loadtxt(file_name_parameters)
        likelihood_list = np.loadtxt(file_name_likelihood)
    else:
        m_samples, likelihood_list = markov_chain_MC(grav_data, x_measurements, N_samples, N_burn_in, boundaries, scaling_factor = 0.05, random_start_guess = False, save_samples = True)

    # calculate cov. matrix
    corr_matrix = calc_corr_matrix(m_samples[np.arange(0,N_samples,iterations_between_samples),:])
    #print("corr. matrix: \n", corr_matrix)
    # Find mean
    m_mean = np.mean(m_samples[np.arange(0,N_samples, iterations_between_samples),:], axis = 0)

    ## Calculate Rel. entropy

    volume = np.prod(np.array([boundaries[:, 1] - boundaries[:, 0]], dtype = 'float'))

    norm1 = calc_harmonic_mean(lambda x: posterior_prop_density(x, grav_data, x_measurements), m_samples, volume)
    norm2 = calc_harmonic_mean(lambda x: posterior_prop_density(x, grav_data[np.arange(0,N_parameters,2)],\
                                                                 x_measurements[np.arange(0,N_parameters,2)]), m_samples, volume)
   


    #CALC1

    rel_list = []
    #it_list = np.arange(50_000,850_000,50_000)
    it_list = np.arange(1,11)
    if 0:
        
        entropy_boundaries = np.array([[1,1001], [1,201], [1,401], [100,1300], [200,1400], [200,1400]])
        entropy_boundaries = np.concatenate((entropy_boundaries, np.flip(entropy_boundaries, axis = 0)))
        for i in it_list:
            rel_entropy = calc_rel_entropy(m_samples[np.arange(0,N_samples,i), :], bin_width = 50, boundaries = entropy_boundaries)
            rel_list.append(rel_entropy)
    

        rel_list = np.array([rel_list]).flatten()
        np.savetxt('rel5.txt', rel_list) #rel3 is lin with const. norm. rel2 is log2 with variying. rel4 lin varying
        rel_list = np.loadtxt('rel5.txt')
        print(rel_list)

        samples = N_samples / it_list
        plt.plot(samples, rel_list, '-x')
        plt.xlabel('No. of sample points')
        plt.ylabel('Rel. entropy (bits)')
        plt.title('Information gain by including all data points')
    # plt.xticks(ticks = np.arange(0,900_000,100_000), labels = ['0', '1e5', '2e5', '3e5', '4e5', '5e5', '6e5', '7e5', '8e5'])
        #plt.yticks(ticks = np.arange(0,int(3.5e7),int(0.5e7)), labels = ['0', '0.5e7', '1e7', '1.5e7', '2e7', '2.5e7', '3e7'])
        plt.show()
        print(rel_entropy)

    it_list = np.arange(6,11)
    rel_list = []
    if 0:
        
        volume = np.prod(np.array([boundaries[:, 1] - boundaries[:, 0]], dtype = 'float'))

       # norm1 = calc_harmonic_mean(lambda x: posterior_prop_density(x, grav_data, x_measurements), m_samples, volume)
       # norm2 = calc_harmonic_mean(lambda x: posterior_prop_density(x, grav_data[np.arange(0,N_parameters,2)], x_measurements[np.arange(0,N_parameters,2)]), m_samples, volume)
       # print(norm1, norm2)

        entropy_boundaries = np.array([[1,1001], [1,201], [1,401], [100,1300], [200,1400], [200,1400]])
        entropy_boundaries = np.concatenate((entropy_boundaries, np.flip(entropy_boundaries, axis = 0)))
        for i in it_list:
            rel_entropy = calc_rel_entropy(m_samples[np.arange(0,N_samples,i), :], bin_width = 50,\
                 boundaries = entropy_boundaries, norm1 = 1, norm2 = 1)
            rel_list.append(rel_entropy)
    
        rel_list = np.array([rel_list]).flatten()
        np.savetxt('rel5.txt', rel_list) #rel3 is lin with const. norm. rel2 is log2 with variying. rel4 lin varying
        #rel_list = np.loadtxt('rel5.txt')
        #print(rel_list)

        samples = N_samples / it_list
        plt.plot(samples, rel_list, '-x')
        plt.xlabel('No. of sample points')
        plt.ylabel('Rel. entropy (bits)')
        plt.title('Information gain by including all data points')
    # plt.xticks(ticks = np.arange(0,900_000,100_000), labels = ['0', '1e5', '2e5', '3e5', '4e5', '5e5', '6e5', '7e5', '8e5'])
        #plt.yticks(ticks = np.arange(0,int(3.5e7),int(0.5e7)), labels = ['0', '0.5e7', '1e7', '1.5e7', '2e7', '2.5e7', '3e7'])
        plt.show()
       # print(rel_entropy)


    # Plot correlations for the first 6 regions
    figc, axc = plt.subplots()
    for i in range(int(N_parameters/2 + 1)):
        correlations = corr_matrix[:int(N_parameters/2)+1, i]
        axc.plot(np.arange(int(N_parameters/2)+1), correlations, '.-', label = f'Correlations for region {i}')

        axc.set_ylim((-0.6,2))
        axc.set(xlabel = 'Region number', ylabel = '(Pearson) corr. coefficient', title = 'Corr. between the regions 0-6')
    
    figc.tight_layout()
    axc.legend(ncol = 2, loc = 'upper right', fontsize = 12)
    plt.show()
    
    ## Plot misfit and likelihood function
    if plot_misfit:
        # Set number of values to plot
        cutoff = int(N_samples/10) + N_burn_in
        iterations = np.arange(0,len(likelihood_list[0:cutoff]), 1)
        #misfit_vals = - np.apply_along_axis(log_likelihood_function, arr = m_samples[0:cutoff,:], axis = 1)
        misfit_vals = - np.log(likelihood_list[:cutoff])

        plt.figure(figsize = (4,4))
        plt.plot(iterations, misfit_vals, 'r.', markersize = 0.5)
        plt.xlabel('Iteration number', fontsize = 18)
        plt.ylim(ymax = 40)
        plt.ylabel('- ln(likelihood function)', fontsize = 18)
        plt.title(f'Misfit for first {cutoff} iterations', fontsize = 18)
      #  plt.plot([iterations[0], iterations[-1]], [0.5,0.5], 'k--', label = 'Distance = 1 * norm(noise)')
       # plt.plot([iterations[0], iterations[-1]], [2,2], 'k--', label = 'Distance = 2 * norm(noise)')
       # plt.plot([iterations[0], iterations[-1]], [8,8], 'k--', label = 'Distance = 3 * norm(noise)')
        plt.plot([iterations[0], iterations[-1]], [misfit_vals.mean(),misfit_vals.mean()], 'b-', label = 'Mean value')
        plt.legend(loc = 'best', fontsize = 14)

        plt.figure()
        plt.plot(iterations, likelihood_list[:cutoff],'r.-', markersize = 0.5)
        plt.xlabel('Iteration', fontsize = 18)
        plt.ylabel('Likelihood function', fontsize = 18)
        plt.title('Value of likelihood function for different iterations', fontsize = 18)
        plt.show()

    if plot_marginal:
        ## Plot marginal histogram of all layers

        # Intialize vector for storing most likely values
        m_most_probable_marginal = np.empty(N_parameters)


        for i in range(2):
            fig = plt.figure(figsize = (13, 11))
            gs = GridSpec(2,3) #2 rows, 3 columns
            ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2]), fig.add_subplot(gs[1,0]), \
                fig.add_subplot(gs[1,1]), fig.add_subplot(gs[1,2])] #, fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1]), \
                    #fig.add_subplot(gs[2,2]), fig.add_subplot(gs[3,0]), fig.add_subplot(gs[3,1]), fig.add_subplot(gs[3,2])]
            
        #  ax = ax.flatten()
            fig.suptitle(f'Sampled height distributions', fontsize = 24)

            #fig.text(0.5, 0.04, 'Height', fontsize = 18, ha='center', va='center')
            #fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize = 18)
            fig.supxlabel('Height', fontsize = 20)
            fig.supylabel('Frequency', fontsize = 20)
            #fig.text(0.02, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize = 18)

            for j in range(int(N_parameters/2)):

                counts, edges, _ = ax[j].hist(m_samples[np.arange(0,N_samples, iterations_between_samples),j + i * int(N_parameters/2)] , \
                    bins = Nbins, range = height_boundary, histtype = 'step', linewidth = 2, density = False, label = f'L{j + i * int(N_parameters/2)}')
            
                ax[j].set_ylim((0,1.1 * np.max(counts)))

                # Extract index with highest counts
                indices = np.argmax(counts)
                # Find coordinate of bin with highest count
                x_indices = height_boundary[0] + indices * bin_width + 0.5 * bin_width
            
                # Set moost probable value equal to coordinate of bin with highest count
                m_most_probable_marginal[j + i * int(N_parameters/2)] = x_indices
        
                ax[j].plot([x_indices, x_indices], [0,counts.max()], 'k-', lw = 2, label = 'Mode')
                ax[j].plot([m_mean[j +  i * int(N_parameters/2)], m_mean[j +  i * int(N_parameters/2)]], [0,counts.max()],'-', lw = 2,label = 'Mean')

                ax[j].legend(loc = 'best', fontsize = 15)
            
        plt.tight_layout()
        plt.show()

    if plot_joint:
        # Initalize vector to store most probable joint values
        m_most_probable_joint = np.empty(N_parameters)

        for i in range (int(N_parameters/2)):
            plt.figure()
            it_scatter = 50
            plt.scatter(m_samples[np.arange(0, N_samples, it_scatter),2*i], \
                m_samples[np.arange(0, N_samples, it_scatter),2*i + 1], alpha = 0.2, s = 15)
            plt.xlim(xmin = 0)
            plt.ylim(ymin = 0)
            plt.ylabel(ylabel = f'Height of region {2*i+1} (m)', fontsize = 18)
            plt.xlabel(xlabel = f'Height of region {2*i} (m)', fontsize = 18)
            plt.title(label = f'Height scatterplot of adjacent regions', fontsize = 18)
            plt.plot([], [], ' ', label=f"(Pearson) correlation = {corr_matrix[2*i,2*i+1]:.2f}")

            plt.text
            plt.legend(loc = 'best', fontsize = 16)

            plt.figure()
            plt.title(f'{i}')
            bounds = ((height_boundary[0], height_boundary[1]),(height_boundary[0],height_boundary[1]))
            bins = (100, 100)
            bin_width = (height_boundary[1] - height_boundary[0]) / bins[0]

            hist_2d,_,_,im =plt.hist2d(m_samples[np.arange(0,N_samples,iterations_between_samples),2*i], \
                m_samples[np.arange(0,N_samples,iterations_between_samples),2*i + 1], cmap = 'Blues', \
                    range = bounds, bins = bins, cmin = 0)

            plt.title(f'Sampled eight-height distribution for regions {2*i} and {2*i+1}', fontsize = 18)
            plt.xlabel(f'Region {2*i} height (m) ', fontsize = 18)
            plt.ylabel(f'Region {2*i+1} height (m) ', fontsize = 18)

            hist_2d[np.isnan(hist_2d)] = 0
            # Extract indices and coordinate of bin with highest count
            indices = np.argwhere(hist_2d == np.max(hist_2d))
            x_indices = height_boundary[0] + indices[0,0] * bin_width + 0.5 * bin_width
            y_indices = height_boundary[0] + indices[0,1] * bin_width + 0.5 * bin_width

            plt.plot(x_indices,y_indices, 'rs', label = 'Bin with highest counts')
            plt.colorbar(im)
            plt.legend(loc = 'best', fontsize = 16)
            # Set most probable values equal to coordinates of bin with highest count
            m_most_probable_joint[2 * i] = x_indices
            m_most_probable_joint[2 * i + 1] = y_indices


        
            #print(corr_matrix)
            plt.show(block = False)

   

    if plot_marginal and plot_joint:    

        print("joint ", m_most_probable_joint)
        print("marg mean ", m_mean)
        print("marg mode ", m_most_probable_marginal)

        data_reconstructed = model(m_most_probable_joint, x_measurements)
        residuals =  data_reconstructed - grav_data

        data_reconstructed_marginal = model(m_most_probable_marginal, x_measurements) 
        residuals_marginal = data_reconstructed_marginal - grav_data 
        data_reconstructed_prior = model(parameter_guess, x_measurements)
        data_reconstructed_prior = model(parameter_guess, x_measurements)

        data_reconstructed_mean = model(m_mean, x_measurements)
        residuals_mean = data_reconstructed_mean - grav_data

   
        
        data_list = [data_reconstructed, data_reconstructed_marginal, data_reconstructed_mean]
        name_list = ['Joint data (mode)', 'Marginal data (mode)', 'Marginal data (mean)']
        p_list = len(data_list) * [None]
        for i, data in enumerate(data_list):
            chi2 = np.sum((data - grav_data) ** 2 / np.diag(data_cov_matrix))
            Ndof = len(grav_data)
            p_list[i] = stats.chi2.sf(chi2, Ndof)
            print(f'For {name_list[i]}:')
            print("Chi2, Ndof, P: ", chi2, Ndof, p_list[i])



        plt.figure()
       
        plt.plot(np.arange(len(grav_data)), 1e5 * data_reconstructed, '.-', \
            color = 'crimson',label = f'Recon. joint data (mode): P (chi2) = {p_list[0]:.2f}')
        plt.plot(np.arange(len(grav_data)), 1e5 * data_reconstructed_marginal, '.-', \
            color = 'plum', label = f'Recon. marg. data (mode): P(chi2) = {p_list[1]:.2f}')
        plt.plot(np.arange(len(grav_data)), 1e5 * data_reconstructed_mean, '.-', \
            color = 'coral', label = f'Recon. marg. data (mean): P (chi2) = {p_list[2]:.2f}')

        plt.plot(np.arange(len(grav_data)), 1e5 * data_reconstructed_prior, '.-', \
                label = f'Recon. data from prior estimate ')
        plt.errorbar(np.arange(len(grav_data)), 1e5 * grav_data, 1e5 * np.sqrt(np.diag(data_cov_matrix)), \
            fmt = '.', label = 'Observed data', ecolor='k', elinewidth=1, capsize=2, capthick=1,)
        plt.legend(fontsize = 14)
        plt.title(f'Observed and reconstructed data', fontsize = 18)
        plt.xlabel('Data point number', fontsize = 18)
        plt.ylabel('Gravity anomaly (mGal)', fontsize = 18)
        plt.xticks(np.arange(0,13,1))

        # Plot residuals
        plt.figure()

        plt.plot(np.arange(len(grav_data)), residuals/np.sqrt(np.diag(data_cov_matrix)),'.-', \
            color = 'crimson', label = 'Joint residuals (mode)')
        plt.plot(np.arange(len(grav_data)), residuals_marginal/np.sqrt(np.diag(data_cov_matrix)),'.-', \
            color = 'plum',label = 'Marg. residuals (mode)')
        plt.plot(np.arange(len(grav_data)), residuals_mean/np.sqrt(np.diag(data_cov_matrix)), '.-',\
            color = 'coral', label = 'Marg. residuals (mean)')
        plt.title('Observed-Reconstructed data residuals', fontsize = 18)
        plt.xlabel('Data point number', fontsize = 18)
        plt.ylabel('Residual (units of noise)', fontsize = 18)
        plt.legend(loc = 'upper right',fontsize = 15)
        plt.xticks(np.arange(0,13,1))
        
        plt.show()
     

if __name__ == '__main__':
    main()
