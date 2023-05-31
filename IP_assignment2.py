# Author: Simon Guldager Andersen
# Date (latest update): 5/1-2023

### SETUP -----------------------------------------------------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import constants


## Change directory to current one
os.chdir('Inverse Problems')

## Define global parameters
# Load data
x_horizontal, grav_data = np.loadtxt('data\\gravdata.txt', unpack = 'True')
x_horizontal *= 1000 # convert to meters

# Set whether to read or write model parameters and log-likehood values from file
load_parameters = False
file_name_parameters = 'data\\MCMC_samples0.txt' #MCMC_SAMPLES er 500_000 dyn. uniform perturb. gode histogrammer, MCMC_SAMPLES0 er 800_000 unif. perturb
file_name_log_likelihood = 'data\\LL_values0.txt' #MCMC_SAMPLES1 500_000 er statisk uniform perturb. 500_000, MCMC_SAMPLE0 er 800_00 dyn uniform
                                            #MCMC_SAMPLESA4 150_000 statisk med lille noise. humlen er uniform, lille noise
### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

def model (parameters):
    cols = parameters.size

    density_fluctuations = parameters[0:int(cols/2)]
    heights = parameters[int(cols/2):]
    x_vertical = np.r_['0', 0, np.cumsum(heights)]

    G = constants.gravitational_constant * np.log ( (x_horizontal[:, np.newaxis] ** 2 + (x_vertical[np.newaxis, 1:]) ** 2 )
                    / (x_horizontal[:, np.newaxis] ** 2 + x_vertical[np.newaxis,:-1]  ** 2 ))
    return G @ density_fluctuations

def likelihood (parameters, noise):
    
    return np.exp(-0.5 * np.linalg.norm((grav_data - model(parameters) ) ** 2 / noise ** 2)  )
    #return np.exp(- np.linalg.norm(grav_data - model(parameters)) ** 2 / (2 * np.linalg.norm(noise) ** 2))

def misfit (parameters, noise):
    return 0.5 * np.linalg.norm((grav_data - model(parameters) ) ** 2 / noise ** 2)
    #return np.linalg.norm(grav_data - model(parameters)) ** 2 / (2 * np.linalg.norm(noise) ** 2)

def acceptance_probability(parameters_old, parameters_new, noise):
    misfit_new = misfit(parameters_new, noise)
    misfit_old = misfit(parameters_old, noise)
    
    if misfit_new <= misfit_old:
        return 1
    elif misfit_new > misfit_old:
        return np.exp(-(misfit_new - misfit_old))

def markov_chain_MC(perturbation_scale, N_samples, start_sampling, boundaries, noise, \
    gaussian_perturbation = False, save_samples = False):

    # Initialize
    N_parameters, _ = boundaries.shape
    m_samples = np.empty([N_samples, N_parameters])
    likelihood_list = []
    N_solutions = 0
    iterations = 0

    # Define the scaling vector as interval length of each dimension times perturbation_scale. This scaling decides the size of the perturbation for each dimension
    scaling = perturbation_scale #* (boundaries[:,1] - boundaries[:,0])


    # Define factor to multiply/divide with scaling to ensure acceptable acceptance rate
    scaling_amplification = 1.0

    # Initalize array to keep whether the last 20 iterations were accepted or rejected
    N_acceptance_history = 20
    acceptance_history = 0.5 * np.ones([N_parameters, N_acceptance_history])
    acceptance_sum = np.zeros(N_parameters)

    ## Initialize random parameter
    # generate N_parameters random numbers
    rvs = np.random
    r = rvs.rand(N_parameters)
    # generate random solution by scaling r with parameter boundaries
    m_old = boundaries[:,0] + r * (boundaries[:,1] - boundaries[:,0])
    
    while N_solutions < N_samples:
        iterations += 1
        if iterations > start_sampling and (iterations - start_sampling)  % (N_samples/100) == 0:
            print('Progress: ', int(100 * ((iterations - start_sampling) / N_samples )), " %")

        # generate random parameter index to perturb
        rand_index = int(np.floor(N_parameters * rvs.rand(1)))
      
        if gaussian_perturbation:
            # generate Gaussian perturbation with mean = 0 and  std = perturbation_scale
            perturbation = np.random.normal(loc = 0, scale = scaling[rand_index])
        else:
            # otherwise generate uniform perturbation
            perturbation = (rvs.rand(1) - 0.5) * scaling[rand_index]

        # add perturbation with periodic boundary conditions
        m_new = m_old.astype('float')
        m_new[rand_index] =  ( m_new[rand_index] + perturbation )
   
        # enforce periodic boundary conditions to ensure that new solution is contained within boundaries
        if m_new[rand_index] < boundaries[rand_index, 0]:
            m_new[rand_index] = min(boundaries[rand_index, 1], boundaries[rand_index, 1] + (m_new[rand_index] - boundaries[rand_index, 0])) 

        elif m_new[rand_index] > boundaries[rand_index, 1]:
            m_new[rand_index] = max(boundaries[rand_index,0], boundaries[rand_index, 0] + (m_new[rand_index] - boundaries[rand_index, 1]))

        # calculate acceptance probability
        if rvs.rand(1) < acceptance_probability(m_old, m_new, noise):
            m_old = m_new.astype('float')
 
            acceptance_history[rand_index, iterations % N_acceptance_history] = 1
            acceptance_sum[rand_index] = acceptance_sum[rand_index] + 1
        else:
            acceptance_history[rand_index, iterations % N_acceptance_history] = 0
        
 
        # change size of perturbation to ensure an acceptance rate near 50%
        if (acceptance_history[rand_index].mean() < 0.35) & (scaling[rand_index] > 0.001 * (boundaries[rand_index,1] - boundaries[rand_index,0])):                
            scaling[rand_index] = scaling[rand_index ] / scaling_amplification
        elif (acceptance_history[rand_index].mean() > 0.65)  & (scaling[rand_index] < 0.25 * (boundaries[rand_index,1] - boundaries[rand_index,0])):
            scaling[rand_index] = scaling[rand_index] * scaling_amplification
        if iterations >= start_sampling:
            #if likelihood(m_old, noise) > 0:
            m_samples[N_solutions, :] = m_old
            N_solutions += 1
            likelihood_list.append(likelihood(m_old, noise))
  
    if save_samples:
        np.savetxt(file_name_parameters, m_samples)
        np.savetxt(file_name_log_likelihood, likelihood_list)

    print("Average acceptance rate: ", acceptance_sum / (iterations / N_parameters))
    
    return m_samples, likelihood_list


### MAIN ------------------------------------------------------------------------------------------------------------------------------------

# Set plotting style
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster
# Set what to plot
plot_misfit = True

def main():
    # Initialize
    N_samples = 500_000
    N_parameters = 8
    start_sampling = 50_000
    height_boundary = [2000, 10000] # meters
    density_boundary = [-2000, 2000] # kg / m^3
    Nbins_density = 120
    Nbins_height = 120
    bin_width_density = (density_boundary[1] - density_boundary[0]) / Nbins_density
    bin_width_height = (height_boundary[1] - height_boundary[0]) / Nbins_height

    noise = np.ones_like(grav_data) 
    noise = 1e-9 * noise / np.linalg.norm(noise)
    
   # noise = 1e-9 * np.ones_like(grav_data)
    #noise[0] =  1e-9
    boundaries = np.block([[density_boundary * (np.ones(int(N_parameters / 2)))[:, np.newaxis]], \
                 [height_boundary * (np.ones(int(N_parameters / 2)))[:, np.newaxis]]])
  
    # Define the perturbation scale of each dimension
    perturbation_scale = 0.015 * (boundaries[:,1] - boundaries[:,0])
    perturbation_scale[:2] *= 0.15
    perturbation_scale[2] *= 0.3
    perturbation_scale[-6:-2] *= 0.7
    perturbation_scale[-2:] *= 1
    perturbation_scale[4] *= 1
    perturbation_scale[-3] *= 1.1
    perturbation_scale[3:5] *= 0.8
    perturbation_scale[3] *= 0.5
    perturbation_scale[6] *= 0.5

    # Load or calculate sample solutions and log likelihood values
    if load_parameters:
        m_samples = np.loadtxt(file_name_parameters)
        likelihood_list = np.loadtxt(file_name_log_likelihood)
    else:
        m_samples, likelihood_list = markov_chain_MC(perturbation_scale = perturbation_scale, N_samples = N_samples, start_sampling = start_sampling, \
        boundaries = boundaries, noise = noise, save_samples = True, gaussian_perturbation=False)

    jump = 10
    likelihood_list = np.array(likelihood_list)[np.arange(0,N_samples, jump)]

    ## Plot misfit and likelihood function
    if plot_misfit:
        # Set number of values to plot
        cutoff = len(likelihood_list)
        iterations = np.arange(0,len(likelihood_list[0:cutoff]), 1)
        misfit_vals = np.apply_along_axis(lambda x: misfit(x,noise), arr = m_samples[:cutoff, :], axis = 1)
 
        plt.figure(figsize = (4,4))
        plt.plot(iterations, misfit_vals, 'r.', markersize = 0.5)
        plt.xlabel('Iteration number', fontsize = 18)
        plt.ylabel('- ln(likelihood function)', fontsize = 18)
        plt.title('- log likelihood function for first 80.000 iterations', fontsize = 18)
        plt.plot([iterations[0], iterations[-1]], [0.5,0.5], 'k--', label = 'Distance = 1 * norm(noise)')
        plt.plot([iterations[0], iterations[-1]], [2,2], 'k--', label = 'Distance = 2 * norm(noise)')
        plt.plot([iterations[0], iterations[-1]], [8,8], 'k--', label = 'Distance = 3 * norm(noise)')
        plt.plot([iterations[0], iterations[-1]], [misfit_vals.mean(),misfit_vals.mean()], 'b-', label = 'Mean value')
        plt.legend(loc = 'best', fontsize = 14)

        plt.figure()
        plt.plot(iterations, likelihood_list[:cutoff],'r.-', markersize = 0.5)
        plt.xlabel('Iteration', fontsize = 18)
        plt.ylabel('Likelihood function', fontsize = 18)
        plt.title('Value of likelihood function for different iterations', fontsize = 18)
        plt.show()


    ## Plot marginal histogram of all layers
    # Set the number of iterations between accepted solutions
    #jump = 1
    noise = 1e-9 * np.ones_like(grav_data)

    name_list = ['Sampled density fluctuation distributions', 'Sampled height distributions']
    xlabel_list = ['Density fluctuation (kg/m^3)', 'Height (m)']
    # Intialize vector for storing most likely values
    m_most_probable_marginal = np.empty(N_parameters)
    for i in range(2):
        if i == 0: # density fluctuations
            bounds = (density_boundary[0],density_boundary[1])
            bins = Nbins_density
        else: # heights
            bounds = (height_boundary[0],height_boundary[1])
            bins = Nbins_height

        fig = plt.figure(figsize = (16,12))
        gs = GridSpec(2,3) #2 rows, 3 columns
        ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2]), fig.add_subplot(gs[1,0]), \
            fig.add_subplot(gs[1,1])]
       
      #  ax = ax.flatten()
        fig.suptitle(f'{name_list[i]}', fontsize = 24)
        
        fig.text(0.5, 0.04, f'{xlabel_list[i]}', fontsize = 18, ha='center', va='center')
        fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize = 18)

        for j in range(int(N_parameters / 2)):
       
            counts, _, _ = ax[j].hist(m_samples[:,j + int(N_parameters / 2 * i)], bins = bins, range = bounds, histtype = 'step', linewidth = 2, density = False, label = f'L{j}')
          #  ax[j].set_ylabel(ylabel ='Frequency', fontsize = 18)
          #  ax[j].set_xlabel(xlabel = f'{xlabel_list[i]}', fontsize = 18)
       
            ax[j].legend(loc = 'best', fontsize = 15)
            ax[j].set_ylim((0,1.2 * np.max(counts)))

            # Extract index with highest counts
            indices = np.argmax(counts)
            # Find coordinate of bin with highest count
            if i == 0:
                x_indices = density_boundary[0] + indices * bin_width_density + 0.5 *bin_width_density
            elif i == 1:
                x_indices = height_boundary[0] + indices * bin_width_height + 0.5 *bin_width_height

            # Set moost probable value equal to coordinate of bin with highest count
            m_most_probable_marginal[j + int(N_parameters / 2 * i)] = x_indices

        plt.tight_layout

    # Initalize vector to store most probable joint values
    m_most_probable_joint = np.empty(N_parameters)

    for i in range (int(N_parameters/2)):
        plt.figure()
        plt.title(f'{i}')
        bounds = ((density_boundary[0],density_boundary[1]),(height_boundary[0],height_boundary[1]))
    

        hist_2d,_,_,im =plt.hist2d(m_samples[np.arange(0,N_samples,jump),i], m_samples[np.arange(0,N_samples,jump),i + int(N_parameters/2)], cmap = 'Blues', range = bounds, bins = (Nbins_density, Nbins_height), cmin = 0)
        plt.title(f'Sampled height-density distribution layer {i}', fontsize = 18)
        plt.xlabel('Denisty fluctuation (kg/m^3)', fontsize = 18)
        plt.ylabel('Layer height (m)', fontsize = 18)

        # Extract indices and coordinate of bin with highest count
        indices = np.argwhere(hist_2d == np.max(hist_2d))
        x_indices = density_boundary[0] + indices[0,0] * bin_width_density + 0.5 *bin_width_density
        y_indices = height_boundary[0] + indices[0,1] * bin_width_height + 0.5 *bin_width_height

        plt.plot(x_indices,y_indices, 'r.', label = 'Bin with highest counts')
        plt.colorbar(im)
        plt.legend(loc = 'best', fontsize = 16)
        # Set most probable values equal to coordinates of bin with highest count
        m_most_probable_joint[i] = x_indices
        m_most_probable_joint[i + int(N_parameters/2)] = y_indices

    # Reconstruct using mean model
    m_mean = np.sum(m_samples,  axis = 0) / N_samples
    data_reconstructed_mean = model(m_mean)

    # Reconstruct solution with using most probable joint and marginal parameters
    data_reconstructed = model(m_most_probable_joint)
    data_reconstructed_marginal = model(m_most_probable_marginal)

    residuals =  data_reconstructed - grav_data
    residuals_marginal = data_reconstructed_marginal - grav_data
    residuals_mean = data_reconstructed_mean - grav_data
    
    plt.figure(figsize = (4,4))
    plt.plot(np.arange(len(grav_data)), data_reconstructed, label = 'Reconstructed joint data')
    plt.plot(np.arange(len(grav_data)), data_reconstructed_mean, label = 'Reconstructed mean data')
    plt.plot(np.arange(len(grav_data)), data_reconstructed_marginal, label = 'Reconstructed marginal data ')
    plt.errorbar(np.arange(len(grav_data)), grav_data, noise, label = 'Observed data', ecolor='k', elinewidth=1, capsize=2, capthick=1,)
    plt.legend(fontsize = 16)
    plt.title(f'Observed and reconstructed data', fontsize = 18)
    plt.xlabel('Data point number', fontsize = 18)
    plt.ylabel('Horizontal gravitational gradient (1/s^2)', fontsize = 18)
    plt.xticks(np.arange(0,19,1))

    # Plot residuals
    plt.figure(figsize =(2,2))
    plt.plot(np.arange(len(grav_data)), residuals/noise, label = '(d_obs - G(m_joint)) / noise ')
    plt.plot(np.arange(len(grav_data)), residuals_mean/noise, label = '(d_obs - G(m_mean)) / noise ')
    plt.plot(np.arange(len(grav_data)), residuals_marginal/noise, label = '(d_obs - G(m_marg)) / noise ')
    plt.title('Observed-Reconstructed data residuals', fontsize = 18)
    plt.xlabel('Data point number', fontsize = 18)
    plt.ylabel('Residual (units of noise)', fontsize = 18)
    plt.legend(loc = 'upper right',fontsize = 16)
    plt.xticks(np.arange(0,19,1))
    
    plt.show()

if __name__ == '__main__':
    main()
