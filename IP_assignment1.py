# Author: Simon Guldager Andersen
# Date (latest update): 11/12-2022

### SETUP -----------------------------------------------------------------------------------------------------------------------------------

## Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable



### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

def tikhonov_solution(A, b, epsilon):
    _, n = A.shape
    Ginv = np.linalg.inv(A.T @ A + epsilon ** 2 * np.eye(n)) @ A.T
    return Ginv @ b

def misfit_function(A, b, noise, epsilon):
    m_est = tikhonov_solution(A, b, epsilon)
    return np.abs((np.linalg.norm(b - A @ m_est) - np.linalg.norm(noise)))

def resolution_matrix(A, epsilon):
    _, n = A.shape
    return np.linalg.inv(A.T @ A + epsilon ** 2 * np.eye(n)) @ A.T @ A

### MAIN ------------------------------------------------------------------------------------------------------------------------------------

## Set parameters:
plot_misfit_function, plot_solution, plot_resolution = True, True, True
plt.style.use('seaborn-darkgrid')


def main():
    ### PART 1: Building the system

    ## Organize data points as (t(left-to-right)_0, ...., t(right-to-left)_0,....)
    time_true = np.empty(20)
    dimensions = [11, 13]
    slowness_deviation = 1/5.200 - 1/5.000
    slowness_data = np.zeros(dimensions)
    ## Initialize rectangular region with non-vanishing slowness deviation (in units of slowness_deviation)
    slowness_data[1:9, 4:7] = 1
    slowness_data = slowness_data.reshape(-1)


    N_time = time_true.size
    N_squares = slowness_data.size
   

        # Build G
    G = np.zeros([N_time, N_squares])
    for i in range(N_time):
        Gi = np.zeros(dimensions)
        if i < 10:   
            ray = np.rot90(np.eye(i+2), 1)
            Gi[0:i+2, 0:i+2] = ray
        else:
            ray = np.eye(i - 8)
            Gi[:(i-8), (dimensions[1] - (i - 8)):] = ray
        Gi = Gi.reshape(-1)
        G[i] = np. sqrt(2) * Gi 

    # Solve forward problem to generate time data
    time_true = G @ slowness_data
    # Print results
    print("Time anomalies obtained by solving the forward problem: ")
    print(time_true)

    # Define Gaussian noise vector
    noise = stats.norm.rvs(size = N_time)

    # Normalize
    noise = noise / np.linalg.norm(noise)
    # Rescale to s.t. norm = 1/18 * norm(t_true)
    noise = 1/18 * np.linalg.norm(time_true) * noise / 100

    # construct time data to include noise
    time_data = time_true + noise


    ### PART 2: 
    # Find a Thikonov solution. The goal is to find an epsilon that minimizes the difference between the residual
    # of the least squares and the norm of the noise ||b - A * m_tikhonov || - ||noise||

    # Examine values from 1e-16 to 1e16. We find the epsilon that minimizes the misfit to be in [0.001,1000].
    exp_bounds = (-3, 3)
    points = 1000
    eps_range = np.logspace(exp_bounds[0],exp_bounds[1], points)

    # Vectorize misfit function
    misfit_vectorized = np.vectorize(lambda epsilon: misfit_function(G, time_data, noise, epsilon))
    misfit_values = misfit_vectorized(eps_range)

    # Find minimum and extract it
    min_index = np.argmin(misfit_values).flatten()
    eps_min = eps_range[min_index]

    # Calculate tikhonov solution
    m_est = tikhonov_solution(G, time_data, eps_min)

    print("The epsilon that minimizes the misfit function is found to be:", float(eps_min))
    print("The value of the misfit function at for this epsilon is: ", float(misfit_values[min_index]))

    if plot_misfit_function:
       
        fig, ax = plt.subplots(figsize = (5,5))

        ax.plot(np.log10(eps_range), np.log10(misfit_values), 'r.-')
        ax.set(xlim = (exp_bounds[0]-0.01, exp_bounds[1]+0.01), xlabel = 'log10(epsilon)', ylabel = 'log10(misfit function)', title = 'Misfit function for different epsilons')

        fig.tight_layout()
        plt.show(block = False)

    # Construct images of the orignal and estimated / reconstructed solutions
    if plot_solution:
        fig2, ax2 = plt.subplots(ncols = 2, figsize = (10,5))
        ax2 = ax2.flatten()
        im_list = [slowness_data, m_est]
        names = ['Original solution', 'Reconstructed solution']
        for i, im in enumerate(im_list):
            divider = make_axes_locatable(ax2[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax2[i].imshow(im.reshape(dimensions), cmap = 'Blues', extent = (0, 13, 11, 0))
            fig2.colorbar(im, cax = cax, orientation = 'vertical')
            ax2[i].set(xlabel = 'x (km)', ylabel = 'z (km)', title = names[i])
            ax2[i].xaxis.tick_top()
        plt.show(block = False)



   ### PART 3: Examine the resolution / smearing of solutions, in which just a single 2x2 square region has a 
   # non-trival slownewss anomaly.

    m_delta = np.zeros(dimensions)

    # Choose a square region such that 1 square is well-resolved (2 rays passing through), 2 squares are semi-resolved (1 ray)
    # and one square is unresolved (0 rays)
    m_delta[4:6,5:7] = 1
    m_delta = m_delta.reshape(-1)

    # Construct the corresponding time data
    time_delta = G @ m_delta

    # Construct the noise vector following the same recipe as before - and add it to the time data
    noise_delta = stats.norm.rvs(size = N_time)
    noise_delta = noise_delta / np.linalg.norm(noise_delta)
    noise_delta = 1/18 * np.linalg.norm(time_delta)
    time_delta += noise_delta 

    # Construct Tikhonov solution
    m_est_delta = tikhonov_solution(G, time_delta, eps_min)


    if plot_resolution:
        fig3, ax3 = plt.subplots(ncols = 2, figsize = (10,5))
        ax3 = ax3.flatten()
        im_list = [m_delta, m_est_delta]
        names = ['Original solution', 'Reconstructed solution']
        for i, im in enumerate(im_list):
            divider = make_axes_locatable(ax3[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax3[i].imshow(im.reshape(dimensions), cmap = 'Blues', extent = (0, 13, 11, 0))
            fig3.colorbar(im, cax = cax, orientation = 'vertical')
            ax3[i].set(xlabel = 'x (km)', ylabel = 'z (km)', title = names[i])
            ax3[i].xaxis.tick_top()
        plt.show()


if __name__ == '__main__':
    main()
