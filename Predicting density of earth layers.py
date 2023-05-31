### SETUP --------------------------------------------------------------------------------------------------------------------

## Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants



### FUNCTIONS ----------------------------------------------------------------------------------------------------------------

def tikhonov_solution (A, b, epsilon):
    _ , n = A.shape
    return np.linalg.inv (A.T @ A + epsilon ** 2 * np.eye(n)) @ A.T @ b 

def object_function (epsilon, tikhonov_solution, A, b, noise):
    """
    """
    return np.linalg.norm (A @ tikhonov_solution(A, b, epsilon) - b) - np.linalg.norm(noise)


### MAIN ---------------------------------------------------------------------------------------------------------------------

## Calculate uncertaintyies on density
## Check stability of solutions against epsilon

def main():
    # Load data
    x_horizontal, grav_data = np.loadtxt('C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\Inverse Problems\\gravdata.txt'
                                , unpack = 'True')

    # Record data points and number of considered slabs
    N_measurements = len(x_horizontal)
    N_predictions = 100

    # Uncertainty on measurements (assuming that horizontal distance measurements are exact)
    error = 1.0e-9

    start_depth = 0
    bottom_depth = 99
    # Decide which depths to consider (marking the depth of the top of the layer)
    x_vertival = np.linspace(start_depth,bottom_depth,N_predictions)

    # Record gravitational constant
    grav_const = constants.G




    # Build transformation matrix G
    G = np.empty([N_measurements, N_predictions])

    G = grav_const * np.log ( (x_horizontal[:, np.newaxis]**2 + (x_vertival[np.newaxis, :] + 1) ** 2 )
                    / (x_horizontal[:, np.newaxis]**2 + x_vertival[np.newaxis,: ]  ** 2 ))

        ## same as
    """
   for i in range(N_measurements):
        G[i,:] = grav_const * np.log ( (x_horizontal[i]**2 + (x_vertival[:] + 1) ** 2 )
                    / (x_horizontal[i]**2 + x_vertival[:]  ** 2 ))
    """

    # Construct noise matrix (diagonal matrix whose entry Noise_ii = uncertainty(measurement_i))
    noise = np.ones(N_measurements) * error

    # Calculate anfd plot the object function in the range 1e-10 to 1e-1
    range_min, range_max, points = -13, -9, 10000 
    # essentially makes uniform linear values in 10*eps space from 10**range_min to 10**range_max
    eps_range = np.logspace(range_min, range_max, points)
 

    object_vectorized = np.vectorize(lambda epsilon: object_function(epsilon, tikhonov_solution, G, grav_data, noise))
    target_values = np.abs(object_vectorized(eps_range))

    plt.style.use('seaborn')
    plt.plot(np.log10(eps_range), np.log10(target_values), 'r.', markersize=3)
    plt.title('Value of objective function against epsilon')
    plt.xlim([range_min,range_max])
    plt.ylim([-14,-6])
    plt.show()

    ## The approximate epsilon minimum is about 1e-11. We do a fine grid search around this point
    min_app = 1e-11
    eps_range2 = np.linspace(0.62*min_app, 0.64*min_app , 1000)

    plt.style.use('seaborn-paper')
    plt.plot(eps_range2, np.log10(np.abs(object_vectorized(eps_range2))), 'b.', markersize = 3)
    plt.title('Value of objective function against epsilon')
    plt.xlim([6.2e-12, 6.4e-12])
    plt.ylim([-13,-9])
    plt.show()
    # We read of the epsilon that minimizes the object function to be
    eps_min = 6.314e-12 
    # Examine the stability of the density fluctuations under perturbations in epsilon
    eps_mins = [6.3e-12, eps_min, 6.325e-12]


    plt.figure()
    plt.errorbar(np.arange(18),grav_data, noise, label = 'Observed data')
    data_reconstructed = G @ tikhonov_solution(G, grav_data, eps_min)
    plt.plot(np.arange(18), data_reconstructed, label = 'Reconstructed data')
    plt.legend()
    plt.show()
    # Ploy density fluctuations
    plt.style.use('seaborn-pastel')
    fig, ax = plt.subplots(figsize = (10,5))
    colors = ['r.-', 'g.-', 'b.-', 'k.-', 'm.-']
    for eps, colors in zip(eps_mins, colors):
        density = tikhonov_solution(G, grav_data, eps)
        ax.plot(x_vertival, density, colors, label = f'epsilon = {eps}')

        # Calculate data correlation matrix to find uncertainty on density
        Ginv = tikhonov_solution(G, np.eye(N_measurements), eps)
        Cd = error ** 2 * np.eye (N_measurements)
        Cm = Ginv @ Cd @ Ginv.T

       # ax.errorbar(x_vertival, density, yerr = np.abs(tikhonov_solution(G, noise, eps)))

    ax.legend()
    ax.set_title('Modeled density fluctuations for different depths')
    ax.set(xlabel = 'depth (km)', ylabel = 'Density fluctuation (kg/m^3)')
    plt.show()

    
if __name__ == '__main__':
    main()