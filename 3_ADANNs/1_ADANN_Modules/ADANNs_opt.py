import os
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from skopt import gp_minimize, forest_minimize
from scipy.stats.qmc import LatinHypercube, Halton, Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from scipy.optimize import brute
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist

sys.path.insert(1, '1_Modules')
from evaluation_utils import evaluate
from utils import save_list_to_file, read_list_from_file , print_to_file_and_console
from training import find_optimal_learningrate_with_documentation

sys.path.insert(1, '1_ADANN_Modules')
from ADANNs import make_adann_creator
from ADANNs_training import adann_training
from ADANNs_grid import CONT_COLORMAP
from documentation_utils import summary


def generate_initial_points(dimensions, n_points=10, sampler_class=Sobol):
    """
    Generate initial points using Latin Hypercube Sampling within the given dimensions.
    Each dimension is a tuple (low, high).
    """
    # Options for sampler_class: LatinHypercube, Halton, Sobol
    sampler = sampler_class(d=len(dimensions))
    sample = sampler.random(n=n_points)
    # Rescale samples to the respective dimensions
    rescaled_sample = np.zeros_like(sample)
    for i in range(sample.shape[1]):
        rescaled_sample[:, i] = dimensions[i][0] + (dimensions[i][1] - dimensions[i][0]) * sample[:, i]
    return rescaled_sample.tolist()


def perturb_point(point, search_space, magnitude=1e-4):
    """
    Slightly perturbs a point within the given search space.
    """
    perturbed_point = point + np.random.uniform(-magnitude, magnitude, size=point.shape)
    # Ensure the perturbed point is still within the search space
    for i in range(len(search_space)):
        low, high = search_space[i]
        perturbed_point[i] = np.clip(perturbed_point[i], low, high)
    return perturbed_point

def find_minimal_from_rbf_brute(points, values, search_space):
    rbf = RBFInterpolator(points, values, kernel='linear')  # You can also try other functions like 'gaussian', 'multiquadric'

    # Objective function: the output of the RBF interpolator
    def objective_function(point):
        return rbf(point.reshape(1, -1))[0]

    # Define the ranges for each parameter
    step_size = 0.01
    ranges = [slice(search_space[i][0], search_space[i][1] + step_size, step_size) for i in range(len(search_space))]

    # Perform the brute force search
    result = brute(objective_function, ranges, full_output=True, finish=None)

    optimal_point = result[0]
    optimal_value = result[1]

    return optimal_point, optimal_value

def generate_evaluation_grid(search_space, n_points_per_dim=100):
    """
    Generates a grid of points for evaluation in N dimensions.
    search_space: List of tuples defining the min and max of each dimension.
    n_points_per_dim: Number of points to evaluate per dimension.
    """
    # Create a linspace for each dimension
    lin_spaces = [np.linspace(dim[0], dim[1], n_points_per_dim) for dim in search_space]
    # Create a meshgrid for all dimensions (unpacking the list of lin_spaces)
    grids = np.meshgrid(*lin_spaces)
    # Reshape and combine to get the grid points in N-dimensions
    grid_points = np.vstack([grid.ravel() for grid in grids]).T
    return grid_points

def sample_minimal_point_from_rbf(points, values, search_space, n_points_per_dim=100, scaling="exponential", plot=True, identifier="", foldername="", show_plot=False):
    """
        points: List of points
        values: List of values at the points
        search_space: List of tuples defining the min and max of each dimension
        n_points_per_dim: Number of points per dim for grid on which next point is sampled
    """

    # Create the RBF interpolator
    # For other options check https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
    rbf = RBFInterpolator(points, values, smoothing=0.01, kernel='linear')

    # Generate list of points corresponding to grid for evaluation
    eval_points = generate_evaluation_grid(search_space, n_points_per_dim=n_points_per_dim)

    # Evaluate the RBF function
    rbf_interpolated_values = rbf(eval_points)

    # Calculate distances to all previously used points
    distances = cdist(eval_points, points)
    penalty_factor = 0.005
    distance_penalties_single = np.min(rbf_interpolated_values) * penalty_factor / (distances + penalty_factor)
    distance_penalties = distance_penalties_single.sum(axis=1)
    penalized_values = rbf_interpolated_values + distance_penalties

    # Convert function values to a pmf
    if scaling == "exponential":
        factor = 100
        min_value = np.min(rbf_interpolated_values)
        shifted_values = np.exp(- factor * (penalized_values - min_value)/np.abs(min_value))
    else:
        scaled_values = rbf_interpolated_values - np.min(penalized_values) + 0.00001 * np.abs(np.min(penalized_values))
        shifted_values = 1.0 / scaled_values

    # Apply the cumulative distance penalty to the shifted values
    pmf = shifted_values / shifted_values.sum()

    # Sample indices according to the PMF
    sampled_index = np.random.choice(len(eval_points), size=None, p=pmf)
    sampled_point = eval_points[sampled_index]
    expected_value = rbf_interpolated_values[sampled_index]

    # Plot results if we are on a 2d grid
    if len(search_space) == 2 and plot:

        rbf_interpol_reshaped = rbf_interpolated_values.reshape(n_points_per_dim, n_points_per_dim)

        # Plot points and values
        plt.scatter([p[0] for p in points], [p[1] for p in points], c=values, cmap='viridis', s=100)
        plt.colorbar()
        plt.title("Points and values")
        if show_plot:
            plt.show()
        else:
            plt.close()

        # Plot RBF interpolation
        plt.imshow(rbf_interpol_reshaped, vmin=rbf_interpol_reshaped.min(), vmax=rbf_interpol_reshaped.max(), origin='lower',
                   extent=[search_space[0][0], search_space[0][1], search_space[1][0], search_space[1][1]], cmap='viridis')
        plt.colorbar()
        plt.plot(sampled_point[0], sampled_point[1], 'kx', markersize=10)
        plt.title("RBF interpolation for error surface")
        plt.savefig(foldername + f"/X_rbf_interpolation_{identifier}.pdf", bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()

        # Plot penalized values
        penalized_values_reshaped = penalized_values.reshape(n_points_per_dim, n_points_per_dim)
        plt.imshow(penalized_values_reshaped, origin='lower', extent=[search_space[0][0], search_space[0][1], search_space[1][0], search_space[1][1]], cmap='viridis')
        plt.colorbar()
        plt.plot(sampled_point[0], sampled_point[1], 'kx', markersize=10)
        plt.title("Penalized values for sampling")
        plt.savefig(foldername + f"/X_penalized_values_{identifier}.pdf", bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()

        # Plot shifted values
        shifted_values_reshaped = shifted_values.reshape(n_points_per_dim, n_points_per_dim)
        plt.imshow(shifted_values_reshaped, origin='lower', extent=[search_space[0][0], search_space[0][1], search_space[1][0], search_space[1][1]], cmap='viridis')
        plt.colorbar()
        plt.plot(sampled_point[0], sampled_point[1], 'kx', markersize=10)
        plt.title("Shifted values for sampling")
        if show_plot:
            plt.show()
        else:
            plt.close()

        # Plot the PMF
        pmf_reshaped = pmf.reshape(n_points_per_dim, n_points_per_dim)
        plt.imshow(pmf_reshaped, origin='lower', extent=[search_space[0][0], search_space[0][1], search_space[1][0], search_space[1][1]], cmap='viridis')
        plt.colorbar()
        plt.plot(sampled_point[0], sampled_point[1], 'kx', markersize=10)
        plt.title("Density for sampling")
        if show_plot:
            plt.show()
        else:
            plt.close()

    return sampled_point, expected_value


def rbf_brute_minimize(
    objective_function,
    search_space,
    n_calls=100,
    n_initial_points=10,
    identifier="",
    foldername=""
):

    # Generate initial points
    points = generate_initial_points(search_space, n_points=n_initial_points)

    # Evaluate function at initial points
    values = [objective_function(point) for point in points]

    for i in range(len(points)):
        if np.isnan(values[i]):
            values[i] = 100

    # Perform the optimization
    for i in range(n_calls - n_initial_points):
        # next_point, expected_value = find_minimal_from_rbf_brute(points, values, search_space)
        next_point, expected_value = sample_minimal_point_from_rbf(points, values, search_space, identifier=f"{identifier}_run_{i}", foldername=foldername)

        if any(np.all(np.isclose(next_point, p), axis=-1) for p in points):  # Check if point was seen before
            print("Needs to be perturbed")
            step_size = 0.01
            next_point = perturb_point(next_point, search_space, step_size)  # Perturb the point slightly

        # Evaluate the function at the new point
        next_value = objective_function(next_point)

        print("Expected value: ", expected_value)
        print("Next value: ", next_value)

        # Add the new point to the list
        points.append(next_point)
        values.append(next_value)

    # Get the best point:
    best_index = np.argmin(values)
    best_point = points[best_index]
    best_value = values[best_index]

    return best_point, best_value, points


def adann_opt_training(
        adann_creator,
        opt_params,
        training_samples_generator,
        optimizer_class,
        loss_fn,
        base_training_kwargs,
        base_lr_search_parameters,
        diff_training_kwargs: dict,
        diff_lr_search_parameters,
        output_folder_dir,
        methods,
        methods_data,
        test_input_values,
        test_ref_sol,
        validation_input_values=None,
        validation_ref_sol=None,
        pde_name="",
        space_size=1,
        dim=1,
        base_nr_timesteps=None,
        only_train_base=False,
        local_learning_rates=False,
        opt_strategy = "rbf_brute" # Other options: "gp_minimize", "forest_minimize"
):
    '''
        Only works for 2D params
    '''

    # Unpack parameters
    p1_low, p1_high, p2_low, p2_high, n_calls, p1_low_start, p1_high_start, p2_low_start, p2_high_start, n_random_starts = opt_params

    if diff_training_kwargs['max_trainsteps'] == 0:
        only_train_base = True
    if only_train_base:
        diff_training_kwargs = diff_training_kwargs.copy()
        diff_training_kwargs['max_trainsteps'] = 0

    # Deduce names
    base_identifier = f"{base_nr_timesteps}_tsteps"
    names = [f"ADANN base - EE ({base_nr_timesteps} time steps)"] if only_train_base else [f"ADANN full - EE ({base_nr_timesteps} time steps)"]

    # Prepare placeholders to track the procedure
    init_params = []
    adann_models = []  # Not used because of memory issues. Kept here in case I need it at some point
    opt_start_errors = []
    opt_end_errors_base = []
    opt_end_errors_base_validation = []
    opt_end_errors = []
    opt_end_errors_validation = []
    base_traintimes = []
    diff_traintimes = []
    base_initial_learning_rates = []
    diff_initial_learning_rates = []

    best_model_opt = None
    best_model_run_nr = None

    # Folder to store results
    opt_foldername = output_folder_dir + "Results_opt"
    if not os.path.exists(opt_foldername):
        os.makedirs(opt_foldername)

    # opt procedure
    with open(opt_foldername + f"/opt_simulation_results_{pde_name}_{base_identifier}.txt", 'w+') as f:
        start_time = time.perf_counter()

        if base_training_kwargs['initial_lr'] is None and not local_learning_rates:
            middle_adann_model = adann_creator(None)
            middle_best_learningrate = find_optimal_learningrate_with_documentation(middle_adann_model.base_model,
                                                                                    training_samples_generator,
                                                                                    optimizer_class,
                                                                                    loss_fn,
                                                                                    base_training_kwargs['initial_batchsize'],
                                                                                    validation_input_values,
                                                                                    validation_ref_sol,
                                                                                    base_lr_search_parameters,
                                                                                    f"base_{base_identifier}_opt_middle",
                                                                                    f,
                                                                                    opt_foldername)
            # Create a copy of kwargs to avoid changing the original
            base_initial_lr = middle_best_learningrate / 4.
            base_training_kwargs = base_training_kwargs.copy()
            base_training_kwargs['initial_lr'] = base_initial_lr
            print_to_file_and_console(f"Choosing nonlocal learning rate: {base_initial_lr}", file=f)

        # Define objective function that we want to minimize
        # It creates and traines a model for a given set of initialization parameters and returns the validation error
        # It keeps track of parameters and corresponding errors (even though gp_optimize will do this too)
        def objective_function(params):
            """
                params = [p1, p2]
            """

            nonlocal best_model_opt, best_model_run_nr  # Needed to access the best model from the outer scope

            run_number = len(opt_end_errors_base)

            print_to_file_and_console(f"\n------------------------------RUN {run_number} : {params}------------------------------", file=f)

            adann_model = adann_creator(params)

            start_error = evaluate(adann_model.base_model, test_input_values, test_ref_sol, space_size=space_size, dim=dim, train_or_test="test")
            print_to_file_and_console(f"Start error: {start_error}", file=f)

            base_traintime, diff_traintime, this_base_initial_lr, this_diff_initial_lr = (
                adann_training(
                    adann_model=adann_model,
                    trainings_samples_generator=training_samples_generator,
                    optimizer_class=optimizer_class,
                    loss_fn=loss_fn,
                    base_training_kwargs=base_training_kwargs,
                    diff_training_kwargs=diff_training_kwargs,
                    instance_identifier=f"{base_identifier}_opt_nr_{run_number}",
                    output_file=f,
                    output_folder=opt_foldername,
                    validation_input_values=validation_input_values,
                    validation_ref_sol=validation_ref_sol,
                    lr_search_parameters=(base_lr_search_parameters, diff_lr_search_parameters)
                )
            )

            end_error_base = evaluate(adann_model.base_model, test_input_values, test_ref_sol, space_size=space_size, dim=dim, train_or_test="test")
            end_error_base_validation = evaluate(adann_model.base_model, validation_input_values, validation_ref_sol, space_size=space_size, dim=dim, train_or_test="validate")
            print_to_file_and_console(f"End error base: {end_error_base_validation}", file=f)

            end_error = evaluate(adann_model, test_input_values, test_ref_sol, space_size=space_size, dim=dim, train_or_test="test")
            end_error_validation = evaluate(adann_model, validation_input_values, validation_ref_sol, space_size=space_size, dim=dim, train_or_test="validate")
            print_to_file_and_console(f"End error: {end_error}", file=f)

            # Track the results
            init_params.append(params)
            # adann_models.append(adann_model)
            opt_start_errors.append(start_error)
            opt_end_errors_base.append(end_error_base)
            opt_end_errors_base_validation.append(end_error_base_validation)
            opt_end_errors.append(end_error)
            opt_end_errors_validation.append(end_error_validation)
            base_traintimes.append(base_traintime)
            diff_traintimes.append(diff_traintime)
            base_initial_learning_rates.append(this_base_initial_lr)
            diff_initial_learning_rates.append(this_diff_initial_lr)

            # Save model if it is the best so far
            if only_train_base:
                if opt_end_errors_base_validation[run_number] == np.nanmin(opt_end_errors_base_validation):
                    best_model_opt = adann_model.base_model
                    best_model_run_nr = run_number
                    print(" We have a new best base model!")
            else:
                if opt_end_errors_validation[run_number] == np.nanmin(opt_end_errors_validation):
                    best_model_opt = adann_model
                    best_model_run_nr = run_number
                    print(" We have a new best model!")

            print("\nSummary of model:")
            summary(adann_model, test_input_values[0:1], test_ref_sol[0:1], plot_file_name=opt_foldername + f"/Z_plot_{run_number}.pdf",  write_file=f)
            print("\n\n", file=f)

            return_value = end_error_base_validation if only_train_base else end_error_validation
            # If return_value is nan or inf, we set it to a very large number to avoid the error in the optimization
            return_value = 1e12 if np.isnan(return_value) or np.isinf(return_value) else return_value

            return return_value

        # Perform the minimization using gp_minimize
        search_space = [(p1_low, p1_high), (p2_low, p2_high)]
        initial_space = [(p1_low_start, p1_high_start), (p2_low_start, p2_high_start)]
        initial_points = generate_initial_points(initial_space, n_points=n_random_starts)

        if opt_strategy == "gp_minimize":
            result = gp_minimize(
                objective_function,  # the function to minimize
                search_space,  # the bounds on each dimension of parameters
                n_calls=n_calls,  # the number of evaluations of objective_function
                n_initial_points=n_random_starts,  # the number of additional random initialization points
                # x0=initial_points,  # initial points for the search
                acq_func="gp_hedge",  # the acquisition function
                acq_optimizer="lbfgs",  # the acquisition function
            )
            best_params, best_error, all_params = result.x, result.fun, result.x_iters

        elif opt_strategy == "forest_minimize":
            result = forest_minimize(
                objective_function,  # the function to minimize
                search_space,  # the bounds on each dimension of parameters
                n_calls=n_calls,  # the number of evaluations of objective_function
                n_initial_points=0,  # the number of additional random initialization points
                # initial_point_generator="lhs",  # "lhs" or "random"
                x0=initial_points,  # initial points for the search
                acq_func="EI",  # the acquisition function
                base_estimator="ET",  # "ET" or "RF",
                random_state=0,  # For reproducibility
            )
            best_params, best_error, all_params = result.x, result.fun, result.x_iters

        elif opt_strategy == "rbf_brute":
            best_params, best_error, all_params = rbf_brute_minimize(
                objective_function,
                search_space,
                n_calls=n_calls,
                n_initial_points=n_random_starts,
                identifier=base_identifier,
                foldername=opt_foldername
            )

        # Save the best model
        best_run_nr = np.nanargmin(opt_end_errors_base_validation)
        methods[names[0]] = best_model_opt
        methods_data.at[names[0], "training_time"] = total_time = time.perf_counter() - start_time

        #######################################
        # Print some summary values to the file
        best_params_opt = best_params
        best_error_opt = best_error
        init_params_opt = all_params
        best_params = init_params[best_run_nr]
        best_error = opt_end_errors_base_validation[best_run_nr] if only_train_base else opt_end_errors_validation[best_run_nr]

        best_run_nr_test = np.nanargmin(opt_end_errors_base)
        best_params_test = init_params[best_run_nr_test]
        best_error_test = opt_end_errors_base[best_run_nr_test]

        print_to_file_and_console("\n\n\n--------------------------------------", file=f)

        print_to_file_and_console("Best validation errors:", file=f)
        print_to_file_and_console(f"    Best parameters: {best_params} (opt: {best_params_opt}) - Run number {best_run_nr} (Sanity check: {best_model_run_nr})", file=f)
        print_to_file_and_console(f"    Best error: {best_error} (opt: {best_error_opt})", file=f)

        print_to_file_and_console("\nBest test errors:", file=f)
        print_to_file_and_console(f"    Best parameters: {best_params_test} - Run number {best_run_nr_test}", file=f)
        print_to_file_and_console(f"    Best error: {best_error_test}", file=f)

        #   Check if best test and best validation are the same:
        if not best_run_nr_test == best_run_nr:
            print_to_file_and_console(f"\033[91m\nWARNING: Best parameters it not the same for test and validation\033[0m", file=f)

        print_to_file_and_console("\nTrain times:", file=f)
        print_to_file_and_console(f"    Total time : {total_time}", file=f)
        print_to_file_and_console(f"    Total train time: {np.sum(base_traintimes) + np.sum(diff_traintimes)}", file=f)
        print_to_file_and_console(f"    Total base train time: {np.sum(base_traintimes)}", file=f)
        print_to_file_and_console(f"    Total diff train time: {np.sum(diff_traintimes)}", file=f)
        print_to_file_and_console("\n    Average train time: ", file=f)
        print_to_file_and_console(f"\n    Average base train time: {np.mean(base_traintimes)}", file=f)
        print_to_file_and_console(f"    Average diff train time: {np.mean(diff_traintimes)}", file=f)

        print(f"---------------------------------------------------------------------------------------\n\n\n")
        print(f"Summary of best trained base {best_run_nr}: {best_params}")
        summary(methods[names[0]], test_input_values[0:1], test_ref_sol[0:1], plot_file_name=opt_foldername + f"/Best_trained_base_plot_{pde_name}.pdf", plot_show=False)

    # Save the trackers to files
    np.savetxt(opt_foldername + f"/Y_opt_start_errors_{base_identifier}.txt", opt_start_errors)
    np.savetxt(opt_foldername + f"/Y_opt_end_errors_base_{base_identifier}.txt", opt_end_errors_base)
    np.savetxt(opt_foldername + f"/Y_opt_end_errors_base_validation_{base_identifier}.txt", opt_end_errors_base_validation)
    np.savetxt(opt_foldername + f"/Y_opt_init_params_{base_identifier}.txt", init_params)
    np.savetxt(opt_foldername + f"/Y_opt_end_errors_{base_identifier}.txt", opt_end_errors)
    np.savetxt(opt_foldername + f"/Y_opt_end_errors_validation_{base_identifier}.txt", opt_end_errors_validation)
    print("base_initial_learning_rates", base_initial_learning_rates)
    np.savetxt(opt_foldername + f"/Y_base_initial_learning_rates_{base_identifier}.txt", base_initial_learning_rates)
    if not only_train_base:
        np.savetxt(opt_foldername + f"/Y_diff_initial_learning_rates_{base_identifier}.txt", diff_initial_learning_rates)


def plot_opt_results_in_axs(
    fig, axs,
    output_folder_dir,
    base_nr_timesteps=None,
    only_train_base=True,
    n_random_starts=0,
    cutoff=0.4,
    size_start=50,
    size_end=200,
    colormap=CONT_COLORMAP,
    fontsize=15
):
    '''
        Plot the results of the opt procedure in the given axes.
        Only works when params are 2D.
    '''

    base_identifier = f"{base_nr_timesteps}_tsteps"
    opt_foldername = output_folder_dir + "Results_opt"

    # Load the data
    opt_end_errors = np.loadtxt(opt_foldername + f"/Y_opt_end_errors_base_{base_identifier}.txt") if only_train_base else np.loadtxt(opt_foldername + f"/Y_opt_end_errors_{base_identifier}.txt")
    init_params = np.loadtxt(opt_foldername + f"/Y_opt_init_params_{base_identifier}.txt")

    # Set all values above the cutoff to nan
    # In order to avoid the cutoff being too high, we set it to the second-smallest value if it is larger than the cutoff
    second_smallest = np.sort(opt_end_errors[~np.isnan(opt_end_errors)])[1]
    threshold = np.max([cutoff, second_smallest])
    opt_end_errors[opt_end_errors > threshold] = np.nan

    # Prepare the data splitting it between random starts and opt search and
    n_calls = len(opt_end_errors)
    random_start_end_errors = opt_end_errors[:n_random_starts]
    random_start_init_params = init_params[:n_random_starts]
    opt_search_end_errors = opt_end_errors[n_random_starts:]
    opt_search_init_params = init_params[n_random_starts:]

    errors_min, errors_max = np.nanmin(opt_end_errors), np.nanmax(opt_end_errors)
    random_start_run_numbers = np.arange(1, n_random_starts + 1)
    opt_search_run_numbers = np.arange(n_random_starts + 1, n_calls + 1)

    # Find indices with NaN values
    random_start_nan_indices = np.isnan(random_start_end_errors)
    random_start_non_nan_indices = ~random_start_nan_indices
    opt_search_nan_indices = np.isnan(opt_search_end_errors)
    opt_search_non_nan_indices = ~opt_search_nan_indices

    # Settings
    size_start = size_start
    size_end = size_end
    colormap = colormap
    fontsize = fontsize

    sizes_random_start = np.linspace(size_start, size_end, n_random_starts)
    sizes_opt_search = np.linspace(size_start, size_end, n_calls - n_random_starts)

    # Spatial scatter plot random start
    scatter1 = axs[0].scatter(random_start_init_params[random_start_non_nan_indices, 0], random_start_init_params[random_start_non_nan_indices, 1],
                              s=sizes_random_start[random_start_non_nan_indices], marker='o',
                              c=random_start_end_errors[random_start_non_nan_indices], vmin=errors_min, vmax=errors_max, cmap=colormap)
    axs[0].scatter(random_start_init_params[random_start_nan_indices, 0], random_start_init_params[random_start_nan_indices, 1],
                       s=sizes_random_start[random_start_nan_indices], marker='x',
                       color='black')
    # Spatial scatter plot opt search
    axs[0].scatter(opt_search_init_params[opt_search_non_nan_indices, 0], opt_search_init_params[opt_search_non_nan_indices, 1],
                                s=sizes_opt_search[opt_search_non_nan_indices], marker='s',
                                c=opt_search_end_errors[opt_search_non_nan_indices], vmin=errors_min, vmax=errors_max, cmap=colormap)
    axs[0].scatter(opt_search_init_params[opt_search_nan_indices, 0], opt_search_init_params[opt_search_nan_indices, 1],
                       s=sizes_opt_search[opt_search_nan_indices], marker='x',
                       color='black')
    # axs[0].set_title('Spatial init params (TBD)')
    axs[0].set_xlabel(r'$p_1$', fontsize=fontsize)
    axs[0].set_ylabel(r'$p_2$', fontsize=fontsize)

    # Function value by point number plot (right plot)
    # axs[1].scatter(run_numbers, opt_start_errors, s=middle_size, marker='x', c='black')
    axs[1].scatter(random_start_run_numbers[random_start_non_nan_indices], random_start_end_errors[random_start_non_nan_indices], s=sizes_random_start[random_start_non_nan_indices], marker='o', c=random_start_end_errors[random_start_non_nan_indices], norm=LogNorm(vmin=errors_min, vmax=errors_max), cmap=colormap)
    axs[1].scatter(opt_search_run_numbers[opt_search_non_nan_indices], opt_search_end_errors[opt_search_non_nan_indices], s=sizes_opt_search[opt_search_non_nan_indices], marker='s', c=opt_search_end_errors[opt_search_non_nan_indices], norm=LogNorm(vmin=errors_min, vmax=errors_max), cmap=colormap)

    ylim = axs[1].get_ylim()
    cross_y_value = ylim[1] * 1.05  # Adjust the multiplier as needed to position the crosses
    crosses_random_start = axs[1].scatter(random_start_run_numbers[random_start_nan_indices],
                     [cross_y_value] * len(random_start_run_numbers[random_start_nan_indices]),
                        s=sizes_random_start[random_start_nan_indices], marker='x', color='black')
    crosses_random_start.set_clip_on(False)
    crosses_opt = axs[1].scatter(opt_search_run_numbers[opt_search_nan_indices],
                   [cross_y_value] * len(opt_search_run_numbers[opt_search_nan_indices]),
                   s=sizes_opt_search[opt_search_nan_indices], marker='x', color='black')
    crosses_opt.set_clip_on(False)
    # Update the y-limit of the plot to include the crosses
    axs[1].set_ylim(bottom=ylim[0], top=cross_y_value)

    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # axs[1].set_title('Error by run number')
    axs[1].set_xlabel('Training run number', fontsize=fontsize)
    # axs[1].set_yscale('log') # This is not needed if we use cutoff

    # Adjust the colorbar position by creating a divider for the left plot
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Move the colorbar to the new axis on the right of the left plot
    cbar = fig.colorbar(scatter1, cax=cax, orientation='vertical')
    # cbar.set_label(r"$L^2$-error", labelpad=17, fontsize=fontsize)

    # Parameter value by run number plot (rightest plot)
    # Select colors for p1 and p2 from the colormap
    colormap = plt.get_cmap(colormap)  # Ensure we are working with a colormap object
    colors = [colormap(0.2), colormap(0.8)]  # Select colors from the colormap, for example at 20% and 80%

    # Scatter plots for parameters with consistent coloring
    axs[2].scatter(random_start_run_numbers[random_start_non_nan_indices], random_start_init_params[random_start_non_nan_indices, 0], color=colors[0], label=r'$p_1$', marker='o')
    axs[2].scatter(opt_search_run_numbers[opt_search_non_nan_indices], opt_search_init_params[opt_search_non_nan_indices, 0], color=colors[0], marker='s')
    axs[2].scatter(random_start_run_numbers[random_start_nan_indices], random_start_init_params[random_start_nan_indices, 0], color=colors[0], label=r'$p_1$', marker='x')
    axs[2].scatter(opt_search_run_numbers[opt_search_nan_indices], opt_search_init_params[opt_search_nan_indices, 0], color=colors[0], marker='x')

    axs[2].scatter(random_start_run_numbers[random_start_non_nan_indices], random_start_init_params[random_start_non_nan_indices, 1], color=colors[1], label=r'$p_2$', marker='o')
    axs[2].scatter(opt_search_run_numbers[opt_search_non_nan_indices], opt_search_init_params[opt_search_non_nan_indices, 1], color=colors[1], marker='s')
    axs[2].scatter(random_start_run_numbers[random_start_nan_indices], random_start_init_params[random_start_nan_indices, 1], color=colors[1], label=r'$p_2$', marker='x')
    axs[2].scatter(opt_search_run_numbers[opt_search_nan_indices], opt_search_init_params[opt_search_nan_indices, 1], color=colors[1], marker='x')

    axs[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[2].set_xlabel("Training run number", fontsize=fontsize)
    axs[2].set_ylabel("Parameter value", fontsize=fontsize, labelpad=10)

    # Adjust the legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=10, label=r'$p_1$', linestyle='None'),
                       plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=10, label=r'$p_2$', linestyle='None')]
    axs[2].legend(handles=legend_elements, fontsize=fontsize, loc='upper left')

    is_there_nan = np.isnan(opt_end_errors).any()
    return is_there_nan


def plot_opt_results_single(
    output_folder_dir,
    pde_name,
    base_nr_timesteps=None,
    only_train_base=True,
    n_random_starts=0,
    cutoff=0.4,
    size_start=50,
    size_end=200,
    colormap=CONT_COLORMAP,
    fontsize=15
):

    opt_foldername = output_folder_dir + "Results_opt"
    base_identifier = f"{base_nr_timesteps}_tsteps"

    # Plot creation
    fig, axs = plt.subplots(1, 3, figsize=(24, 6), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.3})

    plot_opt_results_in_axs(
        fig=fig, axs=axs,
        output_folder_dir=output_folder_dir,
        base_nr_timesteps=base_nr_timesteps,
        only_train_base=only_train_base,
        n_random_starts=n_random_starts,
        cutoff=cutoff,
        size_start=size_start,
        size_end=size_end,
        colormap=colormap,
        fontsize=fontsize
    )


    legend_size = 10
    # Adjusting legend position
    legend_elements = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=legend_size, label='Pseudorandom starting points', linestyle='None'),
                       plt.Line2D([0], [0], marker='s', color='black', markerfacecolor='black', markersize=legend_size, label='Exploration-exploitation points', linestyle='None')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.008), fontsize=fontsize)

    fig.savefig(opt_foldername + f"/opt_scatter_{pde_name}_{base_identifier}.pdf", bbox_inches='tight')
    plt.show()


def plot_opt_results_combined(
        output_folder_dir,
        pde_name,
        list_base_nr_timesteps,
        only_train_base=True,
        n_random_starts=0,
        cutoff=0.4,
        size_start=50,
        size_end=200,
        colormap=CONT_COLORMAP,
        fontsize=15
):
    opt_foldername = output_folder_dir + "Results_opt"
    nr_of_base = len(list_base_nr_timesteps)
    cutoffs = [cutoff] * nr_of_base if not isinstance(cutoff, list) else cutoff

    # Plot creation
    fig, axs = plt.subplots(len(list_base_nr_timesteps), 3, figsize=(24, 6 * len(list_base_nr_timesteps)), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.23})
    axs = np.atleast_2d(axs)

    exists_nan = False
    for i, base_nr_timesteps in enumerate(list_base_nr_timesteps):
        is_there_nan = plot_opt_results_in_axs(
            fig=fig, axs=axs[i],
            output_folder_dir=output_folder_dir,
            base_nr_timesteps=base_nr_timesteps,
            only_train_base=only_train_base,
            n_random_starts=n_random_starts,
            cutoff=cutoffs[i],
            size_start=size_start,
            size_end=size_end,
            colormap=colormap,
            fontsize=fontsize
        )
        exists_nan = exists_nan or is_there_nan

        # Add label for this row
        sec_ax = axs[i, 0].secondary_yaxis('left')
        sec_ax.set_ylabel(f"{base_nr_timesteps} time steps", labelpad=50, fontsize=15)
        sec_ax.tick_params(left=False, labelleft=False)

    # Column Titles
    columns_titles = [r"$L^2$-errors (spatial representation)", r"$L^2$-errors (chronological representation)", r"Coordinates (chronological)"]
    for i, title in enumerate(columns_titles):
        axs[0, i].set_title(title, fontsize=15, pad=20)

    legend_size = 10
    # Adjusting legend position
    legend_elements = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=legend_size, label='Pseudorandom starting points', linestyle='None'),
                       plt.Line2D([0], [0], marker='s', color='black', markerfacecolor='black', markersize=legend_size, label='Exploration-exploitation points', linestyle='None')]
    if exists_nan:
        legend_elements += [plt.Line2D([0], [0], marker='x', color='black', markerfacecolor='black', markersize=legend_size, label='Large or NaN values', linestyle='None')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.1 if nr_of_base == 1 else (0.97 if nr_of_base == 2 else 0.95)), fontsize=fontsize)

    fig.savefig(opt_foldername + f"/opt_scatter_combined_{pde_name}.pdf", bbox_inches='tight')
    plt.show()


def adann_opt(
        ADANN_base_model_class,
        base_model_kwargs,
        diff_model_class,
        diff_model_params,
        list_base_nr_timesteps,
        opt_params,
        training_samples_generator,
        optimizer_class,
        loss_fn,
        base_training_kwargs,
        base_lr_search_parameters,
        diff_training_kwargs,
        diff_lr_search_parameters,
        output_folder_dir,
        methods,
        methods_data,
        test_input_values_rough,
        test_ref_sol_rough,
        validation_input_values_rough,
        validation_ref_sol_rough,
        pde_name,
        space_size,
        dim,
        only_train_base=False,
        local_learning_rates=False
):
    if diff_training_kwargs['max_trainsteps'] == 0:
        only_train_base = True

    for base_nr_timesteps in list_base_nr_timesteps:
        print("------------------------------------------------------------\n")
        print("------------------------------------------------------------\n")
        print(f"Base nr timesteps: {base_nr_timesteps}")
        print("------------------------------------------------------------\n")
        print("------------------------------------------------------------\n")

        this_base_model_kwargs = base_model_kwargs.copy()
        this_base_model_kwargs["nr_timesteps"] = base_nr_timesteps
        adann_creator = make_adann_creator(ADANN_base_model_class, this_base_model_kwargs, diff_model_class, diff_model_params)

        adann_opt_training(
            adann_creator=adann_creator,
            opt_params=opt_params,
            training_samples_generator=training_samples_generator,
            optimizer_class=optimizer_class,
            loss_fn=loss_fn,
            base_training_kwargs=base_training_kwargs,
            base_lr_search_parameters=base_lr_search_parameters,
            diff_training_kwargs=diff_training_kwargs,
            diff_lr_search_parameters=diff_lr_search_parameters,
            output_folder_dir=output_folder_dir,
            methods=methods,
            methods_data=methods_data,
            test_input_values=test_input_values_rough,
            test_ref_sol=test_ref_sol_rough,
            validation_input_values=validation_input_values_rough,
            validation_ref_sol=validation_ref_sol_rough,
            pde_name=pde_name,
            space_size=space_size,
            dim=dim,
            base_nr_timesteps=base_nr_timesteps,
            only_train_base=only_train_base,
            local_learning_rates=local_learning_rates
        )

        plot_opt_results_single(output_folder_dir, pde_name=pde_name, base_nr_timesteps=base_nr_timesteps, only_train_base=only_train_base, n_random_starts=opt_params[-1])

if __name__=="__main__":
    # Tests for the plots
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Sample data setup
    np.random.seed(0)
    n_points = 20
    a, b = -1, 1
    points = (b - a) * np.random.rand(n_points, 2) + a
    function_values = np.sin(points[:, 0]) + np.cos(points[:, 1])
    nr_in_first_category = 10
    points_cat1 = points[:nr_in_first_category]
    points_cat2 = points[nr_in_first_category:]
    func_values_cat1 = function_values[:nr_in_first_category]
    func_values_cat2 = function_values[nr_in_first_category:]
    combined_func_values = np.concatenate([func_values_cat1, func_values_cat2])
    vmin, vmax = combined_func_values.min(), combined_func_values.max()
    sizes_cat1 = np.linspace(10, 200, nr_in_first_category)
    sizes_cat2 = np.linspace(10, 200, n_points - nr_in_first_category)
    point_numbers = np.arange(1, n_points + 1)

    # Plot creation
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.05})

    # Spatial scatter plot (left plot)
    scatter1 = axs[0].scatter(points_cat1[:, 0], points_cat1[:, 1], s=sizes_cat1, marker='o', c=func_values_cat1, vmin=vmin, vmax=vmax, cmap=CONT_COLORMAP)
    scatter2 = axs[0].scatter(points_cat2[:, 0], points_cat2[:, 1], s=sizes_cat2, marker='s', c=func_values_cat2, vmin=vmin, vmax=vmax, cmap=CONT_COLORMAP)
    axs[0].set_title('Spatial Distribution of Points')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Function value')

    # Function value by point number plot (right plot)
    axs[1].scatter(point_numbers[:nr_in_first_category], func_values_cat1, s=sizes_cat1, marker='o', c=func_values_cat1, vmin=vmin, vmax=vmax, cmap=CONT_COLORMAP)
    axs[1].scatter(point_numbers[nr_in_first_category:], func_values_cat2, s=sizes_cat2, marker='s', c=func_values_cat2, vmin=vmin, vmax=vmax, cmap=CONT_COLORMAP)
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[1].set_title('Function Value by Point Number')
    axs[1].set_xlabel('Point Number')
    axs[1].tick_params(axis='y', which='both', left=False, labelleft=False)  # Hide the right plot's Y-axis

    # Create an axes divider for the right plot
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("left", size="5%", pad=0.5)

    # Move the colorbar to the new axis
    cbar = fig.colorbar(scatter1, cax=cax, orientation='vertical')
    cbar.set_label('Function value')
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('left')

    # Adjusting legend position
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Category 1'),
                       plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Category 2')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, title="Categories", bbox_to_anchor=(0.5, 1.02))

    plt.show()