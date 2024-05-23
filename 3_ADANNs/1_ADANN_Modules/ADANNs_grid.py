import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.insert(1, '1_Modules')
from evaluation_utils import evaluate
from utils import print_to_file_and_console
from training import find_optimal_learningrate_with_documentation

sys.path.insert(1, '1_ADANN_Modules')
from ADANNs import make_adann_creator
from ADANNs_training import adann_training
from documentation_utils import summary


CONT_COLORMAP = "plasma"  # 'jet', 'viridis', 'plasma', 'inferno', 'magma'

def adann_grid_training(
    adann_creator,
    param_grid_parameters,
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
    test_input_values,
    test_ref_sol,
    validation_input_values=None,
    validation_ref_sol=None,
    pde_name="",
    space_size=1,
    dim=1,
    base_nr_timesteps=None,
    only_train_base=False,
    local_learning_rates=False
):
    '''
    If I get into memory issues I might want to delete the models and empty cuda cache after each run
    local_learning_rates=False and diff_initial_lr=None is not allowed
    '''

    # Unpack parameters
    grid_p1_low, grid_p1_high, grid_nr_p1_steps, grid_p2_low, grid_p2_high, grid_nr_p2_steps = param_grid_parameters

    if diff_training_kwargs['max_trainsteps'] == 0:
        only_train_base = True
    if only_train_base:
        diff_training_kwargs = diff_training_kwargs.copy()
        diff_training_kwargs['max_trainsteps'] = 0

    # Prepare grid
    p1_values = np.linspace(grid_p1_low, grid_p1_high, grid_nr_p1_steps + 2)[1:-1]
    p2_values = np.linspace(grid_p2_low, grid_p2_high, grid_nr_p2_steps + 2)[1:-1]
    grid_p1s, grid_p2s = np.meshgrid(p1_values, p2_values)

    # Deduce names
    base_identifier = f"{base_nr_timesteps}_tsteps"
    names = [f"ADANN base - grid ({base_nr_timesteps} time steps)"] if only_train_base else [f"ADANN base - grid ({base_nr_timesteps} time steps)", f"ADANN full - grid ({base_nr_timesteps} time steps)"]

    # Prepare placeholders for errors
    grid_start_errors = float('inf') * np.ones_like(grid_p1s)
    grid_end_errors_base = float('inf') * np.ones_like(grid_p1s)
    grid_end_errors_base_validation = float('inf') * np.ones_like(grid_p1s)
    grid_end_errors = float('inf') * np.ones_like(grid_p1s)
    grid_end_errors_validation = float('inf') * np.ones_like(grid_p1s)

    # Prepare placeholders for training times
    base_traintimes = np.zeros_like(grid_p1s)
    diff_traintimes = np.zeros_like(grid_p1s)
    base_initial_learining_rates = np.zeros_like(grid_p1s)
    diff_initial_learining_rates = np.zeros_like(grid_p1s)

    # Prepare placeholders for best models
    best_base_model_grid = None
    best_full_model_grid = None

    # Folder to store results
    grid_foldername = output_folder_dir + "Results_grid"
    if not os.path.exists(grid_foldername):
        os.makedirs(grid_foldername)

    # Loop over the grid
    with open(grid_foldername + f'/grid_simulation_results_{pde_name}_{base_identifier}.txt', 'w+') as f:
        start_time = time.perf_counter()

        if base_training_kwargs['initial_lr'] is None and not local_learning_rates:
            middle_adann_model = adann_creator(None)
            middle_best_learningrate = find_optimal_learningrate_with_documentation(
                model=middle_adann_model.base_model,
                training_samples_generator=training_samples_generator,
                optimizer_class=optimizer_class,
                loss_fn=loss_fn,
                batchsize=base_training_kwargs['initial_batchsize'],
                validation_input_values=validation_input_values,
                validation_ref_sol=validation_ref_sol,
                lr_search_parameters=base_lr_search_parameters,
                identifier=f"base_{base_identifier}_grid_middle",
                output_file=f,
                output_folder=grid_foldername
            )
            base_initial_lr = middle_best_learningrate / 4.
            base_training_kwargs = base_training_kwargs.copy()
            base_training_kwargs['initial_lr'] = base_initial_lr
            print_to_file_and_console(f"Choosing nonlocal learning rate: {base_initial_lr}", file=f)

        lr_search_time = time.perf_counter() - start_time

        for (i, j), p1 in np.ndenumerate(grid_p1s):
            p2 = grid_p2s[i, j]
            params = [p1, p2]
            print_to_file_and_console(f"--------------------------------------RUN ({j},{i}) : {params}--------------------------------------", file=f)

            adann_model = adann_creator(params)

            grid_start_errors[i, j] = evaluate(adann_model.base_model, test_input_values, test_ref_sol, space_size=space_size, dim=dim, train_or_test="test")
            print_to_file_and_console(f"Start error: {grid_start_errors[i, j]}", file=f)

            base_traintimes[i, j], diff_traintimes[i, j], base_initial_learining_rates[i, j], diff_initial_learining_rates[i, j] = (
                adann_training(
                    adann_model=adann_model,
                    trainings_samples_generator=training_samples_generator,
                    optimizer_class=optimizer_class,
                    loss_fn=loss_fn,
                    base_training_kwargs=base_training_kwargs,
                    diff_training_kwargs=diff_training_kwargs,
                    instance_identifier=f"{base_identifier}_grid_{i}-{j}",
                    output_file=f,
                    output_folder=grid_foldername,
                    validation_input_values=validation_input_values,
                    validation_ref_sol=validation_ref_sol,
                    lr_search_parameters=(base_lr_search_parameters, diff_lr_search_parameters)
                )
            )

            grid_end_errors_base[i, j] = evaluate(adann_model.base_model, test_input_values, test_ref_sol, space_size=space_size, dim=dim, train_or_test="test")
            grid_end_errors_base_validation[i, j] = evaluate(adann_model.base_model, validation_input_values, validation_ref_sol, space_size=space_size, dim=dim, train_or_test="validate")
            print_to_file_and_console(f"End error base: {grid_end_errors_base[i, j]}", file=f)

            grid_end_errors[i, j] = evaluate(adann_model, test_input_values, test_ref_sol, space_size=space_size, dim=dim, train_or_test="test")
            grid_end_errors_validation[i, j] = evaluate(adann_model, validation_input_values, validation_ref_sol, space_size=space_size, dim=dim, train_or_test="validate")
            print_to_file_and_console(f"End error: {grid_end_errors[i, j]}", file=f)

            print("\n Summary of model:", file=f)
            summary(adann_model, test_input_values, test_ref_sol, plot_file_name=grid_foldername + f"/Z_plot_{i}-{j}.pdf", write_file=f)
            print_to_file_and_console("\n\n", file=f)

            # Save models if they are the best
            if grid_end_errors_base_validation[i, j] == np.nanmin(grid_end_errors_base_validation):
                best_base_model_grid = adann_model.base_model
                print(" We have a new best base model!")
            if grid_end_errors_validation[i, j] == np.nanmin(grid_end_errors_validation) and not only_train_base:
                best_full_model_grid = adann_model
                print(" We have a new best full model!")

        best_start = np.unravel_index(np.nanargmin(grid_start_errors), grid_start_errors.shape)
        best_trained_base = np.unravel_index(np.nanargmin(grid_end_errors_base), grid_end_errors_base.shape)
        best_trained_base_validation = np.unravel_index(np.nanargmin(grid_end_errors_base_validation), grid_end_errors_base_validation.shape)
        best_full_adann = np.unravel_index(np.nanargmin(grid_end_errors), grid_end_errors.shape)
        best_full_adann_validation = np.unravel_index(np.nanargmin(grid_end_errors_validation), grid_end_errors_validation.shape)

        total_time = time.perf_counter() - start_time

        # Save best models
        methods[names[0]] = best_base_model_grid
        methods_data.at[names[0], "training_time"] = total_time if only_train_base else np.sum(base_traintimes) + lr_search_time

        if not only_train_base:
            methods[names[1]] = best_full_model_grid
            methods_data.at[names[1], "training_time"] = total_time

        #######################################
        # Print some summary values to the file
        print_to_file_and_console("\n\n\n--------------------------------------", file=f)

        print_to_file_and_console("Best test errors:", file=f)
        print_to_file_and_console(f"    Best start at grid point {best_start}: {grid_start_errors[best_start]}", file=f)
        print_to_file_and_console(f"    Best trained base at grid point {best_trained_base}: {grid_end_errors_base[best_trained_base]}", file=f)
        print_to_file_and_console(f"    Best full ADANN at grid point {best_full_adann}: {grid_end_errors[best_full_adann]}", file=f)

        print_to_file_and_console("\nBest validation errors:", file=f)
        print_to_file_and_console(f"    Best trained base at grid point {best_trained_base_validation}: {grid_end_errors_base_validation[best_trained_base_validation]}", file=f)
        print_to_file_and_console(f"    Best full ADANN at grid point {best_full_adann_validation}: {grid_end_errors_validation[best_full_adann_validation]}", file=f)

        #   Check if best test and best validation are the same:
        if not best_trained_base == best_trained_base_validation:
            print_to_file_and_console(f"\033[91m\nWARNING: Best trained base it not the same for test and validation\033[0m", file=f)
        if not best_full_adann == best_full_adann_validation:
            print_to_file_and_console(f"\033[91m\nWARNING: Best full ADANN it not the same for test and validation\033[0m", file=f)

        print_to_file_and_console("\nTrain times:", file=f)
        print_to_file_and_console(f"    Total time : {total_time}", file=f)
        print_to_file_and_console(f"    Total train time: {np.sum(base_traintimes) + np.sum(diff_traintimes)}", file=f)
        print_to_file_and_console(f"    Total base train time: {np.sum(base_traintimes)}", file=f)
        print_to_file_and_console(f"    Total diff train time: {np.sum(diff_traintimes)}", file=f)
        print_to_file_and_console("\n    Average train time: ", file=f)
        print_to_file_and_console(f"    Average base train time: {np.mean(base_traintimes)}", file=f)
        print_to_file_and_console(f"    Average diff train time: {np.mean(diff_traintimes)}", file=f)

        print(f"---------------------------------------------------------------------------------------\n\n\n")
        print(f"Summary of best trained base [{grid_p1s[best_trained_base]}, {grid_p2s[best_trained_base]}] , {best_trained_base}")
        summary(best_base_model_grid, test_input_values[0:1], test_ref_sol[0:1], plot_file_name=grid_foldername + f"/Best_trained_base_plot_grid_{pde_name}.pdf", plot_show=False)

        if not only_train_base:
            print(f"---------------------------------------------------------------------------------------\n\n")
            print(f"Summary of best full ADANN [{grid_p1s[best_full_adann]}, {grid_p2s[best_full_adann]}], {best_full_adann}")
            summary(best_full_model_grid, test_input_values, test_ref_sol, plot_file_name=grid_foldername + f"/Best_full_ADANN_plot_grid_{pde_name}.pdf", plot_show=False)

    # Save error data
    np.savetxt(grid_foldername + f"/Y_grid_start_errors_{base_identifier}.txt", grid_start_errors)
    np.savetxt(grid_foldername + f"/Y_grid_end_errors_base_{base_identifier}.txt", grid_end_errors_base)
    np.savetxt(grid_foldername + f"/Y_grid_end_errors_base_validation_{base_identifier}.txt", grid_end_errors_base_validation)
    np.savetxt(grid_foldername + f"/Y_grid_end_errors_{base_identifier}.txt", grid_end_errors)
    np.savetxt(grid_foldername + f"/Y_grid_end_errors_validation_{base_identifier}.txt", grid_end_errors_validation)
    np.savetxt(grid_foldername + f"/Y_grid_p1s_{base_identifier}.txt", grid_p1s)
    np.savetxt(grid_foldername + f"/Y_grid_p2s_{base_identifier}.txt", grid_p2s)
    np.savetxt(grid_foldername + f"/Y_grid_base_initial_learining_rates_{base_identifier}.txt", base_initial_learining_rates)
    np.savetxt(grid_foldername + f"/Y_grid_diff_initial_learining_rates_{base_identifier}.txt", diff_initial_learining_rates)

    return best_base_model_grid, best_full_model_grid


# Function to crop a 2-dimensional list of the form [[1,2,3],[4,5,6],[7,8,9]] by a list of the form [1,2,1,2]
def crop_heatmap(list_to_be_cropped, crop):
    list_to_be_cropped = list_to_be_cropped[crop[2]:-crop[3]] if crop[3] > 0 else list_to_be_cropped[crop[2]:]
    list_to_be_cropped = [row[crop[0]:-crop[1]] if crop[1] > 0 else row[crop[0]:] for row in list_to_be_cropped]
    return list_to_be_cropped

def plot_heatmap_single(grid_p1s, grid_p2s, grid_errors, title, filename, colormap=CONT_COLORMAP):
    # Create heat maps to show the errors
    grid_errors_masked = np.ma.masked_invalid(grid_errors)
    err_min, err_max = grid_errors_masked[:].min(), grid_errors_masked[:].max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(grid_p1s[:], grid_p2s[:], grid_errors_masked[:], cmap=colormap, vmin=err_min, vmax=err_max)
    ax.set_title(title)
    ax.set_xlabel(r'$p_1$')
    ax.set_ylabel(r'$p_2$')
    fig.colorbar(c, ax=ax)
    fig.savefig(filename)
    plt.show()

    return fig, ax


def plot_error_heat_maps(output_folder_dir, only_train_base, base_nr_timesteps, pde_name="", crop=(0, 0, 0, 0), colormap=CONT_COLORMAP):
    grid_foldername = output_folder_dir + "Results_grid"

    # Column Titles
    columns_titles = [f"$L^2$-errors at initialization", f"$L^2$-errors of trained base models", "$L^2$-errors of trained full ADANN errors"]

    base_identifier = f"{base_nr_timesteps}_tsteps"

    grid_start_errors = np.loadtxt(grid_foldername + f"/Y_grid_start_errors_{base_identifier}.txt", ndmin=2)
    grid_end_errors_base = np.loadtxt(grid_foldername + f"/Y_grid_end_errors_base_{base_identifier}.txt", ndmin=2)
    grid_end_errors_base_validation = np.loadtxt(grid_foldername + f"/Y_grid_end_errors_base_validation_{base_identifier}.txt", ndmin=2)
    grid_end_errors = np.loadtxt(grid_foldername + f"/Y_grid_end_errors_{base_identifier}.txt", ndmin=2)
    grid_end_errors_validation = np.loadtxt(grid_foldername + f"/Y_grid_end_errors_validation_{base_identifier}.txt", ndmin=2)
    grid_p1s = np.loadtxt(grid_foldername + f"/Y_grid_p1s_{base_identifier}.txt", ndmin=2)
    grid_p2s = np.loadtxt(grid_foldername + f"/Y_grid_p2s_{base_identifier}.txt", ndmin=2)

    # Crop the heatmaps
    grid_start_errors = crop_heatmap(grid_start_errors, crop)
    grid_end_errors_base = crop_heatmap(grid_end_errors_base, crop)
    grid_end_errors_base_validation = crop_heatmap(grid_end_errors_base_validation, crop)
    grid_end_errors = crop_heatmap(grid_end_errors, crop)
    grid_end_errors_validation = crop_heatmap(grid_end_errors_validation, crop)
    grid_p1s = crop_heatmap(grid_p1s, crop)
    grid_p2s = crop_heatmap(grid_p2s, crop)

    # Plot heatmaps
    plot_heatmap_single(grid_p1s, grid_p2s, grid_start_errors, columns_titles[0], grid_foldername + f"/grid_start_errors_{pde_name}_{base_identifier}.pdf", colormap=colormap)
    plot_heatmap_single(grid_p1s, grid_p2s, grid_end_errors_base, columns_titles[1], grid_foldername + f"/grid_end_errors_base_{pde_name}_{base_identifier}.pdf", colormap=colormap)
    if not only_train_base:
        plot_heatmap_single(grid_p1s, grid_p2s, grid_end_errors, columns_titles[2], grid_foldername + f"/grid_end_errors_{pde_name}_{base_identifier}.pdf", colormap=colormap)


def plot_heatmap_combined(grid_p1s, grid_p2s, grid_errors, fig, ax, colormap=CONT_COLORMAP):
    grid_errors_masked = np.ma.masked_invalid(grid_errors)
    err_min, err_max = grid_errors_masked[:].min(), grid_errors_masked[:].max()

    c = ax.pcolormesh(grid_p1s[:], grid_p2s[:], grid_errors_masked[:], cmap=colormap, vmin=err_min, vmax=err_max)
    ax.set_xlabel(r'$p_1$')
    ax.set_ylabel(r'$p_2$')
    fig.colorbar(c, ax=ax)
    # ax.set_aspect('equal')

def plot_error_heat_maps_combined(output_folder_dir, only_train_base, list_base_nr_timesteps, pde_name="", crop=(0, 0, 0, 0), colormap=CONT_COLORMAP):
    '''
        Plot the error heatmaps for the grid search in one figure
    '''
    grid_foldername = output_folder_dir + "Results_grid"

    # Create large figure for all plots
    nr_plots_per_base = 2 if only_train_base else 3
    fig, axs = plt.subplots(len(list_base_nr_timesteps), nr_plots_per_base, figsize=(6.2 * nr_plots_per_base, 5 * len(list_base_nr_timesteps)), gridspec_kw={'wspace': 0.2})
    axs = np.atleast_2d(axs)

    # Column Titles
    columns_titles = [f"$L^2$-errors at initialization", f"$L^2$-errors of trained base models", "$L^2$-errors of trained full ADANN models"]
    for i, title in enumerate(columns_titles[:2] if only_train_base else columns_titles):
        axs[0, i].set_title(title, fontsize=15, pad=20)

    for idx, base_nr_timesteps in enumerate(list_base_nr_timesteps):
        base_identifier = f"{base_nr_timesteps}_tsteps"

        grid_start_errors = np.loadtxt(grid_foldername + f"/Y_grid_start_errors_{base_identifier}.txt", ndmin=2)
        grid_end_errors_base = np.loadtxt(grid_foldername + f"/Y_grid_end_errors_base_{base_identifier}.txt", ndmin=2)
        grid_end_errors_base_validation = np.loadtxt(grid_foldername + f"/Y_grid_end_errors_base_validation_{base_identifier}.txt", ndmin=2)
        grid_end_errors = np.loadtxt(grid_foldername + f"/Y_grid_end_errors_{base_identifier}.txt", ndmin=2)
        grid_end_errors_validation = np.loadtxt(grid_foldername + f"/Y_grid_end_errors_validation_{base_identifier}.txt", ndmin=2)
        grid_p1s = np.loadtxt(grid_foldername + f"/Y_grid_p1s_{base_identifier}.txt", ndmin=2)
        grid_p2s = np.loadtxt(grid_foldername + f"/Y_grid_p2s_{base_identifier}.txt", ndmin=2)

        # Crop the heatmaps
        grid_start_errors = crop_heatmap(grid_start_errors, crop)
        grid_end_errors_base = crop_heatmap(grid_end_errors_base, crop)
        grid_end_errors_base_validation = crop_heatmap(grid_end_errors_base_validation, crop)
        grid_end_errors = crop_heatmap(grid_end_errors, crop)
        grid_end_errors_validation = crop_heatmap(grid_end_errors_validation, crop)
        grid_p1s = crop_heatmap(grid_p1s, crop)
        grid_p2s = crop_heatmap(grid_p2s, crop)

        # Plot heatmaps
        plot_heatmap_combined(grid_p1s, grid_p2s, grid_start_errors, fig, axs[idx, 0], colormap=colormap)
        plot_heatmap_combined(grid_p1s, grid_p2s, grid_end_errors_base, fig, axs[idx, 1], colormap=colormap)
        if not only_train_base:
            plot_heatmap_combined(grid_p1s, grid_p2s, grid_end_errors, fig, axs[idx, 2], colormap=colormap)

        # Add label for this row
        sec_ax = axs[idx, 0].secondary_yaxis('left')
        sec_ax.set_ylabel(f"{base_nr_timesteps} time steps", labelpad=50, fontsize=15)
        sec_ax.tick_params(left=False, labelleft=False)

    fig.tight_layout()
    fig.savefig(grid_foldername + f"/grid_error_overview_{pde_name}.pdf")
    fig.show()


def adann_grid(
        ADANN_base_model_class,
        base_model_kwargs,
        diff_model_class,
        diff_model_params,
        list_base_nr_timesteps,
        param_grid_parameters,
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

        adann_grid_training(
            adann_creator=adann_creator,
            param_grid_parameters=param_grid_parameters,
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
        plot_error_heat_maps(output_folder_dir, only_train_base, base_nr_timesteps, pde_name=pde_name)