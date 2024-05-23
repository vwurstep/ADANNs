import os
import sys
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import torch
from torch.nn import Module as MyModel

sys.path.insert(1, '1_Modules')
from utils import print_to_file_and_console
from training import find_optimal_learningrate_with_documentation, whole_training
from training_samples_generators import get_data, TrainingSamplesGeneratorFromSolutions

sys.path.insert(1, '1_ADANN_Modules')
from ADANNs import make_adann_creator


def adanns_learning_rate(
    ADANN_base_model_class,
    base_model_kwargs,
    diff_model_class,
    diff_model_params,
    list_nr_timesteps,
    list_nr_spacediscr,
    train_space_resolution_step,
    test_space_resolution_step,
    get_higher_nr_spacediscr,
    initial_value_generator,
    reference_algorithm,
    nr_train_samples,
    nr_validation_samples,
    train_nr_timesteps,
    test_nr_timesteps,
    reduce_dimension,
    train_batchsize,
    nr_tests_per_lr,
    optimizer_class,
    loss_fn,
    base_lr_search_parameters,
    output_folder_dir
):

    # Folder to store results
    learning_rate_foldername = output_folder_dir + "Learning_rates"
    if not os.path.exists(learning_rate_foldername):
        os.makedirs(learning_rate_foldername)

    best_learning_rates = np.zeros((len(list_nr_spacediscr), len(list_nr_timesteps), nr_tests_per_lr))

    with (open(learning_rate_foldername + f'/learning_rate_results.txt', 'w+') as f):
        for (i, local_nr_spacediscr) in enumerate(list_nr_spacediscr):

            print_to_file_and_console(f"---------------------------------\nNr spacediscr: {local_nr_spacediscr}\n---------------------------------\n", file=f)

            # Produce train and test data
            local_train_nr_spacediscr = get_higher_nr_spacediscr(local_nr_spacediscr, train_space_resolution_step)
            local_test_nr_spacediscr = get_higher_nr_spacediscr(local_nr_spacediscr, test_space_resolution_step)

            print("Generating train samples")
            _, _, local_train_initial_values_rough, local_train_ref_sol_rough = (
                get_data(
                    initial_value_generator, reference_algorithm,
                    nr_train_samples, local_train_nr_spacediscr, train_nr_timesteps,
                    reduce_dimension, train_space_resolution_step, 'train',
                    None, True, None, None, True
                ))
            local_training_samples_generator = TrainingSamplesGeneratorFromSolutions(local_train_initial_values_rough, local_train_ref_sol_rough)

            print("Generating validation samples")
            _, _, local_validation_initial_values_rough, local_validation_ref_sol_rough = (
                get_data(
                    initial_value_generator, reference_algorithm,
                    nr_validation_samples, local_test_nr_spacediscr, test_nr_timesteps,
                    reduce_dimension, test_space_resolution_step, 'validate',
                    None, True, None, None, True
                ))

            # Loop over different base models
            for (j, base_nr_timesteps) in enumerate(list_nr_timesteps):

                print_to_file_and_console(f"Base nr timesteps: {base_nr_timesteps}", file=f)

                local_base_model_kwargs = base_model_kwargs.copy()
                local_base_model_kwargs["nr_spacediscr"] = local_nr_spacediscr
                local_base_model_kwargs["nr_timesteps"] = base_nr_timesteps
                local_adann_creator = make_adann_creator(ADANN_base_model_class, local_base_model_kwargs, diff_model_class, diff_model_params)

                adann_model = local_adann_creator(None)

                for k in range(nr_tests_per_lr):
                    print_to_file_and_console(f"Test {k + 1}/{nr_tests_per_lr}", file=f)

                    best_learning_rates[i, j, k] = find_optimal_learningrate_with_documentation(
                        model=adann_model.base_model,
                        training_samples_generator=local_training_samples_generator,
                        optimizer_class=optimizer_class,
                        loss_fn=loss_fn,
                        batchsize=train_batchsize,
                        validation_input_values=local_validation_initial_values_rough,
                        validation_ref_sol=local_validation_ref_sol_rough,
                        lr_search_parameters=base_lr_search_parameters,
                        identifier=f"spacediscr_{local_nr_spacediscr}_tsteps_{base_nr_timesteps}_test_{k}",
                        output_file=f,
                        output_folder=learning_rate_foldername,
                        plot_title=f"{int(local_nr_spacediscr)} space steps, {base_nr_timesteps} time steps"
                    )

    # Save results
    np.save(learning_rate_foldername + "/best_learning_rates.npy", best_learning_rates)
    np.save(learning_rate_foldername + "/list_nr_timesteps.npy", list_nr_timesteps)
    np.save(learning_rate_foldername + "/list_nr_spacediscr.npy", list_nr_spacediscr)

    return best_learning_rates


def plot_best_lr(output_folder_dir, shade="minmax"):
    learning_rate_foldername = output_folder_dir + "Learning_rates"

    # Load results
    best_learning_rates = np.load(learning_rate_foldername + "/best_learning_rates.npy")
    list_nr_timesteps = np.load(learning_rate_foldername + "/list_nr_timesteps.npy")
    list_nr_spacediscr = np.load(learning_rate_foldername + "/list_nr_spacediscr.npy")

    timesteps_mesh, spacestep_mesh = np.meshgrid(list_nr_timesteps, list_nr_spacediscr)
    timestepssize_mesh = 1 / timesteps_mesh
    spacestepsize_mesh = 1 / spacestep_mesh
    means = np.mean(best_learning_rates, axis=2)
    std_dev = np.std(best_learning_rates, axis=2)
    min_values = np.min(best_learning_rates, axis=2)
    max_values = np.max(best_learning_rates, axis=2)
    errors_bottom = means - min_values
    errors_top = max_values - means

    # Plotting
    colors = sns.color_palette(None, len(list_nr_spacediscr))
    fig_errors, ax_errors = plt.subplots()
    for i, nr_spacediscr in enumerate(list_nr_spacediscr):
        # Plot means
        ax_errors.scatter(list_nr_timesteps, means[i, :], label=f"{int(nr_spacediscr)} space steps", color=colors[i])
        # Shade the spread
        if shade == "minmax":
            ax_errors.errorbar(list_nr_timesteps, means[i, :], yerr=[errors_bottom[i], errors_top[i]], fmt='none', capsize=5, color=colors[i], alpha=0.5)
            ax_errors.fill_between(list_nr_timesteps, min_values[i, :], max_values[i, :], color=colors[i], alpha=0.3)
        if shade == "stddev":
            ax_errors.errorbar(list_nr_timesteps, means[i, :], yerr=std_dev[i, :], fmt='none', capsize=5, color=colors[i], alpha=0.7)
            ax_errors.fill_between(list_nr_timesteps, means[i, :] - std_dev[i, :], means[i, :] + std_dev[i, :], color=colors[i], alpha=0.3)

    ax_errors.set_xlabel("Number of time steps for ADANN base model")
    ax_errors.set_ylabel("Approximate optimal learning rate")
    ax_errors.set_yscale("log")
    ax_errors.set_xscale("log")
    ax_errors.set_xticks(list_nr_timesteps)
    ax_errors.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_errors.set_title(f"Optimal learning rates in dependence of number of space and time steps")
    ax_errors.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_errors.savefig(learning_rate_foldername + f"/Best_LR_plot_{shade}.pdf", bbox_inches='tight')
    fig_errors.show()

def regression_for_means(factor1_mesh, factor2_mesh, values_mesh, degree=3):
    # 1. Flatten the meshes and the target data
    X = np.vstack((factor1_mesh.flatten(), factor2_mesh.flatten())).T
    y = values_mesh.flatten()

    # 2. Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # print(f"Polynomial features: {poly_features.get_feature_names_out(['time_steps', 'space_discr'])}")
    # print(f"X_poly shape: {X_poly.shape}"
    #       f"\nX_poly: {X_poly}")

    # 3. Perform the regression
    model = LinearRegression()
    model.fit(X_poly, y)

    # 4. Predict (you can predict on the original data to evaluate or on new data)
    y_pred = model.predict(X_poly)

    # Evaluation
    r2 = r2_score(y, y_pred)
    print(f"R-squared: {r2}")

    return model, poly_features


def regression_on_dominant_factors(factor1_mesh, factor2_mesh, values_mesh, dominant_factors_powers=[[1, 2]]):
    # Initialize an empty list to hold the features
    features = []

    # Generate features based on the specified powers
    for powers in dominant_factors_powers:
        feature = (factor1_mesh.flatten() ** powers[0]) * (factor2_mesh.flatten() ** powers[1])
        features.append(feature)

    # Stack the features horizontally to create a feature matrix
    X = np.column_stack(features)
    y = values_mesh.flatten()

    # Fit the regression model without intercept
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # Predict and evaluate using the dominant features
    y_pred = model.predict(X)

    # Evaluate
    r2 = r2_score(y, y_pred)
    print(f"R-squared (dominant factors): {r2}")

    # Print coefficients for each dominant factor
    for i, coef in enumerate(model.coef_):
        print(f"Coefficient for factor {dominant_factors_powers[i]}: {coef}")

    return model


class regression_model(MyModel):
    def __init__(self, input_dim, output_dim):
        super(regression_model, self).__init__()
        # Get parameters
        self.log_weights = torch.nn.Parameter(-5 * torch.ones(input_dim, output_dim))
        self.log_bias = torch.nn.Parameter(-5 * torch.ones(output_dim))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return torch.matmul(x, torch.exp(self.log_weights)) + torch.exp(self.log_bias)


def trained_regression_with_custom_loss(factor1_mesh, factor2_mesh, values_mesh, degree=3):
    # 0. Define the costum loss. Just redefine l2 loss by hand
    loss_fn = lambda y_pred, y: torch.mean((torch.log(y_pred) - torch.log(y)) ** 2)

    # 1. Flatten the meshes and the target data
    X = np.vstack((factor1_mesh.flatten(), factor2_mesh.flatten())).T
    y = values_mesh.flatten()

    # 2. Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    training_samples_generator = TrainingSamplesGeneratorFromSolutions(X_poly, y)
    batch_size = len(X_poly)

    # 3. Perform the regression
    input_dim = X_poly.shape[1]  # Number of input features
    output_dim = 1  # Predicting a single value
    model = regression_model(input_dim, output_dim)

    # 4. Train the model
    nr_trainsteps = 20000
    output_steps = 1000
    training_output = f"Z Outputs/Regression_with_custom_loss/"
    whole_training(
        model=model,
        training_samples_generator=training_samples_generator,
        optimizer_class=torch.optim.Adam,
        loss_fn=loss_fn,
        initial_lr=0.0001,
        initial_batchsize=batch_size,
        eval_steps=nr_trainsteps + 10,
        max_trainsteps=nr_trainsteps,
        max_batchsize=batch_size,
        output_steps=output_steps,
        output=training_output
    )

    return model


# Old function to plot regression with tries to find a rule
def plot_best_lr_with_regression(output_folder_dir, minmax=True, stddev=False):
    learning_rate_foldername = output_folder_dir + "Learning_rates"

    # Load results
    best_learning_rates = np.load(learning_rate_foldername + "/best_learning_rates.npy")
    list_nr_timesteps = np.load(learning_rate_foldername + "/list_nr_timesteps.npy")
    list_nr_spacediscr = np.load(learning_rate_foldername + "/list_nr_spacediscr.npy")

    timesteps_mesh, spacestep_mesh = np.meshgrid(list_nr_timesteps, list_nr_spacediscr)
    timestepssize_mesh = 1 / timesteps_mesh
    spacestepsize_mesh = 1 / spacestep_mesh
    means = np.mean(best_learning_rates, axis=2)
    std_dev = np.std(best_learning_rates, axis=2)
    min_values = np.min(best_learning_rates, axis=2)
    max_values = np.max(best_learning_rates, axis=2)
    errors_bottom = means - min_values
    errors_top = max_values - means

    # Do my own logistic loss based regression
    log_regr_model = trained_regression_with_custom_loss(timestepssize_mesh, spacestepsize_mesh, means, degree=3)

    # Get coefficients:
    coefficients = np.exp(log_regr_model.log_weights.detach().numpy().flatten())
    intercept = np.exp(log_regr_model.log_bias.detach().numpy().flatten())
    print("Coefficients of log regression:")
    print("Intercept:", intercept)
    for i, coef in enumerate(coefficients):
        print(f"Coef for feature {i}: {coef}")

    # Do full regression
    model, poly_features = regression_for_means(timestepssize_mesh, spacestepsize_mesh, means, degree=3)

    # Accessing the coefficients
    coefficients = model.coef_
    intercept = model.intercept_

    # Generating names for the polynomial features
    feature_names = poly_features.get_feature_names_out(['time_step', 'space_step'])

    # Printing the coefficients with their corresponding names
    print("Intercept:", intercept)
    for coef, name in zip(coefficients, feature_names):
        print(f"{name}: {coef}")

    # Do a regression only with dominant feature
    dominant_factors_powers = [[1, 3], [2, 1]]
    model_dominant = regression_on_dominant_factors(timestepssize_mesh, spacestepsize_mesh, means, dominant_factors_powers=dominant_factors_powers)

    print("Dominant feature intercept:", model_dominant.intercept_)
    print("Dominant feature coefficient:", model_dominant.coef_)

    # Plotting
    colors = sns.color_palette(None, len(list_nr_spacediscr))

    fig_errors, ax_errors = plt.subplots()
    for i, nr_spacediscr in enumerate(list_nr_spacediscr):
        space_step = 1 / nr_spacediscr
        # Actual data points
        ax_errors.scatter(list_nr_timesteps, means[i, :], label=f"{int(nr_spacediscr)} space steps", color=colors[i])

        # Error bars and fill (if applicable)
        if minmax:
            ax_errors.errorbar(list_nr_timesteps, means[i, :], yerr=[errors_bottom[i], errors_top[i]], fmt='none', capsize=5, color=colors[i], alpha=0.5)
            ax_errors.fill_between(list_nr_timesteps, min_values[i, :], max_values[i, :], color=colors[i], alpha=0.3)
        if stddev:
            ax_errors.errorbar(list_nr_timesteps, means[i, :], yerr=std_dev[i, :], fmt='none', capsize=5, color=colors[i], alpha=0.7)
            ax_errors.fill_between(list_nr_timesteps, means[i, :] - std_dev[i, :], means[i, :] + std_dev[i, :], color=colors[i], alpha=0.3)

        # Compute regression line for full regression
        X_pred = np.vstack((timestepssize_mesh[i, :], spacestepsize_mesh[i, :])).T
        X_pred_poly = poly_features.transform(X_pred)
        y_pred = model.predict(X_pred_poly)

        y_pred_log = log_regr_model(X_pred_poly).detach().numpy()

        # Plot the regression lines
        # ax_errors.plot(list_nr_timesteps, y_pred, label=f"{int(nr_spacediscr)} space steps (regression)", color=colors[i], linestyle='--', linewidth=2)
        print(f"Regression line for spacediscr {int(nr_spacediscr)}: {y_pred}")

        ax_errors.plot(list_nr_timesteps, y_pred_log, label=f"{int(nr_spacediscr)} space steps (log regression)", color=colors[i], linestyle='-', linewidth=2)
        print(f"Log Regression line for spacediscr {int(nr_spacediscr)}: {y_pred_log}")

        # Compute regression line for dominant feature
        features = []

        # Generate features based on the specified powers
        for powers in dominant_factors_powers:
            feature = (timestepssize_mesh[i, :] ** powers[0]) * (spacestepsize_mesh[i, :] ** powers[1])
            features.append(feature)

        # Stack the features horizontally to create a feature matrix
        X_pred_dominant = np.column_stack(features)
        y_pred_dominant = model_dominant.predict(X_pred_dominant)

        # Plot the regression line
        # ax_errors.plot(list_nr_timesteps, y_pred_dominant, label=f"Dominant Regression # Spacediscr: {int(nr_spacediscr)}", color=colors[i], linestyle='-', linewidth=2)
        # print(f"Dominant Regression line for spacediscr {int(nr_spacediscr)}: {y_pred_dominant}")

    # # Plotting
    # #create as many colors as there are spacediscr
    # colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(list_nr_spacediscr)))
    #
    # fig_errors, ax_errors = plt.subplots()
    # for i, nr_spacediscr in enumerate(list_nr_spacediscr):
    #     ax_errors.scatter(list_nr_timesteps, means[i,:], label=f"# Spacediscr: {int(nr_spacediscr)}", color=colors[i])
    #     if minmax:
    #         ax_errors.errorbar(list_nr_timesteps, means[i,:], yerr=[errors_bottom[i], errors_top[i]], fmt='none', capsize=5, color=colors[i], alpha=0.5)
    #         ax_errors.fill_between(list_nr_timesteps, min_values[i,:], max_values[i,:], color=colors[i], alpha=0.3)
    #     if stddev:
    #         ax_errors.errorbar(list_nr_timesteps, means[i,:], yerr=std_dev[i,:], fmt='none', capsize=5, color=colors[i], alpha=0.7)
    #         ax_errors.fill_between(list_nr_timesteps, means[i,:] - std_dev[i,:], means[i,:] + std_dev[i,:], color=colors[i], alpha=0.3)

    ax_errors.set_xlabel("Number of time steps for ADANN base model")
    ax_errors.set_ylabel("Approximate optimal learning rate")
    ax_errors.set_yscale("log")
    ax_errors.set_xscale("log")
    ax_errors.set_xticks(list_nr_timesteps)
    ax_errors.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_errors.set_title(f"Optimal learning rates in dependence of number of space and time steps")
    ax_errors.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_errors.savefig(learning_rate_foldername + f"/Best_LR_plot_{'minmax' if minmax else 'stddev'}.pdf", bbox_inches='tight')
    fig_errors.show()