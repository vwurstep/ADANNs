import sys
from matplotlib import pyplot as plt
import torch

sys.path.insert(1, '1_Modules')
sys.path.insert(1, '1_ADANN_Modules')

from training import whole_training, find_optimal_learningrate_with_documentation
from utils import print_to_file_and_console, numpy_to_torch
from ADANNs import AdannModel

def adann_training(
    adann_model: AdannModel,
    trainings_samples_generator,
    optimizer_class,
    loss_fn,
    base_training_kwargs,
    diff_training_kwargs,
    instance_identifier=None,
    output_file=sys.stdout,
    output_folder="",
    validation_input_values=None,
    validation_ref_sol=None,
    lr_search_parameters=(None, None)
):
    '''
        Either initial_learning_rates or lr_search_parameters can be none. If both are none, the function will fail.
    '''
    # Unpack parameters
    lr_search_parameters_base, lr_search_parameters_diff = lr_search_parameters

    # Training base model
    print("------ TRAINING BASE------")
    base_traintime, base_last_loss, initial_learning_rate_base = whole_training(
        model=adann_model.base_model,
        training_samples_generator=trainings_samples_generator,
        optimizer_class=optimizer_class,
        loss_fn=loss_fn,
        **base_training_kwargs,
        output=True,
        lr_search_params=lr_search_parameters_base,
        validation_input_values=validation_input_values,
        validation_ref_sol=validation_ref_sol,
        instance_identifier=f"base_{instance_identifier}",
        output_file=output_file,
        output_folder=output_folder
    )

    # if max_trainsteps_diff == 0:
    if diff_training_kwargs['max_trainsteps'] == 0:
        # adann_model.set_diff_factor(trainings_samples_generator, 1) # I removed this by setting diff_factor to 0.01 in the constructor
        return base_traintime, 0., initial_learning_rate_base, 0.
        
    # Training difference model
    print("------ TRAINING DIFF------")

    # Set diff factor and corresponding training set
    # TODO: Use validation error
    adann_model.set_diff_factor(trainings_samples_generator, 2048)
    diff_training_samples_generator = adann_model.get_diff_training_samples_generator(trainings_samples_generator)

    # Calculate validation reference solution difference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        base_model_outputs = adann_model.base_model(numpy_to_torch(validation_input_values, device))
        validation_ref_sol_diff = 1 / adann_model.diff_factor * (numpy_to_torch(validation_ref_sol, device) - base_model_outputs)

    diff_traintime, diff_last_loss, initial_learning_rate_diff = whole_training(
        model=adann_model.diff_model,
        training_samples_generator=diff_training_samples_generator,
        optimizer_class=optimizer_class,
        loss_fn=loss_fn,
        **diff_training_kwargs,
        output=True,
        lr_search_params=lr_search_parameters_diff,
        validation_input_values=validation_input_values,
        validation_ref_sol=validation_ref_sol_diff,
        instance_identifier=f"diff_{instance_identifier}",
        output_file=output_file,
        output_folder=output_folder
    )

    return base_traintime, diff_traintime, initial_learning_rate_base, initial_learning_rate_diff
