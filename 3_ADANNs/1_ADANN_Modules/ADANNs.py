import sys
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import expm

sys.path.insert(1, '1_Modules')

from operator_learning_models import MyModel, ANNModel
from evaluation_utils import evaluate
from ode_methods import SemiLinearODE, second_order_lirk
from utils import numpy_to_torch, nr_trainable_params
from training_samples_generators import TrainingSamplesGeneratorFromSolutions
from PDE_operations import fdm_laplace_operator_periodic, first_order_diff_matrix_trans

class AdannBaseModel(MyModel):
    """
      Abstract base class for ADANN base models
    """

    def __init__(self):
        super().__init__()
        self.base_model_algorithm_name = None


class AdannModel(MyModel):
    """
      Abstract base class for ADANN models
    """

    def __init__(self, base_model=None, diff_model=None, diff_factor=0.01):
        super().__init__()
        self.base_model = base_model  # Expected to be an instance of AdannBaseModel
        self.diff_model = diff_model  # Expected to be an instance of MyModel
        self.diff_factor = diff_factor
        self.modelname = f"AdannModel with \n    base_model: {base_model.modelname}\n\n    diff_model: {diff_model.modelname}"

    def forward(self, input_values):
        return self.base_model(input_values) + self.diff_factor * self.diff_model(input_values)

    def set_diff_factor(self, training_samples_generator, nr_samples):
        input_values, ref_sol = training_samples_generator.generate(nr_samples)
        evaluate(self.base_model, input_values, ref_sol, space_size=1, dim=1, loss_fn=torch.nn.MSELoss(), train_or_test="train")
        self.diff_factor = np.sqrt(self.base_model.evaluations[-1]["loss_value"])

    def get_diff_training_samples_generator(self, training_samples_generator):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            base_model_outputs = self.base_model(training_samples_generator.dataset.input_values_torch.to(device))
            scaled_difference_to_ref_sols = 1 / self.diff_factor * (training_samples_generator.dataset.ref_solutions_torch.to(device) - base_model_outputs)
        diff_training_samples_generator = TrainingSamplesGeneratorFromSolutions(training_samples_generator.dataset.input_values_torch, scaled_difference_to_ref_sols)
        return diff_training_samples_generator


class AbstractLirkBasemodel(AdannBaseModel):
    '''
        Abstract base class for LIRK based ADANN base models
        Implements a model with steps of the form
        U_{m+1} = W_{m, 1} U_{m} + W_{m, 2} f(U_{m}) + W_{m, 3} f(W_{m, 4}U_m + W_{m, 5}f(U_m))
    '''

    def __init__(self, depth, nonlin, w_1_inits, w_2_inits, w_3_inits, w_4_inits, w_5_inits, scale_factor=1., inner_scale_factor=1.):

        super().__init__()
        self.modelname = f"AbstractLirkBasemodel"
        self.depth = depth
        self.nonlin = nonlin
        self.scale_factor = scale_factor
        self.inner_scale_factor = inner_scale_factor

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.w_1_init = [torch.from_numpy((1/scale_factor) * w_1_inits[m]).float().to(device) for m in range(depth)]
        self.w_2_init = [torch.from_numpy((1/scale_factor) * w_2_inits[m]).float().to(device) for m in range(depth)]
        self.w_3_init = [torch.from_numpy((1/scale_factor) * w_3_inits[m]).float().to(device) for m in range(depth)]
        self.w_4_init = [torch.from_numpy(inner_scale_factor * (1/scale_factor) * w_4_inits[m]).float().to(device) for m in range(depth)]
        self.w_5_init = [torch.from_numpy(inner_scale_factor * (1/scale_factor) * w_5_inits[m]).float().to(device) for m in range(depth)]

        # print("At init: self.w_1_init[0]", self.w_1_init[0])

        self.w_1_weights = nn.ParameterDict()
        self.w_2_weights = nn.ParameterDict()
        self.w_3_weights = nn.ParameterDict()
        self.w_4_weights = nn.ParameterDict()
        self.w_5_weights = nn.ParameterDict()

        for m in range(self.depth):
            self.w_1_weights[str(m)] = nn.Parameter(self.w_1_init[m].clone().detach(), requires_grad=True)
            self.w_2_weights[str(m)] = nn.Parameter(self.w_2_init[m].clone().detach(), requires_grad=True)
            self.w_3_weights[str(m)] = nn.Parameter(self.w_3_init[m].clone().detach(), requires_grad=True)
            self.w_4_weights[str(m)] = nn.Parameter(self.w_4_init[m].clone().detach(), requires_grad=True)
            self.w_5_weights[str(m)] = nn.Parameter(self.w_5_init[m].clone().detach(), requires_grad=True)

    def forward(self, initial_values):
        '''
            initial_values: [batch_size, nr_spacediscr]
        '''

        u = initial_values
        for m in range(self.depth):
            inner_stage = (1/self.inner_scale_factor) * self.scale_factor * (
                    torch.matmul(u, self.w_4_weights[str(m)]) +
                    torch.matmul(self.nonlin(u), self.w_5_weights[str(m)])
            )
            u = self.scale_factor * (
                torch.matmul(u, self.w_1_weights[str(m)]) +
                torch.matmul(self.nonlin(u), self.w_2_weights[str(m)]) +
                torch.matmul(self.nonlin(inner_stage), self.w_3_weights[str(m)])
            )
        return u

    def restore_initialization(self):
        with torch.no_grad():
            # print("At restore: self.w_1_init[0]", self.w_1_init[0])
            # print("At restore: self.w_1_weights[str(0)]", self.w_1_weights[str(0)])
            for m in range(self.depth):
                self.w_1_weights[str(m)].copy_(self.w_1_init[m])
                self.w_2_weights[str(m)].copy_(self.w_2_init[m])
                self.w_3_weights[str(m)].copy_(self.w_3_init[m])
                self.w_4_weights[str(m)].copy_(self.w_4_init[m])
                self.w_5_weights[str(m)].copy_(self.w_5_init[m])

        self.done_trainsteps = 0
        self.batch_sizes = []
        self.learning_rates = []
        self.optimizers = []
        self.evaluations = []


def param_transformer(params, nr_timesteps):
    if params is None:
        params = [[0.5], [0.5]]

    if not isinstance(params[0], list):
        p1s = [params[0] for _ in range(nr_timesteps)]
        p2s = [params[1] for _ in range(nr_timesteps)]
    elif len(params[0]) == 1:
        p1s = [params[0][0] for _ in range(nr_timesteps)]
        p2s = [params[1][0] for _ in range(nr_timesteps)]
    else:
        p1s = params[0]
        p2s = params[1]

    return p1s, p2s

class SecondOrderLirkAdannBasemodel(AbstractLirkBasemodel):
    """
        ADANN base model based on second order LIRK for semilinear PDEs
    """

    def __init__(self, T, semilinear_ode : SemiLinearODE, nr_timesteps, params, exp_init = False, scale=False):
        self.modelname = f"SecondOrderLirkAdannBasemodel with \n    T : {T}\n    nr_timesteps: {nr_timesteps}\n    params: {params}\n    exp_init:{exp_init}\n    semilinear_ode: {semilinear_ode.ode_name}\n    scale: {scale}"

        # Init quantities
        self.T = float(T)
        self.semilinear_ode = semilinear_ode
        self.nr_timesteps = nr_timesteps
        self.nr_spacediscr = semilinear_ode.space_dim

        nr_spacediscr = self.nr_spacediscr
        time_step = self.T / nr_timesteps
        # scale_factor = max(time_step ** 3, time_step * (1. / self.nr_spacediscr ** 2)) if scale else 1.
        inner_scale_factor = time_step

        p1s, p2s = param_transformer(params, nr_timesteps)

        linear_operator = semilinear_ode.Linear_operator
        if isinstance(linear_operator, torch.Tensor):
            linear_operator = linear_operator.cpu().numpy()

        implicit_flow_trans = [np.transpose(np.linalg.inv(np.eye(nr_spacediscr) - time_step * p2s[m] * linear_operator)) for m in range(nr_timesteps)]
        operator_trans = np.transpose(linear_operator)

        # Init variables
        if exp_init:
            raise NotImplementedError
            # linear_flow_1_inits = [np.transpose(expm(time_step * linear_operator)) for _ in range(nr_timesteps)]
            # nonlinear_flow_1_inits = [np.transpose(np.zeros_like(linear_operator)) for _ in range(nr_timesteps)]
            # nonlinear_flow_2_inits = [time_step * np.transpose(expm(time_step / 2 * linear_operator)) for _ in range(nr_timesteps)]
            # linear_flow_2_inits = [np.transpose(expm(time_step / 2 * linear_operator)) for _ in range(nr_timesteps)]
            # nonlinear_flow_3_inits = [time_step / 2 * np.transpose(expm(time_step / 2 * linear_operator)) for _ in range(nr_timesteps)]
        else:
            linear_flow_1_inits = [(np.matmul(np.eye(nr_spacediscr) + time_step * (1. - p2s[m]) * operator_trans, implicit_flow_trans[m]) +
                                   time_step * time_step * (0.5 - p2s[m]) * np.matmul(np.matmul(operator_trans, implicit_flow_trans[m]), np.matmul(operator_trans, implicit_flow_trans[m]))) for m in range(nr_timesteps)]
            nonlinear_flow_1_inits = [(time_step * ((1. - 1. / (2 * p1s[m])) * implicit_flow_trans[m] +
                                      time_step * (0.5 - p2s[m]) * np.matmul(implicit_flow_trans[m], np.matmul(operator_trans, implicit_flow_trans[m])))) for m in range(nr_timesteps)]
            nonlinear_flow_2_inits = [(time_step / (2 * p1s[m]) * implicit_flow_trans[m]) for m in range(nr_timesteps)]
            linear_flow_2_inits = [(np.matmul(np.eye(nr_spacediscr) + time_step * (p1s[m] - p2s[m]) * operator_trans, implicit_flow_trans[m])) for m in range(nr_timesteps)]
            nonlinear_flow_3_inits = [(time_step * p1s[m] * implicit_flow_trans[m]) for m in range(nr_timesteps)]

        super().__init__(nr_timesteps, self.semilinear_ode.nonlin, linear_flow_1_inits, nonlinear_flow_1_inits, nonlinear_flow_2_inits, linear_flow_2_inits, nonlinear_flow_3_inits, scale_factor=1., inner_scale_factor=inner_scale_factor)


class SecondOrderLirkFDMPeriodicSemilinearPDEAdannBasemodel(SecondOrderLirkAdannBasemodel):
    """
        ADANN base model based on second order LIRK and FDM for semilinear PDEs
        Only dim=1 or dim=2 supported
    """

    def __init__(self, T, laplace_factor, nonlin, space_size, nr_spacediscr, nr_timesteps, dim=1, exp_init=False, nonlin_name="no name", params=None, scale=False):
        self.dim = dim

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim)
        operator_torch = numpy_to_torch(operator, device)
        semilinear_ode = SemiLinearODE(operator_torch, nonlin)

        super().__init__(T, semilinear_ode, nr_timesteps, params, exp_init, scale=scale)
        self.modelname = f"SecondOrderLirkFDMPeriodicSemilinearPDEAdannBasemodel with \n    T : {T}\n    nr_timesteps: {nr_timesteps}\n    params: {params}\n    exp_init:{exp_init}\n    nonlin: {nonlin_name}\n    laplace_factor: {laplace_factor}\n    nonlin: {nonlin}\n    space_size: {space_size}\n    dim: {dim}\n    scale: {scale}"


class SecondOrderLirkFDMPeriodicBurgersAdannBasemodel(SecondOrderLirkAdannBasemodel):
    """
        ADANN base model based on second order LIRK and FDM for Burgers PDE
        Based on burger_fdm_lirk
    """

    def __init__(self, T, laplace_factor, space_size, nr_spacediscr, nr_timesteps, diff_version = 2, conservative=True, exp_init=False, params=None, scale=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        diff_matrix_trans_torch = numpy_to_torch(first_order_diff_matrix_trans(nr_spacediscr, space_size, diff_version), device)

        if conservative:
            nonlin = lambda u: - 0.5 * torch.matmul(u * u, diff_matrix_trans_torch)
        else:
            nonlin = lambda u: - u * (torch.matmul(u, diff_matrix_trans_torch))

        operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1)
        operator_torch = numpy_to_torch(operator, device)

        semilinear_ode = SemiLinearODE(operator_torch, nonlin)

        super().__init__(T, semilinear_ode, nr_timesteps, params, exp_init, scale=scale)
        self.modelname = f"SecondOrderLirkFDMPeriodicBurgersAdannBasemodel with \n    T : {T}\n    nr_timesteps: {nr_timesteps}\n    params: {params}\n    exp_init:{exp_init}\n    laplace_factor: {laplace_factor}\n    space_size: {space_size}\n    dim: {1}\n    diff_version: {diff_version}\n    conservative: {conservative}\n    scale: {scale}"

class SecondOrderLirkFDMPeriodicBurgersAdannBasemodel_learnFirstOrder(AbstractLirkBasemodel):
    """
        ADANN base model based on second order LIRK and FDM for Burgers PDE where the first order diff matrix is taken into the learnable parameters
    """

    def __init__(self, T, laplace_factor, space_size, nr_spacediscr, nr_timesteps, diff_version=2, params=None, scale=False):

        self.T = float(T)
        self.nr_timesteps = nr_timesteps
        self.nr_spacediscr = nr_spacediscr

        time_step = self.T / nr_timesteps
        scale_factor = max(time_step ** 3, time_step * (1. / self.nr_spacediscr ** 2)) if scale else 1.

        p1s, p2s = param_transformer(params, nr_timesteps)

        nonlin = lambda u: - 0.5 * u * u

        operator_trans = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1)
        implicit_flow_trans = [np.transpose(np.linalg.inv(np.eye(nr_spacediscr) - time_step * p2s[m] * operator_trans)) for m in range(nr_timesteps)]
        diff_matrix_trans = first_order_diff_matrix_trans(nr_spacediscr, space_size, diff_version)

        linear_flow_1_inits = [(np.matmul(np.eye(nr_spacediscr) + time_step * (1. - p2s[m]) * operator_trans, implicit_flow_trans[m]) +
                                time_step * time_step * (0.5 - p2s[m]) * np.matmul(np.matmul(operator_trans, implicit_flow_trans[m]), np.matmul(operator_trans, implicit_flow_trans[m])))
                               for m in range(nr_timesteps)]
        nonlinear_flow_1_inits = [np.matmul(diff_matrix_trans,
                                            time_step * ((1. - 1. / (2 * p1s[m])) * implicit_flow_trans[m] +
                                            time_step * (0.5 - p2s[m]) * np.matmul(implicit_flow_trans[m], np.matmul(operator_trans, implicit_flow_trans[m]))))
                                  for m in range(nr_timesteps)]
        nonlinear_flow_2_inits = [np.matmul(diff_matrix_trans,
                                            time_step / (2 * p1s[m]) * implicit_flow_trans[m])
                                  for m in range(nr_timesteps)]
        linear_flow_2_inits = [(np.matmul(np.eye(nr_spacediscr) + time_step * (p1s[m] - p2s[m]) * operator_trans, implicit_flow_trans[m]))
                               for m in range(nr_timesteps)]
        nonlinear_flow_3_inits = [np.matmul(diff_matrix_trans,
                                            time_step * p1s[m] * implicit_flow_trans[m])
                                  for m in range(nr_timesteps)]

        super().__init__(nr_timesteps, nonlin, linear_flow_1_inits, nonlinear_flow_1_inits, nonlinear_flow_2_inits, linear_flow_2_inits, nonlinear_flow_3_inits, scale_factor=scale_factor)
        self.modelname = f"SecondOrderLirkFDMPeriodicBurgersAdannBasemodel_learnFirstOrder with \n    T : {T}\n    nr_timesteps: {nr_timesteps}\n    params: {params}\n    laplace_factor: {laplace_factor}\n    space_size: {space_size}\n    dim: {1}\n    diff_version: {diff_version}\n    scale: {scale}"

# Based on reaction_diffusion_pde_fdm_lirk
class SecondOrderLirkFDMPeriodicReactionDiffusionAdannBasemodel(AdannBaseModel):
    """
        ADANN base model based on second order LIRK and FDM for reaction diffusion PDE
        Based on reaction_diffusion_pde_fdm_lirk
        Check Ipad notes on RD PDE for derivation
    """

    def __init__(self, T, laplace_factor, reaction_nonlin, space_size, nr_spacediscr, nr_timesteps, dim=1, exp_init=False, nonlin_name="no name", params=None, scale=False):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.modelname = f"SecondOrderLirkFDMPeriodicReactionDiffusionAdannBasemodel with \n    T : {T}\n    nr_timesteps: {nr_timesteps}\n    params: {params}\n    exp_init:{exp_init}\n    nonlin_name: {nonlin_name}\n    laplace_factor: {laplace_factor}\n    space_size: {space_size}\n    dim: {dim}\n    scale: {scale}"

        # Init quantities
        self.T = float(T)
        self.reaction_nonlin = reaction_nonlin
        self.nr_spacediscr = nr_spacediscr
        self.nr_timesteps = nr_timesteps
        time_step = self.T / nr_timesteps
        mesh_step = space_size / nr_spacediscr

        self.scale_factor = time_step if scale else 1.

        p1s, p2s = param_transformer(params, nr_timesteps)

        second_order_diff_quot = np.array([-2, 1] + [0 for _ in range(nr_spacediscr - 3)] + [1])
        operator = laplace_factor / mesh_step / mesh_step * np.stack(
            [np.roll(second_order_diff_quot, n) for n in range(nr_spacediscr)])

        implicit_flow_trans = [np.transpose(np.linalg.inv(np.eye(self.nr_spacediscr) - time_step * p2s[m] * operator)) for m in range(nr_timesteps)]
        operator_trans = np.transpose(operator)

        # Init variables
        if exp_init:
            raise NotImplementedError
            # linear_flow_1_inits = [np.transpose(expm(time_step * operator)) for _ in range(nr_timesteps)]
            # nonlinear_flow_1_inits = [np.transpose(np.zeros_like(operator)) for _ in range(nr_timesteps)]
            # nonlinear_flow_2_inits = [time_step * np.transpose(expm(time_step / 2 * operator)) for _ in
            #                           range(nr_timesteps)]
            # linear_flow_2_inits = [np.transpose(expm(time_step / 2 * operator)) for _ in range(nr_timesteps)]
            # nonlinear_flow_3_inits = [time_step / 2 * np.transpose(expm(time_step / 2 * operator)) for _ in
            #                           range(nr_timesteps)]
        else:
            linear_flow_1_inits = [(np.matmul(np.eye(nr_spacediscr) + time_step * (1. - p2s[m]) * operator_trans, implicit_flow_trans[m]) +
                                   time_step * time_step * (0.5 - p2s[m]) * np.matmul(np.matmul(operator_trans, implicit_flow_trans[m]), np.matmul(operator_trans, implicit_flow_trans[m])))
                                   for m in range(nr_timesteps)]
            nonlinear_flow_1_inits = [(time_step * ((1. - 1. / (2 * p1s[m])) * implicit_flow_trans[m] +
                                      time_step * (0.5 - p2s[m]) * np.matmul(implicit_flow_trans[m], np.matmul(operator_trans, implicit_flow_trans[m]))))
                                      for m in range(nr_timesteps)]
            nonlinear_flow_2_inits = [(time_step / (2 * p1s[m]) * implicit_flow_trans[m]) for m in range(nr_timesteps)]
            linear_flow_2_inits = [(np.matmul(np.eye(nr_spacediscr) + time_step * (p1s[m] - p2s[m]) * operator_trans, implicit_flow_trans[m])) for m in range(nr_timesteps)]
            nonlinear_flow_3_inits = [(time_step * p1s[m] * implicit_flow_trans[m]) for m in range(nr_timesteps)]

        self.linear_flow_1_inits = [torch.from_numpy(linear_flow_1_inits[m]).float().to(device) for m in range(nr_timesteps)]
        self.linear_flow_source_1_inits = [torch.from_numpy(nonlinear_flow_1_inits[m] + nonlinear_flow_2_inits[m]).float().to(device) for m in range(nr_timesteps)]
        self.nonlinear_flow_1_inits = [torch.from_numpy(nonlinear_flow_1_inits[m]).float().to(device) for m in range(nr_timesteps)]
        self.nonlinear_flow_2_inits = [torch.from_numpy(nonlinear_flow_2_inits[m]).float().to(device) for m in range(nr_timesteps)]
        self.linear_flow_2_inits = [torch.from_numpy(linear_flow_2_inits[m]).float().to(device) for m in range(nr_timesteps)]
        self.linear_flow_source_2_inits = [torch.from_numpy(nonlinear_flow_3_inits[m]).float().to(device) for m in range(nr_timesteps)]
        self.nonlinear_flow_3_inits = [torch.from_numpy(nonlinear_flow_3_inits[m]).float().to(device) for m in range(nr_timesteps)]

        self.linear_flow_1_weights = nn.ParameterDict()
        self.linear_flow_source_1_weights = nn.ParameterDict()
        self.nonlinear_flow_1_weights = nn.ParameterDict()
        self.nonlinear_flow_2_weights = nn.ParameterDict()
        self.linear_flow_2_weights = nn.ParameterDict()
        self.linear_flow_source_2_weights = nn.ParameterDict()
        self.nonlinear_flow_3_weights = nn.ParameterDict()

        for m in range(self.nr_timesteps):
            self.linear_flow_1_weights[str(m)] = nn.Parameter(self.linear_flow_1_inits[m].clone().detach(), requires_grad=True)
            self.linear_flow_source_1_weights[str(m)] = nn.Parameter(self.linear_flow_source_1_inits[m].clone().detach(), requires_grad=True)
            self.nonlinear_flow_1_weights[str(m)] = nn.Parameter(self.nonlinear_flow_1_inits[m].clone().detach(), requires_grad=True)
            self.nonlinear_flow_2_weights[str(m)] = nn.Parameter(self.nonlinear_flow_2_inits[m].clone().detach(), requires_grad=True)
            self.linear_flow_2_weights[str(m)] = nn.Parameter(self.linear_flow_2_inits[m].clone().detach(), requires_grad=True)
            self.linear_flow_source_2_weights[str(m)] = nn.Parameter(self.linear_flow_source_2_inits[m].clone().detach(), requires_grad=True)
            self.nonlinear_flow_3_weights[str(m)] = nn.Parameter(self.nonlinear_flow_3_inits[m].clone().detach(), requires_grad=True)

    def forward(self, source_terms):
        '''
            source_terms: [batch_size, nr_spacediscr]
        '''

        u = torch.zeros(source_terms.shape, device=source_terms.device)
        for m in range(self.nr_timesteps):
            inner_stage = (
                    torch.matmul(u, self.linear_flow_2_weights[str(m)]) +
                    torch.matmul(source_terms, self.linear_flow_source_2_weights[str(m)]) +
                    torch.matmul(self.reaction_nonlin(u), self.nonlinear_flow_3_weights[str(m)])
            )
            u = (
                torch.matmul(u, self.linear_flow_1_weights[str(m)]) +
                torch.matmul(source_terms, self.linear_flow_source_1_weights[str(m)]) +
                torch.matmul(self.reaction_nonlin(u), self.nonlinear_flow_1_weights[str(m)]) +
                torch.matmul(self.reaction_nonlin(inner_stage), self.nonlinear_flow_2_weights[str(m)])
            )
        return u

    def restore_initialization(self):
        with torch.no_grad():
            for m in range(self.nr_timesteps):
                self.linear_flow_1_weights[str(m)].copy_(self.linear_flow_1_inits[m])
                self.linear_flow_source_1_weights[str(m)].copy_(self.linear_flow_source_1_inits[m])
                self.nonlinear_flow_1_weights[str(m)].copy_(self.nonlinear_flow_1_inits[m])
                self.nonlinear_flow_2_weights[str(m)].copy_(self.nonlinear_flow_2_inits[m])
                self.linear_flow_2_weights[str(m)].copy_(self.linear_flow_2_inits[m])
                self.linear_flow_source_2_weights[str(m)].copy_(self.linear_flow_source_2_inits[m])
                self.nonlinear_flow_3_weights[str(m)].copy_(self.nonlinear_flow_3_inits[m])

        self.done_trainsteps = 0
        self.batch_sizes = []
        self.learning_rates = []
        self.optimizers = []
        self.evaluations = []

def make_adann_creator(ADANN_base_model_class, base_model_kwargs, diff_model_class, diff_model_params):
    """
      Returns a function which can create ADANNs for different parameters
    """
    def create_adann(params):
        """
        Creates an ADANN for given parameters
        """
        base_model_kwargs_with_params = base_model_kwargs.copy()
        base_model_kwargs_with_params["params"] = params

        # Create base model
        base_model = ADANN_base_model_class(**base_model_kwargs_with_params)

        # Create difference model
        diff_model = diff_model_class(*diff_model_params)

        # Create ADANN
        adann = AdannModel(base_model, diff_model)

        # Store ADANN on device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        adann.to(device)

        return adann

    return create_adann


def make_adam_adann_creator_from_basemodel(diff_model_class, diff_model_params):
    """
      Returns a function which can create ADANNs based on a base model
    """
    def create_adann(base_model):
        """
        Creates an ADANN based on a base model
        """
        # Create difference model
        diff_model = diff_model_class(*diff_model_params)

        # Create ADANN
        adann = AdannModel(base_model, diff_model)

        # Store ADANN on device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        adann.to(device)

        return adann

    return create_adann

if __name__ == '__main__':
    from random_function_generators import RandnFourierSeriesGenerator
    from semilinear_heat_multi_d_classical_methods import periodic_semilinear_pde_fdm_lirk
    from burgers_classical_methods import burger_fdm_lirk
    from RD_classical_methods import reaction_diffusion_pde_fdm_lirk
    from documentation_utils import plot_reference_solutions
    from PDE_operations import *

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)


    ###################################################
    print("\nTest SecondOrderLirkAdannBasemodel against second_order_lirk method")
    T = 2.
    nonlin = lambda x: torch.sin(x)
    nonlin_name = "Sine"
    dim = 1

    nr_spacediscr = 30
    linear_operator = 1/100 * torch.randn(nr_spacediscr, nr_spacediscr)
    semilinear_ode = SemiLinearODE(linear_operator, nonlin, nonlin_name)

    # Create ADANN base model
    nr_timesteps = 10
    p1 = 0.4
    p2 = 0.4
    exp_init = False
    base_model = SecondOrderLirkAdannBasemodel(T, semilinear_ode, nr_timesteps, [[p1], [p2]], exp_init, scale=False)

    # initial value
    var = 5000
    decay_rate = 2
    offset = np.power(var, 1 / decay_rate)
    inner_decay = 1.
    initial_value_generator = RandnFourierSeriesGenerator([var, decay_rate, offset, inner_decay, 1, dim])

    nr_samples = 100
    initial_values = initial_value_generator.generate(nr_samples, nr_spacediscr)
    initial_values_torch = torch.from_numpy(initial_values).float()

    # Compare outputs
    base_model_output = base_model(initial_values_torch)
    method_output = second_order_lirk(T, semilinear_ode, initial_values_torch, nr_timesteps, [p1, p2])

    print(f"base_model_output.shape: {base_model_output.shape}")
    print(f"method_output.shape: {method_output.shape}")
    # print(f"base_model_output: {base_model_output}")
    # print(f"method_output: {method_output}")

    # Evaluate squarred difference
    error = torch.sum((base_model_output - method_output) ** 2)
    print(f"SecondOrderLirkAdannBasemodel deviation: {error}")

    print(f"Number of trainable params: {nr_trainable_params(base_model)}")

    ###################################################
    print("\nTest SecondOrderLirkFDMPeriodicBurgersAdannBasemodel and SecondOrderLirkFDMPeriodicBurgersAdannBasemodel_learnFirstOrder against burger_fdm_lirk method")
    T = 1.
    space_size = 2 * np.pi
    laplace_factor = 0.1
    dim = 1

    # Create ADANN base model
    nr_spacediscr = 128
    nr_timesteps = 10
    p1 = 0.4
    p2 = 0.4
    exp_init = False
    diff_version = 2
    conservative = True #If this is false, then I have a problem for learnFirstOrder
    base_model_1 = SecondOrderLirkFDMPeriodicBurgersAdannBasemodel(T, laplace_factor, space_size, nr_spacediscr, nr_timesteps, diff_version, conservative, exp_init, [[p1], [p2]])
    base_model_2 = SecondOrderLirkFDMPeriodicBurgersAdannBasemodel_learnFirstOrder(T, laplace_factor, space_size, nr_spacediscr, nr_timesteps, diff_version, [[p1], [p2]])

    # initial value
    var = 100
    decay_rate = 2
    offset = np.power(var, 1 / decay_rate)
    inner_decay = 1.
    initial_value_generator = RandnFourierSeriesGenerator([var, decay_rate, offset, inner_decay, space_size, 1])

    nr_samples = 100
    initial_values = initial_value_generator.generate(nr_samples, nr_spacediscr)
    initial_values_torch = torch.from_numpy(initial_values).float()

    # Compare outputs
    base_model_output_1 = base_model_1(initial_values_torch)
    base_model_output_2 = base_model_2(initial_values_torch)
    method_output = burger_fdm_lirk(initial_values_torch, T, laplace_factor, space_size, nr_timesteps, diff_version, conservative, [p1, p2])

    print(f"base_model_output_1.shape: {base_model_output_1.shape}")
    print(f"base_model_output_2.shape: {base_model_output_2.shape}")
    print(f"method_output.shape: {method_output.shape}")
    # print(f"base_model_output: {base_model_output}")
    # print(f"method_output: {method_output}")

    # Evaluate squarred difference
    error_1 = torch.sum((base_model_output_1 - method_output) ** 2)
    error_2 = torch.sum((base_model_output_2 - method_output) ** 2)
    print(f"SecondOrderLirkFDMPeriodicBurgersAdannBasemodel deviation: {error_1}")
    print(f"SecondOrderLirkFDMPeriodicBurgersAdannBasemodel_learnFirstOrder deviation: {error_2}")

    print(f"Number of trainable params for SecondOrderLirkFDMPeriodicBurgersAdannBasemodel: {nr_trainable_params(base_model_1)}")
    print(f"Number of trainable params for SecondOrderLirkFDMPeriodicBurgersAdannBasemodel_learnFirstOrder: {nr_trainable_params(base_model_2)}")



    ###################################################
    print("\nTest SecondOrderLirkFDMPeriodicReactionDiffusionAdannBasemodel against reaction_diffusion_pde_fdm_lirk method")
    T = 1.
    space_size = 1.
    laplace_factor = 0.05
    reaction_rate = 2.
    reaction_nonlin = lambda u: reaction_rate * (u - u ** 3)
    nonlin_name = "u2"
    dim = 1  # Dimension can currently only be 1

    # Create ADANN base model
    nr_spacediscr = 128
    nr_timesteps = 10
    p1 = 0.4
    p2 = 0.4
    exp_init = False
    scale = True
    base_model = SecondOrderLirkFDMPeriodicReactionDiffusionAdannBasemodel(T, laplace_factor, reaction_nonlin, space_size, nr_spacediscr, nr_timesteps, dim, exp_init, nonlin_name, [[p1], [p2]], scale=scale)

    # Source Terms
    var = 5000
    decay_rate = 2
    offset = np.power(var, 1 / decay_rate)
    inner_decay = 1.
    source_term_generator = RandnFourierSeriesGenerator([var, decay_rate, offset, inner_decay, space_size, dim])

    nr_samples = 100
    source_terms = source_term_generator.generate(nr_samples, nr_spacediscr)
    source_terms_torch = torch.from_numpy(source_terms).float()

    # Compare outputs
    base_model_output = base_model(source_terms_torch)
    method_output = reaction_diffusion_pde_fdm_lirk(source_terms_torch, T, laplace_factor, reaction_nonlin, space_size, nr_timesteps, dim, [p1, p2])

    print(f"base_model_output.shape: {base_model_output.shape}")
    print(f"method_output.shape: {method_output.shape}")
    # print(f"base_model_output: {base_model_output}")
    # print(f"method_output: {method_output}")

    plot_reference_solutions(source_terms_torch.detach().numpy(), base_model_output.detach().numpy(), 2, dim, x_values_periodic, space_size)
    plot_reference_solutions(source_terms_torch.detach().numpy(), method_output.detach().numpy(), 2, dim, x_values_periodic, space_size)

    # Evaluate squarred difference
    error = torch.sum((base_model_output - method_output) ** 2)
    print(f"SecondOrderLirkFDMPeriodicReactionDiffusionAdannBasemodel deviation: {error}")

    print(f"Number of trainable params: {nr_trainable_params(base_model)}")

    ###################################################
    print("\nTest SecondOrderLirkFDMPeriodicSemilinearPDEAdannBasemodel against periodic_semilinear_pde_fdm_lirk method")
    T = 2.
    space_size = 1.
    laplace_factor = 0.01
    nonlin = lambda x: torch.sin(x)
    nonlin_name = "Sine"
    dim = 1

    # Create ADANN base model
    nr_spacediscr = 30
    nr_timesteps = 10
    p1 = 0.4
    p2 = 0.4
    exp_init = False
    base_model = SecondOrderLirkFDMPeriodicSemilinearPDEAdannBasemodel(T, laplace_factor, nonlin, space_size,
                                                                       nr_spacediscr, nr_timesteps, dim, exp_init,
                                                                       nonlin_name, [[p1], [p2]])
    # initial value
    var = 5000
    decay_rate = 2
    offset = np.power(var, 1 / decay_rate)
    inner_decay = 1.
    initial_value_generator = RandnFourierSeriesGenerator([var, decay_rate, offset, inner_decay, space_size, dim])
    nr_samples = 100
    initial_values = initial_value_generator.generate(nr_samples, nr_spacediscr)
    initial_values_torch = torch.from_numpy(initial_values).float()

    # Compare outputs
    base_model_output = base_model(initial_values_torch)
    method_output = periodic_semilinear_pde_fdm_lirk(initial_values_torch, T, laplace_factor, nonlin, space_size, nr_timesteps, dim=1, params=[p1, p2])

    print(f"base_model_output.shape: {base_model_output.shape}")
    print(f"method_output.shape: {method_output.shape}")
    # print(f"base_model_output: {base_model_output}")
    # print(f"method_output: {method_output}")

    # Evaluate squarred difference
    error = torch.sum((base_model_output - method_output) ** 2)
    print(f"SecondOrderLirkFDMPeriodicSemilinearPDEAdannBasemodel deviation: {error}")

    print(f"Number of trainable params: {nr_trainable_params(base_model)}")

    ###################################################
    print("\nTest make_adann_creator")
    ADANN_base_model_class = SecondOrderLirkFDMPeriodicSemilinearPDEAdannBasemodel
    base_model_kwargs = {"T": T, "laplace_factor": laplace_factor, "nonlin": nonlin, "space_size": space_size,
                         "nr_spacediscr": nr_spacediscr, "nr_timesteps": nr_timesteps, "nonlin_name": nonlin_name}

    diff_model_class = ANNModel
    diff_model_params = [[nr_spacediscr, 100, 200, 150, nr_spacediscr]]

    adann_creator = make_adann_creator(ADANN_base_model_class, base_model_kwargs, diff_model_class, diff_model_params)

    # Create ADANN
    adann_model = adann_creator([[p1], [p2]])

    # Compare outputs
    adann_output = adann_model(initial_values_torch)

    # Evaluate squarred difference
    error = torch.sum((base_model_output - adann_output) ** 2)
    print(f"Created adann deviation: {error}")



