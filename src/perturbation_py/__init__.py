"""perturbation_py — Perturbation-based solution of DSGE models up to third order.

This package provides solvers, simulation tools, and diagnostic utilities for
Dynamic Stochastic General Equilibrium (DSGE) models using the perturbation
method. Solutions are computed via generalized Schur (QZ) decomposition at
first order, and Sylvester matrix equations at second and third order.

Key references:
    Blanchard and Kahn (1980), Econometrica 48(5).
    Schmitt-Grohe and Uribe (2004), JEDC 28, 755-775.
    Kim, Kim, Schaumburg, and Sims (2008), JEDC 32(11).
    Andreasen, Fernandez-Villaverde, and Rubio-Ramirez (2018), REStud 85(1).
"""

from .benchmarks import (
    DynareFixtureReport,
    LinearRESpec,
    ParityReport,
    ReferenceComparison,
    SecondOrderParityReport,
    compare_policy_to_dynare_reference,
    compare_first_order_to_dynare,
    compare_second_order_to_dynare,
    compare_with_dynare_fixture,
    dynare_available,
    run_dynare_mod_file,
)
from .derivatives import Jacobians, compute_jacobians
from .io import (
    DynareResults,
    IRFComparisonReport,
    compare_reconstructed_irfs_to_dynare,
    DynareModSpec,
    DynareReference,
    load_dynare_reference,
    load_dynare_results,
    parse_dynare_mod_file,
    results_mat_path_for_mod,
)
from .model import DSGEModel
from .model_hessians import ModelHessians, ModelThirdDerivatives, compute_model_hessians, compute_model_third_derivatives
from .moments import MomentsResult, compute_unconditional_moments
from .policy import Policy
from .pruning import PrunedSimulationResult, impulse_response_pruned, simulate_pruned
from .qz import BKDiagnostics
from .simulation import (
    GIRFResult,
    SimulationResult,
    generalized_irf,
    impulse_response,
    impulse_response_with_policy,
    simulate_linear,
    simulate_with_policy,
)
from .tensor_ops import mdot, sdot, solve_generalized_sylvester
from .solver import FirstOrderSolution, solve_first_order
from .solver_second_order import SecondOrderSolution, solve_second_order
from .solver_third_order import ThirdOrderSolution, solve_third_order
from .steady_state import SteadyStateResult, solve_steady_state
from .timing import LeadLagIncidence, build_lead_lag_incidence
from .version import __version__

__all__ = [
    "__version__",
    "DynareFixtureReport",
    "LinearRESpec",
    "ParityReport",
    "ReferenceComparison",
    "DSGEModel",
    "Policy",
    "BKDiagnostics",
    "Jacobians",
    "FirstOrderSolution",
    "SecondOrderSolution",
    "ThirdOrderSolution",
    "LeadLagIncidence",
    "GIRFResult",
    "MomentsResult",
    "ModelHessians",
    "ModelThirdDerivatives",
    "SimulationResult",
    "PrunedSimulationResult",
    "SteadyStateResult",
    "DynareModSpec",
    "DynareReference",
    "DynareResults",
    "IRFComparisonReport",
    "build_lead_lag_incidence",
    "SecondOrderParityReport",
    "compare_first_order_to_dynare",
    "compare_second_order_to_dynare",
    "compare_with_dynare_fixture",
    "dynare_available",
    "run_dynare_mod_file",
    "compare_policy_to_dynare_reference",
    "compute_jacobians",
    "solve_first_order",
    "solve_second_order",
    "solve_third_order",
    "solve_steady_state",
    "simulate_linear",
    "simulate_with_policy",
    "simulate_pruned",
    "impulse_response",
    "impulse_response_with_policy",
    "impulse_response_pruned",
    "generalized_irf",
    "compute_unconditional_moments",
    "compute_model_hessians",
    "compute_model_third_derivatives",
    "sdot",
    "mdot",
    "solve_generalized_sylvester",
    "parse_dynare_mod_file",
    "load_dynare_reference",
    "load_dynare_results",
    "compare_reconstructed_irfs_to_dynare",
    "results_mat_path_for_mod",
]
