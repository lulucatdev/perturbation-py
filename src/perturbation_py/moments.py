"""Ergodic (unconditional) moments from perturbation solutions.

Computes theoretical unconditional mean, variance, standard deviation, and
autocorrelations using the linear state-space representation implied by the
perturbation solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .policy import Policy

Array = np.ndarray


@dataclass(frozen=True)
class MomentsResult:
    """Unconditional (ergodic) moments of a perturbation solution.

    All moments are expressed in levels (steady state + deviation) for the
    mean, and in terms of deviations from steady state for variance,
    standard deviation, and autocorrelations.

    Attributes
    ----------
    mean_states : ndarray, shape (n_s,)
        Unconditional mean of the state vector.  At first order this is
        the steady state (zero deviations); at second order and above it
        includes the risk-adjustment term.
    mean_controls : ndarray, shape (n_x,)
        Unconditional mean of the control vector.  At order >= 2 this
        incorporates the ``0.5 * ghs2`` correction.
    variance_states : ndarray, shape (n_s, n_s)
        Unconditional variance-covariance matrix of the states, obtained
        from the discrete Lyapunov equation.
    variance_controls : ndarray, shape (n_x, n_x)
        Unconditional variance-covariance matrix of the controls.
    std_states : ndarray, shape (n_s,)
        Unconditional standard deviations of the states (square root of
        the diagonal of ``variance_states``).
    std_controls : ndarray, shape (n_x,)
        Unconditional standard deviations of the controls.
    autocorrelations : ndarray or None, shape (max_lag, n_s + n_x)
        Autocorrelation at lags 1, ..., ``max_lag`` for each variable
        (states followed by controls).  ``None`` if ``max_lag == 0``.
    """
    mean_states: Array
    mean_controls: Array
    variance_states: Array
    variance_controls: Array
    std_states: Array
    std_controls: Array
    autocorrelations: Array | None


def compute_unconditional_moments(
    policy: Policy,
    *,
    shock_covariance: Array | None = None,
    max_lag: int = 5,
) -> MomentsResult:
    r"""Compute theoretical unconditional moments from a perturbation solution.

    The first-order state-space representation is::

        s_{t+1} = h_x  s_t  +  h_u  e_{t+1}
        x_t     = g_x  s_t  +  g_u  e_t

    **Variance (discrete Lyapunov equation).**
    The unconditional state covariance ``Sigma_s`` satisfies::

        Sigma_s = h_x  Sigma_s  h_x^T  +  h_u  Sigma_e  h_u^T

    which is the discrete Lyapunov equation solved by
    ``scipy.linalg.solve_discrete_lyapunov``.  The control covariance is
    then ``Sigma_x = g_x Sigma_s g_x^T + g_u Sigma_e g_u^T``.

    **Mean.**
    At first order the unconditional mean equals the deterministic steady
    state.  At second order the mean of the controls is adjusted by the
    risk-correction term ``0.5 * ghs2`` (Schmitt-Grohe and Uribe, 2004).

    **Autocorrelation.**
    For lag *k*, the autocovariance of the states is
    ``Cov(s_t, s_{t-k}) = h_x^k  Sigma_s``.  Dividing by the variance
    yields the autocorrelation coefficient.  Control autocorrelations
    follow via the observation equation ``x_t = g_x s_t + ...``.

    Parameters
    ----------
    policy : Policy
        Perturbation policy (order 1, 2, or 3).
    shock_covariance : ndarray, shape (n_e, n_e), optional
        Shock covariance matrix.  Defaults to the identity matrix.
    max_lag : int, optional
        Maximum lag for autocorrelation computation (default 5).  Set to
        0 to skip autocorrelation calculation.

    Returns
    -------
    MomentsResult
        Dataclass containing mean, variance, standard deviations, and
        (optionally) autocorrelations for both states and controls.

    References
    ----------
    Hamilton (1994), *Time Series Analysis*, Ch. 10 (discrete Lyapunov
    equation for covariance of linear state-space models).
    Schmitt-Grohe and Uribe (2004), JEDC 28, 755-775 (risk correction
    at second order).
    """
    n_s = policy.n_states
    n_x = policy.n_controls
    n_e = policy.n_shocks

    if shock_covariance is not None:
        Sigma_e = np.asarray(shock_covariance, dtype=float)
    else:
        Sigma_e = np.eye(n_e, dtype=float)

    # Transition matrices (first-order)
    h_x = np.asarray(policy.transition, dtype=float)  # (n_s, n_s)
    h_u = np.asarray(policy.shock_impact, dtype=float)  # (n_s, n_e)
    ghx = np.asarray(policy.ghx, dtype=float)  # (n_x, n_s)
    ghu = np.asarray(policy.ghu, dtype=float)  # (n_x, n_e)

    # --- Unconditional mean ---
    # At order >= 2: E[s] ≈ 0 (deviations), E[x] ≈ x_bar + 0.5*ghs2
    mean_states = np.zeros(n_s, dtype=float)
    mean_controls = policy.steady_state_controls.copy()
    if policy.order >= 2 and policy.ghs2 is not None:
        mean_controls = mean_controls + 0.5 * np.asarray(policy.ghs2, dtype=float).reshape(-1)

    # --- State covariance via discrete Lyapunov equation ---
    # s_{t+1} = h_x @ s_t + h_u @ e_t
    # Sigma_s = h_x @ Sigma_s @ h_x.T + h_u @ Sigma_e @ h_u.T
    Q = h_u @ Sigma_e @ h_u.T
    Sigma_s = solve_discrete_lyapunov(h_x, Q)

    # --- Control covariance ---
    # x_t ≈ ghx @ s_t + ghu @ e_t  (first-order approximation for variance)
    Sigma_x = ghx @ Sigma_s @ ghx.T + ghu @ Sigma_e @ ghu.T

    std_states = np.sqrt(np.diag(Sigma_s))
    std_controls = np.sqrt(np.diag(Sigma_x))

    # --- Autocorrelations ---
    autocorrelations = None
    if max_lag > 0:
        n_total = n_s + n_x
        # Build combined var-covar
        # y_t = [s_t; x_t] where x_t = ghx @ s_t + ghu @ e_t
        # Cov(y_t, y_{t-k}) via state propagation
        autocorr = np.zeros((max_lag, n_total), dtype=float)
        # Combined variance for normalization
        var_y = np.concatenate([np.diag(Sigma_s), np.diag(Sigma_x)])

        h_x_power = np.eye(n_s, dtype=float)
        for k in range(1, max_lag + 1):
            h_x_power = h_x_power @ h_x
            # Cov(s_t, s_{t-k}) = h_x^k @ Sigma_s
            cov_s_lag = h_x_power @ Sigma_s
            # autocorrelation for each state: diag(cov_s_lag) / var_s
            for i in range(n_s):
                if var_y[i] > 1e-16:
                    autocorr[k - 1, i] = cov_s_lag[i, i] / var_y[i]

            # Cov(x_t, x_{t-k}) = ghx @ h_x^k @ Sigma_s @ ghx.T
            cov_x_lag = ghx @ cov_s_lag @ ghx.T
            for i in range(n_x):
                if var_y[n_s + i] > 1e-16:
                    autocorr[k - 1, n_s + i] = cov_x_lag[i, i] / var_y[n_s + i]

        autocorrelations = autocorr

    return MomentsResult(
        mean_states=mean_states,
        mean_controls=mean_controls,
        variance_states=Sigma_s,
        variance_controls=Sigma_x,
        std_states=std_states,
        std_controls=std_controls,
        autocorrelations=autocorrelations,
    )
