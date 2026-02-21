"""Lead-lag incidence analysis for DSGE model equations.

Parses symbolic equation strings to determine which endogenous variables
appear with leads and/or lags.  The resulting :class:`LeadLagIncidence`
structure is used to classify variables as predetermined (states) or
forward-looking (controls) and to detect the maximum lead and lag depth
in the model -- information that is essential for building the
appropriate perturbation system.

The notation ``x(+1)`` denotes a one-period lead and ``x(-1)`` a
one-period lag, following the Dynare convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LeadLagIncidence:
    """Lead-lag timing structure for endogenous variables.

    Records which endogenous variables appear with leads (``x(+1)``,
    ``x(+2)``, ...) or lags (``x(-1)``, ``x(-2)``, ...) in the model
    equations.  This information is used to classify variables as
    predetermined (states) or forward-looking (controls) and to
    determine the maximum lead and lag depth.

    Attributes
    ----------
    endogenous : tuple of str
        Ordered names of all endogenous variables.
    max_lag : int
        Maximum lag order found across all variables (0 if no lags).
    max_lead : int
        Maximum lead order found across all variables (0 if no leads).
    current_positions : Mapping[str, int]
        Index of each variable in the endogenous tuple.
    lag_orders : Mapping[str, tuple[int, ...]]
        For each variable that appears with a lag, the sorted tuple of
        lag orders (e.g., ``(1, 2)`` for a variable appearing at
        ``x(-1)`` and ``x(-2)``).
    lead_orders : Mapping[str, tuple[int, ...]]
        For each variable that appears with a lead, the sorted tuple
        of lead orders.
    """

    endogenous: tuple[str, ...]
    max_lag: int
    max_lead: int
    current_positions: Mapping[str, int]
    lag_orders: Mapping[str, tuple[int, ...]]
    lead_orders: Mapping[str, tuple[int, ...]]

    def has_lag(self, name: str) -> bool:
        """Return True if *name* appears with at least one lag."""
        return len(self.lag_orders.get(name, ())) > 0

    def has_lead(self, name: str) -> bool:
        """Return True if *name* appears with at least one lead."""
        return len(self.lead_orders.get(name, ())) > 0


_SHIFTED_SYMBOL = re.compile(r"\b([A-Za-z_]\w*)\s*\(([+-]?\d+)\)")


def build_lead_lag_incidence(
    equations: Sequence[str], *, endogenous: Sequence[str]
) -> LeadLagIncidence:
    """Parse equations to build a lead-lag incidence structure.

    Scans symbolic equation strings for timed references of the form
    ``symbol(+n)`` or ``symbol(-n)`` and records which endogenous
    variables appear at which lead or lag orders.

    Parameters
    ----------
    equations : Sequence[str]
        Model equation strings using the Dynare-style timing convention
        (e.g., ``"c(+1) - beta * r * c"``).
    endogenous : Sequence[str]
        Names of all endogenous variables.

    Returns
    -------
    LeadLagIncidence
        The parsed timing structure.

    Raises
    ------
    ValueError
        If a timed reference mentions a symbol not in *endogenous*.
    """
    endogenous_tuple = tuple(endogenous)
    endo_set = set(endogenous_tuple)

    lag_orders: dict[str, set[int]] = {name: set() for name in endogenous_tuple}
    lead_orders: dict[str, set[int]] = {name: set() for name in endogenous_tuple}

    for eq in equations:
        for symbol, shift_text in _SHIFTED_SYMBOL.findall(eq):
            if symbol not in endo_set:
                raise ValueError(
                    f"Unknown endogenous symbol in timed reference: '{symbol}'"
                )

            shift = int(shift_text)
            if shift < 0:
                lag_orders[symbol].add(abs(shift))
            elif shift > 0:
                lead_orders[symbol].add(shift)

    max_lag = max((max(v) for v in lag_orders.values() if v), default=0)
    max_lead = max((max(v) for v in lead_orders.values() if v), default=0)

    return LeadLagIncidence(
        endogenous=endogenous_tuple,
        max_lag=max_lag,
        max_lead=max_lead,
        current_positions={name: i for i, name in enumerate(endogenous_tuple)},
        lag_orders={name: tuple(sorted(v)) for name, v in lag_orders.items() if v},
        lead_orders={name: tuple(sorted(v)) for name, v in lead_orders.items() if v},
    )
