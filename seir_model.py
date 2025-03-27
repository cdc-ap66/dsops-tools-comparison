#!/usr/bin/env python3
"""
SEIR Compartmental Model using SymPy

This module defines a SEIR (Susceptible-Exposed-Infectious-Recovered) compartmental
model class using SymPy for symbolic mathematics. The model can be pickled and versioned.
"""

import datetime
import pickle
from typing import List, Optional, Tuple, Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


class SEIRModel:
    """
    SEIR Compartmental Model using SymPy for symbolic mathematics.

    This class defines a SEIR (Susceptible-Exposed-Infectious-Recovered) model
    with symbolic parameters and equations. The model can be pickled for versioning
    and persistence.

    Attributes:
        parameters (Dict): Dictionary of model parameters
        variables (Dict): Dictionary of model variables
        equations (Dict): Dictionary of model equations
        version (str): Model version
        created_at (datetime): Model creation timestamp
    """
    def __init__(
        self,
        population_size: float = 1.0,
        beta: float = 0.3,  # Infection rate
        sigma: float = 0.2,  # Incubation rate (1/incubation period)
        gamma: float = 0.1,  # Recovery rate (1/infectious period)
        vax_fraction: Callable = lambda t: 0.0,  # Vaccination rate
        vax_eff: float = 0.9,  # Vaccine efficacy
        version: str = "1.0.0",
    ):
        """
        Initialize the SEIR model with parameters.

        Args:
            population_size: Total population size (normalized to 1.0 by default)
            beta: Infection rate parameter
            sigma: Rate at which exposed individuals become infectious
            gamma: Recovery rate parameter
            vax_fraction: Function of time that returns the fraction of the
                population vaccinated
            vax_eff: Vaccine efficacy (fraction of infections prevented)
            version: Model version string
        """
        self.version = version
        self.created_at = datetime.datetime.now()

        # Define symbolic variables for time and state variables
        t = sp.Symbol("t") # pylint: disable=C0103,W0621
        S = sp.Function("S")(t)  # pylint: disable=E1102,C0103  # Susceptible
        E = sp.Function("E")(t)  # pylint: disable=E1102,C0103  # Exposed
        I = sp.Function("I")(t)  # pylint: disable=E1102,C0103  # Infectious # noqa: E741
        R = sp.Function("R")(t)  # pylint: disable=E1102,C0103  # Recovered

        # Store variables
        self.variables = {"t": t, "S": S, "E": E, "I": I, "R": R}

        # Store the vaccination rate function
        self.vax_fraction = vax_fraction

        # Define parameters symbolically
        beta_sym = sp.Symbol("beta")
        sigma_sym = sp.Symbol("sigma")
        gamma_sym = sp.Symbol("gamma")
        vax_eff_sym = sp.Symbol("vax_eff")
        N_sym = sp.Symbol("N") # pylint: disable=C0103

        # Store parameters
        self.parameters = {
            "beta": beta_sym,
            "sigma": sigma_sym,
            "gamma": gamma_sym,
            "vax_eff": vax_eff_sym,
            "N": N_sym,
        }

        # Parameter values
        self.parameter_values = {
            "beta": beta,
            "sigma": sigma,
            "gamma": gamma,
            "vax_eff": vax_eff,
            "N": population_size,
        }

        # Define the SEIR model equations
        # Create a symbolic expression for vaccination rate
        vax_term = S * sp.Function('vax_rate')(t) # pylint: disable=E1102
        dS_dt = -beta_sym * S * I / N_sym  - ( vax_term * vax_eff_sym)# pylint: disable=C0103
        dE_dt = beta_sym * S * I / N_sym - sigma_sym * E # pylint: disable=C0103
        dI_dt = sigma_sym * E - gamma_sym * I # pylint: disable=C0103
        dR_dt = gamma_sym * I + ( vax_term * vax_eff_sym) # pylint: disable=C0103

        # Store equations
        self.equations = { # pylint: disable=C0103
            "dS_dt": dS_dt,
            "dE_dt": dE_dt,
            "dI_dt": dI_dt,
            "dR_dt": dR_dt,
        }

        # Create lambdified versions of the equations for numerical evaluation
        self._update_lambdified_equations()

    def _update_lambdified_equations(self):
        """Update the lambdified versions of the equations for numerical evaluation."""
        # Substitute parameter values into equations
        subs_eqs = {}
        for eq_name, eq in self.equations.items():
            subs_eq = eq
            for param_name, param_value in self.parameter_values.items():
                subs_eq = subs_eq.subs(
                    self.parameters[param_name], param_value
                )
            subs_eqs[eq_name] = subs_eq

        # Create lambdified functions for numerical evaluation
        S, E, I, R = ( # pylint: disable=C0103 # noqa: E741
            self.variables["S"],
            self.variables["E"],
            self.variables["I"],
            self.variables["R"],
        )
        t = self.variables["t"] # pylint: disable=C0103,W0621

        # We can't directly substitute a Python function into a symbolic expression
        # Instead, we'll keep the symbolic function in the equations and handle it during evaluation
        # vax_rate_func = sp.Function('vax_rate')  # pylint: disable=E1102

        # Create lambdified functions that include the vax_rate function
        self.lambdified_equations = {
            "dS_dt": sp.lambdify((t, S, E, I, R), subs_eqs["dS_dt"], 
                                 {"vax_rate": self.vax_fraction}),
            "dE_dt": sp.lambdify((t, S, E, I, R), subs_eqs["dE_dt"], 
                                 {"vax_rate": self.vax_fraction}),
            "dI_dt": sp.lambdify((t, S, E, I, R), subs_eqs["dI_dt"], 
                                 {"vax_rate": self.vax_fraction}),
            "dR_dt": sp.lambdify((t, S, E, I, R), subs_eqs["dR_dt"], 
                                 {"vax_rate": self.vax_fraction}),
        }

    def update_parameters(self, **kwargs):
        """
        Update model parameters.

        Args:
            **kwargs: Parameter name-value pairs to update
        """
        for param_name, param_value in kwargs.items():
            if param_name in self.parameter_values:
                self.parameter_values[param_name] = param_value
            else:
                raise ValueError(f"Unknown parameter: {param_name}")

        # Update lambdified equations with new parameter values
        self._update_lambdified_equations()

    def get_derivatives(self, t: float, y: List[float]) -> List[float]: # pylint: disable=W0621
        """
        Calculate derivatives for numerical integration.

        Args:
            t: Current time point
            y: Current state vector [S, E, I, R]

        Returns:
            List of derivatives [dS_dt, dE_dt, dI_dt, dR_dt]
        """
        S, E, I, R = y # pylint: disable=C0103 # noqa: E741

        dS_dt = self.lambdified_equations["dS_dt"](t, S, E, I, R) # pylint: disable=C0103
        dE_dt = self.lambdified_equations["dE_dt"](t, S, E, I, R) # pylint: disable=C0103
        dI_dt = self.lambdified_equations["dI_dt"](t, S, E, I, R) # pylint: disable=C0103
        dR_dt = self.lambdified_equations["dR_dt"](t, S, E, I, R) # pylint: disable=C0103

        return [dS_dt, dE_dt, dI_dt, dR_dt]

    def simulate(
        self,
        t_span: Tuple[float, float],
        initial_conditions: List[float], # pylint: disable=W0621
        t_points: Optional[int] = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model using numerical integration.

        Args:
            t_span: (t_start, t_end) tuple
            initial_conditions: Initial values [S0, E0, I0, R0]
            t_points: Number of time points to evaluate

        Returns:
            Tuple of (time_points, solution_array)
        """
        t_eval = np.linspace(t_span[0], t_span[1], t_points)
        solution = solve_ivp(
            self.get_derivatives,
            t_span,
            initial_conditions,
            t_eval=t_eval,
            method="RK45",
        )

        return solution.t, solution.y

    def __getstate__(self):
        """
        Prepare the object for pickling.

        This method is called when the object is being pickled.
        It returns a dict of attributes that will be pickled.
        """
        # Get all attributes of the object
        state = self.__dict__.copy()

        # Remove lambdified equations which might not be pickleable
        if 'lambdified_equations' in state:
            del state['lambdified_equations']

        # Store the symbolic expressions as strings
        state['symbolic_variables'] = {
            name: str(var) for name, var in self.variables.items()
        }
        state['symbolic_parameters'] = {
            name: str(param) for name, param in self.parameters.items()
        }
        state['symbolic_equations'] = {
            name: str(eq) for name, eq in self.equations.items()
        }

        # Remove the original symbolic objects which are not pickleable
        del state['variables']
        del state['parameters']
        del state['equations']

        return state

    def __setstate__(self, state):
        """
        Restore the object from the unpickled state.

        This method is called when the object is being unpickled.
        It restores the object's state from the unpickled dict.
        """
        # # Extract the symbolic expressions
        # symbolic_variables = state.pop('symbolic_variables') # noqa: F841
        # symbolic_parameters = state.pop('symbolic_parameters')
        # symbolic_equations = state.pop('symbolic_equations')

        # Restore the object's state
        self.__dict__.update(state)

        # Recreate the symbolic variables
        t = sp.Symbol("t") # pylint: disable=C0103,W0621
        S = sp.Function("S")(t)  # pylint: disable=C0103,E1102
        E = sp.Function("E")(t)  # pylint: disable=E1102,C0103
        I = sp.Function("I")(t)  # pylint: disable=E1102,C0103 # noqa: E741
        R = sp.Function("R")(t)  # pylint: disable=E1102,C0103

        self.variables = {"t": t, "S": S, "E": E, "I": I, "R": R}

        # Recreate the symbolic parameters
        beta_sym = sp.Symbol("beta")
        sigma_sym = sp.Symbol("sigma")
        gamma_sym = sp.Symbol("gamma")
        vax_eff_sym = sp.Symbol("vax_eff")
        N_sym = sp.Symbol("N")  # pylint: disable=C0103

        self.parameters = {
            "beta": beta_sym,
            "sigma": sigma_sym,
            "gamma": gamma_sym,
            "vax_eff": vax_eff_sym,
            "N": N_sym,
        }

        # Recreate the equations
        vax_term = S * sp.Function('vax_rate')(t)  # pylint: disable=E1102
        dS_dt = -beta_sym * S * I / N_sym - (vax_term * vax_eff_sym)  # pylint: disable=C0103
        dE_dt = beta_sym * S * I / N_sym - sigma_sym * E  # pylint: disable=C0103
        dI_dt = sigma_sym * E - gamma_sym * I  # pylint: disable=C0103
        dR_dt = gamma_sym * I + ( vax_term * vax_eff_sym) # pylint: disable=C0103

        self.equations = {  # pylint: disable=C0103
            "dS_dt": dS_dt,
            "dE_dt": dE_dt,
            "dI_dt": dI_dt,
            "dR_dt": dR_dt,
        }

        # Recreate the lambdified equations
        self._update_lambdified_equations()

    def save(self, filename: str):
        """
        Save the model to a file using pickle.

        Args:
            filename: Path to save the model
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "SEIRModel":
        """
        Load a model from a file.

        Args:
            filename: Path to the saved model

        Returns:
            Loaded SEIRModel instance
        """
        with open(filename, "rb") as f:
            model = pickle.load(f) # pylint: disable=W0621
        return model

    def __str__(self):
        """String representation of the model."""
        equations_str = "\n".join(
            [f"{name} = {eq}" for name, eq in self.equations.items()]
        )
        params_str = "\n".join(
            [
                f"{name} = {value}"
                for name, value in self.parameter_values.items()
            ]
        )

        return (
            f"SEIR Model (version {self.version})\n"
            f"Created: {self.created_at}\n\n"
            f"Parameters:\n{params_str}\n\n"
            f"Equations:\n{equations_str}"
        )


if __name__ == "__main__":
    # Example usage
    model = SEIRModel(population_size=1.0, beta=0.3, sigma=0.2, gamma=0.1)

    # Initial conditions: 99% susceptible, 0% exposed, 1% infectious, 0% recovered
    initial_conditions = [0.99, 0.0, 0.01, 0.0]

    # Simulate for 100 time units
    t, y = model.simulate((0, 100), initial_conditions)

    print(f"Model version: {model.version}")
    print(f"Final state (t={t[-1]}):")
    print(f"S: {y[0][-1]:.4f}")
    print(f"E: {y[1][-1]:.4f}")
    print(f"I: {y[2][-1]:.4f}")
    print(f"R: {y[3][-1]:.4f}")

    # Example of saving and loading the model
    # model.save("seir_model.pkl")
    # loaded_model = SEIRModel.load("seir_model.pkl")
