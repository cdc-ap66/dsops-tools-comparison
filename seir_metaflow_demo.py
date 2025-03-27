#!/usr/bin/env python3
"""
SEIR Model Metaflow Demo

This script demonstrates how to use Metaflow to experiment with different
parameter settings for a SEIR (Susceptible-Exposed-Infectious-Recovered)
compartmental model.

The workflow:
1. Loads COVID-19 vaccination data
2. Creates a vaccination rate model
3. Runs multiple SEIR model experiments with different parameters in parallel
4. Compares the results and selects the best model
5. Visualizes the results

To run this workflow:
    python seir_metaflow_demo.py run
"""

import os
import pickle
import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from metaflow import FlowSpec, step, Parameter, current

# Import the SEIR model
from seir_model import SEIRModel


class VaxTrends:
    """Model for vaccination rate interpolation."""

    def __init__(self, start_date: str, df=None, df_path=None):
        if df is None and df_path is not None:
            df = pd.read_parquet(df_path)

        df.date = pd.to_datetime(df.date).dt.date
        self.start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        self.dates = df.date.to_list()
        self.vax_frax = df.vax_frac.to_list()
        self.days = [i.days for i in (df.date - self.start_date).to_list()]
        self.function = interp1d(self.days, self.vax_frax, kind='linear')

    def __call__(self, t: float):
        if t < max(self.days):
            return float(self.function(t))
        else:
            return float(self.vax_frax[-1])

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class SEIRModelFlow(FlowSpec):
    """
    A flow for experimenting with SEIR model parameters using Metaflow.

    This flow demonstrates how to use Metaflow to:
    1. Process data
    2. Run multiple experiments in parallel
    3. Compare results
    4. Select the best model
    """

    # Define flow parameters
    start_date = Parameter('start_date',
                          help='Start date for the simulation (YYYY-MM-DD)',
                          default='2021-01-01')

    simulation_days = Parameter('simulation_days',
                               help='Number of days to simulate',
                               default=120)

    time_points = Parameter('time_points',
                           help='Number of time points in the simulation',
                           default=500)

    # Initial conditions as parameters
    initial_s = Parameter('initial_s',
                         help='Initial susceptible fraction',
                         default=0.90)

    initial_e = Parameter('initial_e',
                         help='Initial exposed fraction',
                         default=0.02)

    initial_i = Parameter('initial_i',
                         help='Initial infectious fraction',
                         default=0.02)

    initial_r = Parameter('initial_r',
                         help='Initial recovered fraction',
                         default=0.06)

    @step
    def start(self):
        """
        Start the flow and create output directories.
        """
        print(f"Starting SEIR Model Flow at {datetime.datetime.now()}")
        print(f"Run ID: {current.run_id}")

        # Create output directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)

        # Go to the next step
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Load and preprocess the COVID-19 vaccination data.
        """

        print("Loading COVID-19 vaccination data...")

        # load the vaccination data
        df = pd.read_parquet('data/covid19vax_trends_us.parquet')

        # Store the data in the flow
        self.df = df # pylint: disable=W0201

        # Go to the next step
        self.next(self.create_vax_model)

    @step
    def create_vax_model(self):
        """
        Create a model for vaccination rate interpolation.
        """
        print("Creating vaccination rate model...")

        # Create the VaxTrends model
        self.vax_model = VaxTrends(self.start_date, df=self.df) # pylint: disable=W0201

        # Save the model
        self.vax_model.save('models/vax_trends_metaflow.pkl')

        # Define parameter sets for experiments
        self.experiment_params = [ # pylint: disable=W0201
            {
                "name": "Baseline",
                "beta": 0.3,
                "sigma": 0.2,
                "gamma": 0.1,
                "vax_eff": 0.7
            },
            {
                "name": "High_Transmission",
                "beta": 0.5,
                "sigma": 0.2,
                "gamma": 0.1,
                "vax_eff": 0.7
            },
            {
                "name": "Low_Transmission",
                "beta": 0.2,
                "sigma": 0.2,
                "gamma": 0.1,
                "vax_eff": 0.7
            },
            {
                "name": "High_Recovery",
                "beta": 0.3,
                "sigma": 0.2,
                "gamma": 0.2,
                "vax_eff": 0.7
            },
            {
                "name": "High_Vaccine_Efficacy",
                "beta": 0.3,
                "sigma": 0.2,
                "gamma": 0.1,
                "vax_eff": 0.9
            }
        ]

        # Go to the next step - fan out to run experiments in parallel
        self.next(self.run_experiment, foreach='experiment_params')

    @step
    def run_experiment(self):
        """
        Run a SEIR model experiment with specific parameters.

        This step is executed in parallel for each parameter set.
        """
        # Get the parameters for this experiment
        params = self.input

        print(f"Running experiment: {params['name']}")
        print(f"Parameters: beta={params['beta']}, sigma={params['sigma']}, "
              f"gamma={params['gamma']}, vax_eff={params['vax_eff']}")

        # Create the SEIR model with the specified parameters
        model = SEIRModel(
            population_size=1.0,
            beta=params['beta'],
            sigma=params['sigma'],
            gamma=params['gamma'],
            vax_fraction=self.vax_model,
            vax_eff=params['vax_eff'],
            version=f"0.1.0-{params['name']}"
        )

        # Initial conditions
        initial_conditions = [
            self.initial_s,
            self.initial_e,
            self.initial_i,
            self.initial_r
        ]

        # Run the simulation
        t_span = (0, self.simulation_days)
        t, y = model.simulate(t_span, initial_conditions, self.time_points)

        # Calculate metrics
        S, E, I, R = y  # pylint: disable=all # noqa: E741

        # Calculate peak infectious
        peak_infectious = np.max(I)
        peak_time = t[np.argmax(I)]

        # Calculate final recovered fraction
        final_recovered = R[-1]

        # Calculate total infected (final recovered)
        total_infected = final_recovered

        # Store the results
        self.params = params # pylint: disable=W0201
        self.model = model # pylint: disable=W0201
        self.t = t # pylint: disable=W0201
        self.y = y # pylint: disable=W0201
        self.metrics = { # pylint: disable=W0201
            "peak_infectious": peak_infectious,
            "peak_time": peak_time,
            "final_recovered": final_recovered,
            "total_infected": total_infected
        }

        # Save the model
        model_file = f"models/seir_model_{params['name'].lower()}_metaflow.pkl"
        model.save(model_file)

        # Create and save the plot
        self._create_plot(
            t, y, self.vax_model,
            f"SEIR Model - {params['name']}",
            f"results/seir_plot_{params['name'].lower()}.png"
        )

        # Go to the next step - join the parallel branches
        self.next(self.join_results)

    @step
    def join_results(self, inputs):
        """
        Join the results from all experiments and compare them.
        """
        print("Joining results from all experiments...")

        # Collect results from all experiments
        self.all_results = []

        for input_data in inputs:
            self.all_results.append({
                "name": input_data.params["name"],
                "params": input_data.params,
                "metrics": input_data.metrics,
                "model": input_data.model,
                "t": input_data.t,
                "y": input_data.y
            })

        # Find the best model (lowest peak infectious)
        best_result = min(self.all_results, key=lambda x: x["metrics"]["peak_infectious"])
        self.best_model = best_result["model"]
        self.best_model_name = best_result["name"]

        print(f"Best model: {self.best_model_name}")
        print(f"Peak infectious: {best_result['metrics']['peak_infectious']:.4f}")

        # Create a comparison DataFrame
        comparison_data = []
        for result in self.all_results:
            comparison_data.append({
                "name": result["name"],
                "beta": result["params"]["beta"],
                "sigma": result["params"]["sigma"],
                "gamma": result["params"]["gamma"],
                "vax_eff": result["params"]["vax_eff"],
                "peak_infectious": result["metrics"]["peak_infectious"],
                "peak_time": result["metrics"]["peak_time"],
                "total_infected": result["metrics"]["total_infected"]
            })

        self.comparison_df = pd.DataFrame(comparison_data)

        # Save the comparison to CSV
        self.comparison_df.to_csv("results/experiment_comparison.csv", index=False)

        # Go to the next step
        self.next(self.visualize_comparison)

    @step
    def visualize_comparison(self):
        """
        Create visualizations comparing the results of all experiments.
        """
        print("Creating comparison visualizations...")

        # Sort by peak infectious
        df = self.comparison_df.sort_values(by="peak_infectious")

        # Plot comparison of peak infectious values
        plt.figure(figsize=(10, 6), dpi=100)
        plt.bar(df["name"], df["peak_infectious"])
        plt.xlabel("Experiment")
        plt.ylabel("Peak Infectious Fraction")
        plt.title("Comparison of Peak Infectious Values Across Experiments")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("results/peak_infectious_comparison.png")
        plt.close()

        # Plot comparison of total infected values
        plt.figure(figsize=(10, 6), dpi=100)
        plt.bar(df["name"], df["total_infected"])
        plt.xlabel("Experiment")
        plt.ylabel("Total Infected Fraction")
        plt.title("Comparison of Total Infected Values Across Experiments")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("results/total_infected_comparison.png")
        plt.close()

        # Create a plot comparing all models
        self._create_comparison_plot()

        # Go to the next step
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow and print summary information.
        """
        print("\nSEIR Model Flow completed successfully!")
        print(f"Run ID: {current.run_id}")
        print(f"Best model: {self.best_model_name}")

        # Print the comparison table
        print("\nExperiment Comparison:")
        print(self.comparison_df.to_string(index=False))

        print("\nResults saved to:")
        print("- results/experiment_comparison.csv")
        print("- results/peak_infectious_comparison.png")
        print("- results/total_infected_comparison.png")
        print("- results/model_comparison.png")

        print("\nTo view the results, you can run:")
        print("  python -m metaflow.client")
        print("And then in the Python REPL:")
        print(f"  run = SEIRModelFlow('{current.run_id}')")
        print("  run.comparison_df")

    def _create_plot(self, t, y, vax_fraction, title, filename):
        """Helper method to create and save a plot of simulation results."""
        S, E, I, R = y # noqa: E741

        plt.figure(figsize=(10, 6), dpi=100)

        # Plot the S, E, I, R states
        plt.plot(t, [vax_fraction(i) for i in t], 'k--', label='Vaccination Fraction', alpha=0.5)
        plt.plot(t, S, 'b-', label='Susceptible')
        plt.plot(t, E, 'c-', label='Exposed')
        plt.plot(t, I, 'r-', label='Infectious')
        plt.plot(t, R, 'm-', label='Recovered')

        plt.grid(ls='--')
        plt.xlabel('Time (days)')
        plt.ylabel('Population Fraction')
        plt.title(title)
        plt.legend(fontsize=9)

        plt.savefig(filename)
        plt.close()

    def _create_comparison_plot(self):
        """Create a plot comparing the infectious curves of all models."""
        plt.figure(figsize=(12, 8), dpi=100)

        for result in self.all_results:
            name = result["name"]
            t = result["t"]
            I = result["y"][2]  # Infectious curve # noqa: E741

            plt.plot(t, I, label=name)

        plt.grid(ls='--')
        plt.xlabel('Time (days)')
        plt.ylabel('Infectious Population Fraction')
        plt.title('Comparison of Infectious Curves Across Experiments')
        plt.legend()

        plt.savefig("results/model_comparison.png")
        plt.close()


if __name__ == "__main__":
    SEIRModelFlow()
