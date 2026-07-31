# -*- coding: utf-8 -*-
"""
Level 1 - Entropy and Mutual-Information-Based Sensor Selection
---------------------------------------------------------------

This script identifies an optimal subset of sensors from acceleration time
histories by maximizing normalized Shannon entropy and minimizing pairwise
redundancy estimated through mutual information.

The sensor-selection problem is formulated as a Quadratic Unconstrained Binary
Optimization (QUBO) problem and solved with the Quantum Approximate Optimization
Algorithm (QAOA) using Qiskit Aer.

Expected repository structure
-----------------------------
Quantum_Sensor_Selection/
|-- data/
|   |-- case_study_1/
|   |   `-- raw/
|   |       `-- acceleration_data.xlsx
|   `-- case_study_2/
|       `-- raw/
|           `-- acceleration_data.xlsx
|-- scripts/
|   `-- level_1_entropy_mutual_information.py
`-- results/

Only the case-study number and analysis parameters in the USER CONFIGURATION
section should normally be changed.
"""

from __future__ import annotations

import json
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Qiskit Optimization and Aer imports
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import SPSA
from qiskit_optimization.utils import algorithm_globals


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Change CASE_STUDY only in scripts/config.py.
from config import CASE_STUDY, get_case_study_config

CASE_CONFIG = get_case_study_config()
K_SENSORS_TO_SELECT = CASE_CONFIG["number_of_sensors_to_select"]
QUBO_LAMBDA = CASE_CONFIG["qubo_lambda"]
QUBO_ALPHA = CASE_CONFIG["qubo_alpha"]

# QAOA settings shared by both case studies.
QAOA_REPS = 3
QAOA_MAXITER = 100
QAOA_SHOTS = 2048
RANDOM_SEED = 42
USE_SPARSE = False

# Output settings.
VISUALIZE = True
SAVE_FIGURES = True
SAVE_NUMERICAL_RESULTS = True

# =============================================================================
# REPOSITORY PATHS
# =============================================================================

# This script is expected inside: <repository_root>/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / f"case_study_{CASE_STUDY}" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / f"case_study_{CASE_STUDY}" / "level_1"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

ACCELERATION_FILE = DATA_DIR / "acceleration_data.xlsx"

CONVERGENCE_FIGURE = FIGURES_DIR / "qaoa_convergence.png"
SELECTION_FIGURE = FIGURES_DIR / "sensor_selection_visualization.png"
SELECTION_RESULTS_FILE = RESULTS_DIR / "selection_results.json"
SENSOR_METRICS_FILE = TABLES_DIR / "sensor_metrics.csv"


# =============================================================================
# 1. DATA PROCESSING AND INFORMATION-THEORY METRICS
# =============================================================================

def load_and_prepare_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load acceleration data and retain only sensor channels.

    Parameters
    ----------
    filepath:
        Path to the Excel file containing the acceleration time histories.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only numeric sensor channels.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            "Acceleration dataset not found.\n"
            f"Expected file: {filepath}\n"
            "Place the input file in the corresponding case-study raw-data "
            "directory and name it 'acceleration_data.xlsx'."
        )

    try:
        df = pd.read_excel(filepath)
    except Exception as exc:
        raise RuntimeError(f"Unable to read the Excel file '{filepath}': {exc}") from exc

    drop_candidates = {"time", "t", "index", "date", "timestamp"}
    columns_to_drop = [
        column for column in df.columns
        if str(column).strip().lower() in drop_candidates
    ]

    if columns_to_drop:
        print(f"    Dropping non-sensor columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)

    # Keep only numeric columns to prevent accidental use of metadata columns.
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        print(f"    Dropping non-numeric columns: {non_numeric_columns}")
        df = df.drop(columns=non_numeric_columns)

    if df.empty:
        raise ValueError("No numeric sensor channels were found in the input dataset.")

    if df.isna().any().any():
        warnings.warn(
            "Missing values were detected. Rows containing missing values will be removed."
        )
        df = df.dropna(axis=0, how="any")

    if df.empty:
        raise ValueError("No valid samples remain after removing rows with missing values.")

    return df


def compute_metrics_from_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized Shannon entropy and mutual-information redundancy.

    Shannon entropy is estimated under a Gaussian assumption from each sensor's
    variance. Pairwise mutual information is estimated from the Pearson
    correlation coefficient under a bivariate Gaussian assumption.
    """
    variances = df.var().to_numpy(dtype=float)
    variances = np.clip(variances, a_min=1e-10, a_max=None)

    entropy = 0.5 * np.log(2.0 * np.pi * np.e * variances)
    entropy_min = float(np.min(entropy))
    entropy_max = float(np.max(entropy))

    if entropy_max > entropy_min:
        utility = (entropy - entropy_min) / (entropy_max - entropy_min)
    else:
        warnings.warn("All sensors have identical entropy values.")
        utility = np.ones_like(entropy)

    pearson_correlation = df.corr().to_numpy(dtype=float)
    pearson_correlation = np.clip(
        pearson_correlation,
        a_min=-0.9999,
        a_max=0.9999,
    )

    mutual_information = -0.5 * np.log(1.0 - pearson_correlation**2)
    np.fill_diagonal(mutual_information, 0.0)

    maximum_mutual_information = float(np.max(mutual_information))
    if maximum_mutual_information > 0.0:
        redundancy_matrix = mutual_information / maximum_mutual_information
    else:
        redundancy_matrix = mutual_information

    return utility, redundancy_matrix


# =============================================================================
# 2. QUBO FORMULATION
# =============================================================================

def build_qubo_sensor_selection(
    utility: np.ndarray,
    redundancy_matrix: np.ndarray,
    k: int,
    lam: float = 10.0,
    alpha: float = 1.0,
) -> np.ndarray:
    """Build the QUBO matrix for exact-cardinality sensor selection."""
    utility = np.asarray(utility, dtype=float)
    redundancy_matrix = np.asarray(redundancy_matrix, dtype=float)

    number_of_sensors = utility.shape[0]

    if redundancy_matrix.shape != (number_of_sensors, number_of_sensors):
        raise ValueError(
            "The redundancy matrix must be square and consistent with the "
            "utility-vector length."
        )

    if not 1 <= k <= number_of_sensors:
        raise ValueError(
            f"The requested number of sensors must be between 1 and "
            f"{number_of_sensors}; received {k}."
        )

    maximum_redundancy = float(np.max(redundancy_matrix))
    effective_alpha = alpha
    if maximum_redundancy > 0.0:
        effective_alpha = alpha / maximum_redundancy
        print(
            "    Dynamic redundancy scaling applied: "
            f"alpha = {effective_alpha:.4f}."
        )

    qubo_matrix = np.zeros((number_of_sensors, number_of_sensors), dtype=float)

    # Utility term: maximizing utility is equivalent to minimizing its negative.
    qubo_matrix[np.arange(number_of_sensors), np.arange(number_of_sensors)] += -utility

    # Pairwise redundancy penalty.
    for i in range(number_of_sensors):
        for j in range(i + 1, number_of_sensors):
            qubo_matrix[i, j] += effective_alpha * redundancy_matrix[i, j]

    # Exact-cardinality penalty: lambda * (sum_i x_i - k)^2.
    qubo_matrix[np.arange(number_of_sensors), np.arange(number_of_sensors)] += (
        lam * (1 - 2 * k)
    )
    for i in range(number_of_sensors):
        for j in range(i + 1, number_of_sensors):
            qubo_matrix[i, j] += 2 * lam

    return qubo_matrix


def build_qubo_sparse_from_matrix(
    qubo_matrix: np.ndarray,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """Convert an upper-triangular QUBO matrix to sparse dictionaries."""
    number_of_sensors = qubo_matrix.shape[0]

    linear = {
        f"x{i}": float(qubo_matrix[i, i])
        for i in range(number_of_sensors)
        if abs(qubo_matrix[i, i]) > 1e-10
    }

    quadratic = {
        (f"x{i}", f"x{j}"): float(qubo_matrix[i, j])
        for i in range(number_of_sensors)
        for j in range(i + 1, number_of_sensors)
        if abs(qubo_matrix[i, j]) > 1e-10
    }

    return linear, quadratic


# =============================================================================
# 3. QAOA SOLVER
# =============================================================================

def solve_qubo_qaoa(
    qubo_matrix: np.ndarray,
    k: int,
    lam: float,
    reps: int = 3,
    maxiter: int = 100,
    seed: int = 42,
    shots: int = 2048,
    use_sparse: bool = False,
) -> Tuple[np.ndarray, float, List[float]]:
    """Solve the QUBO problem using QAOA and return the binary solution."""
    qubo_matrix = np.asarray(qubo_matrix, dtype=float)
    number_of_sensors = qubo_matrix.shape[0]

    if number_of_sensors > 25:
        warnings.warn(
            f"The problem contains {number_of_sensors} binary variables and may "
            "be computationally demanding on a classical simulator."
        )

    # Preserve the adaptive settings used in the original implementation.
    effective_reps = max(reps, 4) if number_of_sensors > 15 else reps
    effective_maxiter = max(maxiter, 200 if number_of_sensors > 10 else 100)

    if effective_reps != reps:
        print(
            f"    QAOA depth increased from {reps} to {effective_reps} "
            "for the current problem size."
        )
    if effective_maxiter != maxiter:
        print(
            f"    Maximum optimizer iterations increased from {maxiter} to "
            f"{effective_maxiter} for the current problem size."
        )

    algorithm_globals.random_seed = seed

    quadratic_program = QuadraticProgram()
    for index in range(number_of_sensors):
        quadratic_program.binary_var(name=f"x{index}")

    constant_offset = lam * (k**2)

    if use_sparse:
        linear, quadratic = build_qubo_sparse_from_matrix(qubo_matrix)
    else:
        linear = {
            f"x{i}": float(qubo_matrix[i, i])
            for i in range(number_of_sensors)
        }
        quadratic = {
            (f"x{i}", f"x{j}"): float(qubo_matrix[i, j])
            for i in range(number_of_sensors)
            for j in range(i + 1, number_of_sensors)
            if abs(qubo_matrix[i, j]) > 1e-10
        }

    quadratic_program.minimize(
        constant=constant_offset,
        linear=linear,
        quadratic=quadratic,
    )

    optimizer = SPSA(maxiter=effective_maxiter)
    sampler = SamplerV2(seed=seed, default_shots=shots)
    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        seed_transpiler=seed,
    )

    objective_history: List[float] = []

    def qaoa_callback(eval_count, parameters, mean, metadata):
        del eval_count, parameters, metadata
        objective_history.append(float(mean))

    initial_point = np.random.default_rng(seed).uniform(
        0.0,
        2.0 * np.pi,
        2 * effective_reps,
    )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=effective_reps,
        pass_manager=pass_manager,
        initial_point=initial_point,
        callback=qaoa_callback,
    )

    minimum_eigen_optimizer = MinimumEigenOptimizer(qaoa)

    try:
        result = minimum_eigen_optimizer.solve(quadratic_program)
    except Exception as exc:
        raise RuntimeError(f"QAOA execution failed: {exc}") from exc

    return (
        np.asarray(result.x, dtype=int),
        float(result.fval),
        objective_history,
    )


# =============================================================================
# 4. SOLUTION VALIDATION AND OUTPUT
# =============================================================================

def validate_solution(
    binary_solution: np.ndarray,
    utility: np.ndarray,
    redundancy_matrix: np.ndarray,
    k: int,
) -> Tuple[float, float, int]:
    """Compute utility, redundancy, and selected-cardinality diagnostics."""
    selected_count = int(np.sum(binary_solution))
    if selected_count != k:
        warnings.warn(
            f"The solution selected {selected_count} sensors instead of the "
            f"requested {k}."
        )

    total_utility = float(np.sum(binary_solution * utility))
    redundancy_penalty = 0.0

    selected_indices = np.where(binary_solution == 1)[0]
    for position, index_i in enumerate(selected_indices):
        for index_j in selected_indices[position + 1:]:
            redundancy_penalty += float(redundancy_matrix[index_i, index_j])

    return total_utility, redundancy_penalty, selected_count


def plot_convergence(
    history: List[float],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot raw QAOA evaluations and the best objective value found so far."""
    if not history:
        warnings.warn("The convergence history is empty; no plot was generated.")
        return

    history_array = np.asarray(history, dtype=float)
    best_so_far = np.minimum.accumulate(history_array)

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(history_array) + 1),
        history_array,
        color="gray",
        alpha=0.3,
        label="Raw evaluated energy",
    )
    plt.plot(
        range(1, len(best_so_far) + 1),
        best_so_far,
        color="purple",
        linewidth=2.5,
        label="Best energy found so far",
    )

    plt.title("QAOA convergence")
    plt.xlabel("Optimizer function evaluations")
    plt.ylabel("Expectation value (energy)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"    Convergence figure saved to: {save_path}")

    plt.show()
    plt.close()


def visualize_results(
    utility: np.ndarray,
    redundancy_matrix: np.ndarray,
    selected_indices: np.ndarray,
    sensor_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot sensor utilities and the pairwise redundancy matrix."""
    figure, (axis_utility, axis_redundancy) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    selected_set = set(selected_indices.tolist())
    bar_colors = [
        "red" if index in selected_set else "steelblue"
        for index in range(len(utility))
    ]

    bars = axis_utility.bar(
        range(len(utility)),
        utility,
        color=bar_colors,
        alpha=0.7,
        edgecolor="black",
    )
    axis_utility.set_title("Normalized Shannon entropy")
    axis_utility.set_xlabel("Candidate sensor")
    axis_utility.set_ylabel("Normalized utility")
    axis_utility.set_xticks(range(len(utility)))
    axis_utility.set_xticklabels(sensor_names, rotation=45, ha="right", fontsize=8)
    axis_utility.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, utility):
        axis_utility.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    image = axis_redundancy.imshow(
        redundancy_matrix,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axis_redundancy.set_title("Normalized mutual-information redundancy")
    axis_redundancy.set_xlabel("Candidate sensor")
    axis_redundancy.set_ylabel("Candidate sensor")

    colorbar = figure.colorbar(
        image,
        ax=axis_redundancy,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Normalized mutual information")

    for index in selected_indices:
        axis_redundancy.axhline(y=index, color="gold", linewidth=2, alpha=0.5)
        axis_redundancy.axvline(x=index, color="gold", linewidth=2, alpha=0.5)

    axis_redundancy.set_xticks(range(len(sensor_names)))
    axis_redundancy.set_yticks(range(len(sensor_names)))
    axis_redundancy.set_xticklabels(
        sensor_names,
        rotation=45,
        ha="right",
        fontsize=7,
    )
    axis_redundancy.set_yticklabels(sensor_names, fontsize=7)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"    Sensor-selection figure saved to: {save_path}")

    plt.show()
    plt.close()


def print_detailed_results(
    selected_indices: np.ndarray,
    sensor_names: List[str],
    utility: np.ndarray,
    redundancy_matrix: np.ndarray,
) -> None:
    """Print sensor-level utility and redundancy diagnostics."""
    print("\n" + "=" * 70)
    print("DETAILED SELECTION METRICS")
    print("=" * 70)

    print("\nSelected sensors and associated metrics:")
    print(
        f"{'Sensor':<20} {'Utility':<12} "
        f"{'Redundancy with other selected sensors':<40}"
    )
    print("-" * 85)

    for index in selected_indices:
        other_selected = [
            other_index
            for other_index in selected_indices
            if other_index != index
        ]
        redundancy_values = [
            redundancy_matrix[index, other_index]
            for other_index in other_selected
        ]

        if redundancy_values:
            redundancy_text = (
                f"mean: {np.mean(redundancy_values):.3f}, "
                f"maximum: {np.max(redundancy_values):.3f}"
            )
        else:
            redundancy_text = "not applicable"

        print(
            f"{sensor_names[index]:<20} "
            f"{utility[index]:<12.4f} "
            f"{redundancy_text:<40}"
        )

    if len(selected_indices) > 1:
        print("\nMutual-information matrix for the selected sensors:")
        print(" " * 10, end="")
        for index in selected_indices:
            print(f"{sensor_names[index][:8]:>9}", end="")
        print()

        for index_i in selected_indices:
            print(f"{sensor_names[index_i][:8]:>9}", end=" ")
            for index_j in selected_indices:
                print(f"{redundancy_matrix[index_i, index_j]:9.3f}", end="")
            print()


def print_validation_results(
    total_utility: float,
    redundancy_penalty: float,
    selected_count: int,
    k: int,
) -> None:
    """Print summary validation metrics."""
    print("\n" + "=" * 50)
    print("SOLUTION VALIDATION")
    print("=" * 50)
    print(f"Requested number of sensors: {k}")
    print(f"Selected number of sensors:  {selected_count}")
    print(f"Total utility:               {total_utility:.4f} (higher is better)")
    print(
        f"Redundancy penalty:          {redundancy_penalty:.4f} "
        "(lower is better)"
    )

    if selected_count != k:
        print(
            "WARNING: The exact-cardinality constraint was not satisfied. "
            f"Difference: {abs(selected_count - k)} sensor(s)."
        )


def save_numerical_results(
    selected_indices: np.ndarray,
    sensor_names: List[str],
    utility: np.ndarray,
    redundancy_matrix: np.ndarray,
    optimal_value: float,
    total_utility: float,
    redundancy_penalty: float,
    selected_count: int,
) -> None:
    """Save the selected subset and sensor metrics to JSON and CSV files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    selected_sensors = [sensor_names[index] for index in selected_indices]

    result_payload = {
        "analysis_level": 1,
        "case_study": CASE_STUDY,
        "input_file": str(ACCELERATION_FILE.relative_to(PROJECT_ROOT)),
        "requested_sensor_count": K_SENSORS_TO_SELECT,
        "selected_sensor_count": selected_count,
        "selected_indices_zero_based": selected_indices.tolist(),
        "selected_sensors": selected_sensors,
        "optimal_qubo_energy": optimal_value,
        "total_utility": total_utility,
        "redundancy_penalty": redundancy_penalty,
        "qubo_parameters": {
            "lambda": QUBO_LAMBDA,
            "alpha": QUBO_ALPHA,
        },
        "qaoa_parameters": {
            "reps_requested": QAOA_REPS,
            "maxiter_requested": QAOA_MAXITER,
            "shots": QAOA_SHOTS,
            "random_seed": RANDOM_SEED,
            "use_sparse": USE_SPARSE,
        },
    }

    with SELECTION_RESULTS_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(result_payload, output_file, indent=4)

    selected_set = set(selected_indices.tolist())
    sensor_metrics = pd.DataFrame(
        {
            "sensor_index_zero_based": np.arange(len(sensor_names)),
            "sensor_name": sensor_names,
            "normalized_entropy_utility": utility,
            "selected": [index in selected_set for index in range(len(sensor_names))],
            "mean_redundancy_with_all_sensors": np.mean(redundancy_matrix, axis=1),
            "maximum_redundancy_with_all_sensors": np.max(redundancy_matrix, axis=1),
        }
    )
    sensor_metrics.to_csv(SENSOR_METRICS_FILE, index=False)

    print(f"    Numerical results saved to: {SELECTION_RESULTS_FILE}")
    print(f"    Sensor metrics saved to: {SENSOR_METRICS_FILE}")


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Run the complete Level 1 sensor-selection workflow."""
    if CASE_STUDY not in {1, 2}:
        raise ValueError("CASE_STUDY must be set to either 1 or 2.")

    print("\n" + "=" * 70)
    print("LEVEL 1 SENSOR SELECTION: ENTROPY AND MUTUAL INFORMATION")
    print("=" * 70)
    print(f"Repository root: {PROJECT_ROOT}")
    print(f"Selected case study: {CASE_STUDY}")
    print(f"Input data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    print(f"\n[1] Loading acceleration data from: {ACCELERATION_FILE}")
    data = load_and_prepare_data(ACCELERATION_FILE)
    sensor_names = [str(column) for column in data.columns]
    print(f"    Number of candidate sensors: {len(sensor_names)}")
    print(f"    Number of time samples: {len(data)}")

    print("\n[2] Computing Shannon-entropy utility and mutual-information redundancy...")
    utility, redundancy_matrix = compute_metrics_from_data(data)

    print(f"\n[3] Building the QUBO matrix for k = {K_SENSORS_TO_SELECT}...")
    qubo_matrix = build_qubo_sensor_selection(
        utility=utility,
        redundancy_matrix=redundancy_matrix,
        k=K_SENSORS_TO_SELECT,
        lam=QUBO_LAMBDA,
        alpha=QUBO_ALPHA,
    )

    print("\n[4] Solving the QUBO problem with QAOA on Qiskit Aer...")
    binary_solution, optimal_value, objective_history = solve_qubo_qaoa(
        qubo_matrix=qubo_matrix,
        k=K_SENSORS_TO_SELECT,
        lam=QUBO_LAMBDA,
        reps=QAOA_REPS,
        maxiter=QAOA_MAXITER,
        seed=RANDOM_SEED,
        shots=QAOA_SHOTS,
        use_sparse=USE_SPARSE,
    )

    selected_indices = np.where(binary_solution == 1)[0]
    selected_sensors = [sensor_names[index] for index in selected_indices]

    print("\n" + "=" * 50)
    print("OPTIMAL SENSOR-SELECTION RESULTS")
    print("=" * 50)
    print(f"Requested number of sensors: {K_SENSORS_TO_SELECT}")
    print(f"Selected sensors: {selected_sensors}")
    print(f"Optimal QUBO energy: {optimal_value:.6f}")

    total_utility, redundancy_penalty, selected_count = validate_solution(
        binary_solution=binary_solution,
        utility=utility,
        redundancy_matrix=redundancy_matrix,
        k=K_SENSORS_TO_SELECT,
    )

    print_validation_results(
        total_utility=total_utility,
        redundancy_penalty=redundancy_penalty,
        selected_count=selected_count,
        k=K_SENSORS_TO_SELECT,
    )
    print_detailed_results(
        selected_indices=selected_indices,
        sensor_names=sensor_names,
        utility=utility,
        redundancy_matrix=redundancy_matrix,
    )

    if SAVE_NUMERICAL_RESULTS:
        save_numerical_results(
            selected_indices=selected_indices,
            sensor_names=sensor_names,
            utility=utility,
            redundancy_matrix=redundancy_matrix,
            optimal_value=optimal_value,
            total_utility=total_utility,
            redundancy_penalty=redundancy_penalty,
            selected_count=selected_count,
        )

    if VISUALIZE:
        convergence_path = CONVERGENCE_FIGURE if SAVE_FIGURES else None
        selection_path = SELECTION_FIGURE if SAVE_FIGURES else None

        plot_convergence(objective_history, save_path=convergence_path)
        visualize_results(
            utility=utility,
            redundancy_matrix=redundancy_matrix,
            selected_indices=selected_indices,
            sensor_names=sensor_names,
            save_path=selection_path,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        traceback.print_exc()
        raise SystemExit(1) from error
