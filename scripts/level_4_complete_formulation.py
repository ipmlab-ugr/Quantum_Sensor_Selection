# -*- coding: utf-8 -*-
"""
Level 4 - Complete Sensor-Selection Formulation
------------------------------------------------

This script implements the complete sensor-selection formulation presented in
this repository. It identifies an optimal subset of sensors by combining:

- Shannon-entropy utility;
- spectral utility;
- Fisher information matrix (FIM) utility from OMA results;
- time-domain and spectral redundancy;
- a geometry-based proximity penalty;
- a left-right symmetry-balance penalty;
- an exact-cardinality penalty.

The resulting Quadratic Unconstrained Binary Optimization (QUBO) problem is
solved with the Quantum Approximate Optimization Algorithm (QAOA) using
Qiskit Aer.

Expected repository structure
-----------------------------
Quantum_Sensor_Selection/
|-- data/
|   |-- case_study_1/
|   |   `-- raw/
|   |       |-- acceleration_data.xlsx
|   |       |-- sensor_geometry.xlsx
|   |       `-- oma_results.json
|   `-- case_study_2/
|       `-- raw/
|           |-- acceleration_data.xlsx
|           |-- sensor_geometry.xlsx
|           `-- oma_results.json
|-- scripts/
|   `-- level_4_complete_formulation.py
`-- results/

Only the case-study number and the parameters in the USER CONFIGURATION
section should normally be changed.
"""

import json
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

# Qiskit Optimization & Aer imports
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.utils import algorithm_globals
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import SPSA
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Change CASE_STUDY only in scripts/config.py.
from config import CASE_STUDY, get_case_study_config

CASE_CONFIG = get_case_study_config()
SAMPLE_RATE_HZ = CASE_CONFIG["sampling_rate_hz"]
K_SENSORS_TO_SELECT = CASE_CONFIG["number_of_sensors_to_select"]
SYMMETRY_COORDINATE = CASE_CONFIG["symmetry_coordinate"]
SYMMETRY_AXIS_VALUE = CASE_CONFIG["symmetry_axis_value"]
GEOMETRY_LENGTH_SCALE = CASE_CONFIG["geometry_length_scale"]
GEOMETRY_WEIGHT = CASE_CONFIG["geometry_weight"]
SYMMETRY_WEIGHT = CASE_CONFIG["symmetry_weight"]
QUBO_LAMBDA = CASE_CONFIG["qubo_lambda"]
QUBO_ALPHA = CASE_CONFIG["qubo_alpha"]

# Utility blending weights shared by both case studies.
W_ENTROPY = 0.3
W_SPECTRAL = 0.3
W_FIM = 0.4

# Redundancy blending weights shared by both case studies.
W_MUTUAL_INFORMATION = 0.5
W_SPECTRAL_REDUNDANCY = 0.5

# Number of frequency bands used to compute spectral features.
N_SPECTRAL_BANDS = 4

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
RESULTS_DIR = PROJECT_ROOT / "results" / f"case_study_{CASE_STUDY}" / "level_4"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

ACCELERATION_FILE = DATA_DIR / "acceleration_data.xlsx"
GEOMETRY_FILE = DATA_DIR / "sensor_geometry.xlsx"
OMA_RESULTS_FILE = DATA_DIR / "oma_results.json"

CONVERGENCE_FIGURE = FIGURES_DIR / "qaoa_convergence.png"
SELECTION_FIGURE = FIGURES_DIR / "sensor_selection_visualization.png"
SELECTION_RESULTS_FILE = RESULTS_DIR / "selection_results.json"
SENSOR_METRICS_FILE = TABLES_DIR / "sensor_metric_diagnostics.csv"
DISTANCE_MATRIX_FILE = TABLES_DIR / "sensor_distance_matrix.csv"
PROXIMITY_MATRIX_FILE = TABLES_DIR / "sensor_proximity_matrix.csv"

# =============================================================================
# Utility helpers
# =============================================================================
def _safe_minmax_scale(values: np.ndarray, fill_value: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.min(values), np.max(values)
    if vmax > vmin:
        return (values - vmin) / (vmax - vmin)
    warnings.warn("Min-max scaling received a constant vector.")
    return np.full_like(values, fill_value=fill_value, dtype=float)

def _safe_row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= eps, 1.0, row_sums)
    return matrix / row_sums

# =============================================================================
# OMA / FIM JSON loading helpers
# =============================================================================

def from_serializable(obj):
    if isinstance(obj, dict):
        if "__complex__" in obj:
            return complex(obj["real"], obj["imag"])
        elif "__complex_array__" in obj:
            return np.array(obj["real"]) + 1j * np.array(obj["imag"])
        else:
            return {k: from_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        converted = [from_serializable(v) for v in obj]
        if all(isinstance(x, (float, int, complex, np.number)) for x in converted):
            return np.array(converted)
        elif all(isinstance(x, np.ndarray) for x in converted):
            try:
                return np.stack(converted)
            except Exception:
                return converted
        return converted
    return obj

def load_json_serialized(filepath: Union[str, Path]):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            "OMA results file not found.\n"
            f"Expected file: {filepath}\n"
            "Place the OMA JSON file in the corresponding case-study raw-data directory "
            "and name it 'oma_results.json'."
        )
    with filepath.open('r', encoding='utf-8') as file:
        return from_serializable(json.load(file))

def load_oma_ed_from_json(json_path: Union[str, Path], sensor_names: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load Ed vector from an external OMA/FIM JSON file and align it with the
    sensor order used in the current optimization file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file produced by the standalone OMA code.
    sensor_names : List[str]
        Sensor names as they appear in the main optimization dataframe.

    Returns
    -------
    b_fim : np.ndarray
        Normalized Ed-based utility in [0, 1], aligned with sensor_names.
    oma_diag : pd.DataFrame
        Diagnostic dataframe with raw and normalized Ed values.
    """
    oma_data = load_json_serialized(json_path)

    if "Ed" not in oma_data:
        raise KeyError(f"'Ed' not found in OMA JSON file: {json_path}")

    Ed_raw = np.asarray(oma_data["Ed"], dtype=float).reshape(-1)

    if "sensor_names" in oma_data:
        oma_sensor_names = [str(s) for s in oma_data["sensor_names"]]
        ed_map = {name: Ed_raw[i] for i, name in enumerate(oma_sensor_names)}

        try:
            Ed_aligned = np.array([ed_map[name] for name in sensor_names], dtype=float)
        except KeyError as e:
            raise KeyError(f"Sensor {e} found in optimization file but not in OMA JSON.")
    else:
        # fallback: assume same sensor order
        if len(Ed_raw) != len(sensor_names):
            raise ValueError(
                f"Length mismatch: Ed has length {len(Ed_raw)} but sensor list has length {len(sensor_names)}."
            )
        Ed_aligned = Ed_raw.copy()

    Ed_aligned = np.real_if_close(Ed_aligned).astype(float)
    Ed_aligned = np.clip(Ed_aligned, a_min=0.0, a_max=None)
    ed_max = np.max(Ed_aligned)
    if ed_max > 1e-12:
        b_fim = Ed_aligned / ed_max
    else:
        b_fim = np.zeros_like(Ed_aligned)

    print("Ed_raw from JSON:", Ed_aligned)
    print("Ed range:", np.min(Ed_aligned), np.max(Ed_aligned))

    oma_diag = pd.DataFrame({
        "sensor": sensor_names,
        "Ed_raw": Ed_aligned,
        "b_fim": b_fim,
    })

    return b_fim, oma_diag
# =============================================================================
# 1. Data Processing & Information Theory Metrics
# =============================================================================
def load_and_prep_data(filepath: Union[str, Path]) -> pd.DataFrame:
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filepath}' not found.")
    except Exception as e:
        raise Exception(f"Error loading Excel file: {e}")
    
    drop_candidates = ['time', 't', 'index', 'date', 'timestamp']
    cols_to_drop = [c for c in df.columns if str(c).lower() in drop_candidates]
    if cols_to_drop:
        print(f"    [!] Dropping non-sensor columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    if df.empty:
        raise ValueError("No sensor columns found after dropping time/index columns.")
        
    return df

def load_sensor_geometry(filepath: Union[str, Path]) -> pd.DataFrame:
    geom = pd.read_excel(filepath)
    geom.columns = [str(c).strip().lower() for c in geom.columns]

    required = {"label", "x", "y", "z"}
    if not required.issubset(set(geom.columns)):
        raise ValueError("Geometry file must contain at least: label, x, y, z")

    geom = geom.copy()
    geom["label"] = geom["label"].astype(str).str.strip()
    geom["x"] = pd.to_numeric(geom["x"], errors="coerce")
    geom["y"] = pd.to_numeric(geom["y"], errors="coerce")
    geom["z"] = pd.to_numeric(geom["z"], errors="coerce")

    if geom[["x", "y", "z"]].isna().any().any():
        raise ValueError("Geometry file contains invalid x/y/z coordinates.")

    return geom[["label", "x", "y", "z"]]

def compute_entropy_utility(df: pd.DataFrame) -> np.ndarray:
    """Shannon-entropy-like utility based on signal variance."""
    variances = df.var(ddof=1).values
    variances = np.clip(variances, a_min=1e-10, a_max=None)
    entropy = 0.5 * np.log(2.0 * np.pi * np.e * variances)
    return _safe_minmax_scale(entropy)

def compute_spectral_utility(
    df: pd.DataFrame,
    sample_rate_hz: float,
    nperseg: Optional[int] = None,
    n_bands: int = 4,
    peak_prominence_weight: float = 0.4,
    band_energy_weight: float = 0.6,
    low_freq_bias: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Compute a first spectral utility score for each sensor using:
    1) weighted PSD band energy distribution
    2) dominant peak prominence (peak-to-background contrast)

    Returns
    -------
    b_spec : np.ndarray
        Normalized spectral utility in [0, 1].
    diagnostics : pd.DataFrame
        Per-sensor diagnostic values for interpretation.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    signals = df.to_numpy(dtype=float)
    n_samples, n_sensors = signals.shape
    if nperseg is None:
        nperseg = min(1024, max(256, n_samples // 8))
    nperseg = min(nperseg, n_samples)

    band_scores = np.zeros(n_sensors, dtype=float)
    prominence_scores = np.zeros(n_sensors, dtype=float)
    dom_freqs = np.zeros(n_sensors, dtype=float)
    dom_peak_vals = np.zeros(n_sensors, dtype=float)
    band_ratio_matrix = np.zeros((n_sensors, n_bands), dtype=float)

    for idx in range(n_sensors):
        sig = signals[:, idx]
        freqs, psd = welch(sig, fs=sample_rate_hz, nperseg=nperseg, detrend="constant")

        # Remove DC to focus on structural dynamics.
        dyn_mask = freqs > 0
        freqs_dyn = freqs[dyn_mask]
        psd_dyn = psd[dyn_mask]

        if len(freqs_dyn) == 0 or np.all(psd_dyn <= 0):
            continue

        total_energy = np.trapezoid(psd_dyn, freqs_dyn)
        if total_energy <= 0:
            continue

        # 1) Weighted band-energy score
        band_edges = np.linspace(freqs_dyn[0], freqs_dyn[-1], n_bands + 1)
        band_energies = []
        for b in range(n_bands):
            lo, hi = band_edges[b], band_edges[b + 1]
            # Include upper edge in final band.
            if b == n_bands - 1:
                band_mask = (freqs_dyn >= lo) & (freqs_dyn <= hi)
            else:
                band_mask = (freqs_dyn >= lo) & (freqs_dyn < hi)
            if np.any(band_mask):
                e_band = np.trapezoid(psd_dyn[band_mask], freqs_dyn[band_mask])
            else:
                e_band = 0.0
            band_energies.append(e_band)

        band_energies = np.asarray(band_energies, dtype=float)
        band_ratios = band_energies / max(total_energy, 1e-12)
        band_ratio_matrix[idx, :] = band_ratios

        if low_freq_bias:
            # Favor lower bands slightly, because they often carry the most global structural content.
            weights = np.linspace(1.0, 0.5, n_bands)
        else:
            weights = np.ones(n_bands, dtype=float)
        weights = weights / np.sum(weights)
        band_scores[idx] = float(np.dot(weights, band_ratios))

        # 2) Dominant peak prominence score
        bg_level = float(np.median(psd_dyn)) + 1e-12
        peaks, props = find_peaks(psd_dyn, prominence=bg_level * 0.05)
        if len(peaks) > 0:
            prominences = props.get("prominences", np.array([]))
            if prominences.size > 0:
                max_peak_idx = int(np.argmax(prominences))
                prominence_scores[idx] = float(prominences[max_peak_idx] / bg_level)
                peak_loc = peaks[max_peak_idx]
                dom_freqs[idx] = float(freqs_dyn[peak_loc])
                dom_peak_vals[idx] = float(psd_dyn[peak_loc])
        else:
            peak_loc = int(np.argmax(psd_dyn))
            dom_freqs[idx] = float(freqs_dyn[peak_loc])
            dom_peak_vals[idx] = float(psd_dyn[peak_loc])
            prominence_scores[idx] = float(psd_dyn[peak_loc] / bg_level)

    band_scores_n = _safe_minmax_scale(band_scores, fill_value=0.0)
    prominence_scores_n = _safe_minmax_scale(prominence_scores, fill_value=0.0)

    b_spec = band_energy_weight * band_scores_n + peak_prominence_weight * prominence_scores_n
    b_spec = _safe_minmax_scale(b_spec, fill_value=0.0)

    diagnostics = pd.DataFrame(
        {
            "sensor": df.columns,
            "band_score_raw": band_scores,
            "peak_prominence_raw": prominence_scores,
            "dominant_frequency_hz": dom_freqs,
            "dominant_peak_psd": dom_peak_vals,
            "b_spec": b_spec,
        }
    )
    return b_spec, diagnostics, band_ratio_matrix

def compute_band_correlation_from_ratios(
    band_ratio_matrix: np.ndarray,
) -> np.ndarray:
    """
    Build a spectral redundancy matrix from similarity between
    per-sensor band-energy ratio vectors.
    Uses cosine similarity and returns values in [0, 1].
    """
    X = _safe_row_normalize(band_ratio_matrix)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    Xn = X / norms

    Corr_band = Xn @ Xn.T
    Corr_band = np.clip(Corr_band, 0.0, 1.0)
    np.fill_diagonal(Corr_band, 0.0)
    return Corr_band



def compute_metrics_from_data(
    df: pd.DataFrame,
    sample_rate_hz: float,
    w_entropy: float = 0.3,
    w_spectral: float = 0.3,
    w_fim: float = 0.4,
    n_bands: int = 4,
    w_mi: float = 0.5,
    w_band_corr: float = 0.5,
    oma_json_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Compute:
    - blended utility from entropy, spectral, and FIM contributions
    - redundancy Corr from blended time-domain MI + spectral band correlation
    - diagnostics dataframe for transparency
    """
    total_w = w_entropy + w_spectral + w_fim
    if not np.isclose(total_w, 1.0):
        if total_w <= 0:
            raise ValueError("w_entropy + w_spectral + w_fim must be > 0")
        w_entropy = w_entropy / total_w
        w_spectral = w_spectral / total_w
        w_fim = w_fim / total_w
        warnings.warn("Weights normalized so that w_entropy + w_spectral + w_fim = 1.")

    # A) Time-domain individual informativeness
    b_entropy = compute_entropy_utility(df)

    # B) Spectral utility
    b_spec, spectral_diag, band_ratio_matrix = compute_spectral_utility(
    df=df,
    sample_rate_hz=sample_rate_hz,
    n_bands=n_bands,
    )

    # C) OMA/FIM-based utility from external JSON
    if oma_json_path is None:
        raise ValueError("oma_json_path must be provided to load Ed-based FIM utility.")
    b_fim, oma_diag = load_oma_ed_from_json(
        json_path=oma_json_path,
        sensor_names=df.columns.tolist(),
        )

    # Blended utility
    b = w_entropy * b_entropy + w_spectral * b_spec + w_fim * b_fim
    b = _safe_minmax_scale(b)

    # Redundancy from mutual information (same as original script)
    # --- Time-domain redundancy from mutual information ---
    pearson_corr = df.corr().values
    pearson_corr = np.clip(pearson_corr, a_min=-0.9999, a_max=0.9999)
    mutual_info = -0.5 * np.log(1.0 - pearson_corr ** 2)
    np.fill_diagonal(mutual_info, 0.0)
    
    mi_max = np.max(mutual_info)
    Corr_mi = mutual_info / mi_max if mi_max > 0 else mutual_info
    
    # --- Spectral redundancy from band-energy similarity ---
    Corr_band = compute_band_correlation_from_ratios(band_ratio_matrix)
    
    # --- Blend the two redundancy terms ---
    if not np.isclose(w_mi + w_band_corr, 1.0):
        total_corr_w = w_mi + w_band_corr
        if total_corr_w <= 0:
            raise ValueError("w_mi + w_band_corr must be > 0") 
        w_mi = w_mi / total_corr_w
        w_band_corr = w_band_corr / total_corr_w
        warnings.warn("Correlation weights normalized so that w_mi + w_band_corr = 1.")
    
    Corr = w_mi * Corr_mi + w_band_corr * Corr_band
    Corr = _safe_minmax_scale(Corr, fill_value=0.0)
    np.fill_diagonal(Corr, 0.0)

    diagnostics = spectral_diag.copy()
    diagnostics["b_entropy"] = b_entropy
    diagnostics["Ed_raw"] = oma_diag["Ed_raw"].values
    diagnostics["b_fim"] = b_fim
    diagnostics["b_total"] = b
    diagnostics["dominant_band_idx"] = np.argmax(band_ratio_matrix, axis=1)
    diagnostics = diagnostics[
        [
            "sensor",
            "b_entropy",
            "b_spec",
            "b_fim",
            "Ed_raw",
            "b_total",
            "band_score_raw",
            "peak_prominence_raw",
            "dominant_frequency_hz",
            "dominant_peak_psd",
            "dominant_band_idx",
        ]
    ]

    return b, Corr, diagnostics

def build_distance_matrix(sensor_names: List[str], geom_df: pd.DataFrame) -> np.ndarray:
    geom_idx = geom_df.set_index("label")

    coords = []
    for s in sensor_names:
        if s not in geom_idx.index:
            raise ValueError(f"Sensor '{s}' not found in geometry file.")
        coords.append([geom_idx.loc[s, "x"], geom_idx.loc[s, "y"], geom_idx.loc[s, "z"]])

    coords = np.asarray(coords, dtype=float)
    N = len(sensor_names)
    D = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(coords[i] - coords[j])

    return D

def build_proximity_matrix(dist_matrix: np.ndarray, length_scale: float = 20.0) -> np.ndarray:
    Near = np.exp(-dist_matrix / length_scale)
    np.fill_diagonal(Near, 0.0)
    return Near

def infer_left_right_groups(sensor_names: List[str], geom_df: pd.DataFrame,
    symmetry_coord: str = "x", axis_value: Optional[float] = None, tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Infer left and right sensor groups from the selected coordinate.
    The symmetry axis/plane is assumed to be coord = axis_value.
    If axis_value is not provided, it is estimated as the median coordinate.
    Sensors exactly on the axis are excluded from both groups.
    """
    geom_idx = geom_df.set_index("label")

    coord_vals = []
    for s in sensor_names:
        if s not in geom_idx.index:
            raise ValueError(f"Sensor '{s}' not found in geometry file.")
        coord_vals.append(float(geom_idx.loc[s, symmetry_coord]))

    coord_vals = np.asarray(coord_vals, dtype=float)

    if axis_value is None:
        axis_value = float(np.median(coord_vals))

    left_idx = np.where(coord_vals < axis_value - tol)[0]
    right_idx = np.where(coord_vals > axis_value + tol)[0]

    return left_idx, right_idx, axis_value

# =============================================================================
# 2. QUBO Formulation
# =============================================================================
def build_qubo_sensor_selection(b: np.ndarray, Corr: np.ndarray, k: int, prox_matrix: Optional[np.ndarray] = None,
                                left_indices: Optional[np.ndarray] = None, right_indices: Optional[np.ndarray] = None,
                                lam: float = 10.0, alpha: float = 1.0, gamma: float = 1.0, eta_sym: float = 1.0,) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    Corr = np.asarray(Corr, dtype=float)
    N = b.shape[0]
    
    Q = np.zeros((N, N), dtype=float)

    #Utility term 
    Q[np.arange(N), np.arange(N)] += -b
    
    #Redundancy + proximity 
    for i in range(N):
        for j in range(i + 1, N):
            redundancy_penalty = alpha * Corr[i, j]
            geometry_penalty = 0.0
            if prox_matrix is not None:
                geometry_penalty = gamma * prox_matrix[i, j]

            Q[i, j] += redundancy_penalty + geometry_penalty

    # Symmetry balance term:
    # eta_sym * (sum_{i in L} x_i - sum_{j in R} x_j)^2
    if left_indices is not None and right_indices is not None:
        left_indices = np.asarray(left_indices, dtype=int)
        right_indices = np.asarray(right_indices, dtype=int)

        # diagonal terms: x_i^2 = x_i
        for i in left_indices:
            Q[i, i] += eta_sym
        for j in right_indices:
            Q[j, j] += eta_sym

        # same-side pair penalties: +2 eta_sym
        for a in range(len(left_indices)):
            for b_idx in range(a + 1, len(left_indices)):
                i = left_indices[a]
                j = left_indices[b_idx]
                Q[i, j] += 2.0 * eta_sym

        for a in range(len(right_indices)):
            for b_idx in range(a + 1, len(right_indices)):
                i = right_indices[a]
                j = right_indices[b_idx]
                Q[i, j] += 2.0 * eta_sym

        # cross-side reward: -2 eta_sym
        for i in left_indices:
            for j in right_indices:
                ii, jj = min(i, j), max(i, j)
                Q[ii, jj] += -2.0 * eta_sym

    # Cardinality constraint 
    Q[np.arange(N), np.arange(N)] += lam * (1 - 2 * k)
    for i in range(N):
        for j in range(i + 1, N):
            Q[i, j] += 2 * lam

    return Q

def build_qubo_sparse_from_matrix(Q: np.ndarray) -> Tuple[Dict, Dict]:
    N = Q.shape[0]
    linear = {f"x{i}": float(Q[i, i]) for i in range(N) if abs(Q[i, i]) > 1e-10}
    quadratic = {}
    for i in range(N):
        for j in range(i+1, N):
            if abs(Q[i, j]) > 1e-10:
                quadratic[(f"x{i}", f"x{j}")] = float(Q[i, j])
    return linear, quadratic

# =============================================================================
# 3. QAOA Solver
# =============================================================================
def solve_qubo_qaoa(Q: np.ndarray, k: int, lam: float, reps: int = 3, maxiter: int = 100, 
                    seed: int = 42, shots: int = 2048, 
                    use_sparse: bool = False) -> Tuple[np.ndarray, float, List[float]]:
    Q = np.asarray(Q, dtype=float)
    N = Q.shape[0]
    
    if N > 25:
        warnings.warn(f"Problem size {N} is large for QAOA on classical simulator.")
    
    if N > 15 and reps < 4:
        reps = max(reps, 4)
        print(f"    [!] Increasing QAOA reps to {reps} for larger problem size")
    
    maxiter = max(maxiter, 200 if N > 10 else 100)
    algorithm_globals.random_seed = seed

    qp = QuadraticProgram()
    for i in range(N):
        qp.binary_var(name=f"x{i}")

    constant_offset = lam * (k ** 2)

    if use_sparse:
        linear, quadratic = build_qubo_sparse_from_matrix(Q)
        qp.minimize(constant=constant_offset, linear=linear, quadratic=quadratic)
    else:
        linear = {f"x{i}": float(Q[i, i]) for i in range(N)}
        quadratic = {(f"x{i}", f"x{j}"): float(Q[i, j]) 
                     for i in range(N) for j in range(i + 1, N) 
                     if abs(Q[i, j]) > 1e-10}
        qp.minimize(constant=constant_offset, linear=linear, quadratic=quadratic)

    optimizer = SPSA(maxiter=maxiter)
    sampler = SamplerV2(seed=seed, default_shots=shots)
    pass_manager = generate_preset_pass_manager(optimization_level=1, seed_transpiler=seed)

    objective_history = []
    
    def qaoa_callback(eval_count, parameters, mean, metadata):
        objective_history.append(mean)

    qaoa = QAOA(sampler=sampler, 
                optimizer=optimizer, 
                reps=reps, 
                pass_manager=pass_manager,
                initial_point=np.random.default_rng(seed).uniform(0, 2*np.pi, 2*reps),
                callback=qaoa_callback)
    
    meo = MinimumEigenOptimizer(qaoa)
    
    try:
        result = meo.solve(qp)
    except Exception as e:
        raise RuntimeError(f"QAOA solver failed: {e}")
    
    return np.array(result.x, dtype=int), result.fval, objective_history

# =============================================================================
# 4. Solution Validation & Visualization
# =============================================================================
def validate_solution(x_opt: np.ndarray, b: np.ndarray, Corr: np.ndarray, 
                     k: int, prox_matrix: Optional[np.ndarray] = None, gamma: float = 1.0,) -> Tuple[float, float, float, int]:
    selected_count = np.sum(x_opt)
    if selected_count != k:
        warnings.warn(f"Solution selected {selected_count} sensors, expected {k}")
    
    utility_score = -np.sum(x_opt * b)
    redundancy_penalty = 0.0
    geometry_penalty = 0.0
    selected_indices = np.where(x_opt == 1)[0]
    for i, idx_i in enumerate(selected_indices):
        for idx_j in selected_indices[i+1:]:
            redundancy_penalty += Corr[idx_i, idx_j]
            if prox_matrix is not None:
                geometry_penalty += gamma * prox_matrix[idx_i, idx_j]
    
    return utility_score, redundancy_penalty, geometry_penalty, selected_count

def plot_convergence(history: List[float], save_path: Optional[Union[str, Path]] = None):
    """Plot the Best Objective Value so far vs. QAOA Iterations."""
    history_array = np.array(history)
    
    # Compute the running minimum to show the "best so far" curve
    best_so_far = np.minimum.accumulate(history_array)
    
    plt.figure(figsize=(10, 5))
    
    # Plot noisy raw evaluations in the background
    plt.plot(range(1, len(history_array) + 1), history_array, 
             color='gray', alpha=0.3, label='Raw Evaluated Energy')
             
    # Plot the clean "best so far" curve in the foreground
    plt.plot(range(1, len(best_so_far) + 1), best_so_far, 
             color='purple', linewidth=2.5, label='Best Energy So Far')
             
    plt.title('QAOA Convergence: Best Objective Value vs. Iterations', fontsize=12, fontweight='bold')
    plt.xlabel('Optimizer Function Evaluations', fontsize=10)
    plt.ylabel('Expectation Value (Energy)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n    Convergence plot saved to: {save_path}")
    
    plt.show()

def visualize_results(b: np.ndarray, Corr: np.ndarray, selected_indices: np.ndarray,
                     sensor_names: List[str], save_path: Optional[Union[str, Path]] = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['red' if i in selected_indices else 'steelblue' for i in range(len(b))]
    bars = ax1.bar(range(len(b)), b, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Blended Sensor Utility (Selected in Red)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sensor Index', fontsize=10)
    ax1.set_ylabel('Normalized Utility Score', fontsize=10)
    ax1.set_xticks(range(len(b)))
    ax1.set_xticklabels(sensor_names, rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, b):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    im = ax2.imshow(Corr, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax2.set_title('Redundancy Matrix (MI + Band Correlation)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sensor Index', fontsize=10)
    ax2.set_ylabel('Sensor Index', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Blended Redundancy', fontsize=9)
    
    for i in selected_indices:
        ax2.axhline(y=i, color='gold', linewidth=2, alpha=0.5)
        ax2.axvline(x=i, color='gold', linewidth=2, alpha=0.5)
    
    ax2.set_xticks(range(len(sensor_names)))
    ax2.set_yticks(range(len(sensor_names)))
    ax2.set_xticklabels(sensor_names, rotation=45, ha='right', fontsize=7)
    ax2.set_yticklabels(sensor_names, fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n    Metrics visualization saved to: {save_path}")
    
    plt.show()

def print_detailed_results(
    selected_indices: np.ndarray,
    sensor_names: List[str],
    b: np.ndarray,
    Corr: np.ndarray,
    diagnostics: pd.DataFrame,
    prox_matrix: Optional[np.ndarray] = None,
):
    print("\n" + "=" * 110)
    print("DETAILED SELECTION METRICS")
    print("=" * 110)

    print("\nSelected sensors with metrics:")
    print(
        f"{'Sensor':<18} "
        f"{'b_total':<10} {'b_entropy':<10} {'b_spec':<10} {'b_fim':<10}"
        f"{'band_raw':<10} {'peak_raw':<10} "
        f"{'dom_f(Hz)':<10} {'avg Corr':<10} {'max Corr':<10}"
        f"{'avg Prox':<10} {'max Prox':<10}"
    )
    print("-" * 110)

    diagnostics_idx = diagnostics.set_index("sensor")

    for idx in selected_indices:
        name = sensor_names[idx]
        other_selected = [j for j in selected_indices if j != idx]
        corr_values = [Corr[idx, j] for j in other_selected]

        avg_mi = np.mean(corr_values) if corr_values else 0.0
        max_mi = np.max(corr_values) if corr_values else 0.0

        if prox_matrix is not None:
            prox_values = [prox_matrix[idx, j] for j in other_selected]
            avg_prox = np.mean(prox_values) if prox_values else 0.0
            max_prox = np.max(prox_values) if prox_values else 0.0
        else:
            avg_prox = 0.0
            max_prox = 0.0

        row = diagnostics_idx.loc[name]

        print(
            f"{name:<18} "
            f"{row['b_total']:<10.4f} {row['b_entropy']:<10.4f} {row['b_spec']:<10.4f} {row['b_fim']:<10.4f} "
            f"{row['band_score_raw']:<10.4f} {row['peak_prominence_raw']:<10.4f} "
            f"{row['dominant_frequency_hz']:<10.3f} {avg_mi:<10.4f} {max_mi:<10.4f}"
            f"{avg_prox:<10.4f} {max_prox:<10.4f}"
        )

    if len(selected_indices) > 1:
        print("\nBlended redundancy matrix for selected sensors:")
        print(" " * 14, end="")
        for idx in selected_indices:
            print(f"{sensor_names[idx][:10]:>10}", end="")
        print()

        for idx_i in selected_indices:
            print(f"{sensor_names[idx_i][:12]:>12}  ", end="")
            for idx_j in selected_indices:
                print(f"{Corr[idx_i, idx_j]:10.3f}", end="")
            print()

    print("\nInterpretation:")
    print("- b_entropy  : utility contribution from time-domain variability / entropy")
    print("- b_spec     : utility contribution from spectral features")
    print("- b_fim      : utility contribution from OMA/FIM through the Ed vector")
    print("- band_raw   : raw band-energy score before normalization")
    print("- peak_raw   : raw peak-prominence score before normalization")
    print("- dom_f(Hz)  : dominant frequency from PSD")
    print("- avg Corr     : average blended redundancy with the other selected sensors")
    print("- max Corr     : worst-case blended redundancy with the other selected sensors")
    print("- avg Prox  : average geometric proximity with the other selected sensors")
    print("- max Prox  : worst-case geometric proximity with the other selected sensors")

def print_validation_results(utility_score: float, redundancy_penalty: float, geometry_penalty: float, 
                            selected_count: int, k: int):
    print("\n" + "="*50)
    print("SOLUTION VALIDATION")
    print("="*50)
    print(f"Expected sensors (k):    {k}")
    print(f"Selected sensors:        {selected_count}")
    print(f"Utility score:   {-utility_score:.4f} (higher is better)")
    print(f"Redundancy penalty:     {redundancy_penalty:.4f} (lower is better)")
    print(f"Geometry penalty:        {geometry_penalty:.4f} (lower is better)")
    if selected_count != k:
        print(f"WARNING: Constraint violation! Off by {abs(selected_count - k)}")

# =============================================================================
# 5. Result serialization
# =============================================================================
def save_selection_results(
    output_path: Union[str, Path],
    selected_indices: np.ndarray,
    selected_sensors: List[str],
    optimal_value: float,
    utility_score: float,
    redundancy_penalty: float,
    geometry_penalty: float,
    selected_count: int,
    symmetry_axis_value: float,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    sensor_names: List[str],
) -> None:
    """Save the main numerical results and analysis settings to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "case_study": CASE_STUDY,
        "selected_sensor_count_requested": K_SENSORS_TO_SELECT,
        "selected_sensor_count_obtained": int(selected_count),
        "selected_indices_zero_based": [int(i) for i in selected_indices],
        "selected_sensors": selected_sensors,
        "optimal_qubo_energy": float(optimal_value),
        "utility_sum": float(-utility_score),
        "redundancy_penalty_unweighted": float(redundancy_penalty),
        "geometry_penalty_weighted": float(geometry_penalty),
        "sampling_rate_hz": SAMPLE_RATE_HZ,
        "utility_weights": {
            "entropy": W_ENTROPY,
            "spectral": W_SPECTRAL,
            "fim": W_FIM,
        },
        "redundancy_weights": {
            "mutual_information": W_MUTUAL_INFORMATION,
            "spectral_redundancy": W_SPECTRAL_REDUNDANCY,
        },
        "qubo_parameters": {
            "lambda_cardinality": QUBO_LAMBDA,
            "alpha_redundancy": QUBO_ALPHA,
            "gamma_geometry": GEOMETRY_WEIGHT,
            "eta_symmetry": SYMMETRY_WEIGHT,
        },
        "geometry": {
            "length_scale": GEOMETRY_LENGTH_SCALE,
            "symmetry_coordinate": SYMMETRY_COORDINATE,
            "symmetry_axis_value": float(symmetry_axis_value),
            "left_sensors": [sensor_names[int(i)] for i in left_indices],
            "right_sensors": [sensor_names[int(i)] for i in right_indices],
        },
        "qaoa_settings": {
            "reps_requested": QAOA_REPS,
            "maxiter_requested": QAOA_MAXITER,
            "shots": QAOA_SHOTS,
            "random_seed": RANDOM_SEED,
        },
        "input_files": {
            "acceleration_data": str(ACCELERATION_FILE.relative_to(PROJECT_ROOT)),
            "sensor_geometry": str(GEOMETRY_FILE.relative_to(PROJECT_ROOT)),
            "oma_results": str(OMA_RESULTS_FILE.relative_to(PROJECT_ROOT)),
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


# =============================================================================
# 6. Main execution
# =============================================================================
def main() -> None:
    try:
        required_files = [ACCELERATION_FILE, GEOMETRY_FILE, OMA_RESULTS_FILE]
        missing_files = [path for path in required_files if not path.exists()]
        if missing_files:
            missing_list = "\n".join(f"- {path}" for path in missing_files)
            raise FileNotFoundError(
                "One or more required input files are missing:\n"
                f"{missing_list}\n\n"
                "Place the files in the selected case-study raw-data directory:\n"
                f"{DATA_DIR}"
            )

        if SAVE_FIGURES or SAVE_NUMERICAL_RESULTS:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if SAVE_FIGURES:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        if SAVE_NUMERICAL_RESULTS:
            TABLES_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\nCase study: {CASE_STUDY}")
        print(f"Input directory: {DATA_DIR}")
        print(f"Output directory: {RESULTS_DIR}")

        print(f"\n[1] Loading acceleration data from {ACCELERATION_FILE}...")
        df = load_and_prep_data(ACCELERATION_FILE)
        sensor_names = [str(name).strip() for name in df.columns]
        df.columns = sensor_names
        print(f"    Number of candidate sensors: {len(sensor_names)}")

        if not 1 <= K_SENSORS_TO_SELECT <= len(sensor_names):
            raise ValueError(
                "K_SENSORS_TO_SELECT must be between 1 and the number of "
                f"candidate sensors ({len(sensor_names)})."
            )

        print("\n[2] Computing entropy, spectral, and FIM utilities...")
        b, Corr, diagnostics = compute_metrics_from_data(
            df=df,
            sample_rate_hz=SAMPLE_RATE_HZ,
            w_entropy=W_ENTROPY,
            w_spectral=W_SPECTRAL,
            w_fim=W_FIM,
            n_bands=N_SPECTRAL_BANDS,
            w_mi=W_MUTUAL_INFORMATION,
            w_band_corr=W_SPECTRAL_REDUNDANCY,
            oma_json_path=OMA_RESULTS_FILE,
        )

        print("\n[3] Loading geometry and building geometric penalties...")
        geom_df = load_sensor_geometry(GEOMETRY_FILE)
        distance_matrix = build_distance_matrix(sensor_names, geom_df)
        proximity_matrix = build_proximity_matrix(
            distance_matrix,
            length_scale=GEOMETRY_LENGTH_SCALE,
        )

        left_indices, right_indices, symmetry_axis = infer_left_right_groups(
            sensor_names,
            geom_df,
            symmetry_coord=SYMMETRY_COORDINATE,
            axis_value=SYMMETRY_AXIS_VALUE,
        )
        print(
            f"    Symmetry axis: {SYMMETRY_COORDINATE} = "
            f"{symmetry_axis:.6g}"
        )
        print(f"    Left-side sensors: {[sensor_names[i] for i in left_indices]}")
        print(f"    Right-side sensors: {[sensor_names[i] for i in right_indices]}")

        if len(left_indices) == 0 or len(right_indices) == 0:
            warnings.warn(
                "One side of the inferred symmetry partition is empty. "
                "Review SYMMETRY_COORDINATE or SYMMETRY_AXIS_VALUE."
            )

        if SAVE_NUMERICAL_RESULTS:
            diagnostics.to_csv(SENSOR_METRICS_FILE, index=False)
            pd.DataFrame(
                distance_matrix,
                index=sensor_names,
                columns=sensor_names,
            ).to_csv(DISTANCE_MATRIX_FILE)
            pd.DataFrame(
                proximity_matrix,
                index=sensor_names,
                columns=sensor_names,
            ).to_csv(PROXIMITY_MATRIX_FILE)
            print(f"    Sensor metrics saved to: {SENSOR_METRICS_FILE}")

        print(f"\n[4] Building the complete QUBO matrix (k={K_SENSORS_TO_SELECT})...")
        Q = build_qubo_sensor_selection(
            b=b,
            Corr=Corr,
            k=K_SENSORS_TO_SELECT,
            prox_matrix=proximity_matrix,
            left_indices=left_indices,
            right_indices=right_indices,
            lam=QUBO_LAMBDA,
            alpha=QUBO_ALPHA,
            gamma=GEOMETRY_WEIGHT,
            eta_sym=SYMMETRY_WEIGHT,
        )

        print("\n[5] Solving the QUBO with QAOA on Qiskit Aer...")
        x_opt, optimal_value, objective_history = solve_qubo_qaoa(
            Q=Q,
            k=K_SENSORS_TO_SELECT,
            lam=QUBO_LAMBDA,
            reps=QAOA_REPS,
            maxiter=QAOA_MAXITER,
            seed=RANDOM_SEED,
            shots=QAOA_SHOTS,
            use_sparse=USE_SPARSE,
        )

        selected_indices = np.where(x_opt == 1)[0]
        selected_sensors = [sensor_names[i] for i in selected_indices]

        print("\n" + "=" * 56)
        print("OPTIMAL SENSOR-SELECTION RESULTS")
        print("=" * 56)
        print(f"Requested number of sensors: {K_SENSORS_TO_SELECT}")
        print(f"Selected sensors: {selected_sensors}")
        print(f"Optimal QUBO energy: {optimal_value:.6f}")

        utility_score, redundancy_penalty, geometry_penalty, selected_count = validate_solution(
            x_opt=x_opt,
            b=b,
            Corr=Corr,
            k=K_SENSORS_TO_SELECT,
            prox_matrix=proximity_matrix,
            gamma=GEOMETRY_WEIGHT,
        )

        print_validation_results(
            utility_score,
            redundancy_penalty,
            geometry_penalty,
            selected_count,
            K_SENSORS_TO_SELECT,
        )
        print_detailed_results(
            selected_indices,
            sensor_names,
            b,
            Corr,
            diagnostics,
            prox_matrix=proximity_matrix,
        )

        if SAVE_NUMERICAL_RESULTS:
            save_selection_results(
                output_path=SELECTION_RESULTS_FILE,
                selected_indices=selected_indices,
                selected_sensors=selected_sensors,
                optimal_value=optimal_value,
                utility_score=utility_score,
                redundancy_penalty=redundancy_penalty,
                geometry_penalty=geometry_penalty,
                selected_count=selected_count,
                symmetry_axis_value=symmetry_axis,
                left_indices=left_indices,
                right_indices=right_indices,
                sensor_names=sensor_names,
            )
            print(f"    Selection summary saved to: {SELECTION_RESULTS_FILE}")

        if VISUALIZE:
            convergence_path = CONVERGENCE_FIGURE if SAVE_FIGURES else None
            plot_convergence(objective_history, save_path=convergence_path)

            selection_path = SELECTION_FIGURE if SAVE_FIGURES else None
            visualize_results(
                b,
                Corr,
                selected_indices,
                sensor_names,
                save_path=selection_path,
            )

    except Exception as exc:
        print(f"\nERROR: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
