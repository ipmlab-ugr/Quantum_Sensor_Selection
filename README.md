# Hybrid Quantum Sensor Selection for Structural Health Monitoring

This repository contains the Python implementation accompanying the paper on **hybrid quantum optimization for optimal sensor placement in Structural Health Monitoring (SHM)**.

The proposed methodology formulates the Optimal Sensor Placement (OSP) problem as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and solves it using the **Quantum Approximate Optimization Algorithm (QAOA)**. Four progressive optimization formulations are provided, corresponding to the methodology presented in the paper.

---

# Repository Structure

```
Quantum_Sensor_Selection/
│
├── data/
│   ├── case_study_1/
│   │   └── raw/
│   │       ├── acceleration_data.xlsx
│   │       ├── sensor_geometry.xlsx
│   │       └── oma_results.json
│   │
│   └── case_study_2/
│       └── raw/
│           ├── acceleration_data.xlsx
│           ├── sensor_geometry.xlsx
│           └── oma_results.json
│
├── scripts/
│   ├── config.py
│   ├── level_1_entropy_mutual_information.py
│   ├── level_2_entropy_spectral.py
│   ├── level_3_entropy_spectral_fim.py
│   └── level_4_complete_formulation.py
│
├── results/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# Methodology

The repository reproduces the four progressive optimization formulations presented in the paper.

| Script | Description |
|---------|-------------|
| `level_1_entropy_mutual_information.py` | Sensor selection based on Shannon entropy and mutual-information-based redundancy. |
| `level_2_entropy_spectral.py` | Adds spectral utility to the Level 1 formulation. |
| `level_3_entropy_spectral_fim.py` | Incorporates Fisher Information Matrix (FIM) utility. |
| `level_4_complete_formulation.py` | Complete proposed formulation including utility, redundancy, geometric proximity and symmetry constraints. |

The four scripts can be executed independently or sequentially to reproduce the progressive development of the proposed methodology.

---

# Installation

Clone the repository

```bash
git clone https://github.com/fmeligeni/Quantum_Sensor_Selection.git
cd Quantum_Sensor_Selection
```

It is recommended to create a dedicated virtual environment.

Windows PowerShell

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install all required packages

```bash
pip install -r requirements.txt
```

---

# Input Data

Two independent case studies are included.

Each case study requires three input files located inside

```
data/case_study_X/raw/
```

where **X = 1** or **2**.

| File | Description |
|------|-------------|
| `acceleration_data.xlsx` | Time-domain acceleration measurements (one column per candidate sensor). |
| `sensor_geometry.xlsx` | Coordinates of the candidate sensor locations. |
| `oma_results.json` | Operational Modal Analysis results (modal frequencies and mode shapes) used to compute the Fisher Information Matrix. |

The filenames should not be modified unless the corresponding paths are also updated in `scripts/config.py`.

---

# Reproducing the Results

The repository has been designed so that the analyses presented in the paper can be reproduced with only a few steps.

## Step 1 – Select the case study

Open

```
scripts/config.py
```

and modify

```python
CASE_STUDY = 1
```

to

```python
CASE_STUDY = 2
```

if you want to reproduce the second case study.

The configuration file automatically selects:

- the appropriate input directory;
- the corresponding optimization parameters;
- the sampling frequency;
- the number of sensors to be selected;
- the geometric parameters;
- the symmetry settings;
- the output directory.

No modifications are required inside the optimization scripts.

---

## Step 2 – Run the desired optimization level

### Level 1

```bash
python scripts/level_1_entropy_mutual_information.py
```

### Level 2

```bash
python scripts/level_2_entropy_spectral.py
```

### Level 3

```bash
python scripts/level_3_entropy_spectral_fim.py
```

### Level 4

```bash
python scripts/level_4_complete_formulation.py
```

Each script reproduces one of the four formulations discussed in the paper.

To reproduce the complete progressive methodology, simply execute the four scripts sequentially.

---

## Step 3 – Generated Outputs

For each execution, the scripts automatically create the corresponding output directory inside

```
results/
```

The generated outputs may include:

- selected sensor subset;
- normalized utility values;
- redundancy matrices;
- diagnostic tables;
- convergence plots;
- graphical visualization of the selected sensors;
- JSON summary of the optimization results.

The output folders are automatically organized according to the selected case study and optimization level.

---

# Reproducing the Complete Study

To reproduce **Case Study I**:

1. Set `CASE_STUDY = 1` in `scripts/config.py`.
2. Execute one or more optimization levels.
3. Retrieve the generated outputs from `results/case_study_1/`.

To reproduce **Case Study II**:

1. Set `CASE_STUDY = 2` in `scripts/config.py`.
2. Execute one or more optimization levels.
3. Retrieve the generated outputs from `results/case_study_2/`.

No files need to be moved or renamed when switching between the two case studies.

---

# Troubleshooting

If an input file cannot be found:

- verify that the correct case study has been selected in `scripts/config.py`;
- verify that the required files are located inside the corresponding `raw` directory;
- verify that the filenames match exactly those reported above.

The recommended execution format is

```bash
python scripts/script_name.py
```

from the root directory of the repository.

---

# Citation

If you use this repository in your research, please cite the associated publication.

```text
Citation information will be added after publication.
```

---

# License

This project is distributed under the MIT License.
