# TBdMM

**TBdMM** (Tight-Binding d-Band Moment) is a Python package that computes the tight-binding d-band moment \(m_n\) and the corresponding Hamiltonian matrix \(H\) for a given site of a material structure. It is useful for analyzing electronic structures in materials modeling, especially in surface chemistry.

---

## Features

- **Compute tight-binding d-band moment** \(m_n\) for a selected atomic site.
- **Construct tight-binding Hamiltonian matrix** \(H\) based on user-defined structure.
- **Return** both \(m_n\) and \(H\).
- **Command-line interface (CLI)** for straightforward usage in shell scripts or remote servers.
- **Python API** for integration into custom workflows and research pipelines.

---

## Installation

You can install TBdMM in several ways:

### 1. Install via PyPI

    pip install TBdMM

### 2. Install via GitHub

    git clone https://github.com/your_username/TBdMM.git
    cd TBdMM
    pip install -e .

---

## Usage

### Command-Line Interface

After installation, TBdMM provides a command-line tool `tbdmm`. You can run:

    tbdmm -i <input_structure> -s <site_index> -n <order> [-o output_file.npy]

**Arguments**:

- `-i, --input-file`: Path to your structure file (e.g., POSCAR, CIF, XYZ, etc.).
- `-s, --site-index`: The site index of the center atom for which to compute the d-band moment.
- `-n, --order`: The order of the d-band moment.
- `-o, --output-h`: (Optional) The file path to save the Hamiltonian matrix (H) in NumPy `.npy` format.

**Example**:

    tbdmm -i POSCAR -s 10 -n 2 -o H_matrix.npy

Sample output:

    m_2 = 0.123456   # example result
    Hamiltonian saved to H_matrix.npy

### Python API

You can also import and use TBdMM in Python scripts or Jupyter notebooks:

    from ase.io import read
    from TBdMM import TBdMM  # or from TBdMM.calculations import TBdMM

    atoms = read("POSCAR")  # Load the structure into an ASE Atoms object

    m2, H = TBdMM(atoms, site_index=10, n=2)
    print("m2 =", m2)
    print("Hamiltonian shape:", H.shape)

This returns a tuple `(m2, H)`, where:

- `m2` is the second-order d-band moment.
- `H` is the constructed Hamiltonian as a NumPy 2D array.

---

## How to cite

If you are using this package, please consider citing the following papers:

- [Phys. Rev. B 110, L121404 (2024)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.L121404)  
- [J. Chem. Phys. 161, 164702 (2024)](https://pubs.aip.org/aip/jcp/article/161/16/164702/3317708)
