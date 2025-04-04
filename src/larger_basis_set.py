# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis (https://github.com/MatthieuSarkis).
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
This module performs FCI calculations for H$_2$ using a non-minimal basis set (6-31g)
and computes the filtered stabilizer Rényi entropy along with the FCI binding energy.
The simulation results are saved in a folder './logs/'.
A separate plotting function then loads these results and prepares the figure.
The ground state is also printed in ket notation.

Ket notation explanation:
    Each ket is written as:
       |α₀ α₁ … αₙ₋₁ ; β₀ β₁ … βₙ₋₁⟩,
    where the left half (α's) represents the occupation of the alpha spin orbitals
    and the right half (β's) the beta spin orbitals.
    "1" means occupied and "0" means unoccupied.
    The ordering is taken from the chosen active space.

Usage:
    To simulate and save the data:
         python script.py simulate
    To plot the data:
         python script.py plot
    If no argument is provided, both simulation and plotting are run sequentially.
"""

from typing import List, Tuple, Dict
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci
from pyscf.fci import cistring
from scipy.signal import find_peaks
import random

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "figure.figsize": (8, 6),
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 14
})

# Global constants: 2x2 matrices for a single fermionic mode.
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
c = np.array([[0, 0], [1, 0]], dtype=complex)
cd = np.array([[0, 1], [0, 0]], dtype=complex)
eta = c + cd        # Majorana operator: η = c + c†
chi = 1j * (c - cd)   # Majorana operator: χ = i(c - c†)

def local_operator(v1: int, v2: int) -> np.ndarray:
    """Return the local operator for one fermionic mode as η^(v1)*χ^(v2)."""
    op = I2
    if v1 == 1:
        op = np.dot(op, eta)
    if v2 == 1:
        op = np.dot(op, chi)
    return op

def overall_phase(v: List[int]) -> complex:
    """Compute the overall phase factor i^(Σ_{i>j} v_i*v_j) for a binary vector."""
    exponent = 0
    for i in range(1, len(v)):
        for j in range(i):
            exponent += v[i] * v[j]
    return (1j) ** exponent

def P_v_operator(v: List[int]) -> np.ndarray:
    """Construct the full operator P_v as a tensor product of local operators, multiplied by an overall phase factor."""
    if len(v) % 2 != 0:
        raise ValueError("Length of v must be even (two bits per fermionic mode)")
    n_modes = len(v) // 2
    op = 1
    for a in range(n_modes):
        op_local = local_operator(v[2*a], v[2*a+1])
        op = np.kron(op, op_local)
    return overall_phase(v) * op

def full_state_vector(ci: np.ndarray, active_space_size: int, nelec: Tuple[int, int]) -> np.ndarray:
    """Construct the full state vector in the computational basis for the active space."""
    nspin = 2 * active_space_size
    dim = 2 ** nspin
    state = np.zeros(dim, dtype=complex)
    str_alpha = cistring.make_strings(list(range(active_space_size)), nelec[0])
    str_beta = cistring.make_strings(list(range(active_space_size)), nelec[1])
    for i, det_a in enumerate(str_alpha):
        for j, det_b in enumerate(str_beta):
            coeff = ci[i, j]
            index = (det_a << active_space_size) | det_b
            state[index] = coeff
    return state

def expected_value_state(state: np.ndarray, v: List[int]) -> complex:
    """Compute the expectation value ⟨Ψ|P_v|Ψ⟩ for a full state vector."""
    P_v = P_v_operator(v)
    return np.vdot(state, P_v.dot(state))

SAMPLING_THRESHOLD = 1e6

def compute_full_sum(state: np.ndarray, n_modes: int, func) -> Tuple[float, int]:
    """Compute the full sum Σ_{v in {0,1}^{2*n_modes}} f(⟨Ψ|P_v|Ψ⟩)."""
    total = 0.0
    total_terms = 2 ** (2 * n_modes)
    for iv in range(total_terms):
        v = [(iv >> i) & 1 for i in range(2 * n_modes)]
        ev = expected_value_state(state, v)
        total += func(ev)
    return total, total_terms

def compute_mc_sum(state: np.ndarray, n_modes: int, func, n_samples: int) -> Tuple[float, int]:
    """Estimate the sum via Monte Carlo sampling over n_samples random v."""
    total = 0.0
    total_terms = 2 ** (2 * n_modes)
    for _ in range(n_samples):
        iv = random.randint(0, total_terms - 1)
        v = [(iv >> i) & 1 for i in range(2 * n_modes)]
        ev = expected_value_state(state, v)
        total += func(ev)
    return (total / n_samples) * total_terms, total_terms

def filtered_stabilizer_Renyi_entropy_state(state: np.ndarray, alpha: int, n_modes: int, n_samples: int = None) -> float:
    """Compute the filtered stabilizer Rényi entropy from the full state vector."""
    def f(ev, v):
        if all(bit == 0 for bit in v) or all(bit == 1 for bit in v):
            return 0.0
        return (ev ** (2 * alpha)).real

    total = 0.0
    total_terms = 2 ** (2 * n_modes)
    if n_samples is None and total_terms <= SAMPLING_THRESHOLD:
        for iv in range(total_terms):
            v = [(iv >> i) & 1 for i in range(2 * n_modes)]
            total += f(expected_value_state(state, v), v)
    else:
        if n_samples is None:
            n_samples = 10000
        samples = 0
        sum_samples = 0.0
        while samples < n_samples:
            iv = random.randint(0, total_terms - 1)
            v = [(iv >> i) & 1 for i in range(2 * n_modes)]
            if all(bit == 0 for bit in v) or all(bit == 1 for bit in v):
                continue
            sum_samples += f(expected_value_state(state, v), v)
            samples += 1
        total = (sum_samples / n_samples) * (total_terms - 2)
    zeta_filtered = total / (2 ** n_modes - 2)
    return (1 / (1 - alpha)) * np.log(zeta_filtered)

def full_state_to_ket(state: np.ndarray, active_space_size: int, coeff_threshold: float = 1e-3) -> str:
    """Convert a full state vector into a string representing a linear combination of kets."""
    n_spin = 2 * active_space_size
    dim = 2 ** n_spin
    terms = []
    for idx in range(dim):
        coeff = state[idx]
        if abs(coeff) < coeff_threshold:
            continue
        bits = format(idx, '0{}b'.format(n_spin))
        alpha_part = bits[:active_space_size]
        beta_part = bits[active_space_size:]
        ket = f"|{alpha_part};{beta_part}⟩"
        coeff_str = f"({coeff.real:.3g}"
        if abs(coeff.imag) > coeff_threshold:
            coeff_str += f"{coeff.imag:+.3g}j"
        coeff_str += ")"
        terms.append(f"{coeff_str}{ket}")
    if not terms:
        return "0"
    return " + ".join(terms)

def get_ground_state_at_distance(d: float, active_space_size: int) -> Tuple[float, np.ndarray]:
    """Perform an FCI calculation at a given interatomic distance and return the energy and full state vector."""
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {d}',
        basis='6-31g',
        symmetry=True,
        unit='B',
        verbose=0
    )
    mol.spin = 0
    mf = scf.UHF(mol).run(verbose=0)
    cisolver = fci.FCI(mf)
    energy, ci = cisolver.kernel()
    nelec = mol.nelec
    active_space_size_used = mf.mo_coeff[0].shape[1]
    state = full_state_vector(ci, active_space_size_used, nelec)
    return energy, state

# --- Data Generation and Saving ---

def compute_fci_data(distances: np.ndarray, active_space_size: int = None, n_samples: int = None) -> Dict[str, np.ndarray]:
    r"""Compute FCI energies and the filtered magic proxy over a range of interatomic distances.

    For each distance the code:
      - Performs an FCI calculation using a non-minimal basis (6-31g).
      - Reconstructs the full state vector for the chosen active space.
      - Computes the filtered stabilizer Rényi entropy (denoted as F₂).

    Args:
        distances (np.ndarray): Array of interatomic distances.
        active_space_size (int, optional): Number of spatial orbitals in the active space.
            If None, the full set of orbitals from the mean-field is used.
        n_samples (int, optional): Number of Monte Carlo samples if needed.

    Returns:
        Dict[str, np.ndarray]: Dictionary with keys 'energies' and 'F2'.
    """
    energies: List[float] = []
    F2_values: List[float] = []

    for d in distances:
        mol = gto.M(
            atom=f'H 0 0 0; H 0 0 {d}',
            basis='6-31g',
            symmetry=True,
            unit='B',
            verbose=0
        )
        mol.spin = 0
        mf = scf.UHF(mol).run(verbose=0)
        cisolver = fci.FCI(mf)
        e_fci, ci = cisolver.kernel()
        energies.append(e_fci)

        norb = mf.mo_coeff[0].shape[1]
        active_space_used = norb if active_space_size is None else active_space_size
        nelec = mol.nelec
        state = full_state_vector(ci, active_space_used, nelec)
        n_modes = 2 * active_space_used

        F2 = filtered_stabilizer_Renyi_entropy_state(state, alpha=2, n_modes=n_modes, n_samples=n_samples)
        F2_values.append(F2)

    return {
        'energies': np.array(energies),
        'F2': np.array(F2_values)
    }

def simulate_data() -> None:
    """Generate simulation data and save the results in './src/stabilizer_renyi_entropy/logs/'."""
    distances = np.linspace(0.4, 5.3, 30)
    data = compute_fci_data(distances, active_space_size=None, n_samples=None)
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    out_file = os.path.join(log_dir, "simulation_results.npz")
    np.savez_compressed(out_file, distances=distances, energies=data['energies'], F2=data['F2'])
    print("Simulation data saved to", out_file)

def plot_data() -> None:
    """Load simulation data and prepare the figure."""
    log_dir = "./logs/"
    data_file = os.path.join(log_dir, "simulation_results.npz")
    if not os.path.exists(data_file):
        raise FileNotFoundError("Data file not found. Run simulation first.")
    loaded = np.load(data_file)
    distances = loaded['distances']
    energies = loaded['energies']
    F2 = loaded['F2']

    dark_blue = 'midnightblue'
    dark_green = 'darkcyan'

    energy_shifted = energies - energies[-1]
    dE_dx = np.gradient(energy_shifted, distances)
    d2E_dx2 = np.gradient(dE_dx, distances)
    curvature = np.abs(d2E_dx2) / (1 + dE_dx**2)**(3/2)
    peaks, _ = find_peaks(curvature)

    fig, ax1 = plt.subplots()
    ax1.plot(distances, energy_shifted, 'o--', markersize=5, color=dark_blue,
             alpha=0.8, label=r'$\mathcal{E}_\mathrm{FCI}$')
    ax1.set_xlabel(r'Interatomic Distance (Bohr)')
    ax1.set_ylabel(r'FCI Binding Energy (Ha)', color=dark_blue)
    ax1.tick_params(axis='both', labelsize=10, colors=dark_blue)
    ax1.grid(True, ls='--', lw=0.5)
    if len(peaks) > 0:
        ax1.axvline(x=distances[peaks[-1]], color='black', linestyle='--',
                    linewidth=1.5, alpha=0.8)

    ax2 = ax1.twinx()
    ax2.plot(distances, F2, 'o--', markersize=5, color=dark_green,
             alpha=0.6, label=r'$\mathcal{FS}_2$')
    ax2.set_ylabel(r'Magic Proxy', color=dark_green)
    ax2.tick_params(axis='both', labelsize=10, colors=dark_green)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    out_fig = os.path.join(log_dir, "magic_vs_distance_631g.png")
    plt.savefig(out_fig, dpi=600)
    plt.show()
    print("Figure saved to", out_fig)

def main() -> None:
    """
    Main function.
    Use command-line argument 'simulate' to generate data,
    'plot' to load and plot, or if no argument is provided, run both.
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "simulate":
            simulate_data()
        elif sys.argv[1] == "plot":
            plot_data()
        else:
            print("Usage: python script.py simulate|plot")
    else:
        simulate_data()
        plot_data()

if __name__ == '__main__':
    main()
