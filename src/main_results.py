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
This module performs FCI calculations for H$_2$ and computes various non-stabilizerness proxies,
including the stabilizer Rényi entropy and Mana. It also generates plots showing the FCI binding energy,
non-stabilizerness proxies, and the extracted theta values.

Usage:
    Run the module directly to execute the calculations and display/save the plots.
"""

from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci
from pyscf.fci import cistring
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set LaTeX fonts and style for plotting.
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
I2: np.ndarray = np.array([[1, 0], [0, 1]], dtype=complex)
c: np.ndarray = np.array([[0, 0], [1, 0]], dtype=complex)
cd: np.ndarray = np.array([[0, 1], [0, 0]], dtype=complex)
eta: np.ndarray = c + cd        # Majorana operator: η = c + c†
chi: np.ndarray = 1j * (c - cd)   # Majorana operator: χ = i(c - c†)


def local_operator(v1: int, v2: int) -> np.ndarray:
    r"""Return the local operator for one mode: η^(v1) * χ^(v2).

    If v1 or v2 is 0, the corresponding operator is replaced by the identity.

    Args:
        v1 (int): Exponent for η operator (0 or 1).
        v2 (int): Exponent for χ operator (0 or 1).

    Returns:
        np.ndarray: The 2×2 local operator.
    """
    op = I2
    if v1 == 1:
        op = np.dot(op, eta)
    if v2 == 1:
        op = np.dot(op, chi)
    return op


def overall_phase(v: List[int]) -> complex:
    r"""Compute the overall phase factor i^(∑_{i>j} v_i * v_j) for an 8-component binary vector.

    Args:
        v (List[int]): An 8-element binary vector.

    Returns:
        complex: The computed phase factor.
    """
    exponent = 0
    for i in range(1, len(v)):
        for j in range(i):
            exponent += v[i] * v[j]
    return (1j) ** exponent


def P_v_operator(v: List[int]) -> np.ndarray:
    r"""Construct the full 16×16 operator P_v for 4 modes from an 8-bit binary vector.

    The operator is defined as the tensor product of local operators
    multiplied by an overall phase factor.

    Args:
        v (List[int]): A binary vector of length 8.

    Returns:
        np.ndarray: The 16×16 operator P_v.

    Raises:
        ValueError: If the length of v is not 8.
    """
    if len(v) != 8:
        raise ValueError("v must be a binary vector of length 8")
    op = 1  # Start with scalar 1 and build via tensor product.
    for a in range(4):
        op_local = local_operator(v[2 * a], v[2 * a + 1])
        op = np.kron(op, op_local)
    return overall_phase(v) * op


def state_psi(theta: float) -> np.ndarray:
    r"""Construct the state |\psi(θ)⟩ in a 16-dimensional Hilbert space.

    The state is defined as:
        |\psi(θ)⟩ = cos(θ)|1,1,0,0⟩ + sin(θ)|0,0,1,1⟩,
    where the four modes correspond to:
      - Mode 1: α orbital 0,
      - Mode 2: β orbital 0,
      - Mode 3: α orbital 1,
      - Mode 4: β orbital 1.

    Args:
        theta (float): Angle parameter in radians.

    Returns:
        np.ndarray: The 16-dimensional state vector.
    """
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    state_1100 = np.kron(np.kron(ket1, ket1), np.kron(ket0, ket0))
    state_0011 = np.kron(np.kron(ket0, ket0), np.kron(ket1, ket1))
    return np.cos(theta) * state_1100 + np.sin(theta) * state_0011


def expected_value(theta: float, v: List[int]) -> complex:
    r"""Compute the expectation value ⟨ψ(θ)|P_v|ψ(θ)⟩.

    Args:
        theta (float): Angle parameter in radians.
        v (List[int]): An 8-bit binary vector.

    Returns:
        complex: The computed expectation value.
    """
    psi = state_psi(theta)
    P_v = P_v_operator(v)
    return np.vdot(psi, P_v.dot(psi))


def stabilizer_Renyi_entropy(theta: float, alpha: int = 2) -> float:
    r"""Compute the stabilizer Rényi entropy M_α for the state |\psi(θ)⟩.

    The entropy is defined as:
        M_α = 1/(1-α) * ln(ζ_α),
    where
        ζ_α = (1/16) * Σ_v ⟨ψ(θ)|P_v|ψ(θ)⟩^(2α).

    Args:
        theta (float): Angle parameter in radians.
        alpha (int, optional): Order of the Rényi entropy. Defaults to 2.

    Returns:
        float: The computed stabilizer Rényi entropy.
    """
    total = 0.0
    for iv in range(256):
        v = [(iv >> i) & 1 for i in range(8)]
        ev = expected_value(theta, v)
        total += (ev ** (2 * alpha)).real
    zeta_alpha = total / 16.0
    return (1 / (1 - alpha)) * np.log(zeta_alpha)


def filtered_stabilizer_Renyi_entropy(theta: float, alpha: int = 2) -> float:
    r"""Compute the filtered stabilizer Rényi entropy F_α, excluding identity and parity operators.

    The filtered entropy is defined as:
        F_α = 1/(1-α) * ln(ζ_α^filtered),
    where
        ζ_α^filtered = (1/(16-2)) * Σ_{v ≠ (0,...,0), (1,...,1)} ⟨ψ(θ)|P_v|ψ(θ)⟩^(2α).

    Args:
        theta (float): Angle parameter in radians.
        alpha (int, optional): Order of the Rényi entropy. Defaults to 2.

    Returns:
        float: The computed filtered stabilizer Rényi entropy.
    """
    total = 0.0
    for iv in range(256):
        v = [(iv >> i) & 1 for i in range(8)]
        if tuple(v) == (0,) * 8 or tuple(v) == (1,) * 8:
            continue
        ev = expected_value(theta, v)
        total += (ev ** (2 * alpha)).real
    zeta_alpha_filtered = total / (16 - 2)
    return (1 / (1 - alpha)) * np.log(zeta_alpha_filtered)


def mana(theta: float) -> float:
    r"""Compute the Mana of the state |\psi(θ)⟩.

    The Mana is defined as:
        Mana = ln[(1/16) * Σ_v |⟨ψ(θ)|P_v|ψ(θ)⟩|].

    Args:
        theta (float): Angle parameter in radians.

    Returns:
        float: The computed Mana value.
    """
    total = 0.0
    for iv in range(256):
        v = [(iv >> i) & 1 for i in range(8)]
        ev = expected_value(theta, v)
        total += abs(ev)
    norm = total / 16.0
    return np.log(norm)


def extract_theta(ci: np.ndarray, norb: int, nelec: Tuple[int, int], threshold: float = 1e-3) -> float:
    r"""Extract the angle θ from the CI coefficients.

    The extraction is based on identifying the significant determinants corresponding
    to |1,1,0,0⟩ and |0,0,1,1⟩, and then computing:
        θ = arctan2(c2, c1).

    Args:
        ci (np.ndarray): The CI coefficient matrix.
        norb (int): Number of orbitals.
        nelec (Tuple[int, int]): Number of electrons (alpha, beta).
        threshold (float, optional): Threshold for significant coefficients. Defaults to 1e-3.

    Returns:
        float: The extracted angle θ in radians.

    Raises:
        ValueError: If both significant determinants are not found.
    """
    str_alpha: List[int] = cistring.make_strings(list(range(norb)), nelec[0])
    str_beta: List[int] = cistring.make_strings(list(range(norb)), nelec[1])
    c1 = None
    c2 = None
    for i, det_a in enumerate(str_alpha):
        for j, det_b in enumerate(str_beta):
            coeff = ci[i, j]
            if abs(coeff) > threshold:
                # Check for determinant corresponding to |1,1,0,0⟩.
                if det_a == 1 and det_b == 1:
                    c1 = coeff
                # Check for determinant corresponding to |0,0,1,1⟩.
                elif det_a == 2 and det_b == 2:
                    c2 = coeff
    if c1 is None or c2 is None:
        raise ValueError("Did not find both significant determinants!")
    return np.arctan2(c2, c1)


def compute_fci_data(distances: np.ndarray, threshold: float = 1e-3) -> Dict[str, np.ndarray]:
    r"""Compute FCI energies and non-stabilizerness proxies over a range of interatomic distances.

    For each distance, this function performs:
      - FCI calculation using a minimal basis (sto-3g).
      - Extraction of the theta angle from the CI coefficients.
      - Calculation of stabilizer Rényi entropy (M2), filtered entropy (F2), and Mana.

    Args:
        distances (np.ndarray): Array of interatomic distances.
        threshold (float, optional): Threshold for significant CI coefficients. Defaults to 1e-3.

    Returns:
        Dict[str, np.ndarray]: Dictionary with keys:
            'energies', 'M2', 'F2', 'mana', and 'theta', each mapped to an array.
    """
    energies: List[float] = []
    M2_values: List[float] = []
    F2_values: List[float] = []
    mana_values: List[float] = []
    theta_vals: List[float] = []

    for d in distances:
        mol = gto.M(
            atom=f'H 0 0 0; H 0 0 {d}',
            basis='sto-3g',
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
        nelec = mol.nelec

        theta_val = extract_theta(ci, norb, nelec, threshold)
        theta_vals.append(theta_val)

        M2_values.append(stabilizer_Renyi_entropy(theta_val, alpha=2))
        F2_values.append(filtered_stabilizer_Renyi_entropy(theta_val, alpha=2))
        mana_values.append(mana(theta_val))

    return {
        'energies': np.array(energies),
        'M2': np.array(M2_values),
        'F2': np.array(F2_values),
        'mana': np.array(mana_values),
        'theta': np.array(theta_vals)
    }


def plot_results(distances: np.ndarray, data: Dict[str, np.ndarray]) -> None:
    r"""Generate and display plots for FCI binding energy, non-stabilizerness proxies, and theta values.

    This function creates:
      1. A combined twin-axis plot of FCI binding energy and non-stabilizerness proxies, with an inset
         for the second derivative (curvature) analysis.
      2. A plot of FCI binding energy and the extracted theta angle.
      3. A plot of cosine and sine of theta values.

    Args:
        distances (np.ndarray): Array of interatomic distances.
        data (Dict[str, np.ndarray]): Dictionary containing 'energies', 'M2', 'F2', 'mana', and 'theta'.
    """
    energies = data['energies']
    M2_values = data['M2']
    F2_values = data['F2']
    mana_values = data['mana']
    theta_vals = data['theta']

    # Define dark colors for plotting.
    dark_blue = 'midnightblue'
    dark_red = 'darkred'
    dark_green = 'darkcyan'
    dark_purple = 'indigo'

    # Compute shifted energy for curvature analysis.
    energy_shifted = energies - energies[-1]
    dE_dx = np.gradient(energy_shifted, distances)
    d2E_dx2 = np.gradient(dE_dx, distances)
    curvature = np.abs(d2E_dx2) / (1 + dE_dx**2)**(3/2)
    peaks, _ = find_peaks(curvature)

    # ----------------------- Plot 1: Combined Twin-Axis Plot -----------------------
    fig, ax1 = plt.subplots()
    ax1.plot(distances, energy_shifted, 'o--', markersize=5, color=dark_blue,
             alpha=0.8, label=r'$\mathcal E_\textsc{fci}$')
    ax1.set_xlabel(r'Interatomic Distance (Bohr)')
    ax1.set_ylabel(r'FCI Binding Energy (Ha)', color=dark_blue)
    ax1.tick_params(axis='both', which='major', length=4, labelsize=10, colors=dark_blue)
    ax1.grid(True, which='both', ls='--', lw=0.5)

    # Add vertical dashed line at the last curvature peak.
    if len(peaks) > 0:
        ax1.axvline(x=distances[peaks[-1]], color='black', linestyle='--',
                    linewidth=1.5, alpha=0.8)

    ax2 = ax1.twinx()
    ax2.plot(distances, M2_values, 'o--', markersize=5, color=dark_red,
             alpha=0.6, label=r'$\mathcal S_2$')
    ax2.plot(distances, F2_values, 'o--', markersize=5, color=dark_green,
             alpha=0.6, label=r'$\mathcal{FS}_2$')
    ax2.plot(distances, mana_values, 'o--', markersize=5, color=dark_purple,
             alpha=0.6, label=r'$\mathcal M$')
    ax2.set_ylabel(r'Non-stabilizerness proxies', color=dark_red)
    ax2.tick_params(axis='both', which='major', length=4, labelsize=10, colors=dark_red)

    # Combine legends from both axes.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Inset: Plot the second derivative for distances > 1.6 Bohr with a vertical dashed line.
    mask = distances > 1.6
    max_curv = distances[peaks[-1]] if len(peaks) > 0 else None

    inset_ax = inset_axes(ax1, width="27%", height="27%", loc='upper left',
                          bbox_to_anchor=(0.10, -0.02, 1, 1),
                          bbox_transform=ax1.transAxes, borderpad=0)
    inset_ax.plot(distances[mask], d2E_dx2[mask], '--', color=dark_blue, alpha=0.8,
                  label=r'$\frac{\text{d}^2\mathcal E_\textsc{fci}}{\text{d}\ell^2}$')
    if max_curv is not None:
        inset_ax.axvline(x=max_curv, color='black', linestyle='--',
                          linewidth=1.5, alpha=0.8)
    inset_ax.tick_params(axis='both', which='major', labelsize=8)
    inset_ax.tick_params(axis='y', which='both', labelleft=False, length=0)
    inset_ax.set_xlim(1.4, distances[-1])
    inset_ax.grid(True, which='both', ls='--', lw=0.5)

    fig.tight_layout()
    plt.savefig('./logs/magic_vs_distance.png', dpi=600)
    plt.show()

    # ----------------------- Plot 2: FCI Binding Energy and Theta Angle -----------------------
    fig, ax1 = plt.subplots()
    ax1.plot(distances, energy_shifted, 'o--', markersize=5, color=dark_blue,
             alpha=0.8, label=r'$\mathcal E_\textsc{fci}$')
    ax1.set_xlabel(r'Interatomic Distance (Bohr)')
    ax1.set_ylabel(r'FCI Binding Energy (Ha)', color=dark_blue)
    ax1.tick_params(axis='y', labelcolor=dark_blue)
    ax1.grid(True, which='both', ls='--', lw=0.5)

    ax2 = ax1.twinx()
    ax2.plot(distances, theta_vals, 'o--', markersize=5, color=dark_purple,
             alpha=0.8, label=r'$\theta$')
    ax2.set_ylabel(r'$\theta$ (radians)', color=dark_purple)
    ax2.tick_params(axis='y', labelcolor=dark_purple)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    plt.savefig('./logs/binding_theta.png', dpi=600)
    plt.show()


def main() -> None:
    r"""Main function to execute FCI calculations and plot the results."""
    # Define the range of interatomic distances (in Bohr).
    distances = np.linspace(0.4, 5.3, 30)
    # Compute FCI energies and non-stabilizerness proxies.
    data = compute_fci_data(distances, threshold=1e-3)
    # Generate and display plots.
    plot_results(distances, data)


if __name__ == '__main__':
    main()