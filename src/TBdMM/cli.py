# src/TBdMM/cli.py
import argparse
from ase.io import read
from .calculations import TBdMM

def main():
    parser = argparse.ArgumentParser(
        description="Compute d-band moment (m_n) and Hamiltonian H from a structure."
    )
    parser.add_argument("-i", "--input-file", type=str, required=True,
                        help="Path to the input structure file (e.g., POSCAR, CIF, XYZ...).")
    parser.add_argument("-s", "--site-index", type=int, required=True,
                        help="Site index of the center atom.")
    parser.add_argument("-n", "--order", type=int, required=True,
                        help="Order n for the d-band moment.")
    parser.add_argument("-o", "--output-h", type=str, default=None,
                        help="Optional path to save the Hamiltonian matrix (as .npy).")

    args = parser.parse_args()
    image = read(args.input_file)
    m_n, H = TBdMM(image, args.site_index, args.order)

    if m_n is None:
        print("No valid result. Possibly invalid n?")
        return

    print(f"m_{args.order} = {m_n}")
    if args.output_h:
        import numpy as np
        np.save(args.output_h, H)
        print(f"Hamiltonian saved to {args.output_h}")
    else:
        print("Hamiltonian not saved. Use '-o filename.npy' to save.")