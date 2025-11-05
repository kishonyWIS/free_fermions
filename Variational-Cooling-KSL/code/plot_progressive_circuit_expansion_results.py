import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import from variational_circuit_KSL_numba
sys.path.insert(0, os.path.dirname(__file__))

# Data directory path (relative to this file)
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

# Import functions from variational_circuit_KSL_numba
from variational_circuit_KSL_numba import (
    load_optimized_parameters_for_res_p,
    simulate_grid_with_analysis,
    n_cycles_test
)


def main(csv_path: str = None) -> None:
    if csv_path is None:
        # Try new naming first, fall back to old naming for backward compatibility
        new_path = os.path.join(DATA_DIR, "progressive_circuit_expansion_results_monotonic_good_starting_from_p3.csv")
        old_path = os.path.join(DATA_DIR, "grid_search_results_monotonic_good_starting_from_p3.csv")
        if os.path.exists(new_path):
            csv_path = new_path
        elif os.path.exists(old_path):
            csv_path = old_path
        else:
            csv_path = new_path  # Default to new naming
    df = pd.read_csv(csv_path)

    # Use test energy for plots; drop failures
    df = df.dropna(subset=["energy_density_test"])  # keep completed rows

    res_vals = sorted(df["res"].unique())
    p_vals = sorted(df["p"].unique())

    # 1) Energy vs p (one line per res): solid=test, dashed=train (same color)
    plt.figure(figsize=(7, 4))
    for res in res_vals:
        dd = df[df.res == res].sort_values("p")
        # Plot test (solid)
        line_test, = plt.plot(dd.p, dd.energy_density_test, "-o", label=f"res={res}", linewidth=2)
        color = line_test.get_color()
        # Plot train (dashed) with same color
        plt.plot(dd.p, dd.energy_density_train, "--o", color=color, alpha=0.8, linewidth=1.5)
    
    plt.xlabel("p")
    plt.ylabel("Energy density")
    plt.title("Energy vs p by res (solid=test, dashed=train)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="res", ncol=2)
    plt.tight_layout()
    
    # Save the figure BEFORE showing it
    fig_path = os.path.join(os.path.dirname(__file__), "../figures/energy_vs_p_by_res.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved energy_vs_p_by_res.pdf to {fig_path}")
    
    # Print best (test) configuration
    best = df.loc[df["energy_density_test"].idxmin()]
    print(
        "Best (test):",
        {
            "res": int(best["res"]),
            "p": int(best["p"]),
            "n_k_points_train": int(best["n_k_points_train"]),
            "energy_density_train": float(best["energy_density_train"]),
            "energy_density_test": float(best["energy_density_test"]),
        },
    )
    
    # Show the figure (optional, can be removed for headless operation)
    # plt.show()
    plt.close()
    
    # 2) Chern number vs p
    plot_chern_vs_p(df, res_vals)

def plot_chern_vs_p(df, res_vals=None):
    """
    Generate chern_vs_p.pdf - Chern number vs. circuit depth p
    
    Args:
        df: DataFrame with results from progressive circuit expansion
        res_vals: list of res values to plot (default: use all in df)
    """
    import variational_circuit_KSL_numba as vc_module
    
    print("\n" + "="*60)
    print("Generating chern_vs_p.pdf")
    print("="*60)
    
    if res_vals is None:
        res_vals = sorted(df["res"].unique())
    
    # Focus on primary res value (usually res=3)
    # Prefer res=3 if available, otherwise use the first one
    if 3 in res_vals:
        primary_res = 3
    else:
        primary_res = res_vals[0] if len(res_vals) > 0 else 3
    print(f"Focusing on res={primary_res}")
    
    # Get p values for this res
    df_res = df[df.res == primary_res].sort_values("p")
    p_vals = sorted(df_res["p"].unique())
    
    # Create test grid
    n_k_points_test = 1 + 6 * 20  # 121
    kx_list_test = np.linspace(-np.pi, np.pi, n_k_points_test)
    ky_list_test = np.linspace(-np.pi, np.pi, n_k_points_test)
    
    system_chern_numbers = []
    bath_chern_numbers = []
    p_vals_valid = []
    
    for p_val in p_vals:
        print(f"  Evaluating p={p_val}...")
        
        try:
            # Load parameters
            strength_durations = load_optimized_parameters_for_res_p(primary_res, p_val)
            
            # Set global p
            vc_module.p = p_val
            
            # Run simulation
            _, _, _, system_chern, bath_chern, _ = simulate_grid_with_analysis(
                kx_list_test, ky_list_test, strength_durations, n_cycles=n_cycles_test
            )
            
            system_chern_numbers.append(system_chern)
            bath_chern_numbers.append(bath_chern)
            p_vals_valid.append(p_val)
            
            print(f"    System Chern: {system_chern:.4f}, Bath Chern: {bath_chern:.4f}")
            
        except (FileNotFoundError, KeyError) as e:
            print(f"    Warning: Could not evaluate p={p_val}: {e}")
            continue
    
    if len(system_chern_numbers) == 0:
        print("  Error: No valid data points found for Chern number plot")
        return
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(p_vals_valid, system_chern_numbers, 'b-o', linewidth=2, markersize=8, 
             label='System Chern number', color='#1f77b4')
    plt.plot(p_vals_valid, bath_chern_numbers, 'r--s', linewidth=2, markersize=8, 
             label='Bath Chern number', color='#d62728')
    
    # Add horizontal lines at target Chern numbers
    plt.axhline(y=1.0, color='b', linestyle=':', alpha=0.5, linewidth=1)
    plt.axhline(y=-1.0, color='r', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.xlabel('Circuit Depth $p$', fontsize=12)
    plt.ylabel('Chern Number', fontsize=12)
    plt.title(f'Chern Number vs. Circuit Depth\n(res={primary_res}, test grid: {n_k_points_test}$\\times${n_k_points_test} points)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(os.path.dirname(__file__), "../figures/chern_vs_p.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved chern_vs_p.pdf to {fig_path}")


if __name__ == "__main__":
    main()


