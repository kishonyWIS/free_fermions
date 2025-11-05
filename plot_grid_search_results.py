import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path: str = "grid_search_results_monotonic_good_starting_from_p3.csv") -> None:
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
    plt.show()

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

    #save the figure
    plt.savefig("energy_vs_p_by_res.pdf")

if __name__ == "__main__":
    main()


