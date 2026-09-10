import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, FuncFormatter, MaxNLocator
from collections import defaultdict
import os
from scipy.interpolate import make_interp_spline


Y_HIGH_END_STRETCH_GAMMA = 0.3


def smooth_with_best_log_fit(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.asarray(y_values, dtype=float)

    x_min, x_max = np.min(x), np.max(x)
    span = max(x_max - x_min, 1e-6)
    shift_candidates = np.geomspace(max(1e-4, span * 1e-3), span * 2.0, num=20)

    best_mse = np.inf
    best_shift = None
    best_coefs = None

    for shift in shift_candidates:
        t = np.log(x + shift)
        for deg in [1, 2]:
            if len(x) <= deg:
                continue
            try:
                coefs = np.polyfit(t, y, deg=deg)
                y_hat = np.polyval(coefs, t)
                mse = np.mean((y - y_hat) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_shift = shift
                    best_coefs = coefs
            except np.linalg.LinAlgError:
                continue

    if best_coefs is None:
        return np.asarray(y_values, dtype=float)

    y_fit_at_x = np.polyval(best_coefs, np.log(x + best_shift))
    y_smoothed_at_x = 0.8 * y_fit_at_x + 0.2 * y

    y_out = np.asarray(y_values, dtype=float)
    y_out[valid] = y_smoothed_at_x
    y_out = np.maximum(y_out, 1e-4)
    return y_out

def customPlotFun(y, pos):
    if y <= 0.0001:
        return "≤ 0.0001"
    else:
        return f"{y:.5f}".rstrip('0').rstrip('.')


def high_end_stretch_forward(y, gamma=Y_HIGH_END_STRETCH_GAMMA):
    y = np.asarray(y, dtype=float)
    return np.power(np.clip(y, 0.0, None), gamma)


def high_end_stretch_inverse(y, gamma=Y_HIGH_END_STRETCH_GAMMA):
    y = np.asarray(y, dtype=float)
    return np.power(np.clip(y, 0.0, None), 1.0 / gamma)

# Label maps
method_labels = {
    "ours": "Ours",
    "ours_skew": "Ours",
    # "ours_localization": "Ours Loc.",
    "quadric_based": "Gummeson",
    "right_cylinder": "Ding",
    "zhang_4": "Zhang 4 views",
    "zhang_30": "Zhang 30 views",
    "ceres": "Ceres"
}

metric_labels = {
    "delta_f": r"$\Delta f$",
    "delta_uv": r"$\Delta c$",
    "delta_skew": r"$\Delta$ $\alpha$",
    "delta_r": r"$\Delta R$",
    "delta_t": r"$\Delta t$",
    "success_rate": "Success Rate",
}

method_supports = {
    "ours": {"delta_f": True, "delta_uv": True, "delta_skew": False, "delta_r": True, "delta_t": True, "success_rate": True},
    "ours_skew": {"delta_f": False, "delta_uv": False, "delta_skew": True, "delta_r": False, "delta_t": False, "success_rate": False},
    "ours_localization": {"delta_f": False, "delta_uv": False, "delta_skew": False, "delta_r": True, "delta_t": True, "success_rate": True},
    "quadric_based": {"delta_f": False, "delta_uv": False, "delta_skew": False, "delta_r": True, "delta_t": True, "success_rate": True},
    "right_cylinder": {"delta_f": True, "delta_uv": True, "delta_skew": False, "delta_r": False, "delta_t": False, "success_rate": False},
    "zhang_4": {"delta_f": True, "delta_uv": True, "delta_skew": False, "delta_r": True, "delta_t": True, "success_rate": False},
    "zhang_30": {"delta_f": True, "delta_uv": True, "delta_skew": False, "delta_r": True, "delta_t": True, "success_rate": False},
    "ceres": {"delta_f": True, "delta_uv": True, "delta_skew": True, "delta_r": True, "delta_t": False, "success_rate": True},
}

methods = list(method_labels.keys())
metrics = list(metric_labels.keys())
colors = ["blue", "blue", "purple", "green", "orange", "red", "brown"]
linestyles = ["-", "-", "--", "--", "-.", ":", ":", "--"]

# Load data
with open("./synthetic/results.json") as f:
    data = json.load(f)

# Group by metric → method → noise → values
grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for entry in data:
    noise = entry["noise"]
    method = entry["method"]
    for metric in metrics:
        values = entry.get(metric, [])
        if isinstance(values, list):
            for v in values:
                if v is not None:
                    grouped[metric][method][noise].append(v)
        else:
            if values is not None:
                grouped[metric][method][noise].append(values)

# Plot each metric
for metric in metrics:
    plt.figure(figsize=(5, 4))
    plt.rcParams.update({'font.size': 10})

    for idx, method in enumerate(methods):
        # Skip if the method does not support this metric
        if not method_supports.get(method, {}).get(metric, False):
            continue

        noise_levels = sorted(grouped[metric][method].keys())
        if metric in ["delta_f", "delta_uv", "delta_r", "delta_t", "success_rate"]:
            noise_levels = noise_levels[::3]
        means = []
        for noise in noise_levels:
            vals = grouped[metric][method][noise]
            if vals:
                vals = np.array([max(v, 0) for v in vals])  # Ensure non-negative
                if metric == "delta_skew":
                    vals = vals / 10
                variance = np.var(vals)
                q25 = np.percentile(vals, 25)
                q75 = np.percentile(vals, 75)
                vals_iqr = vals[(vals >= q25) & (vals <= q75)]
                mean = np.mean(vals_iqr) if len(vals_iqr) > 0 else np.mean(vals)

                means.append(mean)
                err_low = max(mean - q25, 0)
                err_high = max(q75 - mean,0)

                if (err_low != 0 and err_high != 0):
                    plt.errorbar(
                        noise * 100,
                        mean,
                        yerr=np.array([[err_low], [err_high]]),
                        fmt='o',
                        color=colors[idx],
                        alpha=0.6,
                        markersize=0.1,
                    )
            else:
                means.append(np.nan)

        x_values = np.array(noise_levels) * 100
        line_x = x_values
        line_y = np.array(means)

        if method == "right_cylinder":
            line_y = smooth_with_best_log_fit(x_values, means)

        # for (i, noise) in enumerate(line_x):
        #     if noise == 0.0:
        #         line_y[i] = line_y[i+1] if i+1 < len(line_y) else line_y[i-1]

        plt.plot(
            line_x,
            line_y,
            color=colors[idx],
            label=method_labels[method],
            linestyle=linestyles[idx],
        )
        markerfacecolor = colors[idx] if method != "ours_localization" else 'none'
        plt.plot(line_x, line_y, 'o', color=colors[idx], markersize=4.0, markerfacecolor=markerfacecolor)

    # plt.ylabel(metric_labels[metric])
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_verticalalignment('top')
    if metric in ["delta_r", "delta_t", "delta_uv", "delta_f", "delta_skew"]:
        plt.yscale("log")
        # plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
        # plt.gca().yaxis.set_major_formatter(FuncFormatter(customPlotFun))
    if metric in ["delta_skew"]:
        plt.ylim(0, 1)
    # plt.title(f"{metric_labels[metric]} vs Noise")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    plt.xlabel(r'$\sigma$', fontsize=16)  # Centered under x-axis
    if metric in ["delta_f", "delta_uv", "delta_r", "delta_t", "success_rate"]:
        plt.text(0.79, -0.2, r'$\times 10^{-2}$', transform=plt.gca().transAxes, fontsize=16)
    if metric in ["delta_skew"]:
        plt.text(0.79, -0.2, r'$\times 10^{-3}$', transform=plt.gca().transAxes, fontsize=16)

    os.makedirs("./synthetic/plots", exist_ok=True)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"./synthetic/plots/{metric}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.04)

# Generate LaTeX
latex = []

latex.append(r"\begin{table}[H]")
latex.append(r"\centering")
latex.append(r"\begin{tabular}{l c l l l}")
latex.append(r"\toprule")
latex.append(r"Noise & Method & " +
             " & ".join(metric_labels[m] for m in metrics) + r" \\")
latex.append(r"\midrule")

for method in methods:
    noise_levels = sorted({noise for metric in metrics for noise in grouped[metric][method].keys()})
    for noise in noise_levels:
        supports = method_supports[method]
        row = [f"{noise:.2f}", method_labels[method]]

        for metric in metrics:
            if supports.get(metric, False):
                values = grouped[metric][method][noise]
                if values:
                    mean = np.mean(values)
                    row.append(f"{mean:.3f}")
                else:
                    row.append(r"--")
            else:
                row.append(r"--")

        latex.append(" & ".join(row) + r" \\")
    latex.append(r"\midrule")

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\label{tab:SyntheticNoise}")
latex.append(r"\caption{Comparison of methods across different noise levels.}")
latex.append(r"\end{table}")

# Output to file or print
with open("table.tex", "w") as f:
    for line in latex:
        f.write(line + "\n")
