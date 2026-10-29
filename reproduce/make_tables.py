#!/usr/bin/env python3
"""Regenerate all seven frozen Reaction-paper tables from Tier-A CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(os.environ.get("REACTION_DATA_ROOT", ROOT / "data/reaction-paper-v1"))
METHODS = ("Exact FEM", "FNO", "DeepONet", "GenFNO", "GenDeepONet")
SURROGATES = METHODS[1:]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def percent(value: str, digits: int) -> str:
    return f"{100.0 * float(value):.{digits}f}"


def integer(value: str | float) -> str:
    return f"{int(round(float(value))):d}"


def grouped_percent(values: list[float], digits: int = 1) -> str:
    low = f"{100.0 * min(values):.{digits}f}"
    high = f"{100.0 * max(values):.{digits}f}"
    return low if low == high else f"{low}--{high}"


def table_1(data: Path, output: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Architecture & $\lambda_J$ & Parameters & Epoch & $E_F$ (\%) & $E_J$ (\%) & $S$ (\%) \\",
        r"\midrule",
    ]
    for row in rows(data / "tables/headline_emulator_test.csv"):
        lines.append(
            f"{row['architecture']} & {int(float(row['lambda_J']))} & {int(row['trainable_parameters']):,} & "
            f"{int(row['checkpoint_epoch'])} & {percent(row['E_F_test'], 3)} & "
            f"{percent(row['E_J_test'], 3)} & {percent(row['S_test'], 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(output / "headline_emulator_test.tex", lines)


def table_2(data: Path, output: Path) -> None:
    lines = [
        r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Truth & Method & $k_{\mathrm{or}}$ & $e_m^\star$ (\%) & $e_m^{K}$ (\%) & Surrogate residual (\%) & FEM residual (\%) \\",
        r"\midrule",
    ]
    source_rows = rows(data / "tables/noiseless_headline_T1_T2.csv")
    for index, row in enumerate(source_rows):
        if index and row["truth_id"] != source_rows[index - 1]["truth_id"]:
            lines.append(r"\midrule")
        surrogate = "--" if not row["surrogate_relative_residual_at_oracle"] else percent(row["surrogate_relative_residual_at_oracle"], 3)
        lines.append(
            f"{row['truth_id']} & {row['method']} & {integer(row['oracle_iteration'])} & "
            f"{percent(row['oracle_relative_u_error'], 3)} & {percent(row['final_relative_u_error'], 3)} & "
            f"{surrogate} & {percent(row['fem_relative_residual_at_same_field'], 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", "}"]
    write(output / "noiseless_headline_T1_T2.tex", lines)


def table_3(data: Path, output: Path) -> None:
    lines = [
        r"\begin{tabular}{lrr@{\hspace{10pt}}rrr@{\hspace{10pt}}rrr}", r"\toprule",
        r" & & & \multicolumn{3}{c}{Mat\'ern, $\nu_C=3/2$, $\ell_C=0.1$} & \multicolumn{3}{c}{Identity} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Width/modes & $E_F$ (\%) & $E_J$ (\%) & $e_m^\star$ (\%) & $k_{\rm or}$ & Final/min & $e_m^\star$ (\%) & $k_{\rm or}$ & Final/min \\",
        r"\midrule",
    ]
    for row in rows(data / "tables/fno_capacity_semiconvergence.csv"):
        lines.append(
            f"{row['width']}/{row['modes']} & {percent(row['E_F_test'], 3)} & {percent(row['E_J_test'], 3)} & "
            f"{percent(row['matern_min_error'], 2)} & {integer(row['matern_oracle_iteration'])} & {float(row['matern_final_over_min']):.2f} & "
            f"{percent(row['identity_min_error'], 2)} & {integer(row['identity_oracle_iteration'])} & {float(row['identity_final_over_min']):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(output / "fno_capacity_semiconvergence.tex", lines)


def table_4(data: Path, output: Path) -> None:
    source = sorted(rows(data / "tables/fno_training_size_semiconvergence.csv"), key=lambda r: (int(r["width"]), int(r["train_samples"])))
    lines = [
        r"\begin{tabular}{rrcrr@{\hspace{8pt}}rrr@{\hspace{8pt}}rrr}", r"\toprule",
        r" & & & & & \multicolumn{3}{c}{T2} & \multicolumn{3}{c}{T1} \\",
        r"\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Training fields & Width/modes & $\lambda_J$ & $E_F^{\rm val}$ (\%) & $E_J^{\rm val}$ (\%) & $e_m^\star$ (\%) & $k_{\rm or}$ & Final/min & $e_m^\star$ (\%) & $k_{\rm or}$ & Final/min \\",
        r"\midrule",
    ]
    last_width = None
    for row in source:
        if last_width is not None and row["width"] != last_width:
            lines.append(r"\addlinespace")
        if row["T1_available"] == "1":
            t1 = f"{percent(row['T1_min_error'], 2)} & {integer(row['T1_oracle_iteration'])} & {float(row['T1_final_over_min']):.2f}"
        else:
            t1 = "-- & -- & --"
        lines.append(
            f"{int(row['train_samples']):,} & {row['width']}/{row['modes']} & {int(float(row['lambda_J']))} & "
            f"{percent(row['E_F_val'], 3)} & {percent(row['E_J_val'], 3)} & "
            f"{percent(row['T2_min_error'], 2)} & {integer(row['T2_oracle_iteration'])} & {float(row['T2_final_over_min']):.2f} & {t1} \\\\"
        )
        last_width = row["width"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(output / "fno_training_size_semiconvergence.tex", lines)


def low_smoothness_lookup(data: Path) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    for row in rows(data / "revision/E3/reference_arms.csv"):
        method = "Exact FEM" if row["architecture"] == "ExactFEM" else row["architecture"]
        result[(row["truth"], method, row["configuration_id"])] = float(row["min_relative_coefficient_error"])
    for row in rows(data / "revision/E3/trajectory_outcomes_pre_geometry.csv"):
        if float(row["matern_length_scale"]) != 0.1:
            continue
        method = "Exact FEM" if row["architecture"] == "ExactFEM" else row["architecture"]
        result[(row["truth"], method, row["configuration_id"])] = float(row["min_relative_coefficient_error"])
    return result


def table_5(data: Path, output: Path) -> None:
    lookup = low_smoothness_lookup(data)
    lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r" & & \multicolumn{3}{c}{Mat\'ern, $\ell_C=0.1$} & \\",
        r"\cmidrule(lr){3-5}",
        r"Truth & Method & $\nu_C=3/2$ & $\nu_C=0.25$ & $\nu_C=0.4$ & Identity \\",
        r"\midrule",
    ]
    for truth in ("T1", "T2"):
        if truth == "T2":
            lines.append(r"\midrule")
        for method in METHODS:
            cells = []
            for config in ("baseline_nu1p5_ell0p1", "nu0p25_ell0p1", "nu0p4_ell0p1", "identity"):
                value = lookup.get((truth, method, config))
                cells.append("--" if value is None else f"{100.0 * value:.2f}")
            lines.append(f"{truth} & {method} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(output / "preconditioner_low_smoothness_errors.tex", lines)


def table_6(data: Path, output: Path) -> None:
    selected = [row for row in rows(data / "revision/E1_E2/trajectory_diagnostics_summary.csv") if row["configuration_id"] == "nu0p25_ell0p1"]
    lookup = {(row["truth"], "Exact FEM" if row["architecture"] == "ExactFEM" else row["architecture"]): row for row in selected}
    lines = [
        r"\begin{tabular}{llrrrrcr}", r"\toprule",
        r"Truth & Method & $\zeta_{k_{\mathrm{or}}}$ & $k_\zeta$ & $k_{\mathrm{or}}$ & $e_m(k_\zeta)/e_m(k_{\mathrm{or}})$ & $\M_C$ monotone & $\max_k s_k$ \\",
        r"\midrule",
    ]
    for truth in ("T1", "T2"):
        if truth == "T2":
            lines.append(r"\midrule")
        for method in METHODS:
            row = lookup[(truth, method)]
            crossing = "--" if not row["first_diagnostic_gamma_at_least_half"] else f"{int(float(row['first_diagnostic_gamma_at_least_half'])):,}"
            ratio = "--" if not row["first_gamma_half_error_over_oracle"] else f"{float(row['first_gamma_half_error_over_oracle']):.2f}"
            monotone = "yes" if row["geometry_monotone_to_oracle"] == "1" else "no"
            lines.append(
                f"{truth} & {method} & {float(row['gamma_at_oracle']):.2f} & {crossing} & "
                f"{int(float(row['oracle_iteration'])):,} & {ratio} & {monotone} & {float(row['maximum_s']):.2f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(output / "table_low_smoothness_diagnostics_single.tex", lines)


def table_7(data: Path, output: Path) -> None:
    selected_taus = (1.0, 1.5, 2.0)
    source = [row for row in rows(data / "revision/E5/noisy_tau_stopping.csv") if float(row["tau"]) in selected_taus]
    lookup = {(row["truth"], int(row["noise_percent"]), row["method"], float(row["tau"])): row for row in source}
    lines = [
        r"\begin{tabular}{lllrrrr}", r"\toprule",
        r" & & & & \multicolumn{3}{c}{$e_m(k_\tau)$ (\%)} \\",
        r"\cmidrule(lr){5-7}",
        r"Truth & Noise & Model & $e_m(k_{\mathrm{or}})$ (\%) & $\tau=1$ & $\tau=1.5$ & $\tau=2$ \\",
        r"\midrule",
    ]
    for truth in ("T1", "T2"):
        if truth == "T2":
            lines.append(r"\midrule")
        for noise in (1, 2, 5):
            exact = [lookup[(truth, noise, "Exact FEM", tau)] for tau in selected_taus]
            exact_cells = [f"{100.0 * float(exact[0]['oracle_relative_u_error']):.1f}"]
            exact_cells += ["--" if row["stopped"] != "1" else f"{100.0 * float(row['relative_u_error_at_k_tau']):.1f}" for row in exact]
            lines.append(f"{truth} & {noise}\\% & Exact FEM & " + " & ".join(exact_cells) + r" \\")
            oracle = [float(lookup[(truth, noise, method, 1.0)]["oracle_relative_u_error"]) for method in SURROGATES]
            surrogate_cells = [grouped_percent(oracle)]
            for tau in selected_taus:
                current = [lookup[(truth, noise, method, tau)] for method in SURROGATES]
                stopped = [row for row in current if row["stopped"] == "1"]
                if not stopped:
                    surrogate_cells.append("--")
                else:
                    cell = grouped_percent([float(row["relative_u_error_at_k_tau"]) for row in stopped])
                    if len(stopped) < 4:
                        cell += f" ({len(stopped)}/4)"
                    surrogate_cells.append(cell)
            lines.append(" & & Surrogates & " + " & ".join(surrogate_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(output / "noisy_tau_stopping_grouped.tex", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/tables")
    args = parser.parse_args()
    data = args.data_root.resolve()
    output = args.output_dir.resolve()
    table_1(data, output)
    table_2(data, output)
    table_3(data, output)
    table_4(data, output)
    table_5(data, output)
    table_6(data, output)
    table_7(data, output)
    print(f"Wrote seven tables to {output}")


if __name__ == "__main__":
    main()
