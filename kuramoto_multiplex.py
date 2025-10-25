"""Kuramoto model on a two-layer multiplex network with symmetry reduction.

This script builds the quotient network induced by the graph symmetries,
implements a multiplex Kuramoto model with cooperative inter-layer coupling,
scans the inter-layer strength S, and records the steady-state order parameter
for each layer. The final results are exported as an SVG plot.
"""
from __future__ import annotations

import math
import cmath
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class MultiplexKuramotoParams:
    """Container for all model parameters."""

    adjacency: Tuple[Tuple[float, ...], ...]
    cluster_sizes: Tuple[float, ...]
    intralayer_couplings: Tuple[float, float]
    natural_frequencies_layer1: Tuple[float, ...]
    natural_frequencies_layer2: Tuple[float, ...]
    s_values: Sequence[float]
    time_step: float = 0.02
    transient_time: float = 60.0
    average_time: float = 20.0


def build_quotient_adjacency() -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[float, ...]]:
    """Return the adjacency of the quotient network and cluster sizes."""

    adjacency = (
        (0.0, 2.0, 3.0, 0.0),
        (1.0, 1.0, 0.0, 6.0),
        (1.0, 0.0, 2.0, 6.0),
        (0.0, 2.0, 3.0, 2.0),
    )
    cluster_sizes = (1.0, 2.0, 3.0, 6.0)
    return adjacency, cluster_sizes


def order_parameter(phases: Sequence[float], cluster_sizes: Sequence[float]) -> complex:
    """Compute the Kuramoto order parameter for weighted clusters."""

    total = sum(cluster_sizes)
    result = 0j
    for phase, size in zip(phases, cluster_sizes):
        weight = size / total
        result += weight * cmath.exp(1j * phase)
    return result


def kuramoto_rhs(
    phases_layer1: Sequence[float],
    phases_layer2: Sequence[float],
    params: MultiplexKuramotoParams,
    s_strength: float,
) -> Tuple[List[float], List[float]]:
    """Kuramoto right-hand side for both layers."""

    adjacency = params.adjacency
    lambda1, lambda2 = params.intralayer_couplings
    omega1 = params.natural_frequencies_layer1
    omega2 = params.natural_frequencies_layer2

    r1 = abs(order_parameter(phases_layer1, params.cluster_sizes))
    r2 = abs(order_parameter(phases_layer2, params.cluster_sizes))
    d_1_to_2 = 1.0 + s_strength * r1
    d_2_to_1 = 1.0 + s_strength * r2

    def layer_rhs(phases: Sequence[float], omega: Sequence[float], lambda_intra: float, modulation: float) -> List[float]:
        derivatives: List[float] = []
        for i, (phase_i, omega_i) in enumerate(zip(phases, omega)):
            interaction = 0.0
            for j, weight in enumerate(adjacency[i]):
                if weight == 0.0:
                    continue
                interaction += weight * math.sin(phases[j] - phase_i)
            derivatives.append(omega_i + lambda_intra * modulation * interaction)
        return derivatives

    return (
        layer_rhs(phases_layer1, omega1, lambda1, d_2_to_1),
        layer_rhs(phases_layer2, omega2, lambda2, d_1_to_2),
    )


def vector_add(vec: Sequence[float], other: Sequence[float], scale: float) -> List[float]:
    """Add two vectors with scaling."""

    return [v + scale * o for v, o in zip(vec, other)]


def rk4_step(
    theta1: Sequence[float],
    theta2: Sequence[float],
    dt: float,
    params: MultiplexKuramotoParams,
    s_strength: float,
) -> Tuple[List[float], List[float]]:
    """Advance the system by one RK4 step."""

    k11, k12 = kuramoto_rhs(theta1, theta2, params, s_strength)
    k21, k22 = kuramoto_rhs(
        vector_add(theta1, k11, 0.5 * dt),
        vector_add(theta2, k12, 0.5 * dt),
        params,
        s_strength,
    )
    k31, k32 = kuramoto_rhs(
        vector_add(theta1, k21, 0.5 * dt),
        vector_add(theta2, k22, 0.5 * dt),
        params,
        s_strength,
    )
    k41, k42 = kuramoto_rhs(
        vector_add(theta1, k31, dt),
        vector_add(theta2, k32, dt),
        params,
        s_strength,
    )

    next_theta1 = [
        th + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d)
        for th, a, b, c, d in zip(theta1, k11, k21, k31, k41)
    ]
    next_theta2 = [
        th + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d)
        for th, a, b, c, d in zip(theta2, k12, k22, k32, k42)
    ]
    return next_theta1, next_theta2


def simulate_for_s(params: MultiplexKuramotoParams, s_strength: float) -> Tuple[float, float]:
    """Simulate the system at a given S and return the steady-state order parameters."""

    dt = params.time_step
    n_transient_steps = int(params.transient_time / dt)
    n_average_steps = int(params.average_time / dt)

    theta1 = [0.0 for _ in params.natural_frequencies_layer1]
    theta2 = [0.0 for _ in params.natural_frequencies_layer2]

    for _ in range(n_transient_steps):
        theta1, theta2 = rk4_step(theta1, theta2, dt, params, s_strength)

    r1_values: List[float] = []
    r2_values: List[float] = []
    for _ in range(n_average_steps):
        theta1, theta2 = rk4_step(theta1, theta2, dt, params, s_strength)
        r1_values.append(abs(order_parameter(theta1, params.cluster_sizes)))
        r2_values.append(abs(order_parameter(theta2, params.cluster_sizes)))

    r1_mean = sum(r1_values) / len(r1_values)
    r2_mean = sum(r2_values) / len(r2_values)
    return r1_mean, r2_mean


def scan_s_values(params: MultiplexKuramotoParams) -> Tuple[List[float], List[float], List[float]]:
    """Run simulations for each S value and collect order parameters."""

    s_vals = [float(s) for s in params.s_values]
    r1_results: List[float] = []
    r2_results: List[float] = []

    for s_strength in s_vals:
        r1, r2 = simulate_for_s(params, s_strength)
        r1_results.append(r1)
        r2_results.append(r2)

    return s_vals, r1_results, r2_results


def linspace(start: float, stop: float, num: int) -> List[float]:
    """Simple replacement for numpy.linspace."""

    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def save_plot_svg(s_vals: Sequence[float], r1: Sequence[float], r2: Sequence[float], filename: str) -> None:
    """Save a simple SVG plot showing |z| vs S for both layers."""

    width, height = 720, 420
    margin_left, margin_bottom, margin_right, margin_top = 80, 60, 20, 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_s, max_s = min(s_vals), max(s_vals)
    min_r = min(min(r1), min(r2), 0.0)
    max_r = max(max(r1), max(r2), 1.0)

    def scale_x(value: float) -> float:
        if max_s == min_s:
            return margin_left + plot_width / 2.0
        return margin_left + (value - min_s) / (max_s - min_s) * plot_width

    def scale_y(value: float) -> float:
        if max_r == min_r:
            return margin_top + plot_height / 2.0
        return height - margin_bottom - (value - min_r) / (max_r - min_r) * plot_height

    def polyline(points: Sequence[Tuple[float, float]], color: str) -> str:
        coords = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in points)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'

    tick_count = 5
    x_ticks = linspace(min_s, max_s, tick_count)
    y_ticks = linspace(min_r, max_r, tick_count)

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="black" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="black" stroke-width="1.5"/>',
    ]

    for value in x_ticks:
        x = scale_x(value)
        lines.append(f'<line x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 6}" stroke="black" stroke-width="1"/>')
        lines.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 24}" font-size="14" text-anchor="middle">{value:.2f}</text>'
        )

    for value in y_ticks:
        y = scale_y(value)
        lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left - 6}" y2="{y:.2f}" stroke="black" stroke-width="1"/>')
        lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" font-size="14" text-anchor="end">{value:.2f}</text>'
        )

    for value in x_ticks:
        x = scale_x(value)
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#dddddd" stroke-width="1"/>'
        )
    for value in y_ticks:
        y = scale_y(value)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )

    points_layer1 = list(zip(s_vals, r1))
    points_layer2 = list(zip(s_vals, r2))
    lines.append(polyline(points_layer1, "#1f77b4"))
    lines.append(polyline(points_layer2, "#d62728"))

    legend_x = margin_left + 10
    legend_y = margin_top + 10
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="220" height="60" fill="white" stroke="black" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{legend_x + 15}" y1="{legend_y + 20}" x2="{legend_x + 55}" y2="{legend_y + 20}" stroke="#1f77b4" stroke-width="2"/>'
    )
    lines.append(
        f'<text x="{legend_x + 65}" y="{legend_y + 25}" font-size="14">Layer 1 |z|</text>'
    )
    lines.append(
        f'<line x1="{legend_x + 15}" y1="{legend_y + 40}" x2="{legend_x + 55}" y2="{legend_y + 40}" stroke="#d62728" stroke-width="2"/>'
    )
    lines.append(
        f'<text x="{legend_x + 65}" y="{legend_y + 45}" font-size="14">Layer 2 |z|</text>'
    )

    lines.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 15}" font-size="16" text-anchor="middle">Inter-layer coupling strength S</text>'
    )
    lines.append(
        f'<text x="20" y="{margin_top + plot_height / 2:.2f}" font-size="16" text-anchor="middle" transform="rotate(-90 20,{margin_top + plot_height / 2:.2f})">Steady-state |z|</text>'
    )
    lines.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{margin_top - 15}" font-size="18" text-anchor="middle">Stability vs S</text>'
    )

    lines.append('</svg>')

    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    adjacency, cluster_sizes = build_quotient_adjacency()

    params = MultiplexKuramotoParams(
        adjacency=adjacency,
        cluster_sizes=cluster_sizes,
        intralayer_couplings=(0.9, 1.1),
        natural_frequencies_layer1=(0.9, 1.0, 1.1, 1.05),
        natural_frequencies_layer2=(1.05, 0.95, 1.0, 1.15),
        s_values=[i * 0.2 for i in range(26)],
    )

    s_vals, r1, r2 = scan_s_values(params)
    save_plot_svg(s_vals, r1, r2, "stability_vs_S.svg")


if __name__ == "__main__":
    main()
