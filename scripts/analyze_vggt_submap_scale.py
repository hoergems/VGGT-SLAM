#!/usr/bin/env python3
"""Offline, read-only metric/VGGT per-submap similarity diagnostics.

Inputs are the existing headerless trajectory exports:
  metric:  timestamp_ns x y z qx qy qz qw
  submaps: timestamp_ns submap_id frame_index frame_id x y z qx qy qz qw

Or the scale-diagnostic export:
  timestamp_ns submap_id frame_index frame_id raw_xyz optimized_xyz metric_xyz incoming_scale

The fitted convention is metric_position ~= scale * (R @ vggt_position) + t.
No changes are made to SLAM, the input files, or point clouds.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


def read_rows(path: Path, columns: int, label: str):
    rows = []
    with path.open('r', encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != columns:
                raise ValueError(f'{label} {path}:{line_no}: expected {columns} columns, got {len(parts)}')
            try:
                stamp = int(parts[0])  # never parse nanosecond timestamps through float
                values = np.asarray([float(x) for x in parts[1:]], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f'{label} {path}:{line_no}: invalid numeric value') from exc
            if not np.isfinite(values).all():
                raise ValueError(f'{label} {path}:{line_no}: nonfinite value')
            rows.append((stamp, values))
    if not rows:
        raise ValueError(f'{label}: no rows in {path}')
    return rows


def load_metric(path):
    result = {}
    for stamp, v in read_rows(path, 8, 'metric'):
        if stamp in result:
            raise ValueError(f'metric: duplicate timestamp {stamp}')
        result[stamp] = v[0:3]
    return result


def load_submaps(path):
    groups = {}
    seen = set()
    for stamp, v in read_rows(path, 11, 'submap'):
        # These are integral identifiers; validate before converting.
        submap, frame_index, frame_id = v[:3]
        if not float(submap).is_integer() or not float(frame_index).is_integer():
            raise ValueError(f'noninteger submap/frame_index at timestamp {stamp}')
        submap, frame_index = int(submap), int(frame_index)
        key = (submap, frame_index)
        if key in seen:
            raise ValueError(f'duplicate (submap_id, frame_index): {key}')
        seen.add(key)
        groups.setdefault(submap, []).append((frame_index, stamp, v[3:6]))
    for sid in groups:
        groups[sid].sort(key=lambda row: row[0])
        indices = [row[0] for row in groups[sid]]
        if indices != list(range(len(indices))):
            raise ValueError(f'submap {sid}: frame_index is not contiguous from zero: {indices}')
    return groups


def load_scale_diagnostics(path):
    """Load the 14-column raw/final/metric occurrence export."""
    groups, seen = {}, set()
    with path.open('r', encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 14:
                raise ValueError(f'scale diagnostics {path}:{line_no}: expected 14 columns, got {len(parts)}')
            try:
                stamp = int(parts[0])
                submap, frame_index = int(parts[1]), int(parts[2])
                frame_id = float(parts[3])
                positions = np.asarray([float(value) for value in parts[4:13]], dtype=np.float64)
                incoming = float(parts[13])
            except ValueError as exc:
                raise ValueError(f'scale diagnostics {path}:{line_no}: invalid numeric value') from exc
            if not np.isfinite(positions).all():
                raise ValueError(f'scale diagnostics {path}:{line_no}: nonfinite position')
            key = (submap, frame_index)
            if key in seen:
                raise ValueError(f'scale diagnostics: duplicate (submap_id, frame_index): {key}')
            seen.add(key)
            groups.setdefault(submap, []).append((frame_index, stamp, frame_id, positions[:3], positions[3:6], positions[6:9], incoming))
    if not groups:
        raise ValueError(f'scale diagnostics: no rows in {path}')
    for sid, rows in groups.items():
        rows.sort(key=lambda row: row[0])
        indices = [row[0] for row in rows]
        if indices != list(range(len(indices))):
            raise ValueError(f'submap {sid}: frame_index is not contiguous from zero: {indices}')
        scales = [row[-1] for row in rows]
        if not all(np.isnan(scale) for scale in scales) and not all(np.isfinite(scale) for scale in scales):
            raise ValueError(f'submap {sid}: incoming_pairwise_scale must be consistently finite or nan')
        if any(np.isfinite(scale) and scale <= 0 for scale in scales):
            raise ValueError(f'submap {sid}: incoming_pairwise_scale must be positive')
    return groups


def fit_similarity(visual, metric):
    """Proper-rotation least-squares Umeyama; columns are point coordinates."""
    n = len(visual)
    xmean, ymean = visual.mean(axis=0), metric.mean(axis=0)
    x, y = visual - xmean, metric - ymean
    cov = y.T @ x / n
    u, singular, vt = np.linalg.svd(cov)
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = u @ np.diag(signs) @ vt
    variance = np.sum(x * x) / n
    if variance <= np.finfo(float).eps:
        raise ValueError('VGGT trajectory has zero positional variance')
    scale = np.sum(singular * signs) / variance
    translation = ymean - scale * rotation @ xmean
    fitted = scale * (visual @ rotation.T) + translation
    errors = np.linalg.norm(fitted - metric, axis=1)
    return scale, rotation, translation, fitted, errors


def length_and_displacement(points):
    if len(points) < 2:
        return 0.0, 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()), float(np.linalg.norm(points[-1] - points[0]))


def analyze(groups, metric_by_timestamp):
    results = []
    plot_data = []
    missing = [(sid, stamp) for sid, rows in groups.items() for _, stamp, _ in rows if stamp not in metric_by_timestamp]
    if missing:
        raise ValueError(f'{len(missing)} submap occurrences have no exact metric timestamp; first: {missing[:5]}')
    for sid, rows in groups.items():
        timestamps = [stamp for _, stamp, _ in rows]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError(f'submap {sid}: repeated timestamp within one submap')
        visual = np.asarray([p for _, _, p in rows])
        metric = np.asarray([metric_by_timestamp[t] for t in timestamps])
        ml, md = length_and_displacement(metric)
        vl, vd = length_and_displacement(visual)
        duration = (max(timestamps) - min(timestamps)) * 1e-9
        # Centred spread is a scale-observability proxy. A planar or linear
        # path can still constrain scale; do not reject based on 3D rank.
        centred = visual - visual.mean(axis=0)
        spread = float(np.sqrt(np.mean(np.sum(centred ** 2, axis=1))))
        row = dict(submap_id=sid, frames=len(rows), duration_s=duration,
                   metric_path_m=ml, metric_displacement_m=md,
                   vggt_path_units=vl, vggt_displacement_units=vd,
                   vggt_rms_spread_units=spread,
                   scale_m_per_vggt_unit=float('nan'), rmse_m=float('nan'),
                   median_error_m=float('nan'), max_error_m=float('nan'),
                   relative_rmse_to_metric_path=float('nan'),
                   loo_scale_min=float('nan'), loo_scale_max=float('nan'),
                   fit_status='insufficient_frames')
        fitted = None
        if len(rows) >= 3 and spread > 1e-10:
            try:
                scale, rotation, translation, fitted, errors = fit_similarity(visual, metric)
                row.update(scale_m_per_vggt_unit=float(scale),
                           rmse_m=float(np.sqrt(np.mean(errors ** 2))),
                           median_error_m=float(np.median(errors)),
                           max_error_m=float(errors.max()),
                           relative_rmse_to_metric_path=float(np.sqrt(np.mean(errors ** 2)) / ml) if ml > 0 else float('nan'),
                           fit_status='ok')
                # Leave-one-out spread exposes sensitivity to individual frames.
                if len(rows) >= 4:
                    loo = []
                    for omit in range(len(rows)):
                        mask = np.arange(len(rows)) != omit
                        try:
                            s, *_ = fit_similarity(visual[mask], metric[mask])
                            if np.isfinite(s):
                                loo.append(float(s))
                        except (ValueError, np.linalg.LinAlgError):
                            pass
                    if loo:
                        row['loo_scale_min'], row['loo_scale_max'] = min(loo), max(loo)
            except (ValueError, np.linalg.LinAlgError) as exc:
                row['fit_status'] = f'fit_failed:{exc}'
        results.append(row)
        plot_data.append((sid, visual, metric, fitted))
    return results, plot_data


def _fit_fields(source, target, prefix, units):
    result = {f'{prefix}_scale': float('nan'), f'{prefix}_rmse_{units}': float('nan'),
              f'{prefix}_max_error_{units}': float('nan'), f'{prefix}_loo_min': float('nan'),
              f'{prefix}_loo_max': float('nan')}
    if len(source) < 3:
        return result
    spread = float(np.sqrt(np.mean(np.sum((source - source.mean(axis=0)) ** 2, axis=1))))
    if spread <= 1e-10:
        return result
    scale, _, _, _, errors = fit_similarity(source, target)
    result.update({f'{prefix}_scale': float(scale), f'{prefix}_rmse_{units}': float(np.sqrt(np.mean(errors ** 2))),
                   f'{prefix}_max_error_{units}': float(errors.max())})
    if len(source) >= 4:
        loo = []
        for omit in range(len(source)):
            try:
                scale, *_ = fit_similarity(source[np.arange(len(source)) != omit], target[np.arange(len(target)) != omit])
                loo.append(float(scale))
            except (ValueError, np.linalg.LinAlgError):
                pass
        if loo:
            result[f'{prefix}_loo_min'], result[f'{prefix}_loo_max'] = min(loo), max(loo)
    return result


def analyze_scale_diagnostics(groups):
    """Fit raw→metric, raw→final, and final→metric per ordinary submap."""
    results, trajectories, previous_raw_metric_scale = [], [], None
    for sid, entries in groups.items():
        timestamps = [entry[1] for entry in entries]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError(f'submap {sid}: repeated timestamp within one submap')
        raw = np.asarray([entry[3] for entry in entries])
        optimized = np.asarray([entry[4] for entry in entries])
        metric = np.asarray([entry[5] for entry in entries])
        incoming = entries[0][6] if np.isfinite(entries[0][6]) else float('nan')
        ml, md = length_and_displacement(metric)
        ol, od = length_and_displacement(optimized)
        row = dict(submap_id=sid, frames=len(entries), duration_s=(max(timestamps) - min(timestamps)) * 1e-9,
                   metric_path_m=ml, metric_displacement_m=md,
                   optimized_path_vggt_units=ol, optimized_displacement_vggt_units=od,
                   incoming_pairwise_scale=incoming)
        try:
            row.update(_fit_fields(raw, metric, 'raw_to_metric', 'm'))
            row.update(_fit_fields(raw, optimized, 'raw_to_final', 'vggt_units'))
            row.update(_fit_fields(optimized, metric, 'final_to_metric', 'm'))
            row['fit_status'] = 'ok' if np.isfinite(row['raw_to_metric_scale']) else 'insufficient_frames'
        except (ValueError, np.linalg.LinAlgError) as exc:
            row['fit_status'] = f'fit_failed:{exc}'
        raw_metric, raw_final, final_metric = (row[name] for name in ('raw_to_metric_scale', 'raw_to_final_scale', 'final_to_metric_scale'))
        composed = raw_final * final_metric
        row['composed_scale'] = composed
        row['composition_ratio'] = composed / raw_metric if np.isfinite(composed) and np.isfinite(raw_metric) and raw_metric != 0 else float('nan')
        row['composition_relative_error'] = abs(composed - raw_metric) / abs(raw_metric) if np.isfinite(row['composition_ratio']) else float('nan')
        if np.isfinite(incoming) and previous_raw_metric_scale is not None and previous_raw_metric_scale != 0 and np.isfinite(raw_metric):
            implied = raw_metric / previous_raw_metric_scale
            row['metric_implied_pairwise_scale'] = implied
            row['pairwise_scale_ratio'] = incoming / implied
            row['pairwise_relative_error'] = abs(incoming - implied) / abs(implied)
        else:
            row['metric_implied_pairwise_scale'] = row['pairwise_scale_ratio'] = row['pairwise_relative_error'] = float('nan')
        if np.isfinite(raw_metric):
            previous_raw_metric_scale = raw_metric
        results.append(row)
        trajectories.append((sid, optimized, metric, None))
    return results, trajectories


def make_plots(rows, trajectories, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = [r['submap_id'] for r in rows]
    diagnostic_mode = 'raw_to_metric_scale' in rows[0]
    scale_key = 'raw_to_metric_scale' if diagnostic_mode else 'scale_m_per_vggt_unit'
    low_key = 'raw_to_metric_loo_min' if diagnostic_mode else 'loo_scale_min'
    high_key = 'raw_to_metric_loo_max' if diagnostic_mode else 'loo_scale_max'
    scale = [r[scale_key] for r in rows]
    low = [r[low_key] for r in rows]
    high = [r[high_key] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ids, scale, 'o-', label='Per-submap fitted scale')
    for sid, lo, hi in zip(ids, low, high):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([sid, sid], [lo, hi], linewidth=2, alpha=0.45)
    ax.set(xlabel='Actual submap ID', ylabel='Metres / raw VGGT unit', title='Local similarity scales (bars: leave-one-out range)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'submap_scales.png', dpi=170)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axs[0].bar(ids, [r['metric_path_m'] for r in rows], width=10, label='Metric path')
    axs[0].plot(ids, [r['metric_displacement_m'] for r in rows], 'o-', label='End-to-end displacement')
    axs[0].set(ylabel='Metres', title='Local translation baseline')
    axs[0].legend()
    rmse_key = 'raw_to_metric_rmse_m' if diagnostic_mode else 'rmse_m'
    max_key = 'raw_to_metric_max_error_m' if diagnostic_mode else 'max_error_m'
    axs[1].plot(ids, [100 * r[rmse_key] for r in rows], 'o-', label='RMSE')
    axs[1].plot(ids, [100 * r[max_key] for r in rows], 's--', label='Max error')
    axs[1].set(xlabel='Actual submap ID', ylabel='Centimetres', title='Local Sim(3) position residuals')
    axs[1].legend()
    for ax in axs: ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'submap_motion_and_residuals.png', dpi=170)
    plt.close(fig)

    # Translation and orientation are independently fitted for each submap.
    # Therefore each panel has its own metric-odom coordinate frame.
    cols = 3
    count = len(trajectories)
    fig, axes = plt.subplots((count + cols - 1) // cols, cols, figsize=(14, 3.5 * ((count + cols - 1) // cols)), squeeze=False)
    for ax, (sid, visual, metric, fitted) in zip(axes.flat, trajectories):
        ax.plot(metric[:, 0], metric[:, 1], 'o-', markersize=2.5, label='Go2 metric')
        if fitted is not None:
            ax.plot(fitted[:, 0], fitted[:, 1], 'x--', markersize=3, label='VGGT locally aligned')
        ax.set_title(f'Submap {sid} (XY in metric odom frame)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    for ax in list(axes.flat)[count:]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'submap_local_xy_alignments.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metric', type=Path, help='Canonical metric optical-camera trajectory (legacy mode)')
    parser.add_argument('--vggt-submaps', type=Path, help='Overlap-preserving optimized VGGT trajectory (legacy mode)')
    parser.add_argument('--scale-diagnostics', type=Path, help='14-column raw/final/metric VGGT scale diagnostic export')
    parser.add_argument('--output-dir', type=Path, default=Path('submap_scale_diagnostics'))
    parser.add_argument('--no-plots', action='store_true', help='Only write CSV (no matplotlib required)')
    args = parser.parse_args(argv)

    if args.scale_diagnostics is not None:
        if args.metric is not None or args.vggt_submaps is not None:
            parser.error('--scale-diagnostics cannot be combined with --metric or --vggt-submaps')
        groups = load_scale_diagnostics(args.scale_diagnostics)
        rows, trajectories = analyze_scale_diagnostics(groups)
    else:
        if args.metric is None or args.vggt_submaps is None:
            parser.error('legacy mode requires both --metric and --vggt-submaps')
        metric = load_metric(args.metric)
        groups = load_submaps(args.vggt_submaps)
        rows, trajectories = analyze(groups, metric)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'submap_scale_diagnostics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    occurrences = sum(len(group) for group in groups.values())
    unique = len({entry[1] for group in groups.values() for entry in group})
    print(f'Ordinary submaps: {len(groups)} | occurrences: {occurrences} | unique timestamps: {unique} | overlap occurrences: {occurrences - unique}')
    if args.scale_diagnostics is not None:
        print(f'{"Submap":>7} {"N":>4} {"raw->m":>10} {"raw->final":>12} {"final->m":>10} {"compose ratio":>14} {"pairwise ratio":>15}')
        for r in rows:
            print(f'{r["submap_id"]:7d} {r["frames"]:4d} {r["raw_to_metric_scale"]:10.4f} {r["raw_to_final_scale"]:12.4f} {r["final_to_metric_scale"]:10.4f} {r["composition_ratio"]:14.4f} {r["pairwise_scale_ratio"]:15.4f}')
    else:
        print(f'{"Submap":>7} {"N":>4} {"path(m)":>9} {"disp(m)":>9} {"scale(m/u)":>11} {"RMSE(cm)":>10} {"max(cm)":>9} {"LOO range(m/u)":>20}')
        for r in rows:
            lo, hi = r['loo_scale_min'], r['loo_scale_max']
            loo = f'{lo:.3f}..{hi:.3f}' if np.isfinite(lo) and np.isfinite(hi) else 'n/a'
            print(f'{r["submap_id"]:7d} {r["frames"]:4d} {r["metric_path_m"]:9.3f} {r["metric_displacement_m"]:9.3f} {r["scale_m_per_vggt_unit"]:11.3f} {100*r["rmse_m"]:10.2f} {100*r["max_error_m"]:9.2f} {loo:>20}')
    print(f'CSV: {csv_path}')
    if not args.no_plots:
        make_plots(rows, trajectories, args.output_dir)
        print(f'Plots: {args.output_dir / "submap_scales.png"}, {args.output_dir / "submap_motion_and_residuals.png"}, {args.output_dir / "submap_local_xy_alignments.png"}')
    print('Interpretation: scales are independent local diagnostics, NOT a global map scale. Short/low-baseline segments and large residuals warrant caution.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
