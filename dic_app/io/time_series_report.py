"""Maps, CSV export and PDF report of a multi-epoch (time series) analysis.

The same ``draw_map`` is used by the GUI tab and by the PDF report, so
what is printed is exactly what was looked at on screen.
"""

import csv
import datetime as _dt
import os

import numpy as np

from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)

MAP_MODES = [
    ('cum', "Spostamento cumulato"),
    ('vel', "Velocità media"),
    ('vlast', "Velocità ultimo intervallo"),
    ('acc', "Accelerazione (ultimo / medio)"),
    ('onset', "Data del crollo"),
    ('class', "Classi"),
]


def units(ts):
    """(factor px->unit, displacement unit, velocity unit)."""
    per = '/anno' if ts.has_dates else '/rilievo'
    if ts.gsd:
        return ts.gsd * 100.0, 'cm', 'cm' + per
    return 1.0, 'px', 'px' + per


def _extent(ts):
    gx, gy = ts.grid_x, ts.grid_y
    sx = float(gx[0, 1] - gx[0, 0]) if gx.shape[1] > 1 else 1.0
    sy = float(gy[1, 0] - gy[0, 0]) if gy.shape[0] > 1 else 1.0
    return (gx[0, 0] - sx / 2, gx[0, -1] + sx / 2, gy[-1, 0] + sy / 2, gy[0, 0] - sy / 2)


def _robust_max(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    return float(max(np.percentile(a, 99), 1e-6))


def onset_colors(n):
    import matplotlib
    cmap = matplotlib.colormaps['tab10'] if hasattr(matplotlib, 'colormaps') \
        else matplotlib.cm.get_cmap('tab10')
    return [cmap(i % 10) for i in range(n)]


def draw_map(fig, ax, ts, base, mode, k=None, vmax=None, outlines=True,
             colorbar=True, alpha=0.7):
    """Draw one map of the series on *ax* (image coordinates)."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    f, du, vu = units(ts)
    if base is not None:
        if base.ndim == 2:
            ax.imshow(base, cmap='gray', interpolation='bilinear')
        else:
            ax.imshow(base, interpolation='bilinear')
    ext = _extent(ts)
    onset = ts.collapse_onset()
    title, cb_label, im = '', '', None
    if k is None:
        k = ts.n - 1

    if mode == 'cum':
        U, V = ts.cumulative_stack()
        field = np.hypot(U[k], V[k]) * f
        vm = vmax or _robust_max(field)
        im = ax.imshow(np.ma.masked_invalid(field), cmap='turbo', vmin=0, vmax=vm,
                       extent=ext, alpha=alpha, interpolation='nearest')
        title = (f"Spostamento cumulato {ts.epochs[ts.ref_pos].date_str()} → "
                 f"{ts.epochs[k].date_str()}")
        cb_label = f"spostamento [{du}]"
    elif mode in ('vel', 'vlast'):
        v = ts.velocity() if mode == 'vel' else ts.last_interval_velocity()
        field = v['vmag'] * f
        vm = vmax or _robust_max(field)
        im = ax.imshow(np.ma.masked_invalid(field), cmap='turbo', vmin=0, vmax=vm,
                       extent=ext, alpha=alpha, interpolation='nearest')
        title = "Velocità media (regressione su tutti i rilievi)" if mode == 'vel' \
            else "Velocità nell'ultimo intervallo misurato"
        cb_label = f"velocità [{vu}]"
    elif mode == 'acc':
        field = ts.last_interval_velocity()['ratio']
        im = ax.imshow(np.ma.masked_invalid(field), cmap='RdBu_r', vmin=0, vmax=2,
                       extent=ext, alpha=alpha, interpolation='nearest')
        title = "Accelerazione: velocità ultimo intervallo / velocità media"
        cb_label = "<1 rallenta · 1 costante · >1 accelera"
    elif mode == 'onset':
        present = [i for i in range(ts.n) if (onset == i).any()]
        if present:
            cols = onset_colors(ts.n)
            lut = np.full(ts.n, -1)
            lut[present] = np.arange(len(present))
            idx = np.where(onset >= 0, lut[np.clip(onset, 0, None)], -1)
            cmap = ListedColormap([cols[i] for i in present])
            norm = BoundaryNorm(np.arange(-0.5, len(present) + 0.5), len(present))
            im = ax.imshow(np.ma.masked_less(idx, 0), cmap=cmap, norm=norm,
                           extent=ext, alpha=0.85, interpolation='nearest')
        title = "Crolli: rilievo in cui compaiono per la prima volta"
        cb_label = ''
    elif mode == 'class':
        from dic_app.core.change_detection import CLASS_COLORS, NODATA
        pc = ts.cumulative_class(k)
        if pc is not None:
            rgba = np.zeros(pc.shape + (4,), float)
            for c, col in CLASS_COLORS.items():
                m = pc == c
                rgba[m, :3] = np.array(col) / 255.0
                rgba[m, 3] = 0 if c == NODATA else alpha
            ax.imshow(rgba, extent=ext, interpolation='nearest')
        title = f"Classi {ts.epochs[ts.ref_pos].date_str()} → {ts.epochs[k].date_str()}"

    if mode == 'onset' and im is None:
        ax.text(0.5, 0.02, "Nessun crollo rilevato", transform=ax.transAxes,
                ha='center', color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5))
    if outlines and (onset >= 0).any():
        ax.contour(ts.grid_x, ts.grid_y, (onset >= 0).astype(float), levels=[0.5],
                   colors='red', linewidths=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if base is not None:
        ax.set_xlim(0, base.shape[1])
        ax.set_ylim(base.shape[0], 0)

    if colorbar and im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
        if mode == 'onset':
            prev = (lambda i: ts.epochs[i - 1]) if ts.has_intervals \
                else (lambda i: ts.epochs[ts.ref_pos])
            cb.set_ticks(range(len(present)))
            cb.set_ticklabels([f"{prev(i).date_str()} → {ts.epochs[i].date_str()}"
                               for i in present])
            cb.ax.tick_params(labelsize=7)
        else:
            cb.set_label(cb_label, fontsize=8)
            cb.ax.tick_params(labelsize=7)
    return im


def draw_point_series(ax, ts, r, c):
    """Displacement history of grid node (r, c)."""
    f, du, vu = units(ts)
    s = ts.point_series(r, c)
    if ts.has_dates:
        x = [e.date for e in ts.epochs]
    else:
        x = np.arange(ts.n)
    ok = np.isfinite(s['mag'])
    ax.plot([xi for xi, o in zip(x, ok) if o], s['mag'][ok] * f, 'o-', color='#c62828',
            label='totale')
    ax.plot([xi for xi, o in zip(x, ok) if o], s['u'][ok] * f, '.--', color='#1565c0',
            lw=0.8, label='Est (x)')
    ax.plot([xi for xi, o in zip(x, ok) if o], s['v'][ok] * f, '.--', color='#2e7d32',
            lw=0.8, label='Sud (y immagine)')
    miss = [xi for xi, o in zip(x, ok) if not o]
    if miss:
        ax.plot(miss, [0] * len(miss), 'x', color='gray', label='non misurato')
    if s['onset'] >= 0:
        ax.axvline(x[s['onset']], color='red', lw=1.5, alpha=0.6, label='crollo rilevato')
    v = ts.velocity()
    lv = ts.last_interval_velocity()
    vm, vl, acc = v['vmag'][r, c] * f, lv['vmag'][r, c] * f, lv['ratio'][r, c]
    gx, gy = ts.grid_x[r, c], ts.grid_y[r, c]
    t = f"Punto ({gx:.0f}, {gy:.0f}) px"
    if np.isfinite(vm):
        t += f" — velocità media {vm:.2f} {vu}"
    if np.isfinite(vl):
        t += f", ultimo intervallo {vl:.2f}"
    if np.isfinite(acc):
        t += f" (×{acc:.2f})"
    ax.set_title(t, fontsize=9)
    ax.set_ylabel(f"spostamento [{du}]", fontsize=8)
    ax.axhline(0, color='k', lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')
    ax.tick_params(labelsize=7)


# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------

def export_csv(ts, path_points, path_collapses=None):
    f, du, vu = units(ts)
    U, V = ts.cumulative_stack()
    v = ts.velocity()
    lv = ts.last_interval_velocity()
    onset = ts.collapse_onset()
    gx, gy = ts.grid_x, ts.grid_y
    labels = [e.date.isoformat() if e.date else e.label for e in ts.epochs]
    with open(path_points, 'w', newline='') as fh:
        w = csv.writer(fh, delimiter=';')
        head = ['riga', 'colonna', 'x_px', 'y_px']
        head += [f"spost_{l}_{du}" for l in labels]
        head += [f"vel_media_{vu}", f"vel_ultimo_{vu}", "accelerazione_rapporto",
                 "crollo_rilievo"]
        w.writerow(head)
        ny, nx = gx.shape
        for r in range(ny):
            for c in range(nx):
                mags = np.hypot(U[:, r, c], V[:, r, c]) * f
                if not np.isfinite(mags).any() or (np.isfinite(mags).sum() <= 1
                                                   and onset[r, c] < 0):
                    continue
                row = [r, c, f"{gx[r, c]:.1f}", f"{gy[r, c]:.1f}"]
                row += ['' if not np.isfinite(m) else f"{m:.4f}" for m in mags]
                for val in (v['vmag'][r, c] * f, lv['vmag'][r, c] * f, lv['ratio'][r, c]):
                    row.append('' if not np.isfinite(val) else f"{val:.4f}")
                row.append(labels[onset[r, c]] if onset[r, c] >= 0 else '')
                w.writerow(row)
    if path_collapses:
        with open(path_collapses, 'w', newline='') as fh:
            w = csv.writer(fh, delimiter=';')
            w.writerow(['da', 'a', 'zona', 'area_m2', 'x_px', 'y_px', 'qualita_media'])
            for row in ts.collapse_table():
                for z in row['zones']:
                    w.writerow([row['from'].date_str(), row['to'].date_str(), z['id'],
                                '' if z.get('area_m2') is None else f"{z['area_m2']:.2f}",
                                f"{z['centroid_x_px']:.0f}", f"{z['centroid_y_px']:.0f}",
                                f"{z.get('mean_quality', float('nan')):.3f}"])
    logger.info(f"Time series CSV exported: {path_points}")


# ----------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------

def summary_lines(ts):
    f, du, vu = units(ts)
    lines = []
    v = ts.velocity()['vmag'] * f
    floor = ts.velocity_noise_floor() * f
    if np.isfinite(v).any():
        moving = np.isfinite(v) & (v > floor)
        lines.append(f"Velocità media massima (99° percentile): "
                     f"{np.nanpercentile(v, 99):.2f} {vu}")
        lines.append(f"Soglia di stabilità (rumore): {floor:.2f} {vu}; "
                     f"punti sopra soglia: {100 * moving.sum() / max(np.isfinite(v).sum(), 1):.1f}%")
        acc = ts.last_interval_velocity()['ratio']
        n_acc = int(np.sum(acc > 1.5))
        if n_acc:
            lines.append(f"Punti in accelerazione (velocità ultimo intervallo > 1,5× "
                         f"media): {n_acc}")
    rows = ts.collapse_table()
    if rows:
        tot = rows[-1]['cum_area_m2']
        nz = sum(r['n_zones'] for r in rows)
        lines.append(f"Crolli rilevati: {nz} zone"
                     + (f", area totale {tot:.1f} m²" if tot is not None else ""))
    if not ts.has_dates:
        lines.append("Date dei rilievi non complete: le velocità sono per rilievo, "
                     "non per anno.")
    return lines


def export_pdf(ts, base, path, info=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    info = info or {}
    f, du, vu = units(ts)
    A4L = (11.69, 8.27)
    with PdfPages(path) as pdf:
        # 1. title + epochs + summary
        fig = Figure(figsize=A4L)
        ax = fig.add_subplot(111)
        ax.axis('off')
        y = 0.95
        ax.text(0.02, y, "Serie temporale DIC", fontsize=20, weight='bold',
                transform=ax.transAxes)
        y -= 0.06
        sub = " · ".join(x for x in (info.get('site'), info.get('analyst'),
                                     _dt.date.today().strftime('%d/%m/%Y')) if x)
        ax.text(0.02, y, sub, fontsize=10, color='#555', transform=ax.transAxes)
        y -= 0.08
        ax.text(0.02, y, "Rilievi", fontsize=13, weight='bold', transform=ax.transAxes)
        y -= 0.045
        for i, e in enumerate(ts.epochs):
            role = "riferimento" if e.is_reference else ""
            ax.text(0.04, y, f"{i + 1}. {e.date_str():<14}  {e.label}  {role}",
                    fontsize=9, family='monospace', transform=ax.transAxes)
            y -= 0.035
        y -= 0.03
        ax.text(0.02, y, "Sintesi", fontsize=13, weight='bold', transform=ax.transAxes)
        y -= 0.045
        for line in summary_lines(ts):
            ax.text(0.04, y, "• " + line, fontsize=9, transform=ax.transAxes)
            y -= 0.035
        p = ts.params
        if p is not None:
            y -= 0.03
            ax.text(0.02, y, "Parametri", fontsize=13, weight='bold', transform=ax.transAxes)
            y -= 0.045
            gsd = f"GSD {ts.gsd * 100:.2f} cm/px · " if ts.gsd else ""
            ax.text(0.04, y, f"{gsd}metodo {p.method.value} · subset {p.subset_size} px · "
                    f"passo {p.step_size} px · analisi intervalli: "
                    f"{'sì' if ts.has_intervals else 'no'}",
                    fontsize=9, transform=ax.transAxes)
        pdf.savefig(fig)

        # 2. velocity + acceleration
        for modes in (('vel', 'acc'), ('vlast', 'onset')):
            fig = Figure(figsize=A4L)
            for j, m in enumerate(modes):
                ax = fig.add_subplot(1, 2, j + 1)
                draw_map(fig, ax, ts, base, m)
            fig.tight_layout()
            pdf.savefig(fig)

        # 3. collapsed area per interval
        rows = ts.collapse_table()
        if rows and ts.gsd:
            fig = Figure(figsize=A4L)
            ax = fig.add_subplot(111)
            lab = [f"{r['from'].date_str()}\n→ {r['to'].date_str()}" for r in rows]
            cols = onset_colors(ts.n)
            ax.bar(range(len(rows)), [r['new_area_m2'] or 0 for r in rows],
                   color=[cols[r['pos']] for r in rows])
            ax2 = ax.twinx()
            ax2.plot(range(len(rows)), [r['cum_area_m2'] for r in rows], 'k.-',
                     label='area cumulata')
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels(lab, fontsize=8)
            ax.set_ylabel("nuova area crollata [m²]")
            ax2.set_ylabel("area cumulata [m²]")
            ax.set_title("Area dei crolli per intervallo")
            ax.grid(alpha=0.3, axis='y')
            fig.tight_layout()
            pdf.savefig(fig)

        # 4. cumulative maps, 4 per page, same colour scale
        U, V = ts.cumulative_stack()
        ks = [k for k in range(ts.n) if k != ts.ref_pos]
        vmax = _robust_max(np.hypot(U, V) * f)
        for i0 in range(0, len(ks), 4):
            fig = Figure(figsize=A4L)
            for j, k in enumerate(ks[i0:i0 + 4]):
                ax = fig.add_subplot(2, 2, j + 1)
                draw_map(fig, ax, ts, base, 'cum', k, vmax=vmax)
            fig.tight_layout()
            pdf.savefig(fig)

        # 5. history of the fastest points
        v = ts.velocity()['vmag']
        if np.isfinite(v).any():
            flat = np.where(np.isfinite(v), v, -1).ravel()
            picks, used = [], []
            step = max(ts.grid_x.shape) // 10 + 1
            for idx in np.argsort(flat)[::-1]:
                if flat[idx] <= 0 or len(picks) >= 4:
                    break
                r, c = divmod(int(idx), v.shape[1])
                if all(abs(r - a) + abs(c - b) > step for a, b in used):
                    picks.append((r, c))
                    used.append((r, c))
            if picks:
                fig = Figure(figsize=A4L)
                for j, (r, c) in enumerate(picks):
                    ax = fig.add_subplot(2, 2, j + 1)
                    draw_point_series(ax, ts, r, c)
                fig.suptitle("Storia dei punti più veloci", fontsize=12)
                fig.tight_layout()
                pdf.savefig(fig)
    logger.info(f"Time series report written: {path}")
