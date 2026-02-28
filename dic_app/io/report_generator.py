"""PDF report generation with matplotlib figures and data export."""

import os
import io
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from datetime import datetime

from dic_app.core.dic_engine import DICResult
from dic_app.core.postprocessing import DisplacementStatistics
from dic_app.utils.helpers import (
    displacement_colormap, overlay_heatmap, draw_vector_field,
    compute_magnitude, setup_logger
)

logger = setup_logger(__name__)


class ReportGenerator:
    """Generate comprehensive analysis reports in multiple formats.

    Supports:
    - Multi-page PDF report with figures, tables, and metadata
    - CSV export of displacement data
    - GeoTIFF export of georeferenced results
    - Individual PNG images of all visualization maps
    """

    def __init__(self, result: DICResult, config: dict,
                 base_image: np.ndarray = None,
                 deformed_image: np.ndarray = None,
                 deformed_image_rgb: np.ndarray = None,
                 strain_data: dict = None,
                 active_zones: list = None,
                 pipeline_info: str = "",
                 gsd: float = None,
                 gps_info: dict = None):
        self.result = result
        self.config = config
        self.base_image = base_image
        self.deformed_image = deformed_image
        self.deformed_image_rgb = deformed_image_rgb
        self.strain_data = strain_data
        self.active_zones = active_zones or []
        self.pipeline_info = pipeline_info
        self.gsd = gsd
        self.gps_info = gps_info or {}
        self._progress_callback = None

    def set_progress_callback(self, callback):
        self._progress_callback = callback

    def _report_progress(self, percent, msg=""):
        if self._progress_callback:
            self._progress_callback(int(percent), msg)

    def generate_all(self):
        """Generate all requested outputs."""
        output_dir = self.config.get('output_dir', '.')
        os.makedirs(output_dir, exist_ok=True)
        formats = self.config.get('formats', {})

        step = 0
        total_steps = sum(formats.values())

        if formats.get('images', False):
            self._report_progress(10, "Generazione immagini PNG...")
            self.export_images(output_dir)
            step += 1

        if formats.get('csv', False):
            self._report_progress(30, "Esportazione CSV...")
            self.export_csv(os.path.join(output_dir, "displacement_data.csv"))
            step += 1

        if formats.get('pdf', False):
            self._report_progress(50, "Generazione report PDF...")
            self.generate_pdf(os.path.join(output_dir, "report_dic.pdf"))
            step += 1

        if formats.get('geotiff', False):
            self._report_progress(80, "Esportazione GeoTIFF...")
            self.export_geotiff(output_dir)
            step += 1

        self._report_progress(100, "Report completato!")

    # ------------------------------------------------------------------
    # PDF Report
    # ------------------------------------------------------------------

    def generate_pdf(self, output_path: str):
        """Generate multi-page PDF report using matplotlib."""
        import matplotlib
        matplotlib.use('Agg')

        from matplotlib.backends.backend_pdf import PdfPages

        sections = self.config.get('sections', {})

        page_generators = []
        page_generators.append(('title', self._add_title_page))

        if sections.get('overview', True):
            page_generators.append(('overview', self._add_overview_page))
        if sections.get('parameters', True):
            page_generators.append(('parameters', self._add_parameters_page))
        if sections.get('displacement', True):
            page_generators.append(('displacement', self._add_displacement_pages))
        if sections.get('vectors', True):
            page_generators.append(('vectors', self._add_vector_page))
        if sections.get('correlation', True):
            page_generators.append(('correlation', self._add_correlation_page))
        if sections.get('strain', True) and self.strain_data:
            page_generators.append(('strain', self._add_strain_pages))
        if sections.get('zones', True) and self.active_zones:
            page_generators.append(('zones', self._add_zones_page))
        if sections.get('zone_details', True) and self.active_zones:
            page_generators.append(('zone_details', self._add_zone_detail_pages))
        if sections.get('statistics', True):
            page_generators.append(('statistics', self._add_statistics_page))

        with PdfPages(output_path) as pdf:
            for name, func in page_generators:
                try:
                    func(pdf)
                except Exception as e:
                    logger.error(f"Error generating '{name}' page: {e}")
                    # Add error page instead of crashing
                    try:
                        fig, ax = plt.subplots(figsize=(8.27, 11.69))
                        ax.axis('off')
                        ax.text(0.5, 0.5,
                                f"Errore nella generazione della pagina '{name}':\n{e}",
                                transform=ax.transAxes, ha='center', va='center',
                                fontsize=12, color='red',
                                bbox=dict(boxstyle='round', facecolor='lightyellow'))
                        pdf.savefig(fig)
                        plt.close(fig)
                    except Exception:
                        pass

        logger.info(f"PDF report saved: {output_path}")

    def _add_title_page(self, pdf):
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
        ax.axis('off')

        title = self.config.get('title', 'Report Analisi DIC')
        ax.text(0.5, 0.7, title,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=24, fontweight='bold')

        info_text = []
        if self.config.get('site'):
            info_text.append(f"Sito: {self.config['site']}")
        if self.config.get('analyst'):
            info_text.append(f"Analista: {self.config['analyst']}")
        info_text.append(f"Data: {self.config.get('date', datetime.now().strftime('%Y-%m-%d'))}")
        info_text.append(f"Metodo: {self.result.parameters.method.value}")
        info_text.append(f"Tempo calcolo: {self.result.computation_time_s:.1f} s")

        ax.text(0.5, 0.45, '\n'.join(info_text),
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, linespacing=1.8)

        if self.config.get('notes'):
            ax.text(0.5, 0.15, f"Note: {self.config['notes']}",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, style='italic',
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))

        ax.text(0.5, 0.02, "DIC Landslide Monitor v1.0",
                transform=ax.transAxes, ha='center', fontsize=8, color='gray')

        pdf.savefig(fig)
        plt.close(fig)

    def _add_overview_page(self, pdf):
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle("Panoramica Immagini", fontsize=16, fontweight='bold')

        if self.base_image is not None:
            if self.base_image.ndim == 2:
                axes[0].imshow(self.base_image, cmap='gray')
            else:
                axes[0].imshow(self.base_image)
            axes[0].set_title("Immagine di Riferimento")
        axes[0].axis('off')

        # GPS info
        info_text = "Informazioni Geolocalizzazione:\n\n"
        if self.gps_info:
            for key, val in self.gps_info.items():
                info_text += f"{key}: {val}\n"
        else:
            info_text += "Nessun dato GPS disponibile\n"

        if self.gsd:
            info_text += f"\nGSD: {self.gsd:.4f} m/pixel"
            info_text += f"\n     ({self.gsd * 100:.2f} cm/pixel)"

        axes[1].text(0.1, 0.5, info_text, transform=axes[1].transAxes,
                     fontsize=11, verticalalignment='center',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1].set_title("Metadati")
        axes[1].axis('off')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _add_parameters_page(self, pdf):
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        ax.set_title("Parametri Analisi DIC", fontsize=16, fontweight='bold', pad=20)

        params = self.result.parameters
        text = (
            f"Metodo:              {params.method.value}\n"
            f"Dimensione Subset:   {params.subset_size} pixel\n"
            f"Passo Griglia:       {params.step_size} pixel\n"
            f"Soglia Correlazione: {params.correlation_threshold}\n"
            f"Sub-pixel:           {params.subpixel_method.value}\n"
        )

        if params.method.value == 'template_ncc':
            text += (
                f"\nRaggio Ricerca X:    {params.search_radius_x} pixel\n"
                f"Raggio Ricerca Y:    {params.search_radius_y} pixel\n"
            )
        elif params.method.value == 'optical_flow':
            text += (
                f"\nScala Piramide:      {params.of_pyr_scale}\n"
                f"Livelli Piramide:    {params.of_levels}\n"
                f"Finestra:            {params.of_winsize}\n"
                f"Iterazioni:          {params.of_iterations}\n"
            )

        if self.pipeline_info:
            text += f"\n\nFiltri Preprocessing:\n{self.pipeline_info}"

        shape = self.result.ref_shape
        if shape and shape[0] > 0:
            text += f"\n\nDimensioni Immagine: {shape[1]} x {shape[0]} pixel"
            ny, nx = self.result.u.shape
            text += f"\nPunti Griglia:       {nx} x {ny} = {nx * ny:,}"

        ax.text(0.1, 0.85, text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                fontfamily='monospace')

        pdf.savefig(fig)
        plt.close(fig)

    def _add_displacement_pages(self, pdf):
        """Add displacement magnitude, U, and V maps."""
        mag = compute_magnitude(self.result.u, self.result.v)

        for field, title in [
            (mag, "Magnitudine Spostamento"),
            (self.result.u, "Spostamento U (orizzontale)"),
            (self.result.v, "Spostamento V (verticale)"),
        ]:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))

            if self.base_image is not None:
                if self.base_image.ndim == 2:
                    ax.imshow(self.base_image, cmap='gray', alpha=0.5)
                else:
                    ax.imshow(self.base_image, alpha=0.5)

            # Plot field as scatter on grid points
            valid = ~np.isnan(field)
            if np.any(valid):
                sc = ax.scatter(
                    self.result.grid_x[valid], self.result.grid_y[valid],
                    c=field[valid], cmap='jet', s=2, alpha=0.7)
                plt.colorbar(sc, ax=ax, label='pixel',
                             shrink=0.6, aspect=30)

            unit_label = "pixel"
            if self.gsd:
                unit_label = f"pixel (1px = {self.gsd * 1000:.1f} mm)"

            ax.set_title(f"{title} ({unit_label})", fontsize=14, fontweight='bold')
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
            ax.invert_yaxis()

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    def _add_vector_page(self, pdf):
        fig, ax = plt.subplots(figsize=(11.69, 8.27))

        if self.base_image is not None:
            if self.base_image.ndim == 2:
                ax.imshow(self.base_image, cmap='gray')
            else:
                ax.imshow(self.base_image)

        draw_vector_field(ax, self.result.u, self.result.v,
                          self.result.grid_x, self.result.grid_y,
                          step=3, scale=3.0, color='red', width=0.003)

        ax.set_title("Campo Vettoriale Spostamenti", fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _add_correlation_page(self, pdf):
        fig, ax = plt.subplots(figsize=(11.69, 8.27))

        valid = ~np.isnan(self.result.correlation_quality)
        if np.any(valid):
            sc = ax.scatter(
                self.result.grid_x[valid], self.result.grid_y[valid],
                c=self.result.correlation_quality[valid],
                cmap='RdYlGn', s=2, vmin=0, vmax=1)
            plt.colorbar(sc, ax=ax, label='NCC', shrink=0.6)

        ax.set_title("Mappa Qualita Correlazione", fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _add_strain_pages(self, pdf):
        for key, title in [
            ('E_xx', 'Strain E_xx (Green-Lagrangian)'),
            ('E_yy', 'Strain E_yy (Green-Lagrangian)'),
            ('E_xy', 'Strain E_xy (Green-Lagrangian)'),
            ('principal_1', 'Strain Principale 1'),
            ('max_shear', 'Taglio Massimo'),
            ('von_mises', 'Strain Equivalente Von Mises'),
        ]:
            if key not in self.strain_data:
                continue

            field = self.strain_data[key]
            fig, ax = plt.subplots(figsize=(11.69, 8.27))

            valid = ~np.isnan(field)
            if np.any(valid):
                sc = ax.scatter(
                    self.result.grid_x[valid], self.result.grid_y[valid],
                    c=field[valid], cmap='coolwarm', s=2)
                plt.colorbar(sc, ax=ax, label='strain', shrink=0.6)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # ------------------------------------------------------------------
    # Zone helpers
    # ------------------------------------------------------------------

    def _zone_to_pixel_bbox(self, zone, margin=None):
        """Convert a zone's grid-index bbox to image-pixel coordinates.

        Parameters
        ----------
        zone : dict with 'bbox' = (row0, col0, row1, col1) in grid indices.
        margin : int, pixel margin for context. Defaults to subset_size.

        Returns
        -------
        (x0, y0, x1, y1) in image pixel coordinates, clamped to image bounds.
        """
        row0, col0, row1, col1 = zone['bbox']
        grid_x = self.result.grid_x
        grid_y = self.result.grid_y
        ny, nx = grid_x.shape

        r0 = max(0, min(row0, ny - 1))
        r1 = max(0, min(row1 - 1, ny - 1))
        c0 = max(0, min(col0, nx - 1))
        c1 = max(0, min(col1 - 1, nx - 1))

        x0 = int(grid_x[r0, c0])
        y0 = int(grid_y[r0, c0])
        x1 = int(grid_x[r1, c1])
        y1 = int(grid_y[r1, c1])

        if margin is None:
            margin = getattr(self.result.parameters, 'subset_size', 30)

        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)

        # Clamp to image dimensions
        if self.base_image is not None:
            h_img, w_img = self.base_image.shape[:2]
            x1 = min(w_img, x1 + margin)
            y1 = min(h_img, y1 + margin)
        else:
            x1 += margin
            y1 += margin

        return x0, y0, x1, y1

    @staticmethod
    def _compute_diff_image(ref_crop, def_crop):
        """Compute JET-colormapped absolute difference between two image crops.

        Handles both RGB and grayscale inputs. Returns an RGB uint8 image.
        """
        r, d = ref_crop, def_crop

        # Match sizes
        if r.shape[:2] != d.shape[:2]:
            h = min(r.shape[0], d.shape[0])
            w = min(r.shape[1], d.shape[1])
            r, d = r[:h, :w], d[:h, :w]

        # Convert to grayscale if RGB
        if r.ndim == 3:
            r_gray = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
            d_gray = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
        else:
            r_gray, d_gray = r, d

        diff = cv2.absdiff(r_gray, d_gray)

        # Normalize contrast
        diff_float = diff.astype(np.float32)
        dmax = diff_float.max()
        if dmax > 0:
            diff_enhanced = (diff_float / dmax * 255).astype(np.uint8)
        else:
            diff_enhanced = diff.astype(np.uint8)

        # Apply JET colormap
        diff_color = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
        diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
        return diff_color

    # ------------------------------------------------------------------
    # Zone pages
    # ------------------------------------------------------------------

    def _add_zones_page(self, pdf):
        """Synoptic overview: all validated zones on the full reference image."""
        # Filter to validated zones (fallback to all if none validated)
        display_zones = [z for z in self.active_zones
                         if z.get('status') == 'validated']
        if not display_zones:
            display_zones = self.active_zones

        # --- Page 1: Full-width synoptic map ---
        fig, ax = plt.subplots(figsize=(11.69, 8.27))

        # Base image
        if self.base_image is not None:
            if self.base_image.ndim == 2:
                ax.imshow(self.base_image, cmap='gray')
            else:
                ax.imshow(self.base_image)

        # Displacement overlay (semi-transparent)
        mag = compute_magnitude(self.result.u, self.result.v)
        valid = ~np.isnan(mag)
        if np.any(valid):
            ax.scatter(
                self.result.grid_x[valid], self.result.grid_y[valid],
                c=mag[valid], cmap='jet', s=1, alpha=0.3)

        # Draw zone boxes with labels
        n_zones = max(len(display_zones), 1)
        colors = plt.cm.Set1(np.linspace(0, 1, n_zones))
        for i, zone in enumerate(display_zones):
            x0, y0, x1, y1 = self._zone_to_pixel_bbox(zone, margin=0)
            color = colors[i % len(colors)]

            # Rectangle with semi-transparent fill
            rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2.5, edgecolor=color,
                facecolor=(*color[:3], 0.08))
            ax.add_patch(rect)

            # Label with zone ID and displacement
            label = f"Z{zone['id']}"
            if self.gsd and self.gsd > 0:
                disp_str = f"{zone['max_displacement'] * self.gsd * 1000:.1f} mm"
            else:
                disp_str = f"{zone['max_displacement']:.2f} px"

            ax.text(x0 + 3, y0 - 8, f"{label}: {disp_str}",
                    color='white', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor=color, alpha=0.85, edgecolor='none'))

        n_val = sum(1 for z in display_zones
                    if z.get('status') == 'validated')
        ax.set_title(
            f"Tavola Sinottica Zone Attive ({n_val} validate su "
            f"{len(self.active_zones)} totali)",
            fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Zone summary table ---
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.set_title("Riepilogo Zone Attive", fontsize=16,
                      fontweight='bold', pad=20)

        header = ["ID", "Stato", "Area\n(punti)", "Max\n(px)", "Media\n(px)"]
        if self.gsd:
            header.extend(["Max\n(mm)", "Media\n(mm)"])

        rows = []
        for zone in display_zones:
            status_label = {'validated': 'Validata', 'rejected': 'Rigettata',
                            'pending': 'Pendente'}.get(
                zone.get('status', 'pending'), zone.get('status', ''))
            row = [
                str(zone['id']),
                status_label,
                str(zone['area_points']),
                f"{zone['max_displacement']:.2f}",
                f"{zone['mean_displacement']:.2f}",
            ]
            if self.gsd:
                row.extend([
                    f"{zone['max_displacement'] * self.gsd * 1000:.1f}",
                    f"{zone['mean_displacement'] * self.gsd * 1000:.1f}",
                ])
            rows.append(row)

        if rows:
            table = ax.table(
                cellText=rows, colLabels=header,
                loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.8)
            # Style header row
            for j in range(len(header)):
                table[0, j].set_facecolor('#4CAF50')
                table[0, j].set_text_props(color='white', fontweight='bold')

        pdf.savefig(fig)
        plt.close(fig)

    def _add_zone_detail_pages(self, pdf):
        """Add one page per validated zone with ref/def/diff crop images."""
        if self.deformed_image is None and self.deformed_image_rgb is None:
            logger.warning("No deformed image available; skipping zone detail pages")
            return

        validated_zones = [z for z in self.active_zones
                           if z.get('status') == 'validated']
        if not validated_zones:
            return

        for zone in validated_zones:
            x0, y0, x1, y1 = self._zone_to_pixel_bbox(zone)
            if x1 <= x0 or y1 <= y0:
                continue  # degenerate bbox

            # Extract reference crop (prefer RGB base_image)
            if self.base_image is None:
                continue
            h_img, w_img = self.base_image.shape[:2]
            ref_crop = self.base_image[
                max(0, y0):min(h_img, y1),
                max(0, x0):min(w_img, x1)].copy()

            # Extract deformed crop (prefer RGB)
            def_src = (self.deformed_image_rgb
                       if self.deformed_image_rgb is not None
                       else self.deformed_image)
            if def_src is None:
                continue
            dh, dw = def_src.shape[:2]
            def_crop = def_src[
                max(0, y0):min(dh, y1),
                max(0, x0):min(dw, x1)].copy()

            # Compute difference image
            diff_crop = self._compute_diff_image(ref_crop, def_crop)

            # Build page: 1 row x 3 columns, landscape A4
            fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
            fig.suptitle(
                f"Zona {zone['id']} — Dettaglio Ispezione",
                fontsize=14, fontweight='bold', y=0.98)

            # Zone info subtitle
            info_parts = [
                f"Area: {zone['area_points']} punti",
                f"Max: {zone['max_displacement']:.2f} px",
                f"Media: {zone['mean_displacement']:.2f} px",
            ]
            if self.gsd and self.gsd > 0:
                max_mm = zone['max_displacement'] * self.gsd * 1000
                mean_mm = zone['mean_displacement'] * self.gsd * 1000
                info_parts.append(f"({max_mm:.1f} / {mean_mm:.1f} mm)")
            fig.text(0.5, 0.93, "  |  ".join(info_parts),
                     ha='center', fontsize=10, color='gray')

            # Plot the three images
            titles = ["Riferimento", "Deformata", "Differenza (JET)"]
            images = [ref_crop, def_crop, diff_crop]

            for ax, img, title in zip(axes, images, titles):
                if img.ndim == 2:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
                ax.set_title(title, fontsize=11)
                ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.91])
            pdf.savefig(fig)
            plt.close(fig)

    def _add_statistics_page(self, pdf):
        mag = compute_magnitude(self.result.u, self.result.v)
        stats = DisplacementStatistics.compute_statistics(
            self.result.u, self.result.v, mag,
            self.result.correlation_quality, self.gsd)

        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig.suptitle("Riepilogo Statistico", fontsize=16, fontweight='bold')

        # Histogram
        ax = axes[0, 0]
        valid_mag = mag[~np.isnan(mag)]
        if len(valid_mag) > 0:
            # Handle edge case: all values identical (histogram needs range)
            if np.ptp(valid_mag) < 1e-12:
                ax.hist(valid_mag, bins=1, color='steelblue', edgecolor='white', alpha=0.8)
            else:
                ax.hist(valid_mag, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_xlabel("Spostamento (pixel)")
        ax.set_ylabel("Frequenza")
        ax.set_title("Distribuzione Spostamenti")
        ax.axvline(np.nanmean(valid_mag) if len(valid_mag) > 0 else 0,
                    color='red', linestyle='--', label='Media')
        ax.legend()

        # Rose diagram
        ax = axes[0, 1]
        rose_data = DisplacementStatistics.generate_rose_diagram_data(
            self.result.u, self.result.v)
        if rose_data['angles']:
            angles_rad = np.radians(rose_data['angles'])
            counts = rose_data['counts']
            width = 2 * np.pi / len(counts)
            ax_polar = fig.add_subplot(2, 2, 2, projection='polar')
            axes[0, 1].remove()
            ax_polar.bar(angles_rad, counts, width=width,
                         color='coral', edgecolor='white', alpha=0.8)
            ax_polar.set_title("Direzione Spostamenti", pad=15)

        # Quality histogram
        ax = axes[1, 0]
        valid_q = self.result.correlation_quality[~np.isnan(self.result.correlation_quality)]
        if len(valid_q) > 0:
            if np.ptp(valid_q) < 1e-12:
                ax.hist(valid_q, bins=1, color='seagreen', edgecolor='white', alpha=0.8)
            else:
                ax.hist(valid_q, bins=50, color='seagreen', edgecolor='white', alpha=0.8)
        ax.set_xlabel("Coefficiente Correlazione")
        ax.set_ylabel("Frequenza")
        ax.set_title("Distribuzione Qualita Correlazione")

        # Stats table
        ax = axes[1, 1]
        ax.axis('off')
        table_data = [
            ["Punti validi", f"{stats.get('n_valid_points', 0):,}"],
            ["Copertura", f"{stats.get('coverage_percent', 0):.1f}%"],
            ["Spost. medio", f"{stats.get('mean_displacement_px', 0):.3f} px"],
            ["Spost. max", f"{stats.get('max_displacement_px', 0):.3f} px"],
            ["Dev. std", f"{stats.get('std_displacement_px', 0):.3f} px"],
            ["P95", f"{stats.get('p95_displacement_px', 0):.3f} px"],
            ["Qualita media", f"{stats.get('mean_quality', 0):.4f}"],
        ]
        if self.gsd:
            table_data.append(["Spost. medio", f"{stats.get('mean_displacement_mm', 0):.1f} mm"])
            table_data.append(["Spost. max", f"{stats.get('max_displacement_mm', 0):.1f} mm"])

        table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        ax.set_title("Statistiche", fontsize=12, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    # CSV Export
    # ------------------------------------------------------------------

    def export_csv(self, output_path: str):
        """Export displacement data as CSV."""
        gx = self.result.grid_x
        gy = self.result.grid_y
        u = self.result.u
        v = self.result.v
        mag = compute_magnitude(u, v)
        q = self.result.correlation_quality

        ny, nx = u.shape

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            header = ['grid_row', 'grid_col', 'pixel_x', 'pixel_y',
                      'u_px', 'v_px', 'magnitude_px', 'correlation']
            if self.gsd:
                header.extend(['u_m', 'v_m', 'magnitude_m',
                                'u_mm', 'v_mm', 'magnitude_mm'])
            writer.writerow(header)

            for iy in range(ny):
                for ix in range(nx):
                    if np.isnan(u[iy, ix]):
                        continue
                    row = [iy, ix,
                           int(gx[iy, ix]), int(gy[iy, ix]),
                           f"{u[iy, ix]:.4f}", f"{v[iy, ix]:.4f}",
                           f"{mag[iy, ix]:.4f}",
                           f"{q[iy, ix]:.6f}" if not np.isnan(q[iy, ix]) else ""]
                    if self.gsd:
                        row.extend([
                            f"{u[iy, ix] * self.gsd:.6f}",
                            f"{v[iy, ix] * self.gsd:.6f}",
                            f"{mag[iy, ix] * self.gsd:.6f}",
                            f"{u[iy, ix] * self.gsd * 1000:.2f}",
                            f"{v[iy, ix] * self.gsd * 1000:.2f}",
                            f"{mag[iy, ix] * self.gsd * 1000:.2f}",
                        ])
                    writer.writerow(row)

        logger.info(f"CSV exported: {output_path}")

    # ------------------------------------------------------------------
    # Image Export
    # ------------------------------------------------------------------

    def export_images(self, output_dir: str):
        """Export individual PNG images of all visualization maps."""
        mag = compute_magnitude(self.result.u, self.result.v)

        for field, name in [
            (mag, "displacement_magnitude"),
            (self.result.u, "displacement_u"),
            (self.result.v, "displacement_v"),
            (self.result.correlation_quality, "correlation_quality"),
        ]:
            fig = None
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                if self.base_image is not None:
                    if self.base_image.ndim == 2:
                        ax.imshow(self.base_image, cmap='gray', alpha=0.5)
                    else:
                        ax.imshow(self.base_image, alpha=0.5)

                valid = ~np.isnan(field)
                if np.any(valid):
                    sc = ax.scatter(
                        self.result.grid_x[valid], self.result.grid_y[valid],
                        c=field[valid], cmap='jet', s=2)
                    plt.colorbar(sc, ax=ax, shrink=0.6)

                ax.set_title(name.replace('_', ' ').title())
                ax.invert_yaxis()
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150)
            except Exception as e:
                logger.error(f"Error exporting image '{name}': {e}")
            finally:
                if fig is not None:
                    plt.close(fig)

        # Vector field
        fig, ax = plt.subplots(figsize=(12, 8))
        if self.base_image is not None:
            if self.base_image.ndim == 2:
                ax.imshow(self.base_image, cmap='gray')
            else:
                ax.imshow(self.base_image)
        draw_vector_field(ax, self.result.u, self.result.v,
                          self.result.grid_x, self.result.grid_y,
                          step=3, scale=3.0, color='red')
        ax.set_title("Vector Field")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "vector_field.png"), dpi=150)
        plt.close(fig)

        # Strain maps
        if self.strain_data:
            for key in ['E_xx', 'E_yy', 'E_xy', 'principal_1', 'max_shear', 'von_mises']:
                if key not in self.strain_data:
                    continue
                field = self.strain_data[key]
                fig, ax = plt.subplots(figsize=(12, 8))
                valid = ~np.isnan(field)
                if np.any(valid):
                    sc = ax.scatter(
                        self.result.grid_x[valid], self.result.grid_y[valid],
                        c=field[valid], cmap='coolwarm', s=2)
                    plt.colorbar(sc, ax=ax, shrink=0.6)
                ax.set_title(f"Strain {key}")
                ax.invert_yaxis()
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"strain_{key}.png"), dpi=150)
                plt.close(fig)

        logger.info(f"Images exported to: {output_dir}")

    # ------------------------------------------------------------------
    # GeoTIFF Export
    # ------------------------------------------------------------------

    def export_geotiff(self, output_dir: str):
        """Export georeferenced displacement and strain rasters."""
        try:
            from dic_app.core.geo_utils import GeoReferencer
            from dic_app.io.image_loader import GPSData
        except ImportError:
            logger.warning("Cannot export GeoTIFF: missing dependencies")
            return

        if not self.gps_info or not self.gsd:
            logger.warning("Cannot export GeoTIFF: no GPS data or GSD")
            return

        georef = GeoReferencer()
        gps = GPSData(
            latitude=self.gps_info.get('latitude'),
            longitude=self.gps_info.get('longitude'),
            altitude=self.gps_info.get('altitude')
        )

        if gps.latitude is None or gps.longitude is None:
            logger.warning("Cannot export GeoTIFF: incomplete GPS data")
            return

        # Validate GPS coordinates are within plausible range
        if not (-90 <= gps.latitude <= 90) or not (-180 <= gps.longitude <= 180):
            logger.warning(
                f"GPS coordinates out of range: lat={gps.latitude}, "
                f"lon={gps.longitude}. Skipping GeoTIFF export.")
            return

        image_shape = self.result.ref_shape
        mag = compute_magnitude(self.result.u, self.result.v)

        for field, name in [
            (mag, "displacement_magnitude"),
            (self.result.u, "displacement_u"),
            (self.result.v, "displacement_v"),
        ]:
            filepath = os.path.join(output_dir, f"{name}.tif")
            georef.create_geotiff(field, filepath, gps, self.gsd, image_shape)

        if self.strain_data:
            for key in ['E_xx', 'E_yy', 'E_xy', 'principal_1', 'max_shear']:
                if key in self.strain_data:
                    filepath = os.path.join(output_dir, f"strain_{key}.tif")
                    georef.create_geotiff(
                        self.strain_data[key], filepath, gps, self.gsd, image_shape)

        logger.info(f"GeoTIFF files exported to: {output_dir}")
