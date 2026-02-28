"""ROI and mask management for DIC analysis regions."""

import logging
import json
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ROIType(Enum):
    POLYGON = "polygon"
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    FREEHAND = "freehand"


class ROIMode(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


@dataclass
class ROIDefinition:
    """Definition of a single Region of Interest."""
    roi_type: ROIType
    mode: ROIMode
    points: List[Tuple[int, int]]
    name: str = ""
    color: Tuple[int, int, int] = (0, 255, 0)

    def to_dict(self):
        return {
            'roi_type': self.roi_type.value,
            'mode': self.mode.value,
            'points': self.points,
            'name': self.name,
            'color': list(self.color),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            roi_type=ROIType(d['roi_type']),
            mode=ROIMode(d['mode']),
            points=[tuple(p) for p in d['points']],
            name=d.get('name', ''),
            color=tuple(d.get('color', (0, 255, 0))),
        )


class MaskManager:
    """Manages a collection of ROIs and produces a combined binary mask.

    Mask logic:
    - If any INCLUDE ROIs exist, start with all-black (0).
      INCLUDE ROIs paint white (255).
    - If only EXCLUDE ROIs exist, start with all-white (255).
    - EXCLUDE ROIs always paint black (0) on top.

    The final mask is uint8: 255 = analyze, 0 = skip.
    """

    def __init__(self, image_shape: Tuple[int, int]):
        """
        Parameters
        ----------
        image_shape : (height, width)
        """
        self.image_shape = image_shape
        self.rois: List[ROIDefinition] = []

    def add_roi(self, roi: ROIDefinition):
        """Add an ROI definition after validation."""
        if not roi.points or len(roi.points) < 2:
            logger.warning(f"ROI '{roi.name}' has too few points ({len(roi.points) if roi.points else 0}), skipping")
            return
        if roi.roi_type in (ROIType.POLYGON, ROIType.FREEHAND) and len(roi.points) < 3:
            logger.warning(f"Polygon/freehand ROI '{roi.name}' needs at least 3 points, skipping")
            return
        self.rois.append(roi)
        logger.debug(f"Added ROI '{roi.name}' ({roi.roi_type.value}, {roi.mode.value}, {len(roi.points)} points)")

    def remove_roi(self, index: int):
        """Remove ROI by index."""
        if 0 <= index < len(self.rois):
            self.rois.pop(index)

    def clear_all(self):
        """Remove all ROIs."""
        self.rois.clear()

    def update_roi(self, index: int, roi: ROIDefinition):
        """Replace ROI at index."""
        if 0 <= index < len(self.rois):
            self.rois[index] = roi

    def generate_mask(self) -> np.ndarray:
        """Generate combined binary mask from all ROIs.

        Returns
        -------
        np.ndarray (H, W) uint8, 255 = analyze, 0 = skip
        """
        if not self.rois:
            # No ROIs defined: analyze entire image
            return np.full(self.image_shape, 255, dtype=np.uint8)

        has_include = any(r.mode == ROIMode.INCLUDE for r in self.rois)

        if has_include:
            mask = np.zeros(self.image_shape, dtype=np.uint8)
        else:
            mask = np.full(self.image_shape, 255, dtype=np.uint8)

        # First pass: draw INCLUDE ROIs
        for roi in self.rois:
            if roi.mode == ROIMode.INCLUDE:
                self._rasterize_roi(roi, mask, 255)

        # Second pass: draw EXCLUDE ROIs
        for roi in self.rois:
            if roi.mode == ROIMode.EXCLUDE:
                self._rasterize_roi(roi, mask, 0)

        return mask

    def _rasterize_roi(self, roi: ROIDefinition, canvas: np.ndarray, value: int):
        """Draw a single ROI onto the canvas with the given fill value."""
        pts = np.array(roi.points, dtype=np.int32)

        if roi.roi_type == ROIType.POLYGON or roi.roi_type == ROIType.FREEHAND:
            if len(pts) >= 3:
                cv2.fillPoly(canvas, [pts], value)

        elif roi.roi_type == ROIType.RECTANGLE:
            if len(pts) >= 2:
                pt1 = (int(pts[0][0]), int(pts[0][1]))
                pt2 = (int(pts[1][0]), int(pts[1][1]))
                cv2.rectangle(canvas, pt1, pt2, value, -1)

        elif roi.roi_type == ROIType.ELLIPSE:
            if len(pts) >= 2:
                # pts[0] = center, pts[1] = point on the boundary defining axes
                center = (int(pts[0][0]), int(pts[0][1]))
                rx = abs(int(pts[1][0]) - center[0])
                ry = abs(int(pts[1][1]) - center[1])
                if rx > 0 and ry > 0:
                    cv2.ellipse(canvas, center, (rx, ry), 0, 0, 360, value, -1)

    def get_roi_overlay(self, alpha=0.3) -> np.ndarray:
        """Generate an RGBA overlay showing ROIs with color coding.

        INCLUDE ROIs: green
        EXCLUDE ROIs: red

        Returns
        -------
        np.ndarray (H, W, 4) uint8 RGBA
        """
        overlay = np.zeros((*self.image_shape, 4), dtype=np.uint8)

        for roi in self.rois:
            pts = np.array(roi.points, dtype=np.int32)
            if roi.mode == ROIMode.INCLUDE:
                color = (0, 200, 0, int(alpha * 255))
            else:
                color = (200, 0, 0, int(alpha * 255))

            temp = np.zeros((*self.image_shape, 4), dtype=np.uint8)

            if roi.roi_type in (ROIType.POLYGON, ROIType.FREEHAND):
                if len(pts) >= 3:
                    cv2.fillPoly(temp, [pts], color)
            elif roi.roi_type == ROIType.RECTANGLE:
                if len(pts) >= 2:
                    cv2.rectangle(temp, tuple(pts[0]), tuple(pts[1]), color, -1)
            elif roi.roi_type == ROIType.ELLIPSE:
                if len(pts) >= 2:
                    center = tuple(pts[0])
                    rx = abs(pts[1][0] - center[0])
                    ry = abs(pts[1][1] - center[1])
                    if rx > 0 and ry > 0:
                        cv2.ellipse(temp, center, (rx, ry), 0, 0, 360, color, -1)

            # Combine: later ROIs overwrite earlier ones
            mask_nonzero = temp[:, :, 3] > 0
            overlay[mask_nonzero] = temp[mask_nonzero]

            # Draw outline
            outline_color = (0, 255, 0, 255) if roi.mode == ROIMode.INCLUDE else (255, 0, 0, 255)
            if roi.roi_type in (ROIType.POLYGON, ROIType.FREEHAND):
                if len(pts) >= 3:
                    cv2.polylines(overlay, [pts], True, outline_color, 2)
            elif roi.roi_type == ROIType.RECTANGLE:
                if len(pts) >= 2:
                    cv2.rectangle(overlay, tuple(pts[0]), tuple(pts[1]), outline_color, 2)
            elif roi.roi_type == ROIType.ELLIPSE:
                if len(pts) >= 2:
                    center = tuple(pts[0])
                    rx = abs(pts[1][0] - center[0])
                    ry = abs(pts[1][1] - center[1])
                    if rx > 0 and ry > 0:
                        cv2.ellipse(overlay, center, (rx, ry), 0, 0, 360, outline_color, 2)

        return overlay

    def save_mask(self, filepath: str):
        """Save the generated mask as a PNG image."""
        mask = self.generate_mask()
        cv2.imwrite(filepath, mask)

    def load_mask(self, filepath: str):
        """Load an external mask image and create a single polygon-free ROI.

        The loaded mask is stored as an INCLUDE ROI covering the non-zero region.
        """
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {filepath}")

        # Resize if needed
        if mask.shape != self.image_shape:
            mask = cv2.resize(mask, (self.image_shape[1], self.image_shape[0]))

        # Threshold
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours and add as polygon ROIs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 10:
                # Simplify contour
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                roi = ROIDefinition(
                    roi_type=ROIType.POLYGON,
                    mode=ROIMode.INCLUDE,
                    points=points,
                    name=f"Imported_{i}",
                )
                self.add_roi(roi)

    def export_rois(self, filepath: str):
        """Export ROI definitions to JSON."""
        data = {
            'image_shape': list(self.image_shape),
            'rois': [roi.to_dict() for roi in self.rois],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_rois(self, filepath: str):
        """Import ROI definitions from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        for roi_data in data.get('rois', []):
            self.add_roi(ROIDefinition.from_dict(roi_data))

    def to_dict(self):
        """Serialize for project state."""
        return {
            'image_shape': list(self.image_shape),
            'rois': [roi.to_dict() for roi in self.rois],
        }

    @classmethod
    def from_dict(cls, d, image_shape=None):
        """Deserialize from project state."""
        shape = tuple(d.get('image_shape', image_shape or (0, 0)))
        manager = cls(shape)
        for roi_data in d.get('rois', []):
            manager.add_roi(ROIDefinition.from_dict(roi_data))
        return manager

    def __len__(self):
        return len(self.rois)

    def __repr__(self):
        inc = sum(1 for r in self.rois if r.mode == ROIMode.INCLUDE)
        exc = sum(1 for r in self.rois if r.mode == ROIMode.EXCLUDE)
        return f"MaskManager({inc} include, {exc} exclude, shape={self.image_shape})"
