"""Image loading with EXIF GPS metadata extraction."""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image, ExifTags
from dic_app.utils.helpers import dms_to_decimal, setup_logger

logger = setup_logger(__name__)

# Pillow is used only to read EXIF metadata of local files chosen by the
# user.  Its "decompression bomb" guard refuses images above ~179 MP (and
# warns above ~89 MP), which blocks metadata of large orthophotos.
Image.MAX_IMAGE_PIXELS = None


@dataclass
class GPSData:
    """GPS coordinates extracted from image EXIF data."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class ImageData:
    """Container for a loaded image with metadata.

    The RGB image is loaded lazily from disk on first access via
    the ``image_rgb`` property, to avoid keeping ~72 MB per image
    in memory when only the grayscale version is needed (which is
    the case for all DIC computation).
    """
    filepath: str
    image_gray: np.ndarray = field(repr=False)
    _image_rgb: Optional[np.ndarray] = field(default=None, repr=False)
    width: int = 0
    height: int = 0
    gps: Optional[GPSData] = None
    camera_model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    capture_time: Optional[str] = None
    gsd: Optional[float] = None
    # Set when the file is a preliminary crop (content of its .precrop.json)
    precrop: Optional[dict] = field(default=None, repr=False)

    @property
    def original_width(self) -> int:
        """Width of the ORIGINAL photo (before a preliminary crop).

        GSD formulas relate the sensor width to the full photo width, so
        they must not use the width of a crop.
        """
        if self.precrop and self.precrop.get('source_original_shape'):
            return int(self.precrop['source_original_shape'][1])
        return self.width

    @property
    def image_rgb(self) -> Optional[np.ndarray]:
        """Lazy-load RGB image from disk on first access."""
        if self._image_rgb is None and self.filepath and os.path.exists(self.filepath):
            try:
                bgr = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
                if bgr is not None:
                    self._image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    logger.debug(f"Lazy-loaded RGB for {self.filename}")
            except Exception as e:
                logger.warning(f"Failed to lazy-load RGB for {self.filepath}: {e}")
        return self._image_rgb

    @image_rgb.setter
    def image_rgb(self, value):
        """Allow direct assignment (e.g. from old code or tests)."""
        self._image_rgb = value

    def release_rgb(self):
        """Explicitly free the RGB image to reclaim memory."""
        self._image_rgb = None

    @property
    def filename(self):
        return os.path.basename(self.filepath)

    @property
    def has_gps(self):
        return (self.gps is not None and
                self.gps.latitude is not None and
                self.gps.longitude is not None)


class ImageLoader:
    """Load images from disk with EXIF metadata extraction."""

    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    @staticmethod
    def load(filepath: str) -> ImageData:
        """Load an image file and extract all available metadata.

        Supports JPEG, PNG, TIFF, and GeoTIFF formats.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ImageLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")

        # Load grayscale only – RGB is lazy-loaded on demand to save memory
        image_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise IOError(f"Failed to load image: {filepath}")

        # Warn if image has very low contrast (may cause poor DIC results)
        img_std = float(image_gray.std())
        if img_std < 10:
            logger.warning(
                f"Image '{os.path.basename(filepath)}' has very low contrast "
                f"(std={img_std:.1f}). DIC results may be unreliable.")

        h, w = image_gray.shape

        # Extract EXIF metadata
        gps_data = ImageLoader._extract_exif_gps(filepath)
        camera_info = ImageLoader._extract_camera_info(filepath)

        gsd = ImageLoader._extract_geotiff_gsd(filepath)

        # Preliminary crops (PNG) lose EXIF: recover it from the sidecar
        from dic_app.core.precrop import read_sidecar
        precrop = read_sidecar(filepath)
        if precrop:
            src_meta = precrop.get('source_metadata') or {}
            g = src_meta.get('gps')
            if gps_data is None and g and g.get('latitude') is not None:
                gps_data = GPSData(latitude=g.get('latitude'),
                                   longitude=g.get('longitude'),
                                   altitude=g.get('altitude'),
                                   timestamp=g.get('timestamp'))
            for key in ('camera_model', 'focal_length', 'capture_time'):
                if camera_info.get(key) is None and src_meta.get(key) is not None:
                    camera_info[key] = src_meta[key]
            # a crop has the pixel size of the GUIDE grid it was resampled to
            if gsd is None and precrop.get('guide_gsd'):
                gsd = float(precrop['guide_gsd'])

        img_data = ImageData(
            filepath=filepath,
            image_gray=image_gray,
            width=w,
            height=h,
            gps=gps_data,
            camera_model=camera_info.get('camera_model'),
            focal_length_mm=camera_info.get('focal_length'),
            capture_time=camera_info.get('capture_time'),
            precrop=precrop,
            gsd=gsd,
        )

        logger.info(f"Loaded {img_data.filename} ({w}x{h})"
                     f" GPS={'yes' if img_data.has_gps else 'no'}"
                     f" [RGB lazy-load]")
        return img_data

    @staticmethod
    def _extract_exif_gps(filepath: str) -> Optional[GPSData]:
        """Parse EXIF GPSInfo tags from image file."""
        try:
            pil_img = Image.open(filepath)
            exif_data = pil_img._getexif()
            if exif_data is None:
                return None

            gps_info = {}
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == 'GPSInfo':
                    for gps_tag_id, gps_value in value.items():
                        gps_tag_name = ExifTags.GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_info[gps_tag_name] = gps_value

            if not gps_info:
                return None

            gps_data = GPSData()

            # Latitude
            if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                lat = gps_info['GPSLatitude']
                lat_ref = gps_info['GPSLatitudeRef']
                gps_data.latitude = dms_to_decimal(
                    float(lat[0]), float(lat[1]), float(lat[2]), lat_ref)

            # Longitude
            if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                lon = gps_info['GPSLongitude']
                lon_ref = gps_info['GPSLongitudeRef']
                gps_data.longitude = dms_to_decimal(
                    float(lon[0]), float(lon[1]), float(lon[2]), lon_ref)

            # Altitude
            if 'GPSAltitude' in gps_info:
                alt = gps_info['GPSAltitude']
                gps_data.altitude = float(alt)
                if gps_info.get('GPSAltitudeRef', 0) == 1:
                    gps_data.altitude = -gps_data.altitude

            # Timestamp
            if 'GPSTimeStamp' in gps_info:
                ts = gps_info['GPSTimeStamp']
                gps_data.timestamp = f"{int(ts[0]):02d}:{int(ts[1]):02d}:{float(ts[2]):.1f}"

            return gps_data

        except Exception as e:
            logger.debug(f"No GPS EXIF data in {filepath}: {e}")
            return None

    @staticmethod
    def _extract_geotiff_gsd(filepath: str) -> Optional[float]:
        """Pixel size from the GeoTIFF ModelPixelScale tag (33550), if any.

        Only for projected CRSs in metres (the usual case for orthophotos);
        values that look like degrees (< 1e-4) are ignored.
        """
        if os.path.splitext(filepath)[1].lower() not in ('.tif', '.tiff'):
            return None
        try:
            with Image.open(filepath) as im:
                scale = im.tag_v2.get(33550)
            if scale and len(scale) >= 2:
                sx, sy = float(scale[0]), float(scale[1])
                if sx > 1e-4 and abs(sx - sy) / sx < 0.01:
                    return sx
        except Exception as e:
            logger.debug(f"No GeoTIFF pixel scale in {filepath}: {e}")
        return None

    @staticmethod
    def _extract_camera_info(filepath: str) -> dict:
        """Extract camera model, focal length, and capture timestamp from EXIF."""
        info = {}
        try:
            pil_img = Image.open(filepath)
            exif_data = pil_img._getexif()
            if exif_data is None:
                return info

            tag_names = {v: k for k, v in ExifTags.TAGS.items()}

            # Camera model
            model_tag = tag_names.get('Model')
            if model_tag and model_tag in exif_data:
                info['camera_model'] = str(exif_data[model_tag])

            # Focal length
            fl_tag = tag_names.get('FocalLength')
            if fl_tag and fl_tag in exif_data:
                fl = exif_data[fl_tag]
                info['focal_length'] = float(fl)

            # Capture time
            dt_tag = tag_names.get('DateTimeOriginal')
            if dt_tag and dt_tag in exif_data:
                info['capture_time'] = str(exif_data[dt_tag])

        except Exception as e:
            logger.debug(f"Error reading EXIF from {filepath}: {e}")

        return info

    @staticmethod
    def compute_gsd(altitude_m: float, focal_length_mm: float,
                    sensor_width_mm: float, image_width_px: int) -> float:
        """Compute Ground Sampling Distance from flight parameters.

        GSD = (altitude * sensor_width) / (focal_length * image_width)

        Parameters
        ----------
        altitude_m : float, drone altitude above ground (meters)
        focal_length_mm : float, camera focal length (mm)
        sensor_width_mm : float, camera sensor width (mm)
        image_width_px : int, image width in pixels

        Returns
        -------
        float, GSD in meters/pixel
        """
        if focal_length_mm <= 0 or image_width_px <= 0:
            raise ValueError("Focal length and image width must be positive")
        if altitude_m <= 0:
            raise ValueError("Altitude must be positive")
        if sensor_width_mm <= 0:
            raise ValueError("Sensor width must be positive")
        gsd = (altitude_m * sensor_width_mm) / (focal_length_mm * image_width_px)
        if gsd < 0.001 or gsd > 100:
            logger.warning(
                f"Computed GSD={gsd:.6f} m/px is outside expected range "
                f"[0.001, 100]. Check input parameters: "
                f"alt={altitude_m}m, fl={focal_length_mm}mm, "
                f"sensor={sensor_width_mm}mm, width={image_width_px}px")
        return gsd

    @staticmethod
    def load_geotiff(filepath: str):
        """Load a GeoTIFF file and return array with georeferencing metadata.

        Returns
        -------
        tuple: (np.ndarray image, dict with 'transform', 'crs', 'bounds')
        """
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                data = src.read(1)  # First band
                meta = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'bounds': src.bounds,
                    'width': src.width,
                    'height': src.height,
                    'res': src.res,
                }
                return data, meta
        except ImportError:
            logger.warning("rasterio not installed; GeoTIFF georeferencing unavailable")
            # Fallback: load as regular image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            return img, {}
