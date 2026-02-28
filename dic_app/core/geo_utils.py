"""Geolocation utilities: GPS handling, CRS transforms, GeoTIFF export."""

import math
import numpy as np
from typing import Optional, Tuple
from dic_app.io.image_loader import GPSData
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)


class GeoReferencer:
    """Handle coordinate transformations and georeferencing of DIC results."""

    def __init__(self):
        self.source_crs = None
        self.target_crs = None
        self.transformer = None

    def set_crs(self, source_epsg: int = 4326, target_epsg: int = 32632):
        """Initialize coordinate transformer between CRS.

        Parameters
        ----------
        source_epsg : int, source EPSG code (default 4326 = WGS84)
        target_epsg : int, target EPSG code (default 32632 = UTM 32N)
        """
        try:
            from pyproj import CRS, Transformer
            self.source_crs = CRS.from_epsg(source_epsg)
            self.target_crs = CRS.from_epsg(target_epsg)
            self.transformer = Transformer.from_crs(
                self.source_crs, self.target_crs, always_xy=True)
            logger.info(f"CRS transformer set: EPSG:{source_epsg} -> EPSG:{target_epsg}")
        except ImportError:
            logger.warning("pyproj not installed; coordinate transformations unavailable")

    def latlon_to_projected(self, lon: float, lat: float) -> Tuple[float, float]:
        """Transform lat/lon to projected coordinates (e.g., UTM).

        Returns
        -------
        (easting, northing) in meters
        """
        if self.transformer is None:
            raise RuntimeError("CRS transformer not initialized. Call set_crs() first.")
        return self.transformer.transform(lon, lat)

    def pixel_to_geo(self, pixel_x: float, pixel_y: float,
                     gps_origin: GPSData, gsd: float,
                     image_shape: Tuple[int, int],
                     image_bearing: float = 0.0) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates.

        Parameters
        ----------
        pixel_x, pixel_y : float, pixel coordinates
        gps_origin : GPSData, GPS position of image center
        gsd : float, ground sampling distance (meters/pixel)
        image_shape : (height, width) of the image
        image_bearing : float, degrees clockwise from north

        Returns
        -------
        (longitude, latitude) in decimal degrees
        """
        if gps_origin.latitude is None or gps_origin.longitude is None:
            raise ValueError("GPS origin must have valid lat/lon")

        h, w = image_shape
        cx, cy = w / 2.0, h / 2.0

        # Offset from image center in pixels
        dx_px = pixel_x - cx
        dy_px = -(pixel_y - cy)  # flip Y (image Y is downward)

        # Convert to meters
        dx_m = dx_px * gsd
        dy_m = dy_px * gsd

        # Rotate by bearing
        bearing_rad = math.radians(image_bearing)
        east_m = dx_m * math.cos(bearing_rad) - dy_m * math.sin(bearing_rad)
        north_m = dx_m * math.sin(bearing_rad) + dy_m * math.cos(bearing_rad)

        # Approximate conversion to lat/lon offset
        lat_offset = north_m / 111320.0
        lon_offset = east_m / (111320.0 * math.cos(math.radians(gps_origin.latitude)))

        lat = gps_origin.latitude + lat_offset
        lon = gps_origin.longitude + lon_offset

        return lon, lat

    @staticmethod
    def displacement_to_meters(u_px, v_px, gsd: float):
        """Convert pixel displacement fields to meters.

        Parameters
        ----------
        u_px, v_px : np.ndarray, displacement in pixels
        gsd : float, meters per pixel

        Returns
        -------
        (u_m, v_m) displacement in meters
        """
        return u_px * gsd, v_px * gsd

    @staticmethod
    def geodesic_distance(lat1, lon1, lat2, lon2):
        """Compute geodesic distance between two GPS points.

        Uses Haversine formula.

        Returns
        -------
        float, distance in meters
        """
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)

        a = (math.sin(dphi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def create_geotiff(self, data: np.ndarray, filepath: str,
                       gps_origin: GPSData, gsd: float,
                       image_shape: Tuple[int, int],
                       image_bearing: float = 0.0,
                       nodata: float = -9999.0,
                       epsg: int = 4326):
        """Write a georeferenced GeoTIFF from displacement/strain data.

        Parameters
        ----------
        data : np.ndarray (NY, NX) float, the data field to export
        filepath : str, output path
        gps_origin : GPSData, GPS position of image center
        gsd : float, meters/pixel
        image_shape : (H, W) of original image
        image_bearing : degrees clockwise from north
        nodata : float, value for no-data pixels
        epsg : int, EPSG code for output CRS
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS
        except ImportError:
            logger.warning("rasterio not installed; GeoTIFF export unavailable")
            return

        if gps_origin.latitude is None or gps_origin.longitude is None:
            logger.warning("No GPS data available for GeoTIFF export")
            return

        h_img, w_img = image_shape
        ny, nx = data.shape

        # Compute geographic bounds
        corners = [
            (0, 0), (w_img, 0), (w_img, h_img), (0, h_img)
        ]
        lons, lats = [], []
        for px, py in corners:
            lon, lat = self.pixel_to_geo(
                px, py, gps_origin, gsd, image_shape, image_bearing)
            lons.append(lon)
            lats.append(lat)

        west = min(lons)
        east = max(lons)
        south = min(lats)
        north = max(lats)

        transform = from_bounds(west, south, east, north, nx, ny)

        # Replace NaN with nodata
        data_out = np.where(np.isnan(data), nodata, data).astype(np.float32)

        crs = CRS.from_epsg(epsg)

        with rasterio.open(
            filepath, 'w', driver='GTiff',
            height=ny, width=nx, count=1,
            dtype='float32', crs=crs,
            transform=transform, nodata=nodata,
        ) as dst:
            dst.write(data_out, 1)

        logger.info(f"GeoTIFF saved: {filepath} ({nx}x{ny})")

    @staticmethod
    def compute_gsd_from_two_images(gps1: GPSData, gps2: GPSData,
                                     pixel_shift: Tuple[float, float]) -> Optional[float]:
        """Estimate GSD from two overlapping images with known GPS and pixel shift.

        Parameters
        ----------
        gps1, gps2 : GPSData, GPS positions of image centers
        pixel_shift : (dx_px, dy_px), pixel displacement between images

        Returns
        -------
        float, estimated GSD in meters/pixel, or None if insufficient data
        """
        if (gps1.latitude is None or gps1.longitude is None or
                gps2.latitude is None or gps2.longitude is None):
            return None

        dist_m = GeoReferencer.geodesic_distance(
            gps1.latitude, gps1.longitude,
            gps2.latitude, gps2.longitude
        )

        pixel_dist = math.sqrt(pixel_shift[0] ** 2 + pixel_shift[1] ** 2)
        if pixel_dist < 1.0:
            return None

        return dist_m / pixel_dist

    @staticmethod
    def geotiff_transform_from_gps(gps: GPSData, gsd: float,
                                    image_shape: Tuple[int, int]):
        """Build rasterio Affine transform from GPS origin + GSD.

        Assumes north-up orientation.
        """
        try:
            from rasterio.transform import from_bounds
        except ImportError:
            return None

        h, w = image_shape
        half_w_m = (w / 2.0) * gsd
        half_h_m = (h / 2.0) * gsd

        # Approximate degree extent
        lat_extent = half_h_m / 111320.0
        lon_extent = half_w_m / (111320.0 * math.cos(math.radians(gps.latitude)))

        west = gps.longitude - lon_extent
        east = gps.longitude + lon_extent
        south = gps.latitude - lat_extent
        north = gps.latitude + lat_extent

        return from_bounds(west, south, east, north, w, h)
