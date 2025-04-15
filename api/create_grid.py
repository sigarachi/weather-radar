import math
import netCDF4 as nc
import numpy as np
from scipy.spatial import cKDTree

# Constants
RADIUS_LIMIT = 250  # km
GRID_SIZE = 504
KM_PER_DEG_LAT = 111.32  # 1 degree of latitude is approximately 111.32 km


def calculate_grid(center_lat, center_lon, km_per_pixel=1, grid_size=GRID_SIZE):
    """
    Calculate a grid of coordinates centered at (center_lat, center_lon).
    The grid is limited to 250km radius from the center point.
    """
    # Calculate degrees per kilometer for longitude (depends on latitude)
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))
    deg_per_km_lon = 1 / km_per_deg_lon
    deg_per_km_lat = 1 / KM_PER_DEG_LAT

    # Create coordinate arrays
    lats = np.zeros((grid_size, grid_size))
    lons = np.zeros((grid_size, grid_size))
    mask = np.zeros((grid_size, grid_size), dtype=bool)

    half_size = grid_size // 2

    # Calculate coordinates for each point
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate distance from center in km
            delta_i = (i - half_size) * km_per_pixel
            delta_j = (j - half_size) * km_per_pixel

            # Calculate new coordinates
            lat = center_lat + delta_i * deg_per_km_lat
            lon = center_lon + delta_j * deg_per_km_lon

            # Check if point is within radius
            distance = math.sqrt(delta_i**2 + delta_j**2)
            if distance <= RADIUS_LIMIT:
                lats[i, j] = lat
                lons[i, j] = lon
                mask[i, j] = True
            else:
                lats[i, j] = np.nan
                lons[i, j] = np.nan
                mask[i, j] = False

    valid_points = mask.ravel()
    coords = np.column_stack([lons.ravel(), lats.ravel()])
    valid_coords = coords[valid_points == 1]
    kdtree = cKDTree(valid_coords)

    return lats, lons, mask, kdtree, valid_points


def process_netcdf(input_file, output_file, km_per_pixel=1):
    """
    Process NetCDF file to create a grid of coordinates.
    """
    # Read input NetCDF file
    with nc.Dataset(input_file, 'r') as src:
        center_lat = src.lat_station
        center_lon = src.lon_station

        lats, lons, mask, kdtree, valid_points = calculate_grid(
            center_lat, center_lon, km_per_pixel)

        # Сохраняем плоские индексы для быстрого доступа
        valid_indices = np.where(valid_points)[0]

        # Create output NetCDF file
        with nc.Dataset(output_file, 'w', format='NETCDF4') as dst:
            # Create dimensions
            dst.createDimension('x', GRID_SIZE)
            dst.createDimension('y', GRID_SIZE)

            # Create variables
            lat_var = dst.createVariable('latitude', 'f4', ('y', 'x'))
            lon_var = dst.createVariable('longitude', 'f4', ('y', 'x'))
            mask_var = dst.createVariable('valid_mask', 'i1', ('y', 'x'))

            # Write data
            lat_var[:] = lats
            lon_var[:] = lons
            mask_var[:] = mask.astype(np.int8)

            # Add attributes
            lat_var.units = 'degrees_north'
            lon_var.units = 'degrees_east'
            mask_var.units = 'boolean'
            mask_var.description = '1 if point is within 250km radius, 0 otherwise'

            # Add global attributes
            dst.center_latitude = center_lat
            dst.center_longitude = center_lon
            dst.radius_limit = RADIUS_LIMIT
            dst.grid_size = GRID_SIZE
            dst.km_per_pixel = km_per_pixel
            dst.createVariable('valid_indices', 'i4', ('valid_points',))[
                :] = valid_indices
            dst.kdtree_data = kdtree.data.tobytes()  # Сохраняем данные дерева
            dst.kdtree_shape = kdtree.data.shape


if __name__ == "__main__":
    input_file = "./periods/2023/07/07/00/00/RUDL20230707_000000_1.nc"
    output_file = "grid_coordinates.nc"
    process_netcdf(input_file, output_file)
