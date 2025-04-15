from netCDF4 import Dataset
from environs import Env
from io import BytesIO
from datetime import datetime
import xarray as xr
import numpy as np
import os
import math
from math import radians, degrees, sin, cos, atan2, sqrt, asin, floor, ceil, tan, pi, log
import zipfile
from typing import Tuple, Optional

env = Env()
env.read_env()

TILE_SIZE = 256  # Размер тайла
GRID_RADIUS_KM = 250  # Радиус сетки данных
GRID_SIZE = 512  # Размер сетки (512x512 точек)

# 📌 Вычисляем размер ячейки в километрах
CELL_SIZE_KM = (GRID_RADIUS_KM * 2) / GRID_SIZE

# 📌 Коэффициенты для перевода градусов в километры
KM_PER_DEGREE_LAT = 110.574  # 1° широты ≈ 110.574 км
KM_PER_DEGREE_LON = lambda lat: 111.32 * math.cos(math.radians(lat))  # Долгота зависит от широты

def parse_folder_structure(base_path):
    time_periods = []

    for year in sorted(os.listdir(base_path)):
        year_path = os.path.join(base_path, year)
        if not os.path.isdir(year_path) or not year.isdigit():
            continue

        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            if not os.path.isdir(month_path) or not month.isdigit():
                continue

            for day in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day)
                if not os.path.isdir(day_path) or not day.isdigit():
                    continue

                for hour in sorted(os.listdir(day_path)):
                    hour_path = os.path.join(day_path, hour)
                    if not os.path.isdir(hour_path) or not hour.isdigit():
                        continue

                    for minute in sorted(os.listdir(hour_path)):
                        minute_path = os.path.join(hour_path, minute)
                        if not os.path.isdir(minute_path) or not minute.isdigit():
                            continue

                        try:
                            timestamp = datetime(int(year), int(
                                month), int(day), int(hour), int(minute))
                            time_periods.append(
                                (timestamp.isoformat(), minute_path))
                        except ValueError:
                            print(
                                f"Skipping invalid date: {year}-{month}-{day} {hour}:{minute}")

    return time_periods

def get_file_path(time_data, timestamp, all=False):
    for time_iso, folder_path in time_data:

        if time_iso == timestamp:
            if not all:
                zip_path = find_zip_file(folder_path)
                if not zip_path:
                    raise HTTPException(
                        status_code=404, detail="ZIP file not found")

                nc_file_path = extract_nc_file(zip_path)
                if not nc_file_path:
                    raise HTTPException(
                        status_code=404, detail="No .nc file found in ZIP")

                return nc_file_path
            else:
                return find_all_zip_files(folder_path)

def extract_nc_file(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(".nc"):
                extracted_path = zip_ref.extract(
                    file_name, os.path.dirname(zip_path))
                return extracted_path
    return None


def find_zip_file(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".zip"):
            return os.path.join(folder_path, file)
    return None


def find_all_zip_files(folder_path):
    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".zip"):
            files.append(os.path.join(folder_path, file))
    return files


def calculate_grid(center_lat, center_lon, km_per_pixel=1, grid_size=10):
    # 1 км в градусах широты
    km_per_deg_lat = 111.32
    deg_per_km_lat = 1 / km_per_deg_lat

    # 1 км в градусах долготы (зависит от широты)
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))
    deg_per_km_lon = 1 / km_per_deg_lon

    half_size = grid_size // 2
    grid = []

    for i in range(grid_size):
        for j in range(grid_size):
            # Смещение от центра в км
            delta_i = (i - half_size) * km_per_pixel
            delta_j = (j - half_size) * km_per_pixel

            # Новые координаты
            lat = center_lat + delta_i * deg_per_km_lat
            lon = center_lon + delta_j * deg_per_km_lon

            grid.append((lat, lon))

    return grid

def calc_lon_lat():
    base_directory = './periods'
    periods = parse_folder_structure(base_directory)
    for el in periods:
        location_list = get_file_path(periods, el[0], True)
        for file_path in location_list:
            file = extract_nc_file(file_path)

            
            ds = xr.open_dataset(file)

            result = calculate_grid(ds.attrs.get('lat_station'), ds.attrs.get('lon_station'), 1, 504)

            coords_ds = ds.copy()

            coords_ds[f"coords"] = xr.DataArray(
                result,
                dims=("lat", "lon"),
                attrs={"description": f"Lats, Lons"}
            )
            
            coords_ds.to_netcdf(f"{file}_updated.nc", mode='a')
            coords_ds.close()
            ds.close()

            #precompute_tile_coordinates_optimized(ds, file, [7,8,9,10,11,12], ds.attrs.get('lat_station'), ds.attrs.get('lon_station'))




if __name__ == "__main__":
    calc_lon_lat()