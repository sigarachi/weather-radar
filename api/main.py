import atexit
import os
from functools import lru_cache
from fastapi import FastAPI, Query, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import xarray as xr
import json
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import cartopy.feature as cfeature
from matplotlib.colors import Normalize, LinearSegmentedColormap
from math import radians, degrees, sin, cos, atan2, sqrt, asin, floor, ceil, tan, pi, log
from PIL import Image
from environs import Env
import math
import zipfile
import uuid
import folium
from fastapi.staticfiles import StaticFiles
from shapely.geometry import Polygon, mapping, Point
from shapely.ops import unary_union
import mercantile
from io import BytesIO
from scipy.spatial import cKDTree


env = Env()
env.read_env()
ORIGINS = env.list('ORIGINS')
TILES_DIR = "tiles"

print(os.environ.get("PROJ_DATA"))


app = FastAPI()

# Enable CORS
origins = ORIGINS

app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define color ranges for each variable type
color_ranges = {
    'Zh': {
        'ranges': [-30, -15, -5, 0, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        'colors': ["#ffffff", "#c8c8c8", "#9fc7ff", "#96fdbd", "#46fe8c", "#4485fb", "#0f34f4", "#4d4196", "#050084", "#ffff00", "#ef952a", "#f95b7e", "#fe0300", "#c802cb", "#760076", "#470b46", "#580704", "#270502"]
    },
    'Zv': {
        'ranges': [-32, -28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
        'colors': [
            '#dbdadb', '#a7a7a7', '#aad3ec', '#cfff9a', '#54fe39',
            '#00aafd', '#024dff', '#0000ce', '#010080', '#ffff01',
            '#fe8000', '#ff7e01', '#ff3c36', '#fe0000', '#49dd01',
            '#00a800', '#f400f3', '#aa00ff', '#720000'
        ]
    },
    'Vr': {
        'ranges': [-32, -28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
        'colors': [
            '#013a10', '#005822', '#1b7b3b', '#399a53', '#61b266',
            '#7fc589', '#9ccaa3', '#bcd8c2', '#daf0db', '#ffffff',
            '#fedcd0', '#fdc4b0', '#f7a688', '#fa8267', '#ee603c',
            '#d92d1f', '#b90b0c', '#8b0b0c', '#5f0004'
        ]
    },
    'Zdr': {
        'ranges': [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
        'colors': [
            '#212892', '#105380', '#1CBEFE', '#2FE1FF', '#72FEFB',
            '#74FF88', '#7CB846', '#008601', '#FFE202', '#E0BF4C',
            '#D49802', '#FEA8A7', '#E6594F', '#F81600', '#B90B0A',
            '#BE96AE', '#7F586A'
        ]
    },
    'Fdp': {
        'ranges': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
        'colors': [
            '#F665E6', '#FC90C2', '#FBB59C', '#FCD472', '#F2ED3B',
            '#D0E818', '#A6D912', '#7ACB0D', '#48B90F', '#35A43B',
            '#448A72', '#3B6FA1', '#2A4EC8', '#2827E4', '#591EF2',
            '#8535FA', '#AE40FB', '#DB48F9'
        ]
    },
    'roHV': {
        'ranges': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        'colors': [
            '#441C02', '#833B00', '#C15700', '#FF8001', '#FFC0C0',
            '#70FFFF', '#01D8DB', '#00CB00', '#01E900', '#AAFE50',
            '#CBFE97', '#FFFE80'
        ]
    },
    'Sg': {
        'ranges': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colors': [
            '#E3F7F8', '#CBF8F3', '#9AECEA', '#81DE6C',
            '#F8F649', '#FDCE20', '#F9942A', '#F84E2D', '#E7294D', '#AE1D94'
        ]
    },
    'Sv': {
        'ranges': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colors': [
            '#E3F7F8', '#CBF8F3', '#9AECEA', '#81DE6C',
            '#F8F649', '#FDCE20', '#F9942A', '#F84E2D', '#E7294D', '#AE1D94'
        ]
    },
    'Tb': {
        'ranges': [0, 0.33, 0.66, 1.0],
        'colors': ['#E6E6E6', '#FDCD00', '#FF3300', '#CC0003']
    },
    'DPmap': {
        'ranges': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
        'colors': [
            '#9FA9B2', '#A3C6FF', '#45FF92', '#01C15A', '#009800', '#FFFF81',
            '#4088FE', '#0038FF', '#000074', '#FFAB7F', '#FF557F', '#FF0101',
            '#CA6702', '#894401', '#610000', '#FFAAFF', '#FF54FF', '#C600C7', '#43405D'
        ]
    },
    'Hngo': {
        'ranges': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
        'colors': [
            '#EAF4FE', '#CFE7FF', '#B9DBFE', '#9CCFFE', '#56ABFE', '#027FFF',
            '#006ADA', '#005FBD', '#0052A2', '#024289', '#027500', '#00AB01',
            '#25FF25', '#FFFF01', '#FF9899', '#FE0000', '#A60000', '#710100'
        ]
    },
    'Hvgo': {
        'ranges': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
        'colors': [
            '#07FEA5', '#1BF36C', '#1FBD82', '#229470', '#257164', '#164137',
            '#01CCEA', '#0287E4', '#0136D0', '#05176D', '#D3B51F', '#DA491A',
            '#D91A12', '#84050C', '#43A804', '#033F03', '#A31047', '#DB2F9D'
        ]
    },
    'VIL': {
        'ranges': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
        'colors': [
            '#DCEFFD', '#B1DAFA', '#88C4FF', '#4AA5FE', '#95FDFE', '#4BFFFF',
            '#01D8DA', '#00CB00', '#02EA00', '#A7FF50', '#CBFF98', '#FFFF81',
            '#FF9B8C', '#FF3F40', '#FA6AE3', '#C400C4', '#A60000', '#720000'
        ]
    },
    'R': {
        'ranges': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colors': [
            '#9B9B9B', '#868887', '#0055FE', '#010080', '#FFFF00', '#C9EF04',
            '#FDAB00', '#FF5600', '#FE0000', '#81FF7F', '#02A901', '#FE83F5', '#D000D0'
        ]
    },
    'Qp3': {
        'ranges': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'colors': [
            '#FFAAFF', '#D0E9FF', '#00AAFF', '#004CFE', '#0000C5', '#00007E',
            '#FEFF01', '#FF7F02', '#E5490C', '#FD0100', '#01BE00', '#007500', '#FF00FF', '#CC009D'
        ]
    },
    'Qp6': {
        'ranges': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'colors': [
            '#FFAAFF', '#D0E9FF', '#00AAFF', '#004CFE', '#0000C5', '#00007E',
            '#FEFF01', '#FF7F02', '#E5490C', '#FD0100', '#01BE00', '#007500', '#FF00FF', '#CC009D'
        ]
    },
    'Qp12': {
        'ranges': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'colors': [
            '#FFAAFF', '#D0E9FF', '#00AAFF', '#004CFE', '#0000C5', '#00007E',
            '#FEFF01', '#FF7F02', '#E5490C', '#FD0100', '#01BE00', '#007500', '#FF00FF', '#CC009D'
        ]
    },
    'Qp24': {
        'ranges': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'colors': [
            '#FFAAFF', '#D0E9FF', '#00AAFF', '#004CFE', '#0000C5', '#00007E',
            '#FEFF01', '#FF7F02', '#E5490C', '#FD0100', '#01BE00', '#007500', '#FF00FF', '#CC009D'
        ]
    }
}


def get_color_for_value(variable: str, value: float) -> str:
    """
    Returns the appropriate color for a given value and variable type.
    """
    if variable not in color_ranges:
        return '#000000'  # Default black color for unknown variables

    ranges = color_ranges[variable]['ranges']
    colors = color_ranges[variable]['colors']

    # Handle NaN values
    if np.isnan(value):
        return '#000000'

    # Find the appropriate color range
    for i in range(len(ranges) - 1):
        if ranges[i] <= value < ranges[i + 1]:
            return colors[i]

    # Handle values outside the range
    if value < ranges[0]:
        return colors[0]
    return colors[-1]


def get_custom_cmap(variable: str = ""):
    """
    Creates a colormap based on the variable type and its value ranges.
    """
    if variable not in color_ranges:
        return LinearSegmentedColormap.from_list("custom_gradient", ['#000000'])

    ranges = color_ranges[variable]['ranges']
    colors = color_ranges[variable]['colors']

    # Create a colormap with the specified colors
    return LinearSegmentedColormap.from_list("custom_gradient", colors)


def add_overviews(image_path):
    """
    Add overviews to a GeoTIFF file.

    Args:
        image_path (str): Path to the input GeoTIFF.
    """
    import rasterio

    with rasterio.open(image_path, "r+") as src:
        overview_levels = [2, 4, 8, 16]
        src.build_overviews(overview_levels, Resampling.average)
        src.update_tags(ns="rio_overview", resampling="average")


def calculate_bounds(center_lat, center_lon, width_px, height_px, pixel_size_deg):
    """
    Calculate bounds based on the center coordinates, image dimensions, and pixel size.

    Args:
        center_lat (float): Latitude of the center point.
        center_lon (float): Longitude of the center point.
        width_px (int): Width of the image in pixels.
        height_px (int): Height of the image in pixels.
        pixel_size_deg (float): Size of a pixel in degrees.

    Returns:
        tuple: Bounds in the format (left, bottom, right, top).
    """
    # Преобразуем аргументы в числа, если они переданы как строки
    center_lat = float(center_lat)
    center_lon = float(center_lon)
    width_px = int(width_px)
    height_px = int(height_px)
    pixel_size_deg = float(pixel_size_deg)

    # Вычисляем общую ширину и высоту в градусах
    total_width_deg = width_px * pixel_size_deg
    total_height_deg = height_px * pixel_size_deg

    # Вычисляем границы
    left = center_lon - (total_width_deg / 2)
    right = center_lon + (total_width_deg / 2)
    top = center_lat + (total_height_deg / 2)
    bottom = center_lat - (total_height_deg / 2)

    return (left, bottom, right, top)


@lru_cache(maxsize=128)
def extract_nc_file(zip_path):
    """
    Extracts .nc file from a zip archive and returns its path.
    Uses caching to avoid repeated extractions of the same file.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(".nc"):
                    extracted_path = zip_ref.extract(
                        file_name, os.path.dirname(zip_path))
                    return extracted_path
        return None
    except Exception as e:
        print(f"Error extracting NC file: {e}")
        return None


# Add caching decorators
@lru_cache(maxsize=128)
def parse_folder_structure(base_path):
    time_periods = []
    try:
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
    except Exception as e:
        print(f"Error parsing folder structure: {e}")
        return []
    return time_periods


@lru_cache(maxsize=128)
def find_zip_file(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".zip"):
            return os.path.join(folder_path, file)
    return None


@lru_cache(maxsize=128)
def find_all_zip_files(folder_path):
    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".zip"):
            files.append(os.path.join(folder_path, file))
    return files


# Функция для вычисления расстояния между двумя точками (формула Хаверсина)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Радиус Земли в км
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Возвращает расстояние в км


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


def get_loc_file(location_list, locator_code):
    for loc in location_list:
        if locator_code in loc:
            return loc


# Константы
TILE_SIZE = 256  # Размер тайла в пикселях
EARTH_RADIUS = 6371  # Радиус Земли в метрах
RADIUS_LIMIT = 250000  # Ограничение радиуса 250 км
TILES_XY = 6  # Количество тайлов по X и Y

CACHE_DIR = "./cache"  # Директория для кэширования карт

# Создаем директорию для кэша, если её нет
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cached_map(variable, center_lat, center_lon):
    """Проверяет, существует ли уже сгенерированная карта."""
    filename = f"{CACHE_DIR}/map_{variable}_{center_lat}_{center_lon}.png"
    return filename if os.path.exists(filename) else None


def tile_center(x, y, z):
    """Возвращает точные координаты центра тайла (lat, lon)"""
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


TILE_SIZE = 256  # Размер тайла
GRID_RADIUS_KM = 250  # Радиус сетки данных
GRID_SIZE = 512  # Размер сетки (512x512 точек)

# 📌 Географический центр (широта, долгота)
CENTER_LAT = 55.0
CENTER_LON = 37.0

# 📌 Вычисляем размер ячейки в километрах
CELL_SIZE_KM = (GRID_RADIUS_KM * 2) / GRID_SIZE

# 📌 Коэффициенты для перевода градусов в километры
KM_PER_DEGREE_LAT = 110.574  # 1° широты ≈ 110.574 км


def KM_PER_DEGREE_LON(lat): return 111.32 * \
    math.cos(math.radians(lat))  # Долгота зависит от широты


def calculate_bearing_from_grid(i, j, lat_size, lon_size, pixel_size):
    """
    📌 Вычисляет угол поворота (bearing) относительно центра сетки.
    """
    # 📌 Определяем смещение от центра (dx, dy)
    dx = (j - lon_size // 2) * pixel_size
    dy = (i - lat_size // 2) * pixel_size

    # 📌 Вычисляем геометрический угол (в градусах)
    bearing = math.degrees(math.atan2(dy, dx))

    return bearing


def rotate_point(x, y, angle):
    """
    📌 Поворачивает точку (`x, y`) на `angle` градусов вокруг центра (0,0).
    """
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a

    return x_rot, y_rot


def haversine(lat1, lon1, lat2, lon2):
    """
    📌 Формула Хаверсина – вычисляет расстояние между двумя координатами.
    """
    R = 6371  # Радиус Земли в км
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Расстояние в км


def lonlat_to_nxny(lon, lat, center_lat, center_lon, nx_max, ny_max, bearing_angle):
    """
    📌 Преобразует долготу/широту в индексы `nx, ny`, учитывая угол `bearing_angle`.
    """
    dx_km = (lon - center_lon) * KM_PER_DEGREE_LON(center_lat)
    dy_km = (lat - center_lat) * KM_PER_DEGREE_LAT

    # 🔄 Поворот координат с учетом `bearing`
    dx_km, dy_km = rotate_point(dx_km, dy_km, -bearing_angle)

    nx = nx_max // 2 + int(dx_km / (2 * GRID_RADIUS_KM / nx_max))
    # Инверсия Y (верх -> низ)
    ny = ny_max // 2 - int(dy_km / (2 * GRID_RADIUS_KM / ny_max))

    # 📌 Ограничиваем индексы, чтобы не выйти за границы массива
    nx = max(0, min(nx, nx_max - 1))
    ny = max(0, min(ny, ny_max - 1))

    return nx, ny


def from_pixel_to_lonlat(xp, yp, zoom):
    """
    📌 Преобразует пиксельные координаты (xp, yp) в широту/долготу.
    Поддерживает как скалярные значения, так и массивы.
    """
    PixelsAtZoom = 256 * 2**zoom
    half_size = PixelsAtZoom / 2

    # Преобразуем входные данные в массивы numpy, если они еще не являются таковыми
    xp = np.asarray(xp)
    yp = np.asarray(yp)

    # Вычисляем долготу и широту для всех точек
    lon = (xp - half_size) * (360 / PixelsAtZoom)
    lat_rad = 2 * np.arctan(np.exp((yp - half_size) / -
                            (PixelsAtZoom / (2 * np.pi)))) - np.pi / 2
    lat = lat_rad * (180 / np.pi)

    return lon, lat


def find_closest_node(nx, ny, data):
    """
    📌 Ищет ближайшую доступную точку в сетке (`nx, ny`).
    """
    if 0 <= nx < data.shape[1] and 0 <= ny < data.shape[0]:  # !!! ВАЖНО: `shape = (ny, nx)`
        return data[ny, nx]  # !!! ВАЖНО: `ny` идет первым!

    min_dist = float("inf")
    # print("shape:", data.shape[0], data.shape[1])
    closest_val = np.nan

    # Проходим по `ny`
    for i in range(max(0, ny - 2), min(data.shape[0], ny + 2)):
        # Проходим по `nx`
        for j in range(max(0, nx - 2), min(data.shape[1], nx + 2)):
            if not np.isnan(data[i, j]):
                # !!! ВАЖНО: `(nx, ny) → (j, i)`
                dist = math.sqrt((nx - j) ** 2 + (ny - i) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_val = data[i, j]

    return closest_val


def get_tile_data(nc_file, variable, x, y, zoom, center_lat, center_lon, slice_index=0):
    """
    📌 Генерирует данные для запрашиваемого тайла.
    """
    try:
        ds = xr.open_dataset(nc_file)
        if variable not in ds.variables:
            raise HTTPException(
                status_code=400, detail=f"Переменная {variable} не найдена")

        # 📌 Определяем реальную размерность сетки
        dims = ds.dims
        if "time" in dims:
            time_dim = dims["time"]
        else:
            time_dim = 1  # Если нет временного измерения, считаем, что оно одно

        ny_max, nx_max = dims.get("ny", 0), dims.get(
            "nx", 0)  # Размеры сетки (ny, nx)

        if nx_max == 0 or ny_max == 0:
            raise HTTPException(
                status_code=500, detail="Не удалось определить размеры сетки")

        # 📌 Извлекаем массив данных, как в примере
        data_array = ds[variable].values

        if data_array.ndim == 3:
            if slice_index >= data_array.shape[0]:
                print(
                    f"Индекс временного среза {slice_index} выходит за пределы ({data_array.shape[0]}).")
                return None
            data = data_array[slice_index, :, :]
        else:
            data = data_array[:, :]

        data = np.squeeze(data)  # Убираем лишние размерности
        lat_size, lon_size = data.shape

        # 📌 Определяем границы тайла
        x1, y1 = x * TILE_SIZE, y * TILE_SIZE
        x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE

        # Создаем координатную сетку для тайла
        xi, yi = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))

        # Преобразуем пиксельные координаты в географические
        lons, lats = from_pixel_to_lonlat(xi.ravel(), yi.ravel(), zoom)

        # Векторизованное вычисление расстояний
        distances = np.array([haversine(lat, lon, center_lat, center_lon)
                              for lat, lon in zip(lats, lons)])

        # Создаем маску для точек в пределах радиуса
        in_radius = distances <= GRID_RADIUS_KM

        if not np.any(in_radius):
            print("Тайл полностью вне радиуса 250 км")
            ds.close()
            return None

        # Векторизованное преобразование координат в индексы сетки
        nx = np.clip(np.floor((lons[in_radius] - center_lon) * KM_PER_DEGREE_LON(
            center_lat) / (2 * GRID_RADIUS_KM / nx_max) + nx_max // 2), 0, nx_max - 1).astype(int)
        ny = np.clip(np.floor((lats[in_radius] - center_lat) * KM_PER_DEGREE_LAT / (
            2 * GRID_RADIUS_KM / ny_max) + ny_max // 2), 0, ny_max - 1).astype(int)

        # Формируем тайл
        tile_data = np.full((TILE_SIZE, TILE_SIZE), np.nan)

        # Создаем маску для исходного тайла
        tile_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
        tile_mask.ravel()[in_radius] = True

        # Заполняем только валидные точки в радиусе
        valid_points = (ny >= 0) & (ny < ny_max) & (nx >= 0) & (nx < nx_max)
        if np.any(valid_points):
            # Векторизованное получение значений из data
            tile_data[tile_mask] = data[ny[valid_points], nx[valid_points]]

        ds.close()
        return tile_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add figure caching
figure_cache = {}


def get_cached_figure():
    if 'fig' not in figure_cache:
        figure_cache['fig'] = plt.figure(figsize=(1, 1), dpi=TILE_SIZE)
        figure_cache['ax'] = figure_cache['fig'].add_subplot(111)
        figure_cache['ax'].axis("off")
    return figure_cache['fig'], figure_cache['ax']


def render_tile(data, variable):
    """
    Renders a tile with colors based on value ranges.
    """
    try:
        # Get fixed value ranges for the variable
        if variable in color_ranges:
            ranges = color_ranges[variable]['ranges']
            vmin, vmax = ranges[0], ranges[-1]
        else:
            vmin, vmax = np.nanmin(data), np.nanmax(data)

        # Create a normalized colormap with fixed ranges
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Use cached figure
        fig, ax = get_cached_figure()

        # Clear previous data
        ax.clear()
        ax.axis("off")

        # Get the colormap for the variable
        cmap = get_custom_cmap(variable)

        # Render the image using the colormap
        im = ax.imshow(data, cmap=cmap, norm=norm, origin="upper")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    pad_inches=0, transparent=True)
        buf.seek(0)

        return buf
    except Exception as e:
        print(f"Error rendering tile: {e}")
        return None


# API эндпоинт для тайлов
@app.get("/tiles/{z}/{x}/{y}")
async def get_tile(variable: str, z: int, x: int, y: int, lon: float, lat: float, locator_code: str, slice_index: int, timestamp: str = Query(..., description="Timestamp in ISO format")):
    """Обрабатывает запрос на тайл, создавая его динамически, если он входит в 250 км от центра."""
    try:
        # Проверяем, входит ли запрошенный тайл в радиус 250 км
        time_data = parse_folder_structure('./periods')
        # print(time_data)
        location_list = get_file_path(time_data, timestamp, True)

        zip_location = get_loc_file(location_list, locator_code)
        file_location = extract_nc_file(zip_location)

        data2 = get_tile_data(
            file_location, variable, x, y, z, lat, lon, slice_index)

        if np.isnan(data2).all():
            raise HTTPException(
                status_code=404, detail="Нет данных в пределах этого тайла")

        tile_buf = render_tile(data2, variable)

        return StreamingResponse(tile_buf, media_type="image/png")
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/time-periods")
async def get_time_periods():
    try:
        base_directory = './periods'
        time_data = parse_folder_structure(base_directory)
        json_output = json.dumps({"time_periods": list(time_data)}, indent=4)
        return JSONResponse(content=json_output, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/list_files")
def list_files(timestamp: str = Query(..., description="Timestamp in ISO format"), base_path: str = "path/to/your/folder"):
    try:
        time_data = parse_folder_structure('./periods')
        for time_iso, folder_path in time_data:
            if time_iso == timestamp:
                files = os.listdir(folder_path)
                return JSONResponse(content={"timestamp": timestamp, "files": files}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/variables")
async def get_variables(locator_code: str = "RUDL", timestamp: str = Query(..., description="Timestamp in ISO format"), base_path: str = "path/to/your/folder"):
    try:
        time_data = parse_folder_structure('./periods')
        location_list = get_file_path(time_data, timestamp, True)

        zip_location = get_loc_file(location_list, locator_code)
        file_location = extract_nc_file(zip_location)

        ds = xr.open_dataset(file_location)
        variables = list(ds.data_vars.keys())

        return JSONResponse(content=variables, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Add dataset caching
dataset_cache = {}


def get_cached_dataset(file_path):
    if file_path not in dataset_cache:
        dataset_cache[file_path] = xr.open_dataset(file_path)
    return dataset_cache[file_path]


def clear_dataset_cache():
    for ds in dataset_cache.values():
        ds.close()
    dataset_cache.clear()


# Add periodic cache clearing
atexit.register(clear_dataset_cache)
