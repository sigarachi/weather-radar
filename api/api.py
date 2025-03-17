import os


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
KM_PER_DEGREE_LON = lambda lat: 111.32 * math.cos(math.radians(lat))  # Долгота зависит от широты

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
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
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
    ny = ny_max // 2 - int(dy_km / (2 * GRID_RADIUS_KM / ny_max))  # Инверсия Y (верх -> низ)

    # 📌 Ограничиваем индексы, чтобы не выйти за границы массива
    nx = max(0, min(nx, nx_max - 1))
    ny = max(0, min(ny, ny_max - 1))

    return nx, ny


def from_pixel_to_lonlat(xp, yp, zoom):
    """
    📌 Преобразует пиксельные координаты (xp, yp) в широту/долготу.
    """
    PixelsAtZoom = 256 * 2**zoom
    half_size = PixelsAtZoom / 2

    lon = (xp - half_size) * (360 / PixelsAtZoom)
    lat = (2 * math.atan(math.exp((yp - half_size) / -(PixelsAtZoom / (2 * math.pi)))) - math.pi / 2) * (180 / math.pi)

    return lon, lat


def find_closest_node(nx, ny, data):
    """
    📌 Ищет ближайшую доступную точку в сетке (`nx, ny`).
    """
    if 0 <= nx < data.shape[1] and 0 <= ny < data.shape[0]:  # !!! ВАЖНО: `shape = (ny, nx)`
        return data[ny, nx]  # !!! ВАЖНО: `ny` идет первым!

    min_dist = float("inf")
    closest_val = np.nan

    for i in range(max(0, ny - 2), min(data.shape[0], ny + 2)):  # Проходим по `ny`
        for j in range(max(0, nx - 2), min(data.shape[1], nx + 2)):  # Проходим по `nx`
            if not np.isnan(data[i, j]):
                dist = math.sqrt((nx - j) ** 2 + (ny - i) ** 2)  # !!! ВАЖНО: `(nx, ny) → (j, i)`
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
            raise HTTPException(status_code=400, detail=f"Переменная {variable} не найдена")

        # 📌 Определяем реальную размерность сетки
        dims = ds.dims
        if "time" in dims:
            time_dim = dims["time"]
        else:
            time_dim = 1  # Если нет временного измерения, считаем, что оно одно

        ny_max, nx_max = dims.get("ny", 0), dims.get("nx", 0)  # Размеры сетки (ny, nx)

        if nx_max == 0 or ny_max == 0:
            raise HTTPException(status_code=500, detail="Не удалось определить размеры сетки")

        # 📌 Проверяем, что `slice_index` не выходит за границы
        # if slice_index >= time_dim:
        #     raise HTTPException(status_code=400, detail=f"slice_index {slice_index} выходит за пределы (0-{time_dim-1})")

        # 📌 Извлекаем массив данных, как в примере
        data_array = ds[variable].values

        if data_array.ndim == 3:
            if slice_index >= data_array.shape[0]:
                print(f"Индекс временного среза {slice_index} выходит за пределы ({data_array.shape[0]}).")
                return None
            data = data_array[slice_index, :, :]
        else:
            data = data_array[:, :]

        data = np.squeeze(data)  # Убираем лишние размерности

        lat_size, lon_size = data.shape

        # 📌 Определяем границы тайла
        x1, y1 = x * TILE_SIZE, y * TILE_SIZE
        x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE

        # 📌 Заполняем тайл
        tile_data = np.full((TILE_SIZE, TILE_SIZE), np.nan)

        for yi in range(y1, y2):
            for xi in range(x1, x2):
                lon, lat = from_pixel_to_lonlat(xi, yi, zoom)

                if haversine(lat, lon, center_lat, center_lon) > GRID_RADIUS_KM:
                    continue  # Не обрабатываем этот пиксель

                bearing_angle = calculate_bearing_from_grid(yi, xi, lat_size, lon_size, 1000)
                nx, ny = lonlat_to_nxny(lon, lat, center_lat, center_lon, nx_max, ny_max, bearing_angle)
                tile_data[yi - y1, xi - x1] = find_closest_node(nx, ny, data)

        ds.close()
        return tile_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))