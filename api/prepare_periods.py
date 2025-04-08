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


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Радиус Земли в км
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Возвращает расстояние в км


def from_pixel_to_lonlat(xp, yp, zoom):
    """Преобразует пиксельные координаты в широту/долготу с учетом меркаторской проекции"""
    PixelsAtZoom = 256 * 2**zoom
    half_size = PixelsAtZoom / 2
    
    lon = (xp - half_size) * (360 / PixelsAtZoom)
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (yp / PixelsAtZoom))))
    lat = math.degrees(lat_rad)
    
    return lon, lat

def latlon_to_pixel_xy(lat, lon, zoom):
    """Преобразует lat/lon в пиксельные координаты карты"""
    lat_rad = math.radians(lat)
    pixel_x = ((lon + 180) / 360) * 256 * 2**zoom
    pixel_y = (1 - (math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi)) / 2 * 256 * 2**zoom
    return pixel_x, pixel_y

def calculate_tiles_in_radius(lat, lon, radius_km=250, zoom=12):
    """
    Рассчитывает тайлы карты, попадающие в радиус от заданной точки.
    
    Параметры:
    - lat, lon: широта и долгота центральной точки (в градусах)
    - radius_km: радиус в километрах (по умолчанию 250 км)
    - zoom: уровень масштабирования тайлов (по умолчанию 12)
    
    Возвращает:
    - Список кортежей (x, y) с координатами тайлов
    - Границы области в формате (min_x, max_x, min_y, max_y)
    """
    print(lat, lon, radius_km, zoom)
    
    # Константы
    EARTH_RADIUS = 6371  # радиус Земли в км
    TILE_SIZE = 256  # размер тайла в пикселях
    
    # 1. Переводим координаты центра в радианы
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # 2. Рассчитываем угловое расстояние (в радианах)
    delta = radius_km / EARTH_RADIUS
    
    # 3. Находим границы области в градусах
    min_lat = lat - math.degrees(delta)
    max_lat = lat + math.degrees(delta)
    
    delta_lon = math.asin(math.sin(delta) / math.cos(lat_rad))
    min_lon = lon - math.degrees(delta_lon)
    max_lon = lon + math.degrees(delta_lon)
    
    # 4. Функция для перевода координат в тайлы
    def deg2num(lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi * n))
        return (xtile, ytile)
    
    # 5. Получаем тайлы для угловых точек
    x_min, y_max = deg2num(max_lat, min_lon, zoom)  # верхний левый угол
    x_max, y_min = deg2num(min_lat, max_lon, zoom)  # нижний правый угол
    
    # 6. Генерируем все тайлы в прямоугольной области
    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((x, y))
    
    return tiles

def precalculate_tile_coords_dataset(ds, zoom, x, y, center_lat, center_lon, nx_max, ny_max, tile_size=TILE_SIZE):
    """
    📌 Добавляет в Dataset предрассчитанные координаты только для тайлов, попадающих в радиус.
    Возвращает None если тайл полностью вне радиуса.
    """
    zoom_str = f"zoom_{zoom}"
    radius_km = ds.attrs.get("radius_view", 250.0)

    lat_size, lon_size = 504, 504
    
    lon_arr = np.full((tile_size, tile_size), np.nan)
    lat_arr = np.full((tile_size, tile_size), np.nan)
    nx_arr = np.full((tile_size, tile_size), 0)
    ny_arr = np.full((tile_size, tile_size), 0)
    
    # 📌 Определяем границы тайла
    x1, y1 = x * TILE_SIZE, y * TILE_SIZE
    x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE
    has_valid_pixels = False

    #print(nx_max, ny_max)

    for yi in range(y1, y2):
        for xi in range(x1, x2):
            #print(1)
            
            lon, lat = from_pixel_to_lonlat(xi, yi, zoom)
            distance = haversine(lat, lon, center_lat, center_lon)

            #print(distance)
            
            if distance <= radius_km:
                bearing_angle = calculate_bearing_from_grid(yi, xi, nx_max, ny_max, 1000)
                nx, ny = lonlat_to_nxny(lon, lat, center_lat, center_lon, nx_max, ny_max, bearing_angle)
                
                
                lon_arr[yi - y1, xi - x1] = lon
                lat_arr[yi - y1, xi - x1] = lat
                nx_arr[yi - y1, xi - x1] = nx
                ny_arr[yi - y1, xi - x1] = ny
                has_valid_pixels = True

    updated_ds = ds.copy()
    updated_ds[f"lon_{zoom_str}"] = (("tile_y", "tile_x"), lon_arr)
    updated_ds[f"lat_{zoom_str}"] = (("tile_y", "tile_x"), lat_arr)
    updated_ds[f"nx_{zoom_str}"] = (("tile_y", "tile_x"), nx_arr)
    updated_ds[f"ny_{zoom_str}"] = (("tile_y", "tile_x"), ny_arr)

    if "tile_x" not in updated_ds.coords:
        updated_ds = updated_ds.assign_coords(tile_x=np.arange(tile_size))
    if "tile_y" not in updated_ds.coords:
        updated_ds = updated_ds.assign_coords(tile_y=np.arange(tile_size))

    # Добавляем информацию о тайле
    if "valid_tiles" not in updated_ds.attrs:
        updated_ds.attrs["valid_tiles"] = []
    updated_ds.attrs["valid_tiles"].append(f"{zoom}_{x}_{y}")

    print(f"✅ Добавлены координаты для тайла z={zoom}, x={x}, y={y} (в радиусе {radius_km}км)")
    return updated_ds

KM_PER_DEGREE_LAT = 111.32  # км на градус широты
GRID_RADIUS_KM = 250.0  # Радиус интереса в км

# def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
#     """Вычисляет расстояние между двумя точками на сфере (в км)"""
#     R = 6371  # Радиус Земли в км
#     dlat = math.radians(lat2 - lat1)
#     dlon = math.radians(lon2 - lon1)
#     a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * 
#          math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
#     return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def km_per_degree_lon(lat: float) -> float:
    """Вычисляет км на градус долготы для заданной широты"""
    return math.cos(math.radians(lat)) * KM_PER_DEGREE_LAT

# def from_pixel_to_lonlat(xp: float, yp: float, zoom: int) -> Tuple[float, float]:
#     """Конвертирует пиксельные координаты в географические (Mercator)"""
#     PixelsAtZoom = TILE_SIZE * 2**zoom
#     lon = (xp - PixelsAtZoom/2) * (360 / PixelsAtZoom)
#     lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * yp / PixelsAtZoom)))
#     return lon, math.degrees(lat_rad)

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

def lonlat_to_nxny(lon: float, lat: float, center_lat: float, center_lon: float, 
                  nx_max: int, ny_max: int, bearing_angle: float) -> Tuple[int, int]:
    """Конвертирует lon/lat в индексы сетки nx, ny"""
    dx_km = (lon - center_lon) * km_per_degree_lon(center_lat)
    dy_km = (lat - center_lat) * KM_PER_DEGREE_LAT
    
    # Поворот координат
    angle_rad = math.radians(-bearing_angle)
    dx_rot = dx_km * math.cos(angle_rad) - dy_km * math.sin(angle_rad)
    dy_rot = dx_km * math.sin(angle_rad) + dy_km * math.cos(angle_rad)
    
    nx = nx_max//2 + int(dx_rot / (2 * GRID_RADIUS_KM / nx_max))
    ny = ny_max//2 - int(dy_rot / (2 * GRID_RADIUS_KM / ny_max))  # Инверсия Y
    
    return np.clip(nx, 0, nx_max-1), np.clip(ny, 0, ny_max-1)


def precompute_tile_coordinates(ds, nc_file, zoom_levels, center_lat, center_lon, tile_size=256):
    """
    Предварительно вычисляет и сохраняет координаты x и y для разных уровней масштабирования.
    
    Параметры:
    - nc_file: Путь к исходному NetCDF файлу
    - output_nc_file: Путь к выходному NetCDF файлу для сохранения координат
    - zoom_levels: Список уровней масштабирования (например, [0, 1, 2, 3, 4, 5, 6])
    - center_lat: Широта центра сетки
    - center_lon: Долгота центра сетки
    - tile_size: Размер тайла (по умолчанию 256)
    """
    try:
        # Открываем исходный файл для чтения размерностей
        #ds = xr.open_dataset(f"{nc_file}")
        
        # Получаем размеры сетки
        ny_max, nx_max = ds.dims.get("ny", 0), ds.dims.get("nx", 0)
        if nx_max == 0 or ny_max == 0:
            raise ValueError("Не удалось определить размеры сетки (nx и ny)")
        
        # Создаем новый Dataset для хранения координат
        coords_ds = ds.copy()
        
        for zoom in zoom_levels:
            # Вычисляем количество тайлов на этом уровне масштабирования
            num_tiles = 2 ** zoom
            
            # Создаем массивы для хранения координат
            x_coords = np.zeros((num_tiles, num_tiles, tile_size, tile_size), dtype=np.float32)
            y_coords = np.zeros((num_tiles, num_tiles, tile_size, tile_size), dtype=np.float32)

            print(1)
            
            # Заполняем массивы координат
            for tile_x in range(num_tiles):
                for tile_y in range(num_tiles):
                    # Границы текущего тайла
                    x1, y1 = tile_x * tile_size, tile_y * tile_size
                    x2, y2 = x1 + tile_size, y1 + tile_size

                    print(x1, y1)
                    
                    for yi in range(y1, y2):
                        for xi in range(x1, x2):
                            # Преобразуем пиксели в координаты
                            lon, lat = from_pixel_to_lonlat(xi, yi, zoom)
                            
                            # Проверяем, попадает ли точка в радиус сетки
                            if haversine(lat, lon, center_lat, center_lon) > GRID_RADIUS_KM:
                                x_coords[tile_y, tile_x, yi-y1, xi-x1] = np.nan
                                y_coords[tile_y, tile_x, yi-y1, xi-x1] = np.nan
                                continue
                            
                            # Вычисляем bearing angle (используем упрощенный вариант)
                            bearing_angle = calculate_bearing_from_grid(yi, xi, ny_max, nx_max, 1000)
                            
                            # Преобразуем координаты в индексы сетки
                            nx, ny = lonlat_to_nxny(lon, lat, center_lat, center_lon, nx_max, ny_max, bearing_angle)
                            
                            # Сохраняем координаты
                            x_coords[tile_y, tile_x, yi-y1, xi-x1] = nx
                            y_coords[tile_y, tile_x, yi-y1, xi-x1] = ny
            
            print(2)
            # Добавляем координаты в Dataset
            coords_ds[f"x_zoom_{zoom}"] = xr.DataArray(
                x_coords,
                dims=("tile_y", "tile_x", "pixel_y", "pixel_x"),
                attrs={"description": f"X coordinates for zoom level {zoom}"}
            )
            
            coords_ds[f"y_zoom_{zoom}"] = xr.DataArray(
                y_coords,
                dims=("tile_y", "tile_x", "pixel_y", "pixel_x"),
                attrs={"description": f"Y coordinates for zoom level {zoom}"}
            )
        
        # Сохраняем Dataset в файл
        coords_ds.to_netcdf(f"{nc_file}_updated.nc")
        coords_ds.close()
        ds.close()
        print(3)
        
        print(f"Координаты успешно сохранены в '{nc_file}_updated.nc'")
    
    except Exception as e:
        print(f"Ошибка при предварительном вычислении координат: {str(e)}")
        raise

def calc_lon_lat():
    base_directory = './periods'
    periods = parse_folder_structure(base_directory)
    for el in periods:
        location_list = get_file_path(periods, el[0], True)
        for file_path in location_list:
            file = extract_nc_file(file_path)

            # process_file(file, f"{file}_updated.nc")
            ds = xr.open_dataset(file)
            #print(ds)
            # # print(ds)

            # tiles = calculate_tiles_in_radius(ds.attrs.get('lat_station'), ds.attrs.get('lon_station'), ds.attrs.get('radius_view'), 6)

            # print(tiles)

            # dims = ds.dims
            # ny_max, nx_max = dims.get("ny", 0), dims.get("nx", 0)

            # for x,y in tiles:
            #     ds = precalculate_tile_coords_dataset(ds, 6, x, y, ds.attrs.get('lat_station'), ds.attrs.get('lon_station'), nx_max, ny_max)
                
            # ds.to_netcdf(f"{file}_updated.nc", mode='a')

            precompute_tile_coordinates(ds, file, [7, 8, 9, 10, 11, 12], ds.attrs.get('lat_station'), ds.attrs.get('lon_station'))
            
            
            





if __name__ == "__main__":
    calc_lon_lat()