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
from scipy.spatial import cKDTree
# os.unsetenv('PROJ_DATA')
# os.unsetenv('PROJ_LIB')


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

custom_colors_ZH = [
    '#dbdadb',
    '#a7a7a7',
    '#aad3ec',
    '#cfff9a',
    '#54fe39',
    '#00aafd',
    '#024dff',
    '#0000ce',
    '#010080',
    '#ffff01',
    '#fe8000',
    '#ff7e01',
    '#ff3c36',
    '#fe0000',
    '#49dd01',
    '#00a800',
    '#f400f3',
    '#aa00ff',
    '#720000'
]

custom_colors_VR = [
    '#013a10',
    '#005822',
    '#1b7b3b',
    '#399a53',
    '#61b266',
    '#7fc589',
    '#9ccaa3',
    '#bcd8c2',
    '#daf0db',
    '#ffffff',
    '#fedcd0',
    '#fdc4b0',
    '#f7a688',
    '#fa8267',
    '#ee603c',
    '#d92d1f',
    '#b90b0c',
    '#8b0b0c',
    '#5f0004'
]

custom_colors_ZDR = ['#212892', '#105380', '#1CBEFE', '#2FE1FF', '#72FEFB', '#74FF88', '#7CB846', '#008601',
                     '#FFE202', '#E0BF4C', '#D49802', '#FEA8A7', '#E6594F', '#F81600', '#B90B0A', '#BE96AE', '#7F586A']
custom_colors_FDP = ['#F665E6', '#FC90C2', '#FBB59C', '#FCD472', '#F2ED3B', '#D0E818', '#A6D912', '#7ACB0D',
                     '#48B90F', '#35A43B', '#448A72', '#3B6FA1', '#2A4EC8', '#2827E4', '#591EF2', '#8535FA', '#AE40FB', '#DB48F9']
custom_colors_roHV = ['#441C02', '#833B00', '#C15700', '#FF8001', '#FFC0C0',
                      '#70FFFF', '#01D8DB', '#00CB00', '#01E900', '#AAFE50', '#CBFE97', '#FFFE80']
custom_colors_SV = ['#E3F7F8', '#CBF8F3', '#9AECEA', '#81DE6C',
                    '#F8F649', '#FDCE20', '#F9942A', '#F84E2D', '#E7294D', '#AE1D94']
custom_colors_TB = ['#E6E6E6', '#FDCD00', '#FF3300', '#CC0003']
custom_colors_DPmap = ['#9FA9B2', '#A3C6FF', '#45FF92', '#01C15A', '#009800', '#FFFF81', '#4088FE', '#0038FF', '#000074',
                       '#FFAB7F', '#FF557F', '#FF0101', '#CA6702', '#894401', '#610000', '#FFAAFF', '#FF54FF', '#C600C7', '#43405D']
custom_colors_Hngo = ['#EAF4FE', '#CFE7FF', '#B9DBFE', '#9CCFFE', '#56ABFE', '#027FFF', '#006ADA', '#005FBD',
                      '#0052A2', '#024289', '#027500', '#00AB01', '#25FF25', '#FFFF01', '#FF9899', '#FE0000', '#A60000', '#710100']
custom_colors_VIL = ['#DCEFFD', '#B1DAFA', '#88C4FF', '#4AA5FE', '#95FDFE', '#4BFFFF', '#01D8DA', '#00CB00',
                     '#02EA00', '#A7FF50', '#CBFF98', '#FFFF81', '#FF9B8C', '#FF3F40', '#FA6AE3', '#C400C4', '#A60000', '#720000']
custom_colors_R = ['#9B9B9B', '#868887', '#0055FE', '#010080', '#FFFF00', '#C9EF04',
                   '#FDAB00', '#FF5600', '#FE0000', '#81FF7F', '#02A901', '#FE83F5', '#D000D0']
custom_colors_Qp = ['#FFAAFF', '#D0E9FF', '#00AAFF', '#004CFE', '#0000C5', '#00007E',
                    '#FEFF01', '#FF7F02', '#E5490C', '#FD0100', '#01BE00', '#007500', '#FF00FF', '#CC009D']
custom_colors_Hvgo = ['#07FEA5', '#1BF36C', '#1FBD82', '#229470', '#257164', '#164137', '#01CCEA', '#0287E4',
                      '#0136D0', '#05176D', '#D3B51F', '#DA491A', '#D91A12', '#84050C', '#43A804', '#033F03', '#A31047', '#DB2F9D']

custom_colors_map = {
    'Zh': custom_colors_ZH,
    'Zv': custom_colors_ZH,
    'Vr': custom_colors_VR,
    'Zdr': custom_colors_ZDR,
    'Fdp': custom_colors_FDP,
    'roHV': custom_colors_roHV,
    'Sg': custom_colors_SV,
    'Sv': custom_colors_SV,
    'Tb': custom_colors_TB,
    'DPmap': custom_colors_DPmap,
    'Hngo': custom_colors_Hngo,
    'Hvgo': custom_colors_Hvgo,
    'VIL': custom_colors_VIL,
    'R': custom_colors_R,
    'Qp3': custom_colors_Qp,
    'Qp6': custom_colors_Qp,
    'Qp12': custom_colors_Qp,
    'Qp24': custom_colors_Qp,
}

# custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", custom_colors_ZH)


def get_custom_cmap(variable: str = ""):
    return LinearSegmentedColormap.from_list("custom_gradient", custom_colors_map[variable])


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
    print(data[ny, nx])
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


def get_tile_data_new(nc_file, variable, x, y, zoom, center_lat, center_lon, slice_index=0, locator_code=""):
    """
    📌 Генерирует данные для запрашиваемого тайла.
    """
    try:
        # Сначала проверяем расстояние от центра тайла до центра данных
        tile_center_x = (x + 0.5) * TILE_SIZE
        tile_center_y = (y + 0.5) * TILE_SIZE
        tile_center_lon, tile_center_lat = from_pixel_to_lonlat(
            tile_center_x, tile_center_y, zoom)

        # Проверяем расстояние от центра тайла до центра данных
        center_distance = haversine(
            tile_center_lat, tile_center_lon, center_lat, center_lon)

        # Если центр тайла находится за пределами радиуса + половина диагонали тайла,
        # то весь тайл точно вне радиуса
        tile_diagonal_km = haversine(tile_center_lat, tile_center_lon,
                                     *from_pixel_to_lonlat(tile_center_x + TILE_SIZE/2,
                                                           tile_center_y + TILE_SIZE/2, zoom))

        if center_distance > (GRID_RADIUS_KM + tile_diagonal_km):
            print(
                f"Тайл полностью вне радиуса 250 км (расстояние: {center_distance:.2f} км)")
            return None

        # Открываем файлы только если тайл может содержать данные
        ds_data = xr.open_dataset(nc_file)
        ds_grid = xr.open_dataset(f"grid_coordinates{locator_code}.nc")

        # Проверяем наличие переменной
        if variable not in ds_data.variables:
            raise ValueError(
                f"Переменная {variable} не найдена в файле данных")

        # Загружаем данные
        data_array = ds_data[variable].values
        if data_array.ndim == 3:
            if slice_index >= data_array.shape[0]:
                print(
                    f"Индекс временного среза {slice_index} выходит за пределы ({data_array.shape[0]}).")
                return None
            data = data_array[slice_index, :, :]
        else:
            data = data_array[:, :]

        data = np.squeeze(data)  # Убираем лишние размерности

        # Загружаем предварительно вычисленные данные
        mask = ds_grid['valid_mask'][:]
        valid_indices = ds_grid['valid_indices'][:]
        kdtree_data = ds_grid['kdtree_data'][:]

        print(f"Data shape: {data.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Valid indices shape: {valid_indices.shape}")
        print(f"KDTree data shape: {kdtree_data.shape}")

        # Создаем новое KD-дерево
        kdtree = cKDTree(kdtree_data)

        # Вычисляем границы тайла
        x1, y1 = x * TILE_SIZE, y * TILE_SIZE
        x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE

        # Создаем координатную сетку для тайла
        xi, yi = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))

        # Преобразуем пиксельные координаты в географические
        lons, lats = from_pixel_to_lonlat(xi.ravel(), yi.ravel(), zoom)

        # Проверяем расстояние для каждой точки
        distances = np.array([haversine(lat, lon, center_lat, center_lon)
                              for lat, lon in zip(lats, lons)])

        # Создаем маску для точек в пределах радиуса
        in_radius = distances <= GRID_RADIUS_KM

        if not np.any(in_radius):
            print("Тайл полностью вне радиуса 250 км")
            return None

        # Создаем массив запросов только для точек в радиусе
        query_points = np.column_stack([lons[in_radius], lats[in_radius]])

        # Находим ближайшие точки в сетке только для точек в радиусе
        distances, indices = kdtree.query(query_points)

        # Проверяем границы индексов
        indices = np.clip(indices, 0, len(valid_indices) - 1)
        grid_indices = valid_indices[indices]

        print(f"Max grid index: {np.max(grid_indices)}")
        print(f"Data array size: {data.size}")

        # Преобразуем индексы
        ny, nx = mask.shape
        y_idx = np.clip(grid_indices // nx, 0, ny - 1)
        x_idx = np.clip(grid_indices % nx, 0, nx - 1)

        # Формируем тайл
        tile_data = np.full((TILE_SIZE, TILE_SIZE), np.nan)

        # Заполняем только валидные точки
        valid_mask = (y_idx < ny) & (x_idx < nx) & (mask[y_idx, x_idx] == 1)
        if np.any(valid_mask):
            # Создаем маску для исходного тайла
            tile_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
            tile_mask.ravel()[in_radius] = valid_mask

            # Заполняем только валидные точки в радиусе
            tile_data[tile_mask] = data[y_idx[valid_mask], x_idx[valid_mask]]

        return tile_data

    except Exception as e:
        raise ValueError(f"Ошибка при обработке тайла: {str(e)}")
    finally:
        if 'ds_data' in locals():
            ds_data.close()
        if 'ds_grid' in locals():
            ds_grid.close()


def render_tile(data, variable):
    """
    📌 Рендерит тайл (256x256) с интерполированными значениями.
    """
    norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))

    fig, ax = plt.subplots(figsize=(1, 1), dpi=TILE_SIZE)
    ax.imshow(data, cmap=get_custom_cmap(variable), norm=norm, origin="upper")
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)

    return buf


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

        data2 = get_tile_data_new(
            file_location, variable, x, y, z, lat, lon, slice_index, locator_code)

        if np.isnan(data2).all():
            raise HTTPException(
                status_code=404, detail="Нет данных в пределах этого тайла")

        tile_buf = render_tile(data2, variable)

        return StreamingResponse(tile_buf, media_type="image/png")
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/plot")
async def get_plot(variable: str, locator_code: str = "", lat=0, lon=0, timestamp: str = Query(..., description="Timestamp in ISO format"), base_path: str = "path/to/your/folder", slice_index: int = 1):
    try:
        time_data = parse_folder_structure('./periods')
        # print(time_data)
        location_list = get_file_path(time_data, timestamp, True)

        zip_location = get_loc_file(location_list, locator_code)
        file_location = extract_nc_file(zip_location)
        ds = Dataset(file_location, mode="r")

        print(os.environ.get("TCL_LIBRARY"))
        print(os.environ.get("TK_LIBRARY"))

        if variable not in ds.variables:
            return JSONResponse(content={"error": "Variable not found"}, status_code=400)

        data_array = ds.variables[variable]
        # shape = data_array.shape

        # Generate the plot
        output_file = plot_data_on_map_custom_json_by_color(
            data_array, float(lat), float(lon), variable, slice_index=slice_index)
        ds.close()
        # print(output_file)
        # generate_tiles_from_image(output_file, TILES_DIR, center_lat=lat, center_lon=lon)

        return JSONResponse(content=output_file, status_code=200)
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
