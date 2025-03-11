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

def generate_tiles_from_image(image_path, output_dir, zoom_levels=[8, 9, 10, 11, 12, 13, 14], 
                              center_lat=None, center_lon=None, pixel_size_deg=0.00001, crs="EPSG:4326"):
    """
    Генерация тайлов из изображения с учетом зума.

    Args:
        image_path (str): Путь к исходному изображению.
        output_dir (str): Папка для сохранения тайлов.
        zoom_levels (list): Список уровней зума (по умолчанию от 8 до 14).
        center_lat (float): Центр изображения (широта).
        center_lon (float): Центр изображения (долгота).
        pixel_size_deg (float): Размер пикселя в градусах.
        crs (str): Система координат (например, "EPSG:4326").
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(image_path, output_dir, center_lat, center_lon)

    # Если заданы координаты центра, вычисляем границы и создаем GeoTIFF
    if center_lat is not None and center_lon is not None:
        with Image.open(image_path) as img:
            width_px, height_px = img.size

        bounds = calculate_bounds(center_lat, center_lon, width_px, height_px, pixel_size_deg)

        with Image.open(image_path) as img:
            img_array = np.array(img)

        if len(img_array.shape) == 2:  # Черно-белое изображение
            img_array = np.stack([img_array] * 3, axis=-1)  # Преобразуем в RGB
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        left, bottom, right, top = bounds
        width, height = img_array.shape[1], img_array.shape[0]
        x_res = (right - left) / width
        y_res = (top - bottom) / height
        transform = from_origin(left, top, x_res, y_res)

        temp_image_path = os.path.join(output_dir, "temp_georeferenced.tif")
        with rio_open(
            temp_image_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=img_array.shape[2] if len(img_array.shape) > 2 else 1,
            dtype=img_array.dtype,
            crs=CRS.from_string(crs),
            transform=transform,
        ) as dst:
            if len(img_array.shape) > 2:
                for i in range(img_array.shape[2]):
                    dst.write(img_array[:, :, i], i + 1)
            else:
                dst.write(img_array, 1)

        image_path = temp_image_path  # Используем временный GeoTIFF

    print(f"Обновленный путь к изображению: {image_path}")

    # Генерация тайлов с учетом масштабирования
    with rasterio.open(image_path) as src:
        orig_width = src.width
        orig_height = src.height
        orig_bounds = src.bounds

        for z in zoom_levels:
            scale_factor = 2 ** (z - 8)  # Масштабный коэффициент
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)

            temp_resized_path = os.path.join(output_dir, f"resized_{z}.tif")
            with rasterio.open(
                temp_resized_path,
                "w",
                driver="GTiff",
                height=new_height,
                width=new_width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=src.transform * src.transform.scale(1 / scale_factor, 1 / scale_factor)
            ) as dst:
                for i in range(1, src.count + 1):
                    dst.write(
                        src.read(i, out_shape=(new_height, new_width), resampling=Resampling.bilinear),
                        i
                    )

            # Разбиваем на тайлы
            with COGReader(temp_resized_path) as image:
                max_tiles = 2 ** z  # Число тайлов по каждой оси

                for x in range(max_tiles):
                    for y in range(max_tiles):
                        tile_minx, tile_miny, tile_maxx, tile_maxy = tms.bounds(x, y, z)

                        if (tile_maxx < orig_bounds[0] or tile_minx > orig_bounds[2] or
                                tile_maxy < orig_bounds[1] or tile_miny > orig_bounds[3]):
                            continue  # Пропускаем тайлы за пределами изображения

                        try:
                            tile, mask = image.tile(x, y, z)
                            if tile is not None:
                                tile = np.transpose(tile, (1, 2, 0))  # (3, 256, 256) → (256, 256, 3)
                                tile_image = Image.fromarray(tile)

                                # Создаём директорию для хранения тайлов
                                tile_path = os.path.join(output_dir, str(z), str(x))
                                os.makedirs(tile_path, exist_ok=True)

                                tile_image.save(os.path.join(tile_path, f"{y}.png"))
                                print(f"Сохранён тайл: {z}/{x}/{y}.png")

                        except Exception as e:
                            print(f"Ошибка при обработке тайла ({x}, {y}, {z}): {e}")

            os.remove(temp_resized_path)  # Удаляем временный файл после обработки

    # Удаляем временный GeoTIFF, если он был создан
    if center_lat is not None and center_lon is not None and os.path.exists(temp_image_path):
        os.remove(temp_image_path)


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
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
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
KM_PER_DEGREE_LON = 111.32 * math.cos(math.radians(CENTER_LAT))  # Учитываем широту


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

def lonlat_to_nxny(lon, lat, center_lat, center_lon, nx_max, ny_max):
    """
    📌 Преобразует долготу/широту в индексы `nx, ny` относительно центра.
    """
    dx_km = (lon - center_lon) * KM_PER_DEGREE_LON
    dy_km = (lat - center_lat) * KM_PER_DEGREE_LAT

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

                nx, ny = lonlat_to_nxny(lon, lat, center_lat, center_lon, nx_max, ny_max)
                tile_data[yi - y1, xi - x1] = find_closest_node(nx, ny, data)

        ds.close()
        return tile_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def render_tile(data, variable):
    """
    📌 Рендерит тайл (256x256) с интерполированными значениями.
    """
    norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    
    fig, ax = plt.subplots(figsize=(1, 1), dpi=TILE_SIZE)
    ax.imshow(data, cmap=get_custom_cmap(variable), norm=norm, origin="upper")
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    
    return buf


# API эндпоинт для тайлов
@app.get("/tiles/{z}/{x}/{y}")
async def get_tile(variable: str, z: int, x: int, y: int, lon: float, lat: float, locator_code:str, slice_index:int, timestamp: str = Query(..., description="Timestamp in ISO format")):
    """Обрабатывает запрос на тайл, создавая его динамически, если он входит в 250 км от центра."""
    try:
        # Проверяем, входит ли запрошенный тайл в радиус 250 км
        time_data = parse_folder_structure('./periods')
        # print(time_data)
        location_list = get_file_path(time_data, timestamp, True)

        zip_location = get_loc_file(location_list, locator_code)
        file_location = extract_nc_file(zip_location)
        data = get_tile_data(file_location, variable, x, y, z, lat, lon, slice_index)
    
        if np.isnan(data).all():
            raise HTTPException(status_code=404, detail="Нет данных в пределах этого тайла")

        tile_buf = render_tile(data, variable)
        
        return StreamingResponse(tile_buf, media_type="image/png")
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/plot")
async def get_plot(variable: str, locator_code: str = "", lat=0, lon=0, timestamp: str = Query(..., description="Timestamp in ISO format"), base_path: str = "path/to/your/folder", slice_index:int =1):
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
        #print(output_file)
        #generate_tiles_from_image(output_file, TILES_DIR, center_lat=lat, center_lon=lon)

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
