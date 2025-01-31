from fastapi import FastAPI, Query, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
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
from math import radians, degrees, sin, cos, atan2, sqrt, asin
from PIL import Image
from environs import Env
import math
import os
import zipfile
import uuid

env = Env()
env.read_env() 
ORIGINS = env.list('ORIGINS')

app = FastAPI()

# Enable CORS
origins = ORIGINS


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

custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", custom_colors_ZH)


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
                            timestamp = datetime(int(year), int(month), int(day), int(hour), int(minute))
                            time_periods.append((timestamp.isoformat(), minute_path))
                        except ValueError:
                            print(f"Skipping invalid date: {year}-{month}-{day} {hour}:{minute}")
    
    return time_periods

def extract_nc_file(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(".nc"):
                extracted_path = zip_ref.extract(file_name, os.path.dirname(zip_path))
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

def calculate_destination(lat1, lon1, distance, bearing):
    """
    Рассчитывает координаты точки назначения по расстоянию и направлению (bearing).
    :param lat1: широта начальной точки (в градусах)
    :param lon1: долгота начальной точки (в градусах)
    :param distance: расстояние до точки назначения (в метрах)
    :param bearing: угол направления (bearing) в градусах
    :return: широта и долгота точки назначения (в градусах)
    """
    R = 6371e3  # Радиус Земли в метрах

    # Преобразуем координаты начальной точки и угол в радианы
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    bearing = math.radians(bearing)

    # Вычисляем новую широту (lat2)
    lat2 = math.asin(math.sin(lat1) * math.cos(distance / R) +
                     math.cos(lat1) * math.sin(distance / R) * math.cos(bearing))

    # Вычисляем новую долготу (lon2)
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat1),
                             math.cos(distance / R) - math.sin(lat1) * math.sin(lat2))

    # Преобразуем обратно в градусы
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return lat2, lon2


def plot_data_on_map(data_array, lat_center, lon_center, pixel_size=1000, slice_index=3):
    shape = data_array.shape
    data = data_array[:]
    # max_distance = resolution_km * shape[0]

    if len(shape) == 3:  # Если данные имеют временное измерение
        if slice_index >= shape[0]:
            print(f"Индекс временного среза {slice_index} выходит за пределы первого измерения ({shape[0]}).")
            return
        data = data[slice_index, :, :]
    else:  # Если данные двумерные
        data = data[:, :]

    data = np.squeeze(data)
    lat_size, lon_size = data.shape

    lats = np.zeros((lat_size, lon_size))
    lons = np.zeros((lat_size, lon_size))
    filtered_data = np.zeros_like(data)

    for i in range(lat_size):
        for j in range(lon_size):
            # Вычисляем расстояние от центра
            dx = (j - lon_size // 2) * pixel_size
            dy = (i - lat_size // 2) * pixel_size
            distance = sqrt(dx**2 + dy**2)
            bearing = (degrees(atan2(dx, dy)) + 360) % 360

            # Рассчитываем географические координаты
            lat, lon = calculate_destination(
                lat_center, lon_center, distance, bearing)

            lats[i, j] = lat
            lons[i, j] = lon
            filtered_data[i, j] = data[i, j]

    norm = Normalize(vmin=np.nanmin(data),
                     vmax=np.nanmax(data))

    masked_data = np.ma.masked_where(
        (filtered_data == 0) | (filtered_data == -32), filtered_data)

    fig = plt.figure(figsize=(10, 8), facecolor='none')
    plt.pcolormesh(lons, lats, masked_data,
                   shading='auto', norm=norm, cmap=custom_cmap)
    plt.axis('off')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    output_file = f"./{uuid.uuid4()}.png"
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    return output_file


def get_file_path(time_data, timestamp, all = False):
    for time_iso, folder_path in time_data:
        
        if time_iso == timestamp:
            print(1)
            if not all:
                zip_path = find_zip_file(folder_path)
                if not zip_path:
                    raise HTTPException(status_code=404, detail="ZIP file not found")
                
                nc_file_path = extract_nc_file(zip_path)
                if not nc_file_path:
                    raise HTTPException(status_code=404, detail="No .nc file found in ZIP")
                
                return nc_file_path
            else:
                return find_all_zip_files(folder_path)

def get_loc_file(location_list, locator_code):
    for loc in location_list:
        if locator_code in loc:
            return loc



@app.get("/plot")
async def get_plot(variable: str, locator_code: str = "", lat = 0, lon = 0, timestamp: str = Query(..., description="Timestamp in ISO format"), base_path: str = "path/to/your/folder"):
    try:
        time_data = parse_folder_structure('./periods')
        #print(time_data)
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
        output_file = plot_data_on_map(data_array, float(lat), float(lon))
        ds.close()

        return FileResponse(output_file, media_type="image/png", filename="map.png")
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
