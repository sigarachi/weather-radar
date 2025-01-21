from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import xarray as xr
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from math import radians, degrees, sin, cos, atan2, sqrt, asin
from PIL import Image
from environs import Env
import math
import os

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


def plot_data_on_map(data_array, lat_center, lon_center, pixel_size=1000, slice_index=0):
    shape = data_array.shape
    data = data_array[:]
    # max_distance = resolution_km * shape[0]

    if len(shape) == 3:  # Если данные имеют временное измерение
        if slice_index >= shape[0]:
            print(f"Индекс временного среза {
                  slice_index} выходит за пределы первого измерения ({shape[0]}).")
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
                   shading='auto', norm=norm)
    plt.axis('off')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    output_file = "output_map.png"
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    return output_file


@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        # Save file locally
        file_location = f"./${file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Open the NetCDF file and extract variables
        ds = xr.open_dataset(file_location)
        variables = list(ds.data_vars.keys())

        return JSONResponse(content=variables, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/plot")
async def get_plot(variable: str, lat: float, lon: float, filename: str):
    try:
        # Open the NetCDF file
        file_location = f"./${filename}"  # Replace with actual file location
        ds = Dataset(file_location, mode="r")

        print(os.environ.get("TCL_LIBRARY"))
        print(os.environ.get("TK_LIBRARY"))

        if variable not in ds.variables:
            return JSONResponse(content={"error": "Variable not found"}, status_code=400)

        data_array = ds.variables[variable]
        # shape = data_array.shape

        # Generate the plot
        output_file = plot_data_on_map(data_array, lat, lon)

        return FileResponse(output_file, media_type="image/png", filename="map.png")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
