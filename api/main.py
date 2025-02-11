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

custom_colors_ZDR = ['#212892', '#105380', '#1CBEFE', '#2FE1FF', '#72FEFB', '#74FF88', '#7CB846', '#008601', '#FFE202', '#E0BF4C', '#D49802', '#FEA8A7', '#E6594F', '#F81600', '#B90B0A', '#BE96AE', '#7F586A']
custom_colors_FDP = ['#F665E6', '#FC90C2', '#FBB59C', '#FCD472', '#F2ED3B', '#D0E818', '#A6D912', '#7ACB0D', '#48B90F', '#35A43B', '#448A72', '#3B6FA1', '#2A4EC8', '#2827E4', '#591EF2', '#8535FA', '#AE40FB', '#DB48F9']
custom_colors_roHV = ['#441C02', '#833B00', '#C15700', '#FF8001', '#FFC0C0', '#70FFFF', '#01D8DB', '#00CB00', '#01E900', '#AAFE50', '#CBFE97', '#FFFE80']
custom_colors_SV = ['#E3F7F8', '#CBF8F3', '#9AECEA', '#81DE6C', '#F8F649', '#FDCE20', '#F9942A', '#F84E2D', '#E7294D', '#AE1D94']
custom_colors_TB = ['#E6E6E6', '#FDCD00', '#FF3300', '#CC0003']
custom_colors_DPmap = ['#9FA9B2', '#A3C6FF', '#45FF92', '#01C15A', '#009800', '#FFFF81', '#4088FE', '#0038FF', '#000074', '#FFAB7F', '#FF557F', '#FF0101', '#CA6702', '#894401', '#610000', '#FFAAFF', '#FF54FF', '#C600C7', '#43405D']
custom_colors_Hngo = ['#EAF4FE', '#CFE7FF', '#B9DBFE', '#9CCFFE', '#56ABFE', '#027FFF', '#006ADA', '#005FBD', '#0052A2', '#024289', '#027500', '#00AB01', '#25FF25', '#FFFF01', '#FF9899', '#FE0000', '#A60000', '#710100']
custom_colors_VIL = ['#DCEFFD', '#B1DAFA', '#88C4FF', '#4AA5FE', '#95FDFE', '#4BFFFF', '#01D8DA', '#00CB00', '#02EA00', '#A7FF50', '#CBFF98', '#FFFF81', '#FF9B8C', '#FF3F40', '#FA6AE3', '#C400C4', '#A60000', '#720000']
custom_colors_R = ['#9B9B9B', '#868887', '#0055FE', '#010080', '#FFFF00', '#C9EF04', '#FDAB00', '#FF5600', '#FE0000', '#81FF7F', '#02A901', '#FE83F5', '#D000D0']
custom_colors_Qp = ['#FFAAFF', '#D0E9FF', '#00AAFF', '#004CFE', '#0000C5', '#00007E', '#FEFF01', '#FF7F02', '#E5490C', '#FD0100', '#01BE00', '#007500', '#FF00FF', '#CC009D']
custom_colors_Hvgo = ['#07FEA5', '#1BF36C', '#1FBD82', '#229470', '#257164', '#164137', '#01CCEA', '#0287E4', '#0136D0', '#05176D', '#D3B51F', '#DA491A', '#D91A12', '#84050C', '#43A804', '#033F03', '#A31047', '#DB2F9D']

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

#custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", custom_colors_ZH)

def get_custom_cmap(variable:str =""):
    return LinearSegmentedColormap.from_list("custom_gradient", custom_colors_map[variable])


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
    Рассчитывает координаты точки назначения по расстоянию и направлению (bearing), 
    корректируя долготу с учетом широты.
    """
    R = 6371e3  # Радиус Земли в метрах

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    bearing = math.radians(bearing)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance / R) +
                     math.cos(lat1) * math.sin(distance / R) * math.cos(bearing))

    delta_lon = math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat1),
                           math.cos(distance / R) - math.sin(lat1) * math.sin(lat2))

    lon2 = lon1 + delta_lon / math.cos(lat1)  # Коррекция долготы!

    return math.degrees(lat2), math.degrees(lon2)


def plot_data_on_map(data_array, lat_center, lon_center, variable, pixel_size=1000, slice_index=3):
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

    # masked_data = np.ma.masked_where(
    #     (filtered_data == 0) | (filtered_data == -32), filtered_data)

    fig = plt.figure(figsize=(10, 8), facecolor='none')
    plt.pcolormesh(lons, lats, filtered_data,
                   shading='auto', norm=norm, cmap=get_custom_cmap(variable))
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
        output_file = plot_data_on_map(data_array, float(lat), float(lon), variable)
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
