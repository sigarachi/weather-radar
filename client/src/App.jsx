// React Frontend
import { useState } from 'react';
import { TileLayer, MapContainer, ImageOverlay, Marker, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css'

const App = () => {
  const [netcdfFile, setNetcdfFile] = useState(null);
  const [variables, setVariables] = useState([]);
  const [selectedVariable, setSelectedVariable] = useState(null);
  const [fileName, setFileName] = useState('');
  const [center, setCenter] = useState({ lat: 0, lon: 0 });
  const [mapImage, setMapImage] = useState(null);

  const handleFileUpload = async (event) => {
      const file = event.target.files[0];
      setNetcdfFile(file);
      setFileName(file.name)

      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          body: formData,
      });
      const variables = await response.json();

      setVariables(variables);
  };

  const handleVariableChange = async (event) => {
      const variable = event.target.value;
      setSelectedVariable(variable);

      const queryParams = new URLSearchParams({
          variable: variable,
          lat: center.lat,
          lon: center.lon,
          filename: fileName
      });

      const response = await fetch(`http://localhost:8000/plot?${queryParams}`);
      if (response.ok) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          setMapImage(url);
      } else {
          console.error('Failed to fetch map image');
      }
  };

  const handleCenterChange = (field, value) => {
      setCenter(prev => ({ ...prev, [field]: parseFloat(value) }));
  };

  return (
      <div className='page-wrapper'>
          <h1>NetCDF Viewer</h1>
          <div className='flex row'>
            <input type="file" accept=".nc" onChange={handleFileUpload} />
            {variables.length > 0 && (
                <select onChange={handleVariableChange}>
                    <option value="">Выберите переменную</option>
                    {variables.map((variable) => (
                        <option key={variable} value={variable}>{variable}</option>
                    ))}
                </select>
            )}
          </div>
          <div className='flex row' style={{
            gap: '12px'
          }}>
              <label>
                  Широта:
                  <input
                      type="number"
                      value={center.lat}
                      name="lat"
                      onChange={(e) => handleCenterChange('lat', e.target.value)}
                  />
              </label>
              <label>
                  Долгота:
                  <input
                      type="number"
                      value={center.lon}
                      name="lon"
                      onChange={(e) => handleCenterChange('lon', e.target.value)}
                  />
              </label>
          </div>
          <MapContainer center={[center.lat, center.lon]} zoom={6} style={{ height: '100%', width: '100%' }}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {mapImage && (
                  <ImageOverlay
                      url={mapImage}
                      bounds={[[center.lat - 5, center.lon - 5], [center.lat + 5, center.lon + 5]]}
                      opacity={10}
                  />
              )}
              <Marker position={[center.lat, center.lon]} />
          </MapContainer>
      </div>
  );
};

export default App;

