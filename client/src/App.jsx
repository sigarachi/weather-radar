// React Frontend
import { useMemo, useState } from 'react';
import { TileLayer, MapContainer, ImageOverlay, Marker } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css'
import { ORIGIN } from './globals';
import { usePeriods } from './hooks/use-periods';
import { LOCATOR_MAP, VARIABLE_MAP } from './constants';
import { Selector } from './components/selector';

const App = () => {
 
  const [variables, setVariables] = useState([]);
  
  const [center, setCenter] = useState({ lat: 0, lon: 0 });
  const [mapImage, setMapImage] = useState(null);
  const [selectedPeriod, setSelectedPeriod] = useState('');
  const [locatorOptions, setLocatorOptions] = useState([]);
  const [locator, setLocator] = useState('');

  const locatorCords = useMemo(() => {
    return LOCATOR_MAP.filter((el) => el.code === locator)[0]?.cords ?? {lat: 0, lng: 0}
  }, [locator]);

  const {periods, isLoading, isError, error} = usePeriods();

  const handleVariableChange = async (event) => {
      const variable = event.target.value;
     

      const queryParams = new URLSearchParams({
          variable: variable,
          locator_code: locator,
          timestamp: selectedPeriod[0],
          lat: locatorCords.lat,
          lon: locatorCords.lng,
          base_path: './periods'
      });

      const response = await fetch(`${ORIGIN}/plot?${queryParams}`);
      if (response.ok) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          setMapImage(url);
      } else {
          console.error('Failed to fetch map image');
      }
  };

  const handlePeriodChange = async (event) => {
    const variable = event.target.value.split(',');
    setSelectedPeriod(variable)
   

    const queryParams = new URLSearchParams({
        timestamp: variable[0],
        base_path: variable[1]
    });

    const response = await fetch(`${ORIGIN}/list_files?${queryParams}`);
    if (response.ok) {
        const data = await response.json();
        setLocatorOptions(LOCATOR_MAP.filter((el) => {
            console.log(el)
            if(data.files.find((file) => file.includes(el.code))) return true

            return false
        }))

    } else {
        console.error('Failed to fetch periods');
    }
  };

  const handleLocatorChange = async (event) => {
    const variable = event.target.value;
    setLocator(variable)
   

    const queryParams = new URLSearchParams({
        timestamp: selectedPeriod[0],
        locator_code: variable,
        base_path: selectedPeriod[1]
    });

    const response = await fetch(`${ORIGIN}/variables?${queryParams}`);
    if (response.ok) {
        const data = await response.json();
        setVariables(data)

    } else {
        console.error('Failed to fetch periods');
    }
  };

  return (
      <div className='page-wrapper'>
          <h1>NetCDF Viewer</h1>
          <div className='flex row'>
            {Boolean(!isLoading && periods.length) && 
                <Selector options={periods.map((item) => ({value: item, name: new Date(item[0]).toLocaleDateString('ru-RU', {minute: '2-digit', hour: '2-digit', second: '2-digit'})}))} onChange={handlePeriodChange} value={selectedPeriod} />
            }
            {locatorOptions.length > 0 && (
                <select onChange={handleLocatorChange}>
                    <option value="">Выберите локатор</option>
                    {locatorOptions.map((variable) => (
                        <option key={variable.code} value={variable.code}>{variable.name}</option>
                    ))}
                </select>
            )}
            {variables.length > 0 && (
                <select onChange={handleVariableChange}>
                    <option value="">Выберите переменную</option>
                    {variables.map((variable) => (
                        <option key={variable} value={variable}>{VARIABLE_MAP[variable]}</option>
                    ))}
                </select>
            )}
          </div>
          <MapContainer center={[center.lat,center.lon]} zoom={6} style={{ height: '100%', width: '100%' }}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {mapImage && (
                  <ImageOverlay
                      url={mapImage}
                      bounds={[[locatorCords.lat - 5, locatorCords.lng - 5], [locatorCords.lat + 5, locatorCords.lng + 5]]}
                      opacity={10}
                  />
              )}
              <Marker position={locatorCords} />
          </MapContainer>
      </div>
  );
};

export default App;

