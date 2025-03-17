// React Frontend
import { useMemo, useState } from 'react';
import 'leaflet/dist/leaflet.css';
import './App.css'
import { ORIGIN } from './globals';
import { usePeriods } from './hooks/use-periods';
import { IMAGE_MAP, LOCATOR_MAP, VARIABLE_MAP } from './constants';
import DatePicker, {registerLocale} from 'react-datepicker'
import { MapContainer, TileLayer } from 'react-leaflet';
import "leaflet/dist/leaflet.css";
import "react-datepicker/dist/react-datepicker.css";
import ru from 'date-fns/locale/ru'

registerLocale('ru', ru)


const App = () => {
 
  const [variables, setVariables] = useState([]);
  
  const [selectedPeriod, setSelectedPeriod] = useState('');
  const [selectedDate, setSelectedDate] = useState()
  const [selectedVariable, setSelectedVariable] = useState('');
  const [locatorOptions, setLocatorOptions] = useState([]);
  const [locator, setLocator] = useState('');
  const [sliceIndex, setSliceIndex] = useState(1);

  //const [shapes, setShapes] = useState([]);

  const locatorCords = useMemo(() => {
    return LOCATOR_MAP.filter((el) => el.code === locator)[0]?.cords ?? {lat: 0, lng: 0}
  }, [locator]);

  const showSlice = useMemo(() => Boolean(selectedVariable === 'Zh' || selectedVariable === 'Zv'), [selectedVariable])

  const {periods, isLoading} = usePeriods();
  //const { data } = useVariable(selectedVariable, locator, locatorCords, selectedPeriod[0], sliceIndex);

  const handleVariableChange = async (event) => {
      const variable = event.target.value;

      setSelectedVariable(variable);
  };

  const dateLimits = useMemo(() => periods.length ? [
    new Date(periods[0][0]), new Date(periods[periods.length - 1][0])
  ] : [], [periods]);


  const handlePeriodChange = async () => {
    const variable = periods.find((el) => new Date(el[0]).toLocaleString("ru-RU", {timeZone: 'Europe/Moscow'}) === selectedDate.toLocaleString("ru-RU", {timeZone: 'Europe/Moscow'}));

    if(!variable) return

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
          <div className='flex row'  style={{paddingLeft: '100px'}}>
            {/* {Boolean(!isLoading && periods.length) && 
                <Selector options={periods.map((item) => ({value: item, name: new Date(item[0]).toLocaleDateString('ru-RU', {minute: '2-digit', hour: '2-digit', second: '2-digit'})}))} onChange={handlePeriodChange} value={selectedPeriod} />
            } */}
            {Boolean(!isLoading && periods.length && dateLimits.length) && <DatePicker selected={selectedDate} includeDates={dateLimits} dateFormat="dd/MM/YYYY"  locale="ru" showTimeSelect timeIntervals={10} onChange={(e) => {
                setSelectedDate(e);
                handlePeriodChange()
            }}  />}
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
            {showSlice && 
                <select value={sliceIndex} onChange={(event) => setSliceIndex(event.target.value)}>
                    {Array.from(Array(15).keys()).map((val) => (
                         <option key={val} value={val+1}>{val+1}</option>
                    ))}
                </select>
            }
          </div>
           <MapContainer center={[55.75, 37.62]} zoom={10}  style={{ height: '100vh', width: '100%', zIndex: '1' }}>
                  <TileLayer 
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; OpenStreetMap contributors'
                  />
                  {
                    selectedVariable && <TileLayer opacity={0.6} url={`${ORIGIN}/tiles/{z}/{x}/{y}?${new URLSearchParams({
                        variable: selectedVariable,
                        locator_code: locator,
                        timestamp: selectedPeriod[0],
                        lat: locatorCords.lat,
                        lon: locatorCords.lng,
                        slice_index: sliceIndex
                    })}`} />
                  }
                {/* {data && data.map((shape, idx) => {
                    // shape.coordinates может быть списком многоугольников
                    return shape.polygon.coordinates.map((coords, subIdx) => (
                        <Polygon
                            key={`${idx}-${subIdx}`}
                            positions={coords.map(coord => [coord[0], coord[1]])} // формат [lat, lon]
                            pathOptions={{
                                color: shape.color,
                                fillColor: shape.color,
                                fillOpacity: 0.1,
                                weight: 2
                            }}
                        />
                    ));
                })} */}
            </MapContainer>
          {selectedVariable && 
            <div style={{position: 'absolute', zIndex: 1000, bottom: 0}}>
                <img src={`/img/variables/${IMAGE_MAP[selectedVariable]}`} />
            </div>
          }
      </div>
  );
};

export default App;

