// React Frontend
import { useMemo, useState, useEffect, useCallback } from 'react';
import 'leaflet/dist/leaflet.css';
import './App.css';
import { ORIGIN } from './globals';
import { usePeriods } from './hooks/use-periods';
import { LOCATOR_MAP, VARIABLE_MAP, colorRanges } from './constants';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { ru } from 'date-fns/locale/ru';
import { ToastContainer, toast } from 'react-toastify';
import Legend from './components/legend';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';

//registerLocale('ru', ru);

const App = () => {
	const [variables, setVariables] = useState([]);

	const [selectedPeriod, setSelectedPeriod] = useState('');
	const [selectedDate, setSelectedDate] = useState();
	const [selectedVariable, setSelectedVariable] = useState('');
	const [locatorOptions, setLocatorOptions] = useState([]);
	const [locator, setLocator] = useState('');
	const [sliceIndex, setSliceIndex] = useState(1);
	const [tileLoading, setTileLoading] = useState(false);

	const locatorCords = useMemo(() => {
		return (
			LOCATOR_MAP.filter((el) => el.code === locator)[0]?.cords ?? {
				lat: 0,
				lng: 0,
			}
		);
	}, [locator]);

	const showSlice = useMemo(
		() => Boolean(selectedVariable === 'Zh' || selectedVariable === 'Zv'),
		[selectedVariable]
	);

	const { periods, isLoading } = usePeriods();

	const handleVariableChange = async (event) => {
		const variable = event.target.value;

		setSelectedVariable(variable);
	};

	const dateLimits = useMemo(
		() =>
			periods.length
				? [new Date(periods[0][0]), new Date(periods[periods.length - 1][0])]
				: [],
		[periods]
	);

	const handlePeriodChange = useCallback(
		async (selectedDate) => {
			if (!selectedDate) return;

			const variable = periods.find((el) => {
				const periodDate = new Date(el[0]);
				const selectedDateObj = new Date(selectedDate);

				return periodDate.getTime() === selectedDateObj.getTime();
			});

			if (!variable) {
				toast('Не удалось найти данные для выбранной даты и времени', {
					type: 'error',
				});
				return;
			}

			setSelectedPeriod(variable);

			const queryParams = new URLSearchParams({
				timestamp: variable[0],
				base_path: variable[1],
			});

			const response = await fetch(`${ORIGIN}/list_files?${queryParams}`);
			if (response.ok) {
				const data = await response.json();
				setLocatorOptions(
					LOCATOR_MAP.filter((el) => {
						if (data.files.find((file) => file.includes(el.code))) return true;
						return false;
					})
				);
			} else {
				console.error('Failed to fetch periods');
			}
		},
		[periods]
	);

	useEffect(() => {
		if (selectedVariable && selectedPeriod && locator) {
			console.log('Updating tiles for period:', selectedPeriod[0]);
		}
	}, [selectedVariable, selectedPeriod, locator, sliceIndex]);

	const handleLocatorChange = async (event) => {
		const variable = event.target.value;
		setLocator(variable);

		const queryParams = new URLSearchParams({
			timestamp: selectedPeriod[0],
			locator_code: variable,
			base_path: selectedPeriod[1],
		});

		const response = await fetch(`${ORIGIN}/variables?${queryParams}`);
		if (response.ok) {
			const data = await response.json();
			setVariables(data);
		} else {
			console.error('Failed to fetch periods');
		}
	};

	return (
		<div className="page-wrapper">
			<ToastContainer />
			<div className="controls-overlay">
				<h1>NetCDF Viewer</h1>
				<div className="control-group">
					{Boolean(!isLoading && periods.length && dateLimits.length) && (
						<>
							<label>Выберите дату и время</label>
							<LocalizationProvider
								dateAdapter={AdapterDayjs}
								adapterLocale={ru}>
								<DateTimePicker
									value={selectedDate}
									views={['year', 'day', 'hours', 'minutes']}
									format="DD/MM/YYYY HH:mm"
									ampm={false}
									onChange={(value) => {
										setSelectedDate(value);
										handlePeriodChange(value);
									}}
								/>
							</LocalizationProvider>
						</>
					)}
				</div>
				{locatorOptions.length > 0 && (
					<div className="control-group">
						<label>Выберите локатор</label>
						<select onChange={handleLocatorChange}>
							<option value="">Выберите локатор</option>
							{locatorOptions.map((variable) => (
								<option key={variable.code} value={variable.code}>
									{variable.name}
								</option>
							))}
						</select>
					</div>
				)}
				{variables.length > 0 && (
					<div className="control-group">
						<label>Выберите переменную</label>
						<select onChange={handleVariableChange}>
							<option value="">Выберите переменную</option>
							{variables.map((variable) => (
								<option key={variable} value={variable}>
									{VARIABLE_MAP[variable]}
								</option>
							))}
						</select>
					</div>
				)}
				{showSlice && (
					<div className="control-group">
						<label>Выберите срез</label>
						<select
							value={sliceIndex}
							onChange={(event) => setSliceIndex(event.target.value)}>
							{Array.from(Array(15).keys()).map((val) => (
								<option key={val} value={val + 1}>
									{val + 1}
								</option>
							))}
						</select>
					</div>
				)}
			</div>
			<MapContainer
				center={[55.75, 37.62]}
				zoom={10}
				minZoom={7}
				maxZoom={12}
				style={{ height: '100vh', width: '100%', zIndex: '1' }}>
				<TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
				{tileLoading && (
					<div className="map-loading-overlay">
						<div className="spinner"></div>
					</div>
				)}
				{selectedVariable && selectedPeriod && (
					<TileLayer
						key={`${selectedVariable}-${selectedPeriod[0]}-${sliceIndex}`}
						opacity={0.6}
						eventHandlers={{
							loading: () => setTileLoading(true),
							load: () => setTileLoading(false),
						}}
						url={`${ORIGIN}/tiles/{z}/{x}/{y}?${new URLSearchParams({
							variable: selectedVariable,
							locator_code: locator,
							timestamp: selectedPeriod[0],
							lat: locatorCords.lat,
							lon: locatorCords.lng,
							slice_index: sliceIndex,
						})}`}
					/>
				)}
			</MapContainer>
			{selectedVariable && colorRanges[selectedVariable] && (
				<div className="legend-overlay">
					<Legend
						colorRange={colorRanges[selectedVariable]}
						variable={selectedVariable}
						period={new Date(selectedPeriod[0]).toLocaleDateString('ru-RU', {
							year: '2-digit',
							day: '2-digit',
							month: '2-digit',
							minute: 'numeric',
							hour: 'numeric',
						})}
						locator={
							LOCATOR_MAP.filter((el) => el.code === locator)[0].name ?? ''
						}
					/>
				</div>
			)}
		</div>
	);
};

export default App;
