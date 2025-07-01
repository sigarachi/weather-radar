import { useCallback, useEffect, useState } from 'react';
import { ORIGIN } from '../globals';

export const useVariable = (
	variable,
	locator,
	cords,
	timestamp,
	sliceIndex
) => {
	const [loading, setLoading] = useState(false);
	const [data, setData] = useState(null);
	const [error, setError] = useState('');

	const getVariableData = useCallback(async () => {
		try {
			if (!variable) return;
			setLoading(true);
			const queryParams = new URLSearchParams({
				variable: variable,
				locator_code: locator,
				timestamp: timestamp,
				lat: cords.lat,
				lon: cords.lng,
				slice_index: sliceIndex,
				base_path: './periods',
			});

			const response = await fetch(`${ORIGIN}/plot?${queryParams}`);
			const json = await response.json();
			const data = JSON.parse(json);

			setData(data.shapes);
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	}, [cords.lat, cords.lng, locator, sliceIndex, timestamp, variable]);

	useEffect(() => {
		getVariableData();
	}, [getVariableData]);

	return {
		data,
		loading,
		error,
	};
};
