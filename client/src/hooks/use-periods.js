import PropTypes from 'prop-types';
import { useCallback, useEffect, useState } from 'react';
import { ORIGIN } from '../globals';

export const usePeriods = () => {
    const [loading, setLoading] = useState(false);
    const [periods, setPeriods] = useState([]);
    const [error, setError] = useState('');

    const getTimePeriods = useCallback(async () => {
        try {
            setLoading(true);
            const response = await fetch(`${ORIGIN}/time-periods`);
            if(!response.ok) {
                throw Error(response.statusText)
            }

            const data = await response.json();
            
            setPeriods(JSON.parse(data).time_periods);
        } catch(err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [])

    useEffect(() => {
        getTimePeriods();
    }, [getTimePeriods])

    return {
        periods,
        isLoading: loading,
        isError: error.length,
        error
    }
}

usePeriods.propTypes = {
    periods: PropTypes.arrayOf(PropTypes.string),
    isLoading: PropTypes.bool,
    isError: PropTypes.bool,
    error: PropTypes.string
}