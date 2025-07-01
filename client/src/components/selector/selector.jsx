import PropTypes from 'prop-types';

export const Selector = ({ value, options, onChange }) => {
	return (
		<select value={value} onChange={onChange}>
			<option value="">Выберите период</option>
			{options.map((variable) => (
				<option key={variable.value} value={variable.value}>
					{variable.name}
				</option>
			))}
		</select>
	);
};

Selector.propTypes = {
	options: PropTypes.arrayOf(PropTypes.object),
	value: PropTypes.string,
	onChange: PropTypes.func,
};
