import PropTypes from 'prop-types';


const Legend = ({ colorRange, variable }) => {
  if (!colorRange) return null;
  const { ranges, colors } = colorRange;

  return (
    <div className="legend-bar">
      <div className='flex'>
        {colors.map((color, idx) => (
          <div
            key={idx}
            style={{
              background: color,
              width: 30,
              height: 20,
              border: '1px solid #333',
              boxSizing: 'border-box'
            }}
            title={`${ranges[idx]}${ranges[idx + 1] !== undefined ? ' – ' + ranges[idx + 1] : '+'}`}
          />
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
        {ranges.map((r, idx) => (
          <span key={idx} style={{ width: 30, textAlign: 'end' }}>{r}</span>
        ))}
      </div>
      <div style={{ textAlign: 'center', fontSize: 14, marginTop: 4 }}>
        {variable}
      </div>
    </div>
  );
};

Legend.propTypes = {
  colorRange: PropTypes.shape({
    ranges: PropTypes.arrayOf(PropTypes.number).isRequired,
    colors: PropTypes.arrayOf(PropTypes.string).isRequired
  }),
  variable: PropTypes.string
};

export default Legend; 