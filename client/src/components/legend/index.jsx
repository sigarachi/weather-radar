import PropTypes from 'prop-types';
import { DPmap_CODES } from '../../constants';

const Legend = ({ colorRange, variable }) => {
  if (!colorRange) return null;
  const { ranges, colors } = colorRange;

  const formatValue = (value) => {
    if (variable === 'DPmap') {
      return `${value}: ${DPmap_CODES[value] || 'Неизвестное явление'}`;
    }
    return value;
  };

  return (
    <div className="legend-bar">
      <div className='flex legend-bar-colors'>
        {colors.map((color, idx) => (
          <div
            key={idx}
            style={{
              background: color,
              width: 30,
              height: 20,
              border: '1px solid #333',
              boxSizing: 'border-box',
              position: 'relative'
            }}
          />
        ))}
      </div>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        fontSize: 12,
        position: 'relative',
        marginTop: 4,
        height: 20
      }}>
        {ranges.map((r, idx) => (
          <div 
            key={idx} 
            style={{ 
              position: 'absolute',
              left: `${(idx * 30)}px`,
              width: 30,
              textAlign: 'center',
              whiteSpace: 'nowrap'
            }}
          >
            {formatValue(r)}
          </div>
        ))}
      </div>
      <div style={{ textAlign: 'center', fontSize: 14, marginTop: 20 }}>
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