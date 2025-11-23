import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

function App() {
  const [hotspots, setHotspots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selectedHotspot, setSelectedHotspot] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);

  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // Fetch all hotspots on initial load
  useEffect(() => {
    fetch('http://localhost:8000/api/hotspots')
      .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
      })
      .then(data => {
        setHotspots(data);
        setLoading(false);
      })
      .catch(error => {
        setError(error);
        setLoading(false);
      });
  }, []);

  const handleHotspotClick = (hotspot) => {
    setSelectedHotspot(hotspot);
    // Clear previous forecast and dates when a new hotspot is clicked
    setForecast(null);
    setStartDate('');
    setEndDate('');
  };

  const handleForecastFetch = () => {
    if (!selectedHotspot) {
      alert("Please select a hotspot on the map first.");
      return;
    }
    if (!startDate || !endDate) {
      alert("Please select both a start and end date.");
      return;
    }

    setForecastLoading(true);
    const url = `http://localhost:8000/api/forecast/${selectedHotspot.id}?start_date=${startDate}&end_date=${endDate}`;

    fetch(url)
      .then(response => {
        if (!response.ok) throw new Error('Forecast network response was not ok');
        return response.json();
      })
      .then(data => {
        setForecast(data);
        setForecastLoading(false);
      })
      .catch(error => {
        console.error("Error fetching forecast:", error);
        alert("Failed to fetch forecast. Check the console for details.");
        setForecastLoading(false);
      });
  };

  const chicagoPosition = [41.8781, -87.6298];

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI-Powered Crime Alert System</h1>
      </header>
      <main>
        <div className="map-container">
          {loading && <p>Loading map and hotspots...</p>}
          {error && <p>Error fetching data: {error.message}</p>}
          {!loading && !error && (
            <MapContainer center={chicagoPosition} zoom={11}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              {hotspots.map(hotspot => (
                <CircleMarker
                  key={hotspot.id}
                  center={[hotspot.centroid.lat, hotspot.centroid.lon]}
                  radius={Math.sqrt(hotspot.crime_count) / 2}
                  pathOptions={{ color: 'red', fillColor: '#f03', fillOpacity: 0.5 }}
                  eventHandlers={{ click: () => handleHotspotClick(hotspot) }}
                >
                  <Popup>
                    <b>Hotspot #{hotspot.id}</b><br />
                    Crime Count: {hotspot.crime_count}
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>
          )}
        </div>
        <div className="forecast-panel">
          <h2>Crime Forecast</h2>
          {selectedHotspot ? (
            <div>
              <h3>Selected Hotspot: #{selectedHotspot.id}</h3>
              <div className="date-picker">
                <label>
                  Start Date:
                  <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
                </label>
                <label>
                  End Date:
                  <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
                </label>
                <button onClick={handleForecastFetch} disabled={forecastLoading}>
                  {forecastLoading ? 'Loading...' : 'Get Forecast'}
                </button>
              </div>

              {forecast && forecast.forecast_period && (
                <div className="forecast-results">
                  <h4>
                    Predicted Crimes from {forecast.forecast_period.start} to {forecast.forecast_period.end}:
                    <strong> {Math.floor(forecast.total_predicted_crimes)}</strong>
                  </h4>

                  <div className="forecast-chart-container">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart
                        data={forecast.forecast_breakdown}
                        margin={{ top: 5, right: 20, left: -10, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" className='forecast-chart-grid'/>
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="predicted_crimes" stroke="#8884D8" activeDot={{ r: 8 }} name="Predicted Crimes" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <ul>
                    {forecast.forecast_breakdown.map(f => (
                      <li key={f.date}>
                        {f.date}: <strong>{Math.floor(f.predicted_crimes)}</strong>
                        {f.is_holiday_week && <span className="holiday-label">Holiday Week</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <p>Click on a hotspot on the map to select it.</p>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
