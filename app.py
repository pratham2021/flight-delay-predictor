import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgbm
import datetime
import requests
import pytz
from timezonefinder import TimezoneFinder
from math import radians, sin, cos, sqrt, atan2
from datetime import timedelta
import pydeck as pdk
import time

st.set_page_config("US Flight Delay Predictor")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_items():
    tf = TimezoneFinder()
    model = joblib.load('models/best_model.pkl')
    
    le_carrier = joblib.load('encodings/le_carrier.pkl')
    le_origin_state = joblib.load('encodings/le_origin_state.pkl')
    le_dest_state = joblib.load('encodings/le_dest_state.pkl')
    
    origin_te = joblib.load('encodings/origin_te.pkl')
    dest_te = joblib.load('encodings/dest_te.pkl')
    route_te = joblib.load('encodings/route_te.pkl')
    
    origin_hourly_avg = joblib.load('encodings/origin_hourly_avg.pkl')
    dest_hourly_avg = joblib.load('encodings/dest_hourly_avg.pkl')
    route_hourly_avg = joblib.load('encodings/route_hourly_avg.pkl')
    
    carrier_delay_map = joblib.load('encodings/carrier_delay_map.pkl')
    origin_delay_map = joblib.load('encodings/origin_delay_map.pkl')
    dest_delay_map = joblib.load('encodings/dest_delay_map.pkl')
    route_delay_map = joblib.load('encodings/route_delay_map.pkl')    
    best_threshold = joblib.load('models/best_threshold.pkl')
    
    duration_reg = joblib.load('models/duration_regressor.pkl')
    
    return model, duration_reg, le_carrier, le_origin_state, le_dest_state, origin_te, dest_te, route_te, origin_hourly_avg, dest_hourly_avg, route_hourly_avg, carrier_delay_map, origin_delay_map, dest_delay_map, route_delay_map, best_threshold, tf

model, duration_reg, le_carrier, le_origin_state, le_dest_state, origin_te, dest_te, route_te, origin_hourly_avg, dest_hourly_avg, route_hourly_avg, carrier_delay_map, origin_delay_map, dest_delay_map, route_delay_map, best_threshold, tf = load_items()

def isHolidayPeriod(date):
    if date.month == 1 and 15 <= date.day <= 20:
        return 1
    if date.month == 2 and 13 <= date.day <= 17:
        return 1
    if (date.month == 3 and date.day >= 15) or (date.month == 4 and date.day <= 15):
        return 1
    if date.month == 5 and 23 <= date.day <= 27:
        return 1
    if date.month == 7 and 1 <= date.day <= 7:
        return 1
    if (date.month == 8 and date.day >= 29) or (date.month == 9 and date.day <= 2):
        return 1
    if (date.month == 11 and date.day >= 20) or (date.month == 12 and date.day <= 1):
        return 1
    if (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 5):
        return 1
    return 0

def get_departures(origin_iata, from_time, to_time, api_key):
    url = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/{origin_iata}/{from_time}/{to_time}"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "aerodatabox.p.rapidapi.com"
    }
    params = {
        "withLeg": "true",
        "direction": "Departure",
        "withCancelled": "true",
        "withCodeshared": "true"
    }
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def build_time_window(date_str, local_hour, window_hours=1):
    base = datetime.datetime.strptime(f"{date_str} {local_hour}:00", "%Y-%m-%d %H:%M")
    start = base - timedelta(hours=window_hours)
    end = base + timedelta(hours=window_hours)
    return start.strftime("%Y-%m-%dT%H:%M"), end.strftime("%Y-%m-%dT%H:%M")

def find_matching_flight(departures_response, destination_iata, airline_iata, target_date, target_hour, target_minute, distance, duration_reg, minute_tolerance=10):
    matches = []
    expected_duration = duration_reg.predict([[distance]])[0]
    target_total_minutes = target_hour * 60 + target_minute

    for flight in departures_response.get("departures", []):
        arr_iata = flight.get("arrival", {}).get("airport", {}).get("iata")
        plane_iata = flight.get("airline", {}).get("iata", "")
        if arr_iata != destination_iata or plane_iata != airline_iata:
            continue

        dep_local_str = flight.get("departure", {}).get("scheduledTime", {}).get("local")
        if not dep_local_str:
            continue

        dep_local_dt = datetime.datetime.strptime(dep_local_str[:16], "%Y-%m-%d %H:%M")
        if dep_local_dt.date() != target_date:
            continue

        flight_total_minutes = dep_local_dt.hour * 60 + dep_local_dt.minute
        minute_diff = abs(flight_total_minutes - target_total_minutes)

        if minute_diff <= minute_tolerance:
            dep_str = flight.get("departure", {}).get("scheduledTime", {}).get("utc")
            arr_str = flight.get("arrival", {}).get("scheduledTime", {}).get("utc")

            if dep_str and arr_str:
                dep_utc = datetime.datetime.fromisoformat(dep_str.replace("Z", "+00:00"))
                arr_utc = datetime.datetime.fromisoformat(arr_str.replace("Z", "+00:00"))
                implied_duration = (arr_utc - dep_utc).total_seconds() / 60
                deviation = abs(implied_duration - expected_duration)
            else:
                deviation = float('inf')

            matches.append((minute_diff, deviation, dep_local_dt, flight))

    if not matches:
        return None

    matches.sort(key=lambda x: (x[0], x[1]))
    return matches[0][3]

@st.cache_data(ttl=3600)
def cached_get_departures(origin_iata, from_time, to_time, api_key):
    return get_departures(origin_iata, from_time, to_time, api_key)

@st.cache_data
def get_cleaned_airports():
    airport_data = pd.read_csv('airports/airports_cleaned.csv')
    return airport_data

airport_data = get_cleaned_airports()

@st.cache_data
def get_sorted_airport_codes():
    return sorted(airport_data['iata_code'].tolist())
airport_codes = get_sorted_airport_codes()

@st.cache_data
def get_airport_display_names():  
    return {row['iata_code']: f"{row['iata_code']} - {row['name']}" for _, row in airport_data.iterrows()}

airport_names = get_airport_display_names()

@st.cache_data
def get_airport_timezone(iata_code, _timeZoneFinder):
    row = airport_data[airport_data['iata_code'] == iata_code].iloc[0]
    timezone_str = _timeZoneFinder.timezone_at(lat=row['latitude_deg'], lng=row['longitude_deg'])
    return timezone_str, row['latitude_deg'], row['longitude_deg']

st.markdown("<h1 style='text-align: center;'>US Flight Delay Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Will your flight be delayed?</h3>", unsafe_allow_html=True)

airline_names = { 
    'AA':'American Airlines', 
    'AS': 'Alaska Airlines', 
    'B6':'JetBlue Airways', 
    'DL':'Delta Air Lines', 
    'F9':'Frontier Airlines',
    'G4':'Allegiant Air',
    'HA':'Hawaiian Airlines',
    'MQ':'Envoy Air (American Eagle)',
    'NK':'Spirit Airlines',
    'OH':'PSA Airlines',
    'OO':'Skywest Airlines',
    'UA':'United Airlines',
    'WN':'Southwest Airlines',
    'YX': 'Midwest Express / Republic Airways'
}

col1, col2 = st.columns([1, 1])

with col1:
    origin = st.selectbox(
        "Origin",
        options=airport_codes,
        index=airport_codes.index('AUS'),
        format_func=lambda x: airport_names.get(x, x)
    )
    
    timezone_str, lat, long = get_airport_timezone(origin, tf)
    local_tz = pytz.timezone(timezone_str)
    today_local = datetime.datetime.now(local_tz).date()
    departure_date = st.date_input("Departure Date", min_value=today_local, value=today_local)

with col2:    
    destination = st.selectbox(
        "Destination",
        options=airport_codes,
        index=airport_codes.index('SLC'),
        format_func=lambda x: airport_names.get(x, x)
    )
    
    airline = st.selectbox(
        "Airline",
        options=['AA', 'AS', 'B6', 'DL', 'F9', 'G4', 'HA', 'MQ', 'NK', 'OH', 'OO', 'UA', 'WN', 'YX'],
        format_func=lambda x: airline_names[x],
        index = 3
    )

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    timezone_str, lat, long = get_airport_timezone(origin, tf)
    local_tz = pytz.timezone(timezone_str)
    current_local_hour = datetime.datetime.now(local_tz).hour
    current_local_minute = datetime.datetime.now(local_tz).minute
    
    if departure_date != today_local:
        departure_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=current_local_hour, step=1)
    else:
        departure_hour = st.slider("Departure Hour", min_value=current_local_hour, max_value=23, value=current_local_hour, step=1)
        
    departure_minute = st.slider("Departure Minute", min_value=0, max_value=59, value=0, step=1)
    
st.divider()

_, center_col, _ = st.columns([1, 2, 1])

def haversine_distance(lat1, long1, lat2, long2):
    R = 3958.8 # Earth radius in miles
    latitude_1, longitude_1, latitude_2, longitude_2 = map(radians, [lat1, long1, lat2, long2])
    dlat = latitude_2 - latitude_1
    dlon = longitude_2 - longitude_1
    a = sin(dlat/2) ** 2 + cos(latitude_1) * cos(latitude_2) * sin(dlon/2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def estimate_duration_from_distance(distance):
    prediction = duration_reg.predict([[distance]])[0]
    return int(prediction)

def get_aircraft_current_flight(aircraft_reg, api_key):
    if not aircraft_reg:
        return None
    url = f"https://aerodatabox.p.rapidapi.com/flights/reg/{aircraft_reg}"
    headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "aerodatabox.p.rapidapi.com"}
    params = {"withLocation": "true"}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code == 429:
        time.sleep(1.5)
        resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def find_inbound_leg(flights, origin_iata):
    for flight in flights:
        arr_iata = flight.get("arrival", {}).get("airport", {}).get("iata")
        if arr_iata == origin_iata and "location" in flight:
            return flight  # this is the inbound plane
    return None

def get_flight_track(icao24, api_key=None):
    if not icao24:
        return None
    url = "https://opensky-network.org/api/tracks/all"
    params = { "icao24": icao24.lower(), "time": 0 }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None

def extract_track_points(track_data):
    if not track_data or "path" not in track_data:
        return None
    points = []
    for waypoint in track_data["path"]:
        # waypoint: [time, lat, lon, baro_altitude, true_track, on_ground]
        lat, lon = waypoint[1], waypoint[2]
        if lat is not None and lon is not None:
            points.append([lon, lat])
    return points if points else None

def great_circle_points(lat1, lon1, lat2, lon2, num_points=50):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d = 2 * np.arcsin(np.sqrt( 
        np.sin((lat2 - lat1) / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    ))
    
    if d == 0:
        return [[np.degrees(lon1), np.degrees(lon2)]]

    points = []
    
    for i in range(num_points + 1):
        f = i / num_points
        a = np.sin((1 - f) * d) / np.sin(d)
        b = np.sin(f * d) / np.sin(d)
        x = a * np.cos(lat1) * np.cos(lon1) + b * np.cos(lat2) * np.cos(lon2)
        y = a * np.cos(lat1) * np.sin(lon1) + b * np.cos(lat2) * np.sin(lon2)
        z = a * np.sin(lat1) + b * np.sin(lat2)
        lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        lon = np.degrees(np.arctan2(y, x))
        points.append([lon, lat])

    return points


def show_flight_path(origin_lat, origin_lon, dest_lat, dest_lon, real_path_points=None, inbound_dep_lat=None, inbound_dep_lon=None, inbound_pos_lat=None, inbound_pos_lon=None):

    if real_path_points and len(real_path_points) >= 2:
        curve_points = real_path_points
    else:
        curve_points = great_circle_points(origin_lat, origin_lon, dest_lat, dest_lon)
    
    path_data = pd.DataFrame({'path': [curve_points]})
    line_layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_width=3,
        get_color=[0, 168, 107],
        width_min_pixels=2,
    )
    
    layers = [line_layer]
    
    last_point = curve_points[-1]
    last_lon, last_lat = last_point[0], last_point[1]
    
    last_point_data = pd.DataFrame({'lat': [last_lat], 'lon': [last_lon]})
    
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=last_point_data,
        get_position=["lon", "lat"],
        get_radius=1500,
        radius_min_pixels=8,
        get_fill_color=[255, 0, 0],
    ))
    
    all_lats = [origin_lat, dest_lat] + ([inbound_dep_lat] if inbound_dep_lat is not None else [])
    all_lons = [origin_lon, dest_lon] + ([inbound_dep_lon] if inbound_dep_lon is not None else [])
    
    view_state = pdk.ViewState(
        latitude=sum(all_lats) / len(all_lats),
        longitude=sum(all_lons) / len(all_lons),
        zoom=3,
    )
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9"
    ))

with center_col:
    if st.button("Predict", use_container_width=True):        
        if not origin:
            st.error("You must enter an origin airport!")
            st.stop()
        if not destination:
            st.error("You must enter a destination airport!")
            st.stop()
        if not airline:    
            st.error("You must enter an airline!")
            st.stop()
        
        if departure_date == today_local and departure_hour < current_local_hour:
            st.warning("Please select a current or future departure hour!")
            st.stop()
        
        local_dt = datetime.datetime(departure_date.year, departure_date.month, departure_date.day, departure_hour)
        local_dt_aware = local_tz.localize(local_dt)
        utc_dt = local_dt_aware.astimezone(pytz.utc)
        utc_hour = utc_dt.hour
            
        latitude = airport_data[airport_data['iata_code'] == origin]['latitude_deg'].values[0]
        longitude = airport_data[airport_data['iata_code'] == origin]['longitude_deg'].values[0]
        
        destination_latitude = airport_data[airport_data['iata_code'] == destination]['latitude_deg'].values[0]
        destination_longitude = airport_data[airport_data['iata_code'] == destination]['longitude_deg'].values[0]
        
        distance = haversine_distance(latitude, longitude, destination_latitude, destination_longitude)
        
        today = datetime.date.today()
        departure_date_str = departure_date.strftime('%Y-%m-%d')
        
        if departure_date > today_local + datetime.timedelta(days=16):
            st.error("Weather forecast is only available up to 16 days in advance. Please select an earlier date.")
            st.stop()
        else:
            base_url = "https://api.open-meteo.com/v1/forecast"
        
        with st.spinner("Checking live flight schedule..."):
            try:
                from_time, to_time = build_time_window(departure_date_str, departure_hour, window_hours=1)
                departures = get_departures(origin, from_time, to_time, st.secrets["AERODATABOX_KEY"])
                target_local_dt = datetime.datetime(
                    departure_date.year, departure_date.month, departure_date.day, departure_hour
                )
                print(departures)
                match = find_matching_flight(departures, destination, airline, departure_date, departure_hour, departure_minute, distance, duration_reg)
            except requests.exceptions.RequestException as e:
                st.warning(f"Live lookup unavailable ({e}) — using historical averages instead.")
                match = None
        
        if match:
            st.success(f"Found matching scheduled flight: {match.get('number', 'N/A')}")
            
            estimated_duration = estimate_duration_from_distance(distance)
            
            hour = estimated_duration // 60
            minutes = estimated_duration % 60
            
            if hour == 0:
                st.write(f"Flight Time: {minutes} minutes")
            elif hour == 1:
                if minutes == 0:
                    st.write(f"Flight Time: {hour} hour")
                else:
                    st.write(f"Flight Time: {hour} hour and {minutes} minutes")
            else:
                st.write(f"Flight Time: {hour} hour and {minutes} minutes")
            
            aircraft_reg = match.get('aircraft', {}).get('reg')
            inbound_flight = None
            if aircraft_reg:
               aircraft_flights = get_aircraft_current_flight(aircraft_reg, st.secrets["AERODATABOX_KEY"])
               if aircraft_flights:
                    inbound_flight = find_inbound_leg(aircraft_flights, origin)

            if inbound_flight:
                dep_airport = inbound_flight['departure']['airport']['name']
                dep_time = inbound_flight['departure']['scheduledTime']['local']
                inbound_pos = inbound_flight.get('location')
                
                inbound_dep_lat = inbound_flight['departure']['airport']['location']['lat']
                inbound_dep_lon = inbound_flight['departure']['airport']['location']['lon']
            
                if inbound_pos:
                    inbound_pos_lat = inbound_pos.get('lat')
                    inbound_pos_lon = inbound_pos.get('lon')
                    st.info(f"Your aircraft is currently airborne, inbound from {dep_airport}.")
                    
                    icao24 = match.get('aircraft', {}).get('modeS')
                    track_data = get_flight_track(icao24)
                    real_path_points = extract_track_points(track_data)
                    
                    show_flight_path(latitude, longitude, destination_latitude, destination_longitude, real_path_points=real_path_points, 
                                     inbound_dep_lat=inbound_dep_lat, inbound_dep_lon=inbound_dep_lon, inbound_pos_lat=inbound_pos_lat, inbound_pos_lon=inbound_pos_lon)
                else:
                    st.info(f"Your aircraft is scheduled to arrive from {dep_airport}, departing {dep_time}.")
            else:
                st.caption("No inbound aircraft data available. Aircraft is likely already at the gate.")
        else:
            st.error("No live flight matching inputted details.")
            # st.stop()
        
        url = (
            f"{base_url}?"
            f"latitude={latitude}&"
            f"longitude={longitude}&"
            f"start_date={departure_date_str}&"
            f"end_date={departure_date_str}&"
            f"hourly=precipitation,wind_speed_10m,snowfall,temperature_2m,"
            f"rain,wind_gusts_10m,weather_code,cloud_cover_low"
        )
        
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=60).json()
                break
            except requests.exceptions.ReadTimeout:
                if attempt == 2:
                    st.error("Weather fetch timed out. Please try again.")
                    st.stop()
                
        hourly = response['hourly']
        
        precipitation = hourly['precipitation'][utc_hour]
        rain = hourly['rain'][utc_hour]
        snowfall = hourly['snowfall'][utc_hour]
        wind_speed = hourly['wind_speed_10m'][utc_hour]
        wind_gusts = hourly['wind_gusts_10m'][utc_hour]
        temperature = hourly['temperature_2m'][utc_hour]
        weather_code = hourly['weather_code'][utc_hour]
        cloud_cover_low = hourly['cloud_cover_low'][utc_hour]
        
        MONTH = departure_date.month
        DAY_OF_WEEK = departure_date.weekday()
        IS_HOLIDAY_PERIOD = isHolidayPeriod(departure_date)
        
        origin_state_abr = airport_data[airport_data['iata_code'] == origin]['iso_region'].values[0][3:]
        dest_state_abr = airport_data[airport_data['iata_code'] == destination]['iso_region'].values[0][3:]
        
        airline_encoded = le_carrier.transform([airline])[0]
        origin_state_encoded = le_origin_state.transform([origin_state_abr])[0]
        dest_state_encoded = le_dest_state.transform([dest_state_abr])[0]
        
        route = f"{origin}-{destination}"
        origin_hourly_flights = origin_hourly_avg.get((origin, departure_hour), 0)
        dest_hourly_flights = dest_hourly_avg.get((destination, departure_hour), 0)
        route_hourly_flights = route_hourly_avg.get((route, departure_hour), 0)
        
        carrier_delay_rate = carrier_delay_map.get(airline_encoded, carrier_delay_map.mean())
        origin_delay_rate = origin_delay_map.get(origin, origin_delay_map.mean())
        dest_delay_rate = dest_delay_map.get(destination, dest_delay_map.mean())
        route_delay_rate = route_delay_map.get(route, route_delay_map.mean())
            
        origin_encoded = origin_te.transform([[origin]])[0][0]
        dest_encoded = dest_te.transform([[destination]])[0][0]
        route_encoded = route_te.transform([[route]])[0][0]
        
        feature_vector = pd.DataFrame({
            'DEPARTURE_HOUR': [departure_hour],
            'wind_gusts': [wind_gusts],
            'precipitation': [precipitation],
            'cloud_cover_low': [cloud_cover_low],
            'weather_code': [weather_code],
            'rain': [rain],
            'temperature': [temperature],
            'wind_speed': [wind_speed],
            'snowfall': [snowfall],
            'ORIGIN_HOURLY_FLIGHTS': [origin_hourly_flights],
            'DAY_OF_WEEK': [DAY_OF_WEEK],
            'MONTH': [MONTH],
            'IS_HOLIDAY_PERIOD': [IS_HOLIDAY_PERIOD],
            'DISTANCE': [distance],
            'CRS_ELAPSED_TIME': [estimated_duration],
            'ORIGIN_STATE_ABR': [origin_state_encoded],
            'DEST_STATE_ABR': [dest_state_encoded],
            'ROUTE_HOURLY_FLIGHTS': [route_hourly_flights],
            'OP_UNIQUE_CARRIER': [airline_encoded],
            'DEST_HOURLY_FLIGHTS': [dest_hourly_flights],
            'CARRIER_DELAY_RATE': [carrier_delay_rate],
            'ORIGIN_DELAY_RATE': [origin_delay_rate],
            'ROUTE_DELAY_RATE': [route_delay_rate],
            'DEST_DELAY_RATE': [dest_delay_rate],
            'ORIGIN_ENCODED': [origin_encoded],
            'DEST_ENCODED': [dest_encoded],
            'ROUTE_ENCODED': [route_encoded]
        })
        
        probability = model.predict_proba(feature_vector)[0][1]
        
        if probability >= best_threshold:
            st.error(f"Delayed - {round(probability * 100, 1)}% chance of delay")
        else:
            st.success(f"On Time - {round(probability * 100, 1)}% chance of delay")