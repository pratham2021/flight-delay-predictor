import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgbm
import datetime
import requests
import pytz
from timezonefinder import TimezoneFinder

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

airports = pd.read_csv('airports/airports.csv')
airports = airports.dropna(subset=['iata_code'])
airports = airports[airports['iata_code'] != 'None']



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

@st.cache_data
def get_cleaned_airports():
    df = pd.read_csv("airports/airports.csv")
    df = df.dropna(subset=["iata_code"])
    df = df[df["iata_code"] != "None"]
    return df

airports = get_cleaned_airports()

@st.cache_resource
def load_items():
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
    
    return model, le_carrier, le_origin_state, le_dest_state, origin_te, dest_te, route_te, origin_hourly_avg, dest_hourly_avg, route_hourly_avg, carrier_delay_map, origin_delay_map, dest_delay_map, route_delay_map

model, le_carrier, le_origin_state, le_dest_state, origin_te, dest_te, route_te, origin_hourly_avg, dest_hourly_avg, route_hourly_avg, carrier_delay_map, origin_delay_map, dest_delay_map, route_delay_map = load_items()

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
        options=sorted(airports['iata_code'].tolist()),
        index=sorted(airports['iata_code'].tolist()).index('AUS'),    
        format_func=lambda x: f"{x} - {airports[airports['iata_code'] == x]['name'].values[0]}"        
    )
    departure_date = st.date_input("Departure Date", min_value=datetime.date.today(), value=datetime.date.today())

with col2:    
    destination = st.selectbox(
        "Destination",
        options=sorted(airports['iata_code'].tolist()),
        index=sorted(airports['iata_code'].tolist()).index('SLC'),
        format_func=lambda x: f"{x} - {airports[airports['iata_code'] == x]['name'].values[0]}"        
    )
    airline = st.selectbox(
        "Airline",
        options=['AA', 'AS', 'B6', 'DL', 'F9', 'G4', 'HA', 'MQ', 'NK', 'OH', 'OO', 'UA', 'WN', 'YX'],
        format_func=lambda x: airline_names[x],
        index = 3
    )

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    
    lat = airports[airports['iata_code'] == origin]['latitude_deg'].values[0]
    long = airports[airports['iata_code'] == origin]['longitude_deg'].values[0]

    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=long)
    local_tz = pytz.timezone(timezone_str)
    current_local_hour = datetime.datetime.now(local_tz).hour
    
    departure_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=current_local_hour, step=1)
    
    distance = st.slider("Distance (Miles)", min_value=31.0, max_value=5100.0, value=0.0, step=1.0)
    scheduled_duration = st.slider("Duration (Minutes)", min_value=9.0, max_value=701.0, value = 120.0, step=1.0)
    
st.divider()

_, center_col, _ = st.columns([1, 2, 1])

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
        
        if departure_date == datetime.date.today() and departure_hour < current_local_hour:
            st.warning("Please select a current or future departure hour!")
            st.stop()
            
        latitude = airports[airports['iata_code'] == origin]['latitude_deg'].values[0]
        longitude = airports[airports['iata_code'] == origin]['longitude_deg'].values[0]
        
        today = datetime.date.today()
        departure_date_str = departure_date.strftime('%Y-%m-%d')
        
        if departure_date > today + datetime.timedelta(days=16):
            st.error("Weather forecast is only available up to 16 days in advance. Please select an earlier date.")
            st.stop()
        else:
            base_url = "https://api.open-meteo.com/v1/forecast"
        
        url = (
            f"{base_url}?"
            f"latitude={latitude}&"
            f"longitude={longitude}&"
            f"start_date={departure_date_str}&"
            f"end_date={departure_date_str}&"
            f"hourly=precipitation,wind_speed_10m,snowfall,temperature_2m,"
            f"rain,wind_gusts_10m,weather_code,cloud_cover_low"
        )
        
        response = requests.get(url, timeout=30).json()
        print(response)
        hourly = response['hourly']
        
        precipitation = hourly['precipitation'][departure_hour]
        rain = hourly['rain'][departure_hour]
        snowfall = hourly['snowfall'][departure_hour]
        wind_speed = hourly['wind_speed_10m'][departure_hour]
        wind_gusts = hourly['wind_gusts_10m'][departure_hour]
        temperature = hourly['temperature_2m'][departure_hour]
        weather_code = hourly['weather_code'][departure_hour]
        cloud_cover_low = hourly['cloud_cover_low'][departure_hour]
        
        MONTH = departure_date.month
        DAY_OF_WEEK = departure_date.weekday()
        IS_HOLIDAY_PERIOD = isHolidayPeriod(departure_date)
        
        origin_state_abr = airports[airports['iata_code'] == origin]['iso_region'].values[0][3:]
        dest_state_abr = airports[airports['iata_code'] == destination]['iso_region'].values[0][3:]
        
        airline_encoded = le_carrier.transform([airline])[0]
        origin_state_encoded = le_origin_state.transform([origin_state_abr])[0]
        dest_state_encoded = le_dest_state.transform([origin_state_abr])[0]
        
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
        
        best_threshold = joblib.load('models/best_threshold.pkl')
        
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
            'CRS_ELAPSED_TIME': [scheduled_duration],
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