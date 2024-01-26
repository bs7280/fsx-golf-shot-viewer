import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import json
from plotly import graph_objects as go
from io import StringIO
#from plotly.colors import DEFAULT_PLOTLY_COLORS
#from sklearn.datasets import load_iris
#from sklearn.decomposition import PCA

from utils import get_shot_df, confidence_ellipse

from golfmodel.model import golf_ballstics
from golfmodel.utils import calc_landing_fsx

content = None

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    #bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    content = json.load(stringio)
else:
    with open('example_data/ex_shot_data_wedges.json', 'r') as f:
        content = json.load(f)
    st.warning('Using default example data, upload your own data above', icon="⚠️")

## Air resistance model: 
golf_m = golf_ballstics()

if content is None:
    st.write("Upload a file to see shot data")
else:


    #with open('shot_data.session', 'r') as f:
    #    content = json.load(f)

    df_shots = get_shot_df(content)

    #df_shots = df_shots.head(100)

    fig = go.Figure()

    club_names = df_shots['ClubName'].drop_duplicates().values
    club_color_mapper = dict(df_shots[['ClubName', 'ClubColor']].drop_duplicates().values)


    st.header("Filter Data")
    dt_min, dt_max = st.slider('Select shot time',
            df_shots.Timestamp.dt.time.min(), df_shots.Timestamp.dt.time.max(),
            (df_shots.Timestamp.dt.time.min(), df_shots.Timestamp.dt.time.max())
            )

    distance_min, distance_max = st.slider('Select a distance range',
            df_shots.Carry.min(), df_shots.Carry.max(),
            (df_shots.Carry.min(), df_shots.Carry.max())
            )
    
    st.header("Set wind")
    wind_mph = st.slider("Wind speed MPH", 0, 100, 0)
    wind_direction = st.slider("Wind direction (0 is tailwaind, 180 is headwind)", 0, 360, 0)

    ## State for checkboxes
    if 'selected_clubs' not in st.session_state:
        st.session_state['selected_clubs'] = {}
    ## Toggle state of club
    def update_club_state(club_name):
        if club_name in st.session_state['selected_clubs']:
            current_val = st.session_state['selected_clubs'][club_name]
            st.session_state['selected_clubs'][club_name] = not current_val

    st.header("Select Clubs")
    for club_name in club_names:
        if club_name not in st.session_state['selected_clubs']:
            st.session_state['selected_clubs'][club_name] = True

        st.checkbox(club_name, value=True, on_change=update_club_state, args=(club_name,))
    # Gets list of selected clubs
    active_clubs = [k for k,v in st.session_state['selected_clubs'].items() if v]

    ## Filter data from inputs
    df_shots = df_shots.loc[
        (df_shots.Timestamp.dt.time >= dt_min) &
        (df_shots.Timestamp.dt.time <= dt_max)]
    df_shots = df_shots.loc[
        (df_shots.Carry >= distance_min) &
        (df_shots.Carry <= distance_max)]
    
    ## Add wind if option added
    def add_wind_calcs(shot_data, wind_speed=0, wind_direction=0):
        sd = shot_data.copy()

        for i,_shot in enumerate(sd['Shots']):
            x_distance, y_distance = calc_landing_fsx(
                golf_m,
                windspeed=wind_speed, windheading_deg=wind_direction,
                convert_to_yards=False,
                **_shot['BallData'])
            
            sd['Shots'][i]['FlightData']['CarryDistance_M'] = y_distance
            sd['Shots'][i]['FlightData']['OfflineDistance_M'] = x_distance

        return sd



    if wind_mph > 0:
        new_content = add_wind_calcs(content, wind_speed=wind_mph, wind_direction=wind_direction)
        df_shots = get_shot_df(new_content)

        #st.table(df_wind.tail(1))

    st.header("Vizualize shots")
    shot_data_avg = {}
    for club_name in active_clubs:
        color = club_color_mapper[club_name]
        color = color.replace('0xff', '#')

        x_data = df_shots.loc[df_shots.ClubName==club_name].Offline.values
        y_data = df_shots.loc[df_shots.ClubName==club_name].Carry.values

        # Calculate Avg Values
        shot_data_avg[club_name] = {}
        shot_data_avg[club_name]['avg_distance'] = np.mean(y_data)
        shot_data_avg[club_name]['avg_offline'] = np.mean(x_data)

        fig.add_trace(
            go.Scatter(
                x=x_data, y=y_data,
                name=club_name,
                mode='markers',
                marker={'color': color}
            )
        )
        
        fig.add_shape(type='path',
                    path=confidence_ellipse(x_data, y_data, n_std=1.96/1),
                    line={'dash': 'dot'},
                    line_color=color)
        
    st.write(f"Shot Distances with {wind_mph}mph of wind from {wind_direction} degrees")
    st.table(pd.DataFrame(shot_data_avg).T)

    fig.update_layout(height=700)

    st.plotly_chart(fig,use_container_width=True, height=700)
    #fig.show()

    #st.plotly_chart(fig, use_container_width=True)


    #st.write("Report")
    # TODO - write each club distance + vertical / horizontal tolerance
