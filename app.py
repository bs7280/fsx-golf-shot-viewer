import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import json
from plotly import graph_objects as go
#from plotly.colors import DEFAULT_PLOTLY_COLORS
#from sklearn.datasets import load_iris
#from sklearn.decomposition import PCA

from utils import get_shot_df, confidence_ellipse



#### Second plot - elipses

# Will remove
#iris = load_iris()

#pca = PCA(n_components=2)
#scores = pca.fit_transform(iris.data)

## Will refactor


from io import StringIO

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


if content is None:
    st.write("Upload a file to see shot data")
else:


    #with open('shot_data.session', 'r') as f:
    #    content = json.load(f)

    df_shots = get_shot_df(content)

    #df_shots = df_shots.head(100)

    fig = go.Figure()

    club_names = df_shots['ClubName'].drop_duplicates().values
    club_names = ['Sand Wedge', 'Gap Wedge', 'Pitching wedge', '58']
    club_color_mapper = dict(df_shots[['ClubName', 'ClubColor']].drop_duplicates().values)


    dt_min, dt_max = st.slider('Select shot time',
            df_shots.Timestamp.dt.time.min(), df_shots.Timestamp.dt.time.max(),
            (df_shots.Timestamp.dt.time.min(), df_shots.Timestamp.dt.time.max())
            )

    distance_min, distance_max = st.slider('Select a distance range',
            df_shots.Carry.min(), df_shots.Carry.max(),
            (df_shots.Carry.min(), df_shots.Carry.max())
            )

    st.write(dt_min, dt_max)

    df_shots = df_shots.loc[(df_shots.Timestamp.dt.time >= dt_min) & (df_shots.Timestamp.dt.time <= dt_max)]
    df_shots = df_shots.loc[(df_shots.Carry >= distance_min) & (df_shots.Carry <= distance_max)]

    for club_name in club_names:
        color = club_color_mapper[club_name]
        color = color.replace('0xff', '#')

        x_data = df_shots.loc[df_shots.ClubName==club_name].Offline.values
        y_data = df_shots.loc[df_shots.ClubName==club_name].Carry.values
        #breakpoint()
        fig.add_trace(
            go.Scatter(
                x=x_data, y=y_data,
                #x=scores[iris.target==target_value, 0],
                #y=scores[iris.target==target_value, 1],
                name=club_name,
                mode='markers',
                marker={'color': color}
            )
        )
        
        fig.add_shape(type='path',
                    path=confidence_ellipse(x_data, y_data, n_std=1.96/1),
                    line={'dash': 'dot'},
                    line_color=color)
        
    fig.update_layout(height=1000)

    st.plotly_chart(fig,use_container_width=True, height=1000)
    #fig.show()

    #st.plotly_chart(fig, use_container_width=True)


    #st.write("Report")
    # TODO - write each club distance + vertical / horizontal tolerance
