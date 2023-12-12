import os
import json
import pandas as pd
import time
import numpy as np

METERS_TO_YARDS = 1.09361
MPS_to_MPH = 2.23694



# FROM https://gist.github.com/dpfoose/38ca2f5aee2aea175ecc6e599ca6e973
def confidence_ellipse(x, y, n_std=1.96, size=100):
    """
    Get the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    size : int
        Number of points defining the ellipse

    Returns
    -------
    String containing an SVG path for the ellipse
    
    References (H/T)
    ----------------
    https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
    https://community.plotly.com/t/arc-shape-with-path/7205/5
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack([ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)])
    
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)
  
    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
                                [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    scale_matrix = np.array([[x_scale, 0],
                            [0, y_scale]])
    ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix
        
    path = f'M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}'
    for k in range(1, len(ellipse_coords)):
        path += f'L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}'
    path += ' Z'
    return path


def get_shot_df(shot_data_raw):
    #for x in content['Shots']:
    #    carry = 1.09361 * x['FlightData']['CarryDistance_M']
    #    #print(f"club: {x['ClubName']} carry: {carry}")

    df = pd.json_normalize(shot_data_raw['Shots'])
    
    df['Carry'] = df['FlightData.CarryDistance_M'] * METERS_TO_YARDS
    df['Total'] = df['FlightData.TotalDistance_M'] * METERS_TO_YARDS
    df['Offline'] = df['FlightData.OfflineDistance_M'] * METERS_TO_YARDS
    df['Ballspeed'] = df['BallData.BallSpeed_MS'] * MPS_to_MPH

    df_shots = df[['Timestamp', 'ClubName', 'ClubColor', 'Ballspeed', 'Carry', 'Total', 'Offline']]

    df_shots.Timestamp = pd.to_datetime(shot_data_raw['SessionStartDate'] + ' ' + df_shots.Timestamp)

    #time.strptime('%H:%M:%S', )

    return df_shots


if __name__ == '__main__':
    with open('shot_data.session', 'r') as f:
        content = json.load(f)

    df_shots = get_shot_df(content)