mph_to_ms = 0.44704 # 1 mph = 0.447m/s
meters_to_yards = 1.09361

def calc_landing_fsx(
    golf_m,
    BallSpeed_MS, Azimuth_DEG, LaunchAngle_DEG,
    BackSpin_RPM, SideSpin_RPM, TotalSpin_RPM,
    SpinAxis_DEG, windspeed=0, windheading_deg=0, convert_to_yards=True):
    """
    golf_m: golf_ballistics object
    All other params: Keys in FSX ball flight data, back and side spin keys unused. 

    Method mainly exists so I can do calc_landing_fsx(golf_model, **ball_data)
    """
    x_distance, y_distance = golf_m.get_landingpos(
        velocity=BallSpeed_MS, 
        launch_angle_deg=LaunchAngle_DEG,
        off_center_angle_deg=Azimuth_DEG, 
        spin_rpm=TotalSpin_RPM,
        spin_angle_deg=-1 * SpinAxis_DEG,
        windspeed=windspeed * mph_to_ms,
        windheading_deg=windheading_deg)

    # Convert ouput to Yards from Meters
    if convert_to_yards:
        y_yards = meters_to_yards * y_distance
        x_yards = meters_to_yards * x_distance
        return x_yards, y_yards
    else:
        return x_distance, y_distance