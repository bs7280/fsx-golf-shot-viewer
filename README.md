# FSX Shot Data Viewer
Upload a foresight golf simulator's shot data to view club specific shot data. Currently will show elipitical shot patterns per club.

Features:
- Upload FSX data
- Filter on distance, shot time, or club type
- Add wind speed and direction
- Get table of shot distance averages, and average distance offline
    - Will adjust to wind
- View shot miss pattern elipses

Future:
- Club dispersion variance
- distinguish draws and fades for distance table (and wind)
- Fix wind calculation

**Disclaimer** - Wind model I used is outdated, and is pretty useless above ~200 yards. Code I used was based on a paper in the 90's, since then golf technology has changed substantially.

## Usage
Deployed at [FSX Golf Shoft Viewer](https://fsx-golf-shot-viewer.streamlit.app/)

Input:
![image](screenshots/v0.2_img1.png)

Display:
![image](screenshots/v0.2_img2.png)

## Install

```sh
pip install -r requirements.txt
streamlit run app.py
```

## TODO 
- Give simple summary of distance + tolerance
- Modify shot list to remove outliers
- save shot list
- allow video upload to compare best and worst shots