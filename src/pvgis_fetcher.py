import requests
import pandas as pd

def get_pvgis_data(lat, lon, angle=35, aspect=180, year=2020):
    url = (
        "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
        f"?lat={lat}&lon={lon}"
        f"&startyear={year}&endyear={year}"
        "&outputformat=json"
        f"&angle={angle}&aspect={aspect}"
        "&browser=1"
        "&usehorizon=1"
        "&pvcalculation=0"
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch PVGIS data. Status code: {response.status_code}\n{response.text}")

    data = response.json()
    hourly = data["outputs"]["hourly"]

    # Extract time and convert to datetime
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")

    # Rename columns
    df = df.rename(columns={"G(i)": "GTI", "T2m": "temperature"})

    return df[["timestamp", "GTI", "temperature"]]
