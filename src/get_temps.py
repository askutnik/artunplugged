import requests

def get_temps(lat, lon, month, year=2023, day=1):
    date = f"{year}-{month:02d}-{day:02d}"
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=temperature_2m&timezone=auto"
    )
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        return data["hourly"]["temperature_2m"]
    except Exception as e:
        print("Archive API error:", e)
        return None
