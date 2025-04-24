import requests

def geocode_location(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    try:
        response = requests.get(url, params=params, headers={"User-Agent": "ArtUnplugged/1.0"})
        response.raise_for_status()
        results = response.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            return lat, lon
        else:
            return None, None
    except Exception as e:
        print(f"Geocoding failed: {e}")
        return None, None
