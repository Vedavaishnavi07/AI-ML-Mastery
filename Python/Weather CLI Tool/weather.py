import requests
import json
from pathlib import Path

CACHE_FILE = "cache.json"


WEATHER_CODES = {
    0: "Clear Sky",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Heavy Drizzle",
    61: "Light Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    71: "Light Snow",
    73: "Moderate Snow",
    75: "Heavy Snow",
    80: "Rain Showers",
    95: "Thunderstorm"
}


def load_cache():

    if Path(CACHE_FILE).exists():

        with open(CACHE_FILE, "r") as file:

            try:
                return json.load(file)
            except:
                return {}

    return {}


def save_cache(cache):

    with open(CACHE_FILE, "w") as file:

        json.dump(cache, file, indent=4)


def get_coordinates(city):

    url = "https://geocoding-api.open-meteo.com/v1/search"

    response = requests.get(url, params={"name": city, "count": 1})

    data = response.json()

    if "results" not in data:

        return None

    result = data["results"][0]

    return result["latitude"], result["longitude"], result["name"]


def get_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True
    }

    response = requests.get(url, params=params)

    return response.json()["current_weather"]


def display_weather(city, weather):

    print("\n==============================")

    print("CURRENT WEATHER")

    print("==============================")

    print("City:", city)

    print("Temperature:", weather["temperature"], "°C")

    print("Wind Speed:", weather["windspeed"], "km/h")

    print("Wind Direction:", weather["winddirection"], "°")

    code = weather["weathercode"]

    print("Condition:", WEATHER_CODES.get(code, "Unknown"))

    print("Time:", weather["time"])

    print("==============================\n")


def main():

    print("=" * 40)

    print("WEATHER CLI TOOL")

    print("=" * 40)

    cache = load_cache()

    city = input("\nEnter city name: ").strip().lower()

    if city in cache:

        print("\nUsing cached weather data...")

        display_weather(cache[city]["city"], cache[city]["weather"])

        return

    try:

        location = get_coordinates(city)

        if location is None:

            print("\nCity not found.")

            return

        latitude, longitude, city_name = location

        weather = get_weather(latitude, longitude)

        cache[city] = {
            "city": city_name,
            "weather": weather
        }

        save_cache(cache)

        print("\nFetched from Open-Meteo API.")

        display_weather(city_name, weather)

    except requests.exceptions.RequestException:

        print("\nNetwork error. Check your internet connection.")

    except Exception as e:

        print("\nSomething went wrong.")

        print(e)


if __name__ == "__main__":
    main()