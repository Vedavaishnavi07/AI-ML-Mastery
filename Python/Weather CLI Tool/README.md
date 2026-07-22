# Weather CLI Tool

## Overview

A command-line weather application built with Python that retrieves live weather information using the Open-Meteo API. The application also caches previous searches to reduce unnecessary API requests.

## Features

- Search weather by city
- Uses Open-Meteo Geocoding API
- Displays:
  - Temperature
  - Weather Condition
  - Wind Speed
  - Wind Direction
  - Time
- Stores previous searches in `cache.json`
- Uses cached results for repeat searches
- Handles invalid city names
- Handles network errors

## Technologies Used

- Python 3
- requests
- JSON
- File I/O
- REST API

## How to Run

```bash
python weather.py
```

Enter a city name when prompted.

Example:

```
Enter city name: Chennai
```

## Skills Demonstrated

- API Integration
- JSON Handling
- File Handling
- Error Handling
- HTTP Requests
- Caching