import pandas as pd
import os

ports = [
    {
        "port_name": "Los Angeles",
        "state": "CA",
        "lat": 33.7405,
        "lon": -118.2775
    },
    {
        "port_name": "Long Beach",
        "state": "CA",
        "lat": 33.7542,
        "lon": -118.2165
    },
    {
        "port_name": "New York/New Jersey",
        "state": "NY",
        "lat": 40.6681,
        "lon": -74.0451
    },
    {
        "port_name": "Savannah",
        "state": "GA",
        "lat": 32.0809,
        "lon": -81.0912
    },
    {
        "port_name": "Houston",
        "state": "TX",
        "lat": 29.7604,
        "lon": -95.3698
    },
    {
        "port_name": "Seattle/Tacoma",
        "state": "WA",
        "lat": 47.2529,
        "lon": -122.4443
    }
]

ports_df = pd.DataFrame(ports)

os.makedirs("data/processed", exist_ok=True)

ports_df.to_csv("data/processed/us_ports.csv", index=False)

ports_df