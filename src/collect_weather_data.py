import os
import time
import requests
import pandas as pd


HEADERS = {
    "User-Agent": "supply-chain-disruption-intelligence/1.0 contact@example.com",
    "Accept": "application/geo+json"
}


def get_active_alerts_for_state(state):
    url = "https://api.weather.gov/alerts/active"
    params = {
        "area": state
    }

    response = requests.get(url, headers=HEADERS, params=params, timeout=30)

    if response.status_code != 200:
        print(f"Failed for {state}: {response.status_code}")
        return []

    data = response.json()
    return data.get("features", [])


def summarize_alerts(alerts):
    if not alerts:
        return {
            "active_alert_count": 0,
            "warning_count": 0,
            "watch_count": 0,
            "advisory_count": 0,
            "severe_weather_flag": 0
        }

    warning_count = 0
    watch_count = 0
    advisory_count = 0

    severe_keywords = [
        "Tornado",
        "Hurricane",
        "Storm",
        "Flood",
        "High Wind",
        "Winter Storm",
        "Blizzard",
        "Coastal Flood",
        "Tsunami"
    ]

    severe_weather_flag = 0

    for alert in alerts:
        props = alert.get("properties", {})
        event = props.get("event", "") or ""

        if "Warning" in event:
            warning_count += 1
        if "Watch" in event:
            watch_count += 1
        if "Advisory" in event:
            advisory_count += 1

        if any(keyword.lower() in event.lower() for keyword in severe_keywords):
            severe_weather_flag = 1

    return {
        "active_alert_count": len(alerts),
        "warning_count": warning_count,
        "watch_count": watch_count,
        "advisory_count": advisory_count,
        "severe_weather_flag": severe_weather_flag
    }


def main():
    ports_path = "/Users/navikamaglani/Documents/personal/supply_chain_disruption_ai/data/processed/us_ports.csv"

    if not os.path.exists(ports_path):
        ports_path = "/Users/navikamaglani/Documents/personal/supply_chain_disruption_ai/data/processed/us_ports.csv"

    ports_df = pd.read_csv(ports_path)

    records = []

    for _, row in ports_df.iterrows():
        port_name = row["port_name"]
        state = row["state"]

        print(f"Collecting alerts for {port_name}, {state}")

        alerts = get_active_alerts_for_state(state)
        summary = summarize_alerts(alerts)

        records.append({
            "port_name": port_name,
            "state": state,
            "lat": row["lat"],
            "lon": row["lon"],
            **summary
        })

        time.sleep(1)

    weather_df = pd.DataFrame(records)

    os.makedirs("data/processed", exist_ok=True)
    weather_df.to_csv("data/processed/port_weather_alerts.csv", index=False)

    print("Saved weather alert data:")
    print(weather_df)


if __name__ == "__main__":
    main()