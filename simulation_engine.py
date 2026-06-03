import os
import math
from datetime import datetime
import random
import pandas as pd
import numpy as np

# Cache for dataframe to avoid reloading 24MB CSV on every request
_STATION_DB_CACHE = None

class StationDatabase:
    """Manages the local charging station dataset and handles geospatial queries."""
    
    CSV_PATH = os.path.join(os.path.dirname(__file__), "charging_station.csv")

    @classmethod
    def get_df(cls):
        global _STATION_DB_CACHE
        if _STATION_DB_CACHE is None:
            if not os.path.exists(cls.CSV_PATH):
                raise FileNotFoundError(f"Dataset not found at {cls.CSV_PATH}")
            # Load only required columns to save memory and increase speed
            _STATION_DB_CACHE = pd.read_csv(
                cls.CSV_PATH,
                usecols=["id", "name", "city", "state_province", "country_code", "latitude", "longitude", "ports", "power_kw", "power_class", "is_fast_dc"]
            )
        return _STATION_DB_CACHE

    @classmethod
    def find_nearby_stations(cls, lat: float, lon: float, radius_km: float = 10.0):
        """
        Uses the Haversine formula to search for real-world stations 
        within radius_km from input (lat, lon).
        """
        df = cls.get_df()
        
        # Haversine distance vector calculation
        lon1, lat1, lon2, lat2 = map(np.radians, [lon, lat, df["longitude"].values, df["latitude"].values])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = 6371.0 * c # Earth radius in km
        
        # Filter stations
        nearby_indices = np.where(distances <= radius_km)[0]
        if len(nearby_indices) == 0:
            return []
            
        nearby_df = df.iloc[nearby_indices].copy()
        nearby_df["distance_km"] = distances[nearby_indices]
        
        # Sort by distance
        nearby_df = nearby_df.sort_values(by="distance_km")
        
        # Convert to dictionary format
        return nearby_df.to_dict(orient="records")


class QueuingModel:
    """Implements queuing theory (M/M/c queue) to calculate wait times based on system-time."""
    
    # Diurnal peak multiplier (0.0 to 1.0) based on hour of the day
    DIURNAL_HOURLY_CURVE = {
        0: 0.15, 1: 0.10, 2: 0.05, 3: 0.05, 4: 0.08, 5: 0.15,
        6: 0.30, 7: 0.55, 8: 0.82, 9: 0.88, 10: 0.75, 11: 0.65,
        12: 0.70, 13: 0.72, 14: 0.68, 15: 0.62, 16: 0.70, 17: 0.88,
        18: 0.92, 19: 0.85, 20: 0.70, 21: 0.50, 22: 0.30, 23: 0.20
    }

    @classmethod
    def get_current_multiplier(cls):
        """Gets hourly multiplier based on current machine clock hour."""
        current_hour = datetime.now().hour
        return cls.DIURNAL_HOURLY_CURVE.get(current_hour, 0.5)

    @classmethod
    def simulate_occupancy_and_queue(cls, ports: int, is_fast_dc: bool):
        """
        Uses M/M/c queuing formulas to estimate waiting times and occupied ports.
        c = ports (number of chargers/servers)
        mu = service rate (cars per hour charged). Fast DC charges faster than AC.
        lambda = arrival rate (cars arriving per hour).
        """
        if ports <= 0:
            return {"occupied_ports": 0, "queue_length": 0, "wait_time_minutes": 0.0}

        c = ports
        # Service rate mu (cars per hour per port)
        # Fast DC charger takes ~30 mins (mu = 2.0 cars/hr)
        # AC Level 2 charger takes ~3 hours (mu = 0.33 cars/hr)
        mu = 2.0 if is_fast_dc else 0.33
        
        # Base arrival rate scaled by ports and current hour multiplier
        multiplier = cls.get_current_multiplier()
        
        # Add slight random noise to multiplier to make it feel live and fluctuating on refresh
        noise = random.uniform(-0.05, 0.05)
        multiplier = max(0.02, min(0.98, multiplier + noise))
        
        # Arrival rate lambda (cars/hour)
        # At peak, arrival rate might exceed service capacity (lambda > c * mu)
        base_arrival_ratio = 1.25 # high demand multiplier
        lam = base_arrival_ratio * c * mu * multiplier
        
        # Traffic intensity rho
        rho = lam / (c * mu)
        
        if rho < 0.95:
            # M/M/c queuing calculations
            try:
                # Calculate P0 (probability of 0 cars in the system)
                sum_terms = 0.0
                for n in range(c):
                    sum_terms += ((c * rho) ** n) / math.factorial(n)
                
                div_term = ((c * rho) ** c) / (math.factorial(c) * (1 - rho))
                p0 = 1.0 / (sum_terms + div_term)
                
                # Erlang C formula (probability of queuing)
                pq = div_term * p0
                
                # Average queue length (number of cars waiting)
                lq = pq * (rho / (1.0 - rho))
                
                # Average wait time in queue (hours) -> Little's Law: Wq = Lq / lambda
                wq = lq / lam if lam > 0 else 0.0
                wait_time_min = wq * 60.0
                
                # Simulated occupied ports
                occupied = int(round(c * rho))
                # Add random fluctuation
                occupied = max(0, min(c, occupied + random.choice([-1, 0, 1])))
                
                # Queue length rounded
                queue_len = int(round(lq))
                if occupied < c:
                    queue_len = 0 # Can't have a queue if ports are free!
                    wait_time_min = 0.0
                elif queue_len == 0 and occupied == c:
                    # If all ports are full, we might have a small waiting chance
                    queue_len = random.choice([0, 1])
                    wait_time_min = queue_len * (30.0 / c)
                
            except (ZeroDivisionError, OverflowError):
                # Fallback if calculations overflow
                occupied = c
                queue_len = int(round(c * (rho - 0.9)))
                wait_time_min = queue_len * (30.0 / c)
        else:
            # System is overloaded / gridlocked (rho >= 0.95)
            occupied = c
            queue_len = max(1, int(round(5 * (rho - 0.8) * c)))
            # Average wait time is estimated based on remaining charge times of active cars
            wait_time_min = (queue_len * (35.0 / c)) + random.uniform(5.0, 15.0)

        # Ensure wait time is formatted nicely
        return {
            "occupied_ports": int(occupied),
            "queue_length": int(queue_len),
            "wait_time_minutes": round(wait_time_min, 1),
            "traffic_density_ratio": round(rho, 2)
        }


class GridSimulator:
    """Simulates local transformer power draw, load ratio, and power grid safety warnings."""

    @classmethod
    def simulate_grid_impact(cls, ports: int, occupied_ports: int, power_kw: float):
        """
        Calculates grid impact metrics.
        power_kw is the charging speed of each port.
        """
        if ports <= 0:
            return {"power_draw_kw": 0.0, "transformer_load_percent": 0.0, "status": "Offline", "grid_stability_factor": 1.0}
        
        # If power_kw is NaN or missing, assume standard values
        if pd.isna(power_kw) or power_kw <= 0:
            power_kw = 150.0 # Default Fast DC rating
            
        # Total power currently drawn
        power_draw_kw = occupied_ports * power_kw
        
        # Transformer Capacity Rating (assumed to be sized with some headroom)
        # E.g. A substation transformer for this site is rated for 120% of max capacity
        transformer_capacity_kw = ports * power_kw * 1.15
        
        # Adjust capacity for typical local grid sizes (clamp between 50kW and 2000kW)
        transformer_capacity_kw = max(50.0, min(2000.0, transformer_capacity_kw))
        
        # Transformer Load percentage
        load_pct = (power_draw_kw / transformer_capacity_kw) * 100.0
        
        # Power Factor decreases slightly at high load due to non-linear inverter harmonics
        power_factor = 0.95 - 0.06 * (occupied_ports / ports) if ports > 0 else 0.95
        
        # Determine safety status
        if load_pct >= 90.0:
            status = "CRITICAL OVERLOAD WARNING"
            suggestion = "Substation transformer threshold exceeded! Smart load-shedding activated. Recommend delaying non-essential charging."
            color = "red"
        elif load_pct >= 70.0:
            status = "HIGH GRID STRESS"
            suggestion = "Substation load is high. Power factor correction active. Grid stability is stable but monitored."
            color = "orange"
        else:
            status = "SAFE OPERATION"
            suggestion = "Substation load is within green limits. Grid frequency and voltage are fully stable."
            color = "green"
            
        return {
            "power_draw_kw": round(power_draw_kw, 1),
            "transformer_capacity_kw": round(transformer_capacity_kw, 1),
            "transformer_load_percent": round(load_pct, 1),
            "power_factor": round(power_factor, 2),
            "status": status,
            "suggestion": suggestion,
            "color": color
        }


class CarbonCalculator:
    """Calculates carbon footprint displacement (CO2 saved) based on country energy mixes."""
    
    # Grid emission intensity in grams of CO2 per kWh
    # Source: Standard national statistics (e.g. EEA, NREL)
    GRID_INTENSITIES = {
        "US": 371.0,  # United States
        "IN": 725.0,  # India (heavy coal)
        "CN": 610.0,  # China
        "DE": 385.0,  # Germany
        "FR": 56.0,   # France (mostly nuclear)
        "AD": 45.0,   # Andorra (mostly hydro import)
        "GB": 210.0,  # Great Britain
        "CA": 120.0,  # Canada (mostly hydro)
        "NO": 25.0,   # Norway (extremely clean hydro)
        "UNKNOWN": 320.0 # World average baseline
    }

    @classmethod
    def calculate_co2_savings(cls, country_code: str, power_kw: float, duration_hours: float = 1.0):
        """
        Calculates CO2 emissions saved compared to an average gasoline car.
        Average gasoline car: ~120g CO2 per km.
        EV efficiency: ~5.5 km per kWh (or ~3.4 miles/kWh).
        """
        if pd.isna(power_kw) or power_kw <= 0:
            power_kw = 50.0 # Default fallback
            
        energy_charged_kwh = power_kw * duration_hours
        
        # 1. CO2 emitted by charging the EV (using local grid energy mix)
        grid_intensity = cls.GRID_INTENSITIES.get(str(country_code).upper(), cls.GRID_INTENSITIES["UNKNOWN"])
        ev_co2_emitted_g = energy_charged_kwh * grid_intensity
        
        # 2. CO2 emitted by equivalent gas car driving the same distance
        # Standard EV runs 5.5 km per kWh
        equivalent_distance_km = energy_charged_kwh * 5.5
        # Standard gasoline car emits 120g of CO2 per km (average new passenger car)
        gas_car_co2_emitted_g = equivalent_distance_km * 125.0
        
        # 3. Net savings
        net_saved_g = gas_car_co2_emitted_g - ev_co2_emitted_g
        net_saved_kg = net_saved_g / 1000.0
        
        # Number of trees needed to absorb this CO2 (1 tree absorbs ~22kg of CO2 per year)
        # Tree equivalence for this specific charging hour
        tree_hours = net_saved_kg / (22.0 / 365.0) if net_saved_kg > 0 else 0.0
        
        return {
            "energy_charged_kwh": round(energy_charged_kwh, 1),
            "ev_emissions_kg": round(ev_co2_emitted_g / 1000.0, 2),
            "gas_emissions_kg": round(gas_car_co2_emitted_g / 1000.0, 2),
            "net_saved_kg": round(net_saved_kg, 2),
            "tree_days_absorbed": round(tree_hours / 24.0, 1), # days of one tree absorbing carbon
            "grid_intensity_g_kwh": grid_intensity
        }
