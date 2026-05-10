# ==============================================================================
# Black Swan OS v16.0
# ==============================================================================
# Strategia: GIZA LOCK // Trójkąt Trinity (Egipt-Cypr-Izrael)
# Sensory: Sejsmika IRIS, NASA TEC, Astro-Sync Regulus, Syzygy Gravity Master
# Autorzy: Operator Space & Wingman AI (Black Swan Analysis Squadron)
# ==============================================================================
import os
import time
import threading
import queue
import warnings
import numpy as np
import pandas as pd
import requests
import json
from collections import deque
from datetime import datetime, timezone, timedelta
from scipy.signal import welch, butter, filtfilt, find_peaks
from scipy.stats import entropy
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from obspy.clients.fdsn import Client
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun, get_body
from astropy.time import Time
import astropy.units as u

warnings.filterwarnings('ignore')

# =========================
# GLOBAL CONFIGURATION
# =========================
GIZA_LAT, GIZA_LON = 29.9792, 31.1342
GIZA_LOC = EarthLocation(lat=GIZA_LAT*u.deg, lon=GIZA_LON*u.deg, height=60*u.m)
REGULUS = SkyCoord(ra='10h08m22.3s', dec='+11d58m02s', frame='icrs')

DATE_AUG = datetime(2026, 8, 22, 17, 3, 0, tzinfo=timezone.utc)
DATE_OCT = datetime(2026, 10, 7, 2, 41, 0, tzinfo=timezone.utc)

LOG_FILE = "giza_sentinel_v15.csv"
ENV_LOG_FILE = "giza_environment_log.csv"
GUARDIAN_CONFIG_FILE = "guardian_config_snapshot.json"
FS = 100
CRUST_BASE_OFFSET = 15.0
GRAPH_CONFIG = {"displayModeBar": False}
SYNC_OFFSET = 900  # T-15min sync anchor (seconds)
ENV_LOG_INTERVAL = 60  # Log environment every 60 seconds

STATIONS = [
    {"name": "EGYPT",  "net": "EG", "sta": "HLW",  "lat": 29.8, "lon": 31.3},
    {"name": "CYPRUS", "net": "GE", "sta": "NICO",  "lat": 35.1, "lon": 33.3},
    {"name": "ISRAEL", "net": "IU", "sta": "EIL",   "lat": 29.6, "lon": 34.9}
]

rolling_stats = deque(maxlen=17280)
log_buffer = deque(maxlen=10)

# DEBUG TELEMETRY COUNTER
debug_counter = 0

# Zmień listę kolumn na tę:
if not os.path.exists(LOG_FILE):
    cols = ["timestamp", "sigma", "peak_hz", "tec", "alignment", "entropy", "gravity_load", 
            "h_confidence", "regulus_az", "venus_az", "jupiter_az", "tags"]
    pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)


# =========================
# GUARDIAN: ENVIRONMENTAL TELEMETRY v16.1
# =========================
class GuardianEnvironmental:
    """Lightweight system & weather telemetry for statistical control analysis."""
    
    def __init__(self):
        self.last_env_log = None
        self.cached_weather = {
            "ambient_temp": None,
            "pressure": None,
            "humidity": None,
            "wind_speed": None,
            "last_fetch": None
        }
        self.weather_refresh_interval = 300  # 5 minutes
        
        # Initialize environment log
        if not os.path.exists(ENV_LOG_FILE):
            env_cols = ["timestamp", "cpu_temp", "cpu_usage", "ram_usage", 
                       "ambient_temp", "pressure", "humidity", "wind_speed"]
            pd.DataFrame(columns=env_cols).to_csv(ENV_LOG_FILE, index=False)
    
    def get_system_health(self):
        """
        Get CPU temp, CPU usage, RAM usage.
        Return dict or None if unavailable.
        """
        try:
            import psutil
            cpu_temp = None
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temp = temps['coretemp'][0].current
            except:
                # Fallback to /sys/class/thermal
                try:
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        cpu_temp = float(f.read().strip()) / 1000.0
                except:
                    cpu_temp = None
            
            cpu_usage = psutil.cpu_percent(interval=0.1)
            ram_info = psutil.virtual_memory()
            ram_usage = ram_info.percent
            
            return {
                "cpu_temp": cpu_temp,
                "cpu_usage": cpu_usage,
                "ram_usage": ram_usage
            }
        except Exception as e:
            print(f"[GUARDIAN WARN] System health unavailable: {e}")
            return {
                "cpu_temp": None,
                "cpu_usage": None,
                "ram_usage": None
            }
    
    def get_weather_giza(self):
        """
        Fetch Cairo/Giza weather via Open-Meteo API (free, no auth).
        Cache for 5 minutes. Never block pipeline on failure.
        """
        try:
            now = datetime.now(timezone.utc)
            
            # Check cache freshness
            if (self.cached_weather["last_fetch"] is not None and
                (now - self.cached_weather["last_fetch"]).total_seconds() < self.weather_refresh_interval):
                return {
                    "ambient_temp": self.cached_weather["ambient_temp"],
                    "pressure": self.cached_weather["pressure"],
                    "humidity": self.cached_weather["humidity"],
                    "wind_speed": self.cached_weather["wind_speed"]
                }
            
            # Fetch from Open-Meteo (free API, no key needed)
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": GIZA_LAT,
                "longitude": GIZA_LON,
                "current": "temperature_2m,relative_humidity_2m,pressure_msl,weather_code,wind_speed_10m"
            }
            r = requests.get(url, params=params, timeout=5)
            
            if r.status_code == 200:
                data = r.json()
                current = data.get("current", {})
                
                self.cached_weather.update({
                    "ambient_temp": float(current.get("temperature_2m", 0)),
                    "pressure": float(current.get("pressure_msl", 1013)),
                    "humidity": float(current.get("relative_humidity_2m", 0)),
                    "wind_speed": float(current.get("wind_speed_10m", 0)),
                    "last_fetch": now
                })
                
                return {
                    "ambient_temp": self.cached_weather["ambient_temp"],
                    "pressure": self.cached_weather["pressure"],
                    "humidity": self.cached_weather["humidity"],
                    "wind_speed": self.cached_weather["wind_speed"]
                }
            else:
                # Preserve last valid values
                return {
                    "ambient_temp": self.cached_weather["ambient_temp"],
                    "pressure": self.cached_weather["pressure"],
                    "humidity": self.cached_weather["humidity"],
                    "wind_speed": self.cached_weather["wind_speed"]
                }
        
        except Exception as e:
            print(f"[GUARDIAN WEATHER] API error: {e}")
            # Gracefully return last valid cached values
            return {
                "ambient_temp": self.cached_weather["ambient_temp"],
                "pressure": self.cached_weather["pressure"],
                "humidity": self.cached_weather["humidity"],
                "wind_speed": self.cached_weather["wind_speed"]
            }
    
    def log_environment(self, force=False):
        """
        Log environment to giza_environment_log.csv every 60 seconds (or forced).
        Preserve UTC timestamps, T-15m sync mode.
        """
        now = datetime.now(timezone.utc)
        
        # Check if we should log (60s interval)
        if (not force and 
            self.last_env_log is not None and 
            (now - self.last_env_log).total_seconds() < ENV_LOG_INTERVAL):
            return
        
        # Collect telemetry
        sys_health = self.get_system_health()
        weather = self.get_weather_giza()
        
        # Write to CSV
        try:
            with open(ENV_LOG_FILE, "a") as f:
                f.write(f"{now},"
                       f"{sys_health['cpu_temp'] or 'N/A'},"
                       f"{sys_health['cpu_usage'] or 'N/A'},"
                       f"{sys_health['ram_usage'] or 'N/A'},"
                       f"{weather['ambient_temp'] or 'N/A'},"
                       f"{weather['pressure'] or 'N/A'},"
                       f"{weather['humidity'] or 'N/A'},"
                       f"{weather['wind_speed'] or 'N/A'}\n")
            self.last_env_log = now
        except Exception as e:
            print(f"[GUARDIAN ENV LOG] Write error: {e}")


def save_guardian_config_snapshot():
    """
    On startup, print JSON config snapshot and save to guardian_config_snapshot.json
    """
    config_snapshot = {
        "version": "16.1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sync_offset_seconds": SYNC_OFFSET,
        "env_log_interval_seconds": ENV_LOG_INTERVAL,
        "event_thresholds": {
            "z_score_anomaly": 4.5,
            "alignment_lock": 80.0,
            "h_conf_lock": 50.0,
            "sigma_lock": 10.0
        },
        "scoring_weights": {
            "harmonic_score": 0.45,
            "energy_score": 0.30,
            "persistence_score": 0.15,
            "cluster_score": 0.10
        },
        "astro_sync_mode": "T-15min from now",
        "giza_coordinates": {
            "lat": GIZA_LAT,
            "lon": GIZA_LON,
            "height_m": 60
        },
        "stations": STATIONS
    }
    
    # Print to console
    print("\n[GUARDIAN CONFIG SNAPSHOT]")
    print(json.dumps(config_snapshot, indent=2, default=str))
    print()
    
    # Save to file
    try:
        with open(GUARDIAN_CONFIG_FILE, "w") as f:
            json.dump(config_snapshot, f, indent=2, default=str)
        print(f"[GUARDIAN] Config snapshot saved to {GUARDIAN_CONFIG_FILE}")
    except Exception as e:
        print(f"[GUARDIAN] Config save error: {e}")


# Initialize Guardian system
guardian = GuardianEnvironmental()
save_guardian_config_snapshot()


# =========================
# ASTRO-SYNC & SYZYGY ENGINE
# =========================
class AstroCache:
    """Goal 1: Astro-layer stability with caching and staleness detection."""
    def __init__(self):
        self.last_valid = {
            "reg_az": 0, "ven_az": 0, "jup_az": 0, "grav_load": 25.0
        }
        self.last_update = datetime.now(timezone.utc)
        self.astro_stale = False
    
    def is_stale(self):
        age = (datetime.now(timezone.utc) - self.last_update).total_seconds()
        return age > 120.0
    
    def get_cached(self):
        """Return last valid state, setting ASTRO_STALE if > 120s."""
        if self.is_stale():
            self.astro_stale = True
            print(f"[ASTRO WARN] Cache age > 120s, using last valid state")
        return self.last_valid.copy()
    
    def update(self, new_metrics):
        """Update cache with fresh metrics."""
        self.last_valid = new_metrics.copy()
        self.last_update = datetime.now(timezone.utc)
        self.astro_stale = False

astro_cache = AstroCache()

def get_space_metrics(target_time=None):
    """
    Calculate space metrics (planet positions, gravity load) for a given time.
    
    Args:
        target_time (datetime, optional): Time to calculate metrics for. Defaults to None.
                                         If None, uses current time - 900s (T-15min sync).
    
    Returns:
        dict: Contains reg_az, ven_az, jup_az, grav_load
    """
    try:
        # SYNC T-15min (900s) - Use target_time if provided, otherwise calculate from now
        if target_time is None:
            sync_time = datetime.now(timezone.utc) - timedelta(seconds=SYNC_OFFSET)
        else:
            sync_time = target_time
        
        now = Time(sync_time)
        altaz_frame = AltAz(obstime=now, location=GIZA_LOC)
        
        # Ciała Niebieskie
        sun = get_sun(now).transform_to(altaz_frame)
        moon = get_body('moon', now).transform_to(altaz_frame)
        reg = REGULUS.transform_to(altaz_frame)
        venus = get_body('venus', now).transform_to(altaz_frame)
        jupiter = get_body('jupiter', now).transform_to(altaz_frame)

        # --- SYZYGY 2.0 GRAVITY MODEL ---
        # Liczymy nacisk grawitacyjny (Słońce + Księżyc)
        # Składowa pionowa nacisku: sin(alt)
        sun_press = np.sin(np.radians(max(0, sun.alt.deg))) * 40.0
        moon_press = np.sin(np.radians(max(0, moon.alt.deg))) * 60.0 # Księżyc silniej szarpie pływy
        g_load = (
            sun_press +
            moon_press +
            (np.sin(np.radians(reg.az.deg)) * 15.0) +
            25.0
        )

        g_load = max(0.0, min(100.0, g_load))

        new_metrics = {
            "reg_az": float(reg.az.deg), "ven_az": float(venus.az.deg),
            "jup_az": float(jupiter.az.deg), "grav_load": float(g_load)
        }
        astro_cache.update(new_metrics)
        return new_metrics
    except Exception as e:
        print(f"[ASTRO ERROR] {e}")
        # Goal 1: Use cached state instead of hardcoded fallback
        return astro_cache.get_cached()

# ==============================================================================
# DATA STREAMING & SNIPER ANALYSIS ENGINE
# ==============================================================================
class GizaSniperStreamer:
    def __init__(self):
        self.q = queue.Queue(maxsize=20)
        self.tec_val = 15.0
        self.client = Client("EARTHSCOPE")
        self.running = True
        self.simulation_mode = False
        self.last_end_time = None
        self.persistence = {"LOW": 0, "MID": 0, "HIGH": 0}
        self.last_active_cluster = None
        # Goal 4: Store sub-scores for telemetry diagnostics
        self.last_sub_scores = {
            "harm_score": 0.0, "energy_score": 0.0,
            "persistence_score": 0.0, "cluster_score": 0.0, "noise_floor": 1.0
        }
        # HUD HARMONIC SYNC: Track lock state for elapsed time display
        self.harmonic_lock_start = None
        self.last_lock_state = False

    def tec_worker(self):
        while self.running:
            try:
                r = requests.get(
                    "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
                    timeout=10
                )
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        kp = float(data[-1].get('kp_index', 1.0))
                        self.tec_val = round(5.0 + (kp / 9.0) * 35.0 + np.random.uniform(-1.5, 1.5), 2)
            except Exception as e:
                print(f"[TEC ERROR] {e}")
            time.sleep(300)

    def seismic_worker(self):
        """Nasłuch z 15-minutowym opóźnieniem (900s) dla stabilności danych."""
        while self.running:
            try:
                # 15 MINUT BUFORA - to jest klucz do czystych danych bez timeoutów
                now_stable = datetime.now(timezone.utc) - timedelta(seconds=SYNC_OFFSET)
                
                if self.last_end_time is None:
                    # Start: 10s paczka sprzed 15 minut
                    start_time = now_stable - timedelta(seconds=10)
                else:
                    # Kontynuacja z 2s zakładką
                    start_time = self.last_end_time - timedelta(seconds=2)
                
                end_time = start_time + timedelta(seconds=10)
                
                # Zabezpieczenie przed "dogonieniem" czasu
                if end_time > now_stable:
                    end_time = now_stable
                    start_time = end_time - timedelta(seconds=10)

                self.last_end_time = end_time

                batch = {}
                for s in STATIONS:
                    try:
                        # Wymuszamy czas oczekiwania serwera na 10s
                        st = self.client.get_waveforms(s["net"], s["sta"], "*", "BHZ", start_time, end_time)
                        if len(st) > 0:
                            data = st.data.astype(float)
                            data = (data - np.mean(data)) / (np.std(data) + 1e-6)
                            batch[s["name"]] = data
                            batch[f"{s['name']}_SIM"] = False
                        else:
                            raise ValueError("No data")
                    except Exception:
                        batch[s["name"]] = np.random.normal(0, 1, 1000)
                        batch[f"{s['name']}_SIM"] = True

                if self.q.full(): self.q.get_nowait()
                self.q.put(batch)
            except Exception as e:
                print(f"[STREAMER ERROR] {e}")
            time.sleep(1)


def format_elapsed_harmonic(lock_start_timestamp):
    """
    Format harmonic lock elapsed time safely.
    
    Args:
        lock_start_timestamp: datetime or None
    
    Returns:
        str: Formatted time "XM" (under 60m) or "XH YM" (over 60m), or "0M" if invalid
    """
    if lock_start_timestamp is None:
        return "0M"
    
    try:
        now = datetime.now(timezone.utc)
        elapsed_seconds = (now - lock_start_timestamp).total_seconds()
        
        # Reject impossible elapsed times
        if elapsed_seconds < 0 or elapsed_seconds > 86400:
            return "0M"
        
        elapsed_minutes = int(elapsed_seconds / 60)
        
        if elapsed_minutes < 60:
            return f"{elapsed_minutes}M"
        else:
            hours = elapsed_minutes // 60
            minutes = elapsed_minutes % 60
            return f"{hours}H {minutes}M"
    
    except Exception as e:
        print(f"[HARMONIC FORMAT ERROR] {e}")
        return "0M"


def calculate_giza_alignment(sigs):
    try:
        sig_e = sigs["EGYPT"]
        sig_c = sigs["CYPRUS"]
        sig_i = sigs["ISRAEL"]

        min_l = min(len(sig_e), len(sig_c), len(sig_i))

        corr_ec = np.argmax(np.correlate(sig_e[:min_l], sig_c[:min_l], "full")) - min_l
        corr_ei = np.argmax(np.correlate(sig_e[:min_l], sig_i[:min_l], "full")) - min_l

        expected_ec = (590 / 3.5) * (FS / 1000)
        expected_ei = (400 / 3.5) * (FS / 1000)

        error = abs(corr_ec - expected_ec) + abs(corr_ei - expected_ei)
        alignment = max(0.0, 100.0 - (error / 10.0))
        return float(alignment)
    except Exception as e:
        print(f"[ALIGNMENT ERROR] {e}")
        return 0.0


def analyze_sniper_core(sigs):
    try:
        flags = sigs.get("_flags", {})
        simulated_stations = [k for k, v in flags.items() if v]
        if simulated_stations:
            print(f"[WARN] Symulowane dane dla stacji: {simulated_stations}")

        data_e = sigs["EGYPT"]

        fr, psd = welch(data_e, FS, nperseg=256)

        mask = (fr >= 5) & (fr <= 45)
        peak_idx = np.argmax(psd[mask])
        peak_hz = fr[mask][peak_idx]
        
        pow_peak = psd[mask][peak_idx]
        sigma = pow_peak / (np.mean(psd) + 1e-9)

        # ===== COMPOSITE H_CONF v15.6 (IMPROVED ADAPTIVE SCORING) =====
        # Multi-factor anomaly detection with weighted harmonics and adaptive noise floor
        
        noise_floor = np.mean(psd)
        
        # Goal 3: Adaptive noise floor - normalize against local spectral baseline
        spectral_stdev = np.std(psd[mask])
        adaptive_threshold = noise_floor + spectral_stdev
        
        # Factor 1: HARMONIC COHERENCE with weighted contribution (Goal 2: Not binary)
        harm_score = 0.0
        if pow_peak > adaptive_threshold * 2.0:  # Stronger adaptive threshold to reduce false positives
            p2 = psd[np.argmin(np.abs(fr - peak_hz * 2))]
            p3 = psd[np.argmin(np.abs(fr - peak_hz * 3))]
            harmonic_energy = p2 + p3
            harm_ratio = (harmonic_energy / (pow_peak + 1e-9))
            
            # Goal 2: Weighted harmonic contribution (smooth, partial scoring)
            # Low ratios reduce score instead of zeroing it
            if harm_ratio > 0.1:
                harm_score = (50.0 * harm_ratio) / (1.0 + (harm_ratio ** 2))
            else:
                # Weak harmonics still contribute partially (Goal 2)
                harm_score = harm_ratio * 15.0
            
            harm_score = min(40.0, max(0.0, harm_score))
        else:
            # Goal 2: Weak but coherent narrowband peaks contribute partially
            if pow_peak > adaptive_threshold * 0.5:
                harm_score = 5.0
        
        # Goal 4: Log harmonic score
        sniper.last_sub_scores["harm_score"] = float(harm_score)
        sniper.last_sub_scores["noise_floor"] = float(noise_floor)
        
        # Factor 2: SPECTRAL ENERGY DISTRIBUTION (0-30 weight)
        # === FIX: SAFE ENERGY SCORE NORMALIZATION ===
        e_low = np.sum(psd[(fr >= 7) & (fr <= 14)]) / (np.sum(psd) + 1e-9) * 100
        e_mid = np.sum(psd[(fr >= 15) & (fr <= 25)]) / (np.sum(psd) + 1e-9) * 100
        e_high = np.sum(psd[(fr >= 35) & (fr <= 45)]) / (np.sum(psd) + 1e-9) * 100
        
        # Normalize entropy to [0, 1] range using maximum entropy (log(3) for 3 bands)
        max_entropy = np.log(3.0)  # Maximum entropy for 3 equally distributed bands
        cluster_entropy = -((e_low + 1e-9) * np.log(e_low + 1e-9) + 
                           (e_mid + 1e-9) * np.log(e_mid + 1e-9) + 
                           (e_high + 1e-9) * np.log(e_high + 1e-9)) / max_entropy
        
        # Clamp entropy to [0, 1] to prevent explosion
        cluster_entropy = np.clip(cluster_entropy, 0.0, 1.0)
        
        # Energy score: focused (low entropy) = higher score
        # Safe formula: (1 - normalized_entropy) * max_weight, then clamp
        raw_energy_score = (1.0 - cluster_entropy) * 30.0
        energy_score = np.clip(raw_energy_score, 0.0, 30.0)
        
        sniper.last_sub_scores["energy_score"] = float(energy_score)
        
        # Factor 3: PERSISTENCE ENGINE (0-20 weight)
        active_now = None
        if e_high > e_low and e_high > e_mid and e_high > 2.0: 
            active_now = "HIGH"
        elif e_mid > e_low and e_mid > 2.0: 
            active_now = "MID"
        elif e_low > 2.0: 
            active_now = "LOW"

        if active_now:
            sniper.persistence[active_now] += 1
            for k in sniper.persistence:
                if k != active_now: sniper.persistence[k] = 0
        else:
            for k in sniper.persistence: sniper.persistence[k] = 0

        # Persistence reward: sustained activity
        persistence_score = min(20.0, (sniper.persistence[active_now] / 120.0) * 20.0 if active_now else 0.0)
        sniper.last_sub_scores["persistence_score"] = float(persistence_score)
        
        # Factor 4: MULTI-CLUSTER ACTIVITY (0-10 weight)
        # Anomaly if >2 significant clusters active simultaneously
        active_clusters = sum([
            1 for e in [e_low, e_mid, e_high] if e > 3.0
        ])
        cluster_score = min(10.0, (active_clusters - 1) * 5.0) if active_clusters > 1 else 0.0
        sniper.last_sub_scores["cluster_score"] = float(cluster_score)
        
        # COMPOSITE SCORE (Weighted sum) - Goal 2: Smooth, no sudden jumps
        raw_conf = (
            harm_score * 0.45 +
            energy_score * 0.30 +
            persistence_score * 0.15 +
            cluster_score * 0.10
        )

        # Soft saturation curve to prevent overreacting to stacked medium signals
        h_conf = 100 * (1 - np.exp(-raw_conf / 25))
        h_conf = float(min(100.0, max(0.0, h_conf)))
        
        # ===== DEBUG TELEMETRY: Print every ~20 updates =====
        global debug_counter
        debug_counter += 1
        if debug_counter % 20 == 0:
            print(f"\n[DEBUG TELEMETRY @ {datetime.now().strftime('%H:%M:%S')}]")
            print(f"  raw_conf (sum of weighted scores): {raw_conf:.4f}")
            print(f"  harm_score (0-40, weighted 0.45): {harm_score:.4f}")
            print(f"  energy_score (0-30, weighted 0.30): {energy_score:.4f}")
            print(f"  persistence_score (0-20, weighted 0.15): {persistence_score:.4f}")
            print(f"  cluster_score (0-10, weighted 0.10): {cluster_score:.4f}")
            print(f"  h_conf (final via soft saturation): {h_conf:.4f}")
            print(f"  --- Cluster state: {active_now} | Persistence count: {sniper.persistence}")
            print()

        # Wstępne tagowanie eventów (Negative Case Logging)
        tag = "BASELINE_STABLE"
        alignment = calculate_giza_alignment(sigs)
        
        if sigma > 8.0 and h_conf > 40.0:
            tag = "TECH_RESONANCE"
        elif alignment > 95.0 and h_conf > 55.0:
            tag = "GIZA_LOCK_ON"
        elif sigma > 12.0:
            tag = "HIGH_ENERGY_EVENT"

        psd_norm = psd[mask] / (np.sum(psd[mask]) + 1e-9)
        ent = entropy(psd_norm)

        h_sync = h_conf

        return {
            "sigma": float(sigma),
            "peak_hz": float(peak_hz),
            "ent": float(ent),
            "h_conf": float(h_conf),
            "tag": tag,
            "e_low": float(e_low),
            "e_mid": float(e_mid),   
            "e_high": float(e_high)  
        }
    except Exception as e:
        print(f"[ANALYSIS ERROR] {e}")
        return {
            "sigma": 1.0, "peak_hz": 11.25, "ent": 4.0, "h_conf": 0.0,
            "tag": "ERROR", "e_low": 0.0, "e_mid": 0.0, "e_high": 0.0
        }


# =========================
# START WORKERÓW
# =========================
sniper = GizaSniperStreamer()
threading.Thread(target=sniper.seismic_worker, daemon=True).start()
threading.Thread(target=sniper.tec_worker, daemon=True).start()

# ==============================================================================
# DASH APP
# ==============================================================================
app = dash.Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap"
])

app.layout = html.Div(id="main-container", style={
    "background": "#08090B", "minHeight": "100vh", "padding": "20px"
}, children=[

    html.Div(style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "flex-start",
        "borderBottom": "1px solid #222", "paddingBottom": "15px", "marginBottom": "20px"
    }, children=[
        html.Div([
            html.H1("GIZA_OS // SPACE_SENTINEL v15.0", style={
                "margin": "0", "fontFamily": "Orbitron, monospace",
                "color": "#FFB302", "fontSize": "20px", "letterSpacing": "2px"
            }),
            html.Div(id="sync-indicator", style={
                "color": "#00D4FF", "fontSize": "11px",
                "fontFamily": "Orbitron, monospace", "marginTop": "3px",
                "letterSpacing": "1px"
            }),
            html.Div(id="regulus-hud", style={
                "color": "#00D4FF", "fontSize": "12px",
                "fontFamily": "Orbitron, monospace", "marginTop": "5px"
            })
        ]),

        # SPECTRAL EQUALIZER v15.3 - Dynamic placeholder (updated by callback)
        html.Div(id="spectral-bars", style={
            "display": "flex", "justifyContent": "center", "alignItems": "flex-end",
            "height": "100px", "gap": "20px", "flex": "1", "margin": "0 20px"
        }),

        html.Div(style={"textAlign": "right"}, children=[
            html.Div([
                html.Span("IONOSPHERE: ", style={"color": "#666", "fontSize": "10px"}),
                html.Span(id="tec-val", style={"color": "#00FF41", "fontFamily": "Orbitron, monospace", "fontSize": "18px"})
            ]),
            html.Div([
                html.Span("GRAVITY LOAD: ", style={"color": "#666", "fontSize": "10px"}),
                html.Span(id="gravity-val", style={"color": "#FFB302", "fontFamily": "Orbitron, monospace", "fontSize": "18px"})
            ])
        ]),
    ]),

    html.Div(style={
        "display": "flex", "justifyContent": "space-around",
        "background": "#0D0E12", "padding": "10px",
        "border": "1px solid #222", "marginBottom": "20px"
    }, children=[
        html.Div([
            html.Span("VIRGO WINDOW (AUG 22): ", style={"color": "#888"}),
            html.Span(id="cd-aug", style={"color": "#FFB302", "fontFamily": "Orbitron, monospace"})
        ]),
        html.Div([
            html.Span("RED RISE (OCT 07): ", style={"color": "#888"}),
            html.Span(id="cd-oct", style={"color": "#FFB302", "fontFamily": "Orbitron, monospace"})
        ]),
        html.Div([
            html.Span("GIZA ALIGNMENT: ", style={"color": "#888"}),
            html.Span(id="alignment-val", style={
                "color": "#00FF41", "fontWeight": "bold", "fontFamily": "Orbitron, monospace"
            })
        ])
    ]),

    html.Div(style={"display": "flex", "gap": "20px"}, children=[

        html.Div(style={"width": "65%"}, children=[
            dcc.Graph(id="sigma-plot", config=GRAPH_CONFIG, style={"marginBottom": "20px"}),
            dcc.Graph(id="spec-plot", config=GRAPH_CONFIG, style={"marginBottom": "20px"}),
            html.Div(id="status-msg", style={
                "textAlign": "center", "fontFamily": "Orbitron, monospace", "fontSize": "14px"
            })
        ]),

        html.Div(style={"width": "35%"}, children=[
            html.Div(style={
                "border": "1px solid #1A1C21", "background": "#000", "padding": "5px"
            }, children=[
                dcc.Graph(id="radar-map", config=GRAPH_CONFIG)
            ]),
            html.Div(style={
                "marginTop": "20px", "display": "flex",
                "flexDirection": "column", "gap": "15px"
            }, children=[
                dcc.Graph(id="gauge-sigma", config=GRAPH_CONFIG),
                html.Div(style={
                    "background": "#0D0E12", "padding": "15px", "border": "1px solid #222"
                }, children=[
                    html.Div([
                        html.Span("LOCK DURATION: ", style={"color": "#888"}),
                        html.Span(id="lock-duration-val", style={"color": "#00FF41", "fontFamily": "Orbitron, monospace"})
                    ]),
                    html.Div([
                        html.Span("H-CONF: ", style={"color": "#888"}),
                        html.Span(id="h-conf-val", style={"color": "#FFB302", "fontFamily": "Orbitron, monospace"})
                    ], style={"marginTop": "5px"}),
                    html.Div([
                        html.Span("SIGNAL ORDER: ", style={"color": "#888"}),
                        html.Span(id="entropy-val", style={"color": "#FFB302", "fontFamily": "Orbitron, monospace"})
                    ], style={"marginTop": "5px"})
                ]),
                html.Div(id="log-display", style={
                    "height": "180px", "overflowY": "auto",
                    "background": "#050505", "fontSize": "11px",
                    "padding": "10px", "border": "1px solid #222",
                    "color": "#00D4FF", "fontFamily": "monospace"
                })
            ])
        ])
    ]),

    dcc.Interval(id="heartbeat", interval=5000, n_intervals=0, max_intervals=-1)
])


# ==============================================================================
# MISSION CONTROL CALLBACK
# ==============================================================================
@app.callback(
    [Output("sigma-plot", "figure"), Output("spec-plot", "figure"),
     Output("radar-map", "figure"), Output("gauge-sigma", "figure"),
     Output("status-msg", "children"), Output("status-msg", "style"),
     Output("alignment-val", "children"), Output("lock-duration-val", "children"),
     Output("h-conf-val", "children"), Output("entropy-val", "children"),
     Output("tec-val", "children"), Output("gravity-val", "children"),
     Output("regulus-hud", "children"), Output("cd-aug", "children"),
     Output("cd-oct", "children"), Output("main-container", "style"),
     Output("log-display", "children"), Output("spectral-bars", "children"),
     Output("sync-indicator", "children")],
    [Input("heartbeat", "n_intervals")]
)
def update_mission_control(n):
    sniper.simulation_mode = False

    # GUARDIAN: Log environment data every 60 seconds
    guardian.log_environment()

    try:
        batch = sniper.q.get(timeout=10)
    except Exception as e:
        print(f"[QUEUE TIMEOUT] {e}")
        batch = {s["name"]: np.random.normal(0, 1, 1000) for s in STATIONS}
        batch["_flags"] = {s["name"]: True for s in STATIONS}
    
    # STEP 1: CALLBACK RETURN FIX - Receive all variables from analyze_sniper_core
    result = analyze_sniper_core(batch)
    sigma = result["sigma"]
    peak_hz = result["peak_hz"]
    ent = result["ent"]
    h_conf = result["h_conf"]
    tag = result["tag"]
    e_low = result["e_low"]
    e_mid = result["e_mid"]
    e_high = result["e_high"]
    align = calculate_giza_alignment(batch)
    
    # STEP 2: Calculate synchronized time (T-15min offset) for astro metrics
    sync_time = datetime.now(timezone.utc) - timedelta(seconds=SYNC_OFFSET)
    
    # STEP 3: Pass sync_time to get_space_metrics for historical ephemeris calculation
    s_metrics = get_space_metrics(target_time=sync_time)
    tec = sniper.tec_val
    grav_load = s_metrics['grav_load']

    rolling_stats.append(sigma)
    avg_sigma = np.mean(rolling_stats) if len(rolling_stats) > 10 else 5.0
    std_sigma = np.std(rolling_stats) if len(rolling_stats) > 10 else 1.0
    z_score = (sigma - avg_sigma) / (std_sigma + 1e-6)

    # Jednolita logika statusu
    is_anomaly = z_score > 4.5
    h_sync = result["h_conf"]
    is_locked = align > 80.0 and h_sync > 50.0 and sigma > 10.0
    
    # Track harmonic lock transitions
    if is_locked and not sniper.last_lock_state:
        # FALSE -> TRUE transition: start tracking elapsed time
        sniper.harmonic_lock_start = datetime.now(timezone.utc)
        sniper.last_lock_state = True
    elif not is_locked and sniper.last_lock_state:
        # TRUE -> FALSE transition: reset tracking
        sniper.harmonic_lock_start = None
        sniper.last_lock_state = False

    # Ustalenie koloru tła
    bg_color = "#220000" if is_anomaly else "#08090B"
    container_style = {
        "backgroundColor": bg_color,
        "minHeight": "100vh",
        "padding": "20px",
        "transition": "background 0.6s ease"
    }

    # Przygotowanie tekstu statusu i stylu
    font_orbitron = "Orbitron, monospace"
    if is_locked:
        status_txt = ">> [!] TARGET LOCKED: GIZA CORE ACTIVE [!]"
        status_color = {
            "color": "#FF0000", "fontWeight": "bold",
            "border": "2px solid #FF0000", "padding": "10px",
            "backgroundColor": "rgba(255,0,0,0.1)", "fontFamily": font_orbitron
        }
    elif is_anomaly:
        status_txt = f">> ANOMALY DETECTED (Z-SCORE: {z_score:.2f})"
        status_color = {
            "color": "#FFB302", "border": "1px solid #FFB302",
            "padding": "10px", "fontFamily": font_orbitron
        }
    else:
        status_txt = ">> SCANNING_DEEP_NOISE // BASELINE_STABLE"
        status_color = {
            "color": "#00D4FF", "border": "1px solid #333",
            "padding": "10px", "fontFamily": font_orbitron
        }

    # ZAPIS LOGU v15.2 (Triple Planet + T-15min Sync)
    ts = datetime.now(timezone.utc)

    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{sigma:.4f},{peak_hz:.2f},{tec:.2f},{align:.1f},{ent:.4f},{grav_load:.2f},"
                f"{h_conf:.1f},{s_metrics['reg_az']:.2f},{s_metrics['ven_az']:.2f},{s_metrics['jup_az']:.2f},{tag}\n")

    f_sigma = go.Figure(go.Scatter(
        y=list(rolling_stats)[-100:], mode='lines',
        line=dict(color="#FFB302", width=2),
        fill='tozeroy', fillcolor="rgba(255,179,2,0.1)"
    ))
    f_sigma.update_layout(
        title="SYSTEM_COHERENCE_SIGMA (ROLLING)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#888", size=10),
        margin=dict(t=30, b=10, l=10, r=10), height=200,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#222")
    )

    # REAL PSD SPECTRUM v15.4
    data_e = batch["EGYPT"]
    fr_spec, psd_spec = welch(data_e, FS, nperseg=256)

    mask_spec = (fr_spec >= 5) & (fr_spec <= 45)

    f_spec = go.Figure()

    f_spec.add_trace(go.Scatter(
        x=fr_spec[mask_spec],
        y=psd_spec[mask_spec],
        mode='lines',
        line=dict(color="#00D4FF", width=2),
        fill='tozeroy',
        fillcolor="rgba(0,212,255,0.15)"
    ))

    # Peak marker
    f_spec.add_vline(
        x=peak_hz,
        line_width=2,
        line_dash="dash",
        line_color="#FF0000"
    )

    f_spec.update_layout(
        title=f"SPECTRAL ANALYSIS // PEAK {peak_hz:.2f} Hz",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#888", size=10),
        margin=dict(t=30, b=10, l=10, r=10),
        height=220,
        xaxis=dict(
            title="Frequency (Hz)",
            gridcolor="#222"
        ),
        yaxis=dict(
            title="Power",
            gridcolor="#222"
        )
    )

    f_map = go.Figure()

    # GIZA
    f_map.add_trace(go.Scattermapbox(
        lat=[GIZA_LAT],
        lon=[GIZA_LON],
        mode="markers",
        marker=dict(
            size=30,
            color="#FF0000" if is_locked else "#FFB302",
            opacity=0.9
        ),
        text=["GIZA CORE"],
        name="GIZA"
    ))

    # STACJE
    for s in STATIONS:
        f_map.add_trace(go.Scattermapbox(
            lat=[s["lat"]],
            lon=[s["lon"]],
            mode="markers+text",
            marker=dict(size=14, color="#00D4FF"),
            text=[s["name"]],
            textposition="top center",
            name=s["name"]
        ))

        # Linie coherence
        f_map.add_trace(go.Scattermapbox(
            lat=[GIZA_LAT, s["lat"]],
            lon=[GIZA_LON, s["lon"]],
            mode="lines",
            line=dict(
                width=max(1, align / 25),
                color="rgba(0,212,255,0.35)"
            ),
            hoverinfo="none",
            showlegend=False
        ))

    f_map.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=29.98, lon=31.13), zoom=4),
        paper_bgcolor="rgba(0,0,0,0)", 
        margin=dict(l=0, r=0, t=0, b=0), 
        height=400, 
        showlegend=False
    )

    g_sigma = go.Figure(go.Indicator(
        mode="gauge+number", value=sigma,
        title={'text': "SIGMA_PRIME", 'font': {'family': 'Orbitron', 'size': 12, 'color': '#FFB302'}},
        gauge={
            'axis': {'range': [0, 40], 'tickcolor': "#FFB302"},
            'bar': {'color': "#FFB302"},
            'bgcolor': "#0D0E12",
            'steps': [
                {'range': [0, 15], 'color': 'rgba(0, 212, 255, 0.1)'},
                {'range': [15, 30], 'color': 'rgba(255, 179, 2, 0.1)'},
                {'range': [30, 40], 'color': 'rgba(255, 0, 0, 0.2)'}
            ]
        }
    ))
    g_sigma.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="#FFB302", size=12),
        margin=dict(t=30, b=20, l=30, r=30), 
        height=180
    )

    now = datetime.now(timezone.utc)
    def cd(target):
        diff = target - now
        if diff.total_seconds() <= 0:
            return "WINDOW_OPEN"
        return f"{diff.days}D {diff.seconds // 3600}H"

    # Nowy pasek nawigacyjny HUD
    astro_txt = html.Div([
        html.Span(f"VENUS: {s_metrics['ven_az']:.1f}° ", style={"color": "#E1ADFF", "marginRight": "15px"}),
        html.Span(f"JUPITER: {s_metrics['jup_az']:.1f}° ", style={"color": "#FFD700", "marginRight": "15px"}),
        html.Span(f"REGULUS: {s_metrics['reg_az']:.1f}°", style={"color": "#FF4444"})
    ])
    
    # Goal 1: Add ASTRO_STALE indicator if cache is stale
    if astro_cache.astro_stale:
        astro_txt = html.Div([
            html.Span("[ASTRO_STALE] ", style={"color": "#FF6666", "fontWeight": "bold"}),
            astro_txt
        ])

    # LIVE EVENT LOGGER v15.4
    event_line = f"{datetime.now().strftime('%H:%M:%S')} | {tag} | {peak_hz:.1f}Hz | SIGMA {sigma:.2f}"

    if len(log_buffer) == 0 or log_buffer[-1] != event_line:
        log_buffer.append(event_line)
    
    log_entries = [html.Pre(e, style={"margin": "0", "fontSize": "11px"}) for e in log_buffer]

    bars_children = [
        html.Div(style={
            "width": "8px", "backgroundColor": "#00D4FF",
            "height": f"{max(10, e_low * 2)}px", "opacity": 0.7
        }),
        html.Div(style={
            "width": "8px", "backgroundColor": "#FFB302",
            "height": f"{max(10, e_mid * 2)}px", "opacity": 0.7
        }),
        html.Div(style={
            "width": "8px", "backgroundColor": "#FF0000" if e_high > 20 else "#00FF41",
            "height": f"{max(10, e_high * 2)}px", "opacity": 0.7
        })
    ]

    # HUD LABELS CLARIFICATION: Renamed "HARMONIC SYNC" → "LOCK DURATION"
    lock_duration = format_elapsed_harmonic(sniper.harmonic_lock_start)
    lock_duration_txt = lock_duration if is_locked else "0M"
    h_conf_txt = f"{h_conf:.1f}%"
    
    return (
        f_sigma, f_spec, f_map, g_sigma,
        status_txt, status_color,
        f"{align:.1f}%", lock_duration_txt, h_conf_txt, f"{ent:.3f}",
        f"{tec:.1f}", f"{grav_load:.1f}",
        astro_txt,
        cd(DATE_AUG), cd(DATE_OCT),
        container_style, log_entries,
        bars_children, f"T-15min SYNC // {sync_time.strftime('%H:%M:%S UTC')}"
    )

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
