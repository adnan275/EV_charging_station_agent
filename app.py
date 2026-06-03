import streamlit as st
import os
import time
import joblib
import pickle
import pandas as pd
import numpy as np

# Suppress Streamlit Cloud Protobuf & Telemetry Errors
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangGraphDeprecatedSince.*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv()

# LangChain Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

# Custom Engineering Modules
from simulation_engine import StationDatabase, QueuingModel, GridSimulator, CarbonCalculator
from explainability_engine import ExplainabilityEngine

# Set Page Config
st.set_page_config(page_title="EV Charging AI Agent", page_icon="⚡", layout="wide")

# Custom Premium Styling Injections
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Pure Black Base */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar - Deepest Black */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Premium Card Design */
    .metric-card {
        background: #0a0a0a;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: rgba(0, 255, 136, 0.3);
        background: #0c0c0c;
        transform: translateY(-2px);
    }
    
    /* Typography Refinement */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin-bottom: 10px;
    }
    
    /* Subtle Accents */
    .accent-text {
        color: #00ff88;
        font-weight: 600;
    }
    
    /* Chat Interface - Minimalist */
    .stChatMessage {
        background-color: #080808 !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
    }
    
    /* Input Field */
    [data-testid="stChatInput"] {
        background-color: #0a0a0a !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    /* Buttons - Sleek & Dark */
    .stButton>button {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem !important;
    }
    .stButton>button:hover {
        background-color: #00ff88 !important;
        color: #000000 !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2) !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Map Container */
    [data-testid="stMap"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# API Key Logic
api_key = None
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    pass

if not api_key:
    api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("🔑 **Groq API Key not found.**")
    st.info("Please add `GROQ_API_KEY` to your Streamlit Secrets (Cloud) or `.env` file (Local).")
    st.stop()

# Rate Limiter
RATE_LIMIT_SECONDS = 5

def check_rate_limit():
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0
    current_time = time.time()
    elapsed = current_time - st.session_state.last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        return False, int(RATE_LIMIT_SECONDS - elapsed)
    return True, 0

# ML Models Loading
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_models():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_model = joblib.load('rf_balanced_retrained_fe.joblib')
            xgb_model = joblib.load('xgb_cs_retrained_fe.joblib')
            scaler = joblib.load('scaler.joblib')
            
            with open('label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
            with open('optimal_threshold.pkl', 'rb') as f:
                threshold = pickle.load(f)
                
        return rf_model, xgb_model, scaler, le, float(threshold)
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None, None, None, None, None

rf_model, xgb_model, scaler, le, threshold = load_models()

# RAG Knowledge Base Initialization
@st.cache_resource(show_spinner="Initializing in-memory RAG Knowledge Base...")
def setup_rag():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        docs = [
            # ─── Charging Levels & Speeds ───────────────────────────────────────────────
            "Fast DC charging stations (Level 3) provide 50kW to 350kW, providing 80% charge in under 30 minutes.",
            "AC Charging (Level 2) typically provides 7kW to 22kW and is common for home and workplace charging.",
            "Level 1 charging uses a standard 120V household outlet and delivers 1.4kW to 1.9kW, adding roughly 5 miles of range per hour.",
            "Ultra-fast DC chargers from 150kW to 350kW can add up to 200 miles of range in just 15-20 minutes for compatible vehicles.",
            "Charging speed depends on both the charger output power and the vehicle onboard charger capacity — the lower of the two limits actual speed.",
            "Most modern EVs support AC onboard charging between 7.2kW and 22kW depending on the vehicle built-in charger.",
            "Some heavy-duty electric trucks support charging at 1MW or higher using the MCS (Megawatt Charging System) standard.",
            "Pantograph charging is used for electric buses and trucks, allowing automatic overhead connection without a physical plug.",
            "Wireless or inductive charging pads for EVs operate at 3.3kW to 22kW and eliminate the need for physical cable connection.",
            "Dynamic wireless charging embeds inductive coils into road surfaces, allowing EVs to charge while driving over equipped roads.",
            "Charging efficiency losses of 10-15% are typical from the grid to the battery due to heat and conversion losses in AC/DC conversion.",
            "Bidirectional charging allows an EV to both receive and send power, enabling V2G, V2H, and V2L (vehicle-to-load) functionality.",

            # ─── Connector Standards ────────────────────────────────────────────────────
            "The CCS2 (Combined Charging System 2) standard is the most common connector for Fast DC charging in Europe.",
            "In North America, Tesla's NACS and CCS1 are the dominant Fast DC connector standards.",
            "The CHAdeMO connector standard, developed in Japan, is used by Nissan Leaf and some older EVs but is being phased out in favor of CCS.",
            "Tesla's NACS (North American Charging Standard) has been adopted by Ford, GM, Rivian, and other major automakers as the de facto US standard.",
            "The GB/T standard is the national EV charging connector standard mandated in China for both AC and DC charging.",
            "Type 2 (Mennekes) connector is the standard AC charging plug across Europe, supporting both single-phase and three-phase charging.",
            "Type 1 (SAE J1772) is the standard AC connector used in North America and Japan for Level 1 and Level 2 charging.",
            "CCS (Combined Charging System) combines the AC Type 1 or Type 2 plug with two additional DC pins for fast DC charging through one port.",
            "The MCS (Megawatt Charging System) connector is being standardized for heavy-duty electric trucks and commercial vehicles.",
            "Adapter dongles allow cross-standard compatibility, such as CCS-to-NACS or CHAdeMO-to-CCS adapters for wider network access.",
            "The IEC 62196 is the international standard that governs EV connector types, including Type 1, Type 2, and CCS variants.",
            "Proprietary charging connectors like early Tesla designs are being phased out in favor of open, standardized connectors globally.",

            # ─── Infrastructure & Locations ─────────────────────────────────────────────
            "High-density urban areas and highway corridors are primary locations for Fast DC infrastructure.",
            "Highway charging corridors typically space Fast DC stations every 50 to 100 miles to support long-distance EV travel.",
            "Retail locations such as shopping malls, grocery stores, and restaurants are increasingly popular sites for Level 2 charging.",
            "Workplace charging programs are expanding rapidly with Level 2 chargers installed in office and corporate campus parking lots.",
            "Public parking garages in city centers are a growing location for both Level 2 and Fast DC charging infrastructure.",
            "Fleet charging depots use high-power DC chargers to rapidly recharge commercial EVs, delivery vans, and electric buses overnight.",
            "Rural charging infrastructure remains underdeveloped in many regions, creating range anxiety for EV drivers outside urban centers.",
            "Hotels and hospitality venues are increasingly offering Level 2 EV charging as an amenity for overnight guests.",
            "Airport long-term parking lots are an ideal location for Level 2 charging, allowing vehicles to charge over multi-day stays.",
            "Supermarkets and big-box retailers like Walmart and Target are deploying thousands of charging stalls across their US parking lots.",
            "Charging hubs — large dedicated sites with 20 or more fast chargers — are being developed in major cities as EV adoption grows.",
            "Gas stations are progressively adding EV charging bays alongside traditional fuel pumps to serve both ICE and EV customers.",
            "Sports stadiums and entertainment venues install Level 2 chargers to serve EV-driving fans attending events.",
            "Multi-unit dwellings (apartments and condos) face unique infrastructure challenges for EV charging due to shared electrical systems.",
            "Curbside charging using lamp posts or dedicated street units is being piloted in cities to serve residents without home charging access.",
            "School and university campuses are adding Level 2 chargers to support students, staff, and fleet electrification.",

            # ─── Charging Networks & Operators ──────────────────────────────────────────
            "Tesla Supercharger is the largest proprietary Fast DC charging network globally with over 50,000 connectors worldwide.",
            "ChargePoint is one of the largest open charging networks operating Level 2 and DC fast chargers across North America and Europe.",
            "Electrify America operates a high-power DC fast charging network across the United States with chargers up to 350kW.",
            "IONITY is a joint venture by major European automakers providing high-power 350kW charging across European highway corridors.",
            "EVgo, Blink, and Shell Recharge are among the major public charging network operators in North America.",
            "Many charging networks offer RFID cards, mobile apps, and Plug & Charge (ISO 15118) authentication for seamless access.",
            "BP Pulse, Pod Point, and Osprey are major charging network operators expanding rapidly across the United Kingdom.",
            "Fastned operates a European fast charging network focused exclusively on renewable energy-powered stations.",
            "Roaming agreements between charging networks allow EV drivers to access multiple networks with a single account or card.",
            "Open Charge Point Protocol (OCPP) is the industry-standard communication protocol connecting chargers to network management systems.",
            "Open Charge Point Interface (OCPI) enables roaming between different charging networks through a standardized data exchange protocol.",
            "Charging network uptime reliability is a critical metric, with industry targets generally set above 97% operational availability.",
            "Tesla opened its Supercharger network to non-Tesla vehicles in many countries starting in 2022, significantly expanding access.",
            "Charging-as-a-Service (CaaS) business models allow businesses to deploy chargers with no upfront cost in exchange for revenue sharing.",

            # ─── Battery Technology ──────────────────────────────────────────────────────
            "Modern EV battery packs typically range from 40 kWh to over 100 kWh providing real-world ranges of 150 to 400+ miles.",
            "Battery degradation in EVs typically results in a 2-3% capacity loss per year under normal usage and charging conditions.",
            "Lithium Iron Phosphate (LFP) batteries are safer, have longer cycle lives, and can be charged to 100% daily without significant degradation.",
            "NMC (Nickel Manganese Cobalt) batteries offer higher energy density than LFP and are common in long-range EVs.",
            "Thermal management systems in EV batteries maintain optimal temperature ranges to maximize charging speed and battery longevity.",
            "DC fast charging at very high power levels generates significant heat, so many EVs pre-condition the battery before arriving at a fast charger.",
            "The 80% charging rule is recommended for Fast DC sessions because charging slows significantly above 80% to protect battery health.",
            "Solid-state batteries promise higher energy density, faster charging, and improved safety over current liquid-electrolyte lithium-ion cells.",
            "Battery Management Systems (BMS) monitor cell voltage, temperature, and state of charge to optimize performance and safety.",
            "Second-life EV batteries — packs removed from vehicles — are repurposed for stationary energy storage applications.",
            "Sodium-ion batteries are an emerging alternative to lithium-ion, using abundant sodium and offering improved low-temperature performance.",
            "Cell-to-pack (CTP) and cell-to-body (CTB) designs eliminate battery module layers, increasing energy density and reducing manufacturing cost.",
            "Battery swap technology, championed by NIO in China and Gogoro for scooters, allows a depleted pack to be exchanged for a full one in minutes.",
            "EV battery warranties typically guarantee at least 70% capacity retention over 8 years or 100,000 miles in most major markets.",
            "Cylindrical, prismatic, and pouch cells are the three main form factors used in EV battery pack construction.",
            "Fast charging repeatedly at very high C-rates (charge rates) accelerates lithium plating, which can permanently reduce battery capacity.",
            "Battery pre-conditioning — heating or cooling the pack to optimal temperature — is essential for maximizing DC fast charging speed.",

            # ─── Smart Charging & Grid Integration ──────────────────────────────────────
            "Smart charging systems communicate between EVs, chargers, and the grid to optimize energy usage and reduce peak demand costs.",
            "Vehicle-to-Grid (V2G) technology allows EVs to discharge stored energy back to the electrical grid during peak demand periods.",
            "Vehicle-to-Home (V2H) technology enables EVs to power a household during outages or high electricity price periods.",
            "Vehicle-to-Load (V2L) allows EVs to power external devices and appliances directly from the car's battery via an outlet adapter.",
            "Time-of-use (TOU) electricity pricing incentivizes EV owners to charge during off-peak hours typically late at night.",
            "ISO 15118 is the international communication standard enabling Plug & Charge, automated billing, and V2G functionality.",
            "Demand response programs allow utilities to manage charging loads by incentivizing EV owners to shift charging times.",
            "Grid-scale EV charging hubs increasingly incorporate on-site battery storage to reduce peak demand charges and grid strain.",
            "Managed charging software platforms allow fleet operators to schedule and optimize charging across large numbers of vehicles.",
            "EV aggregation platforms virtually pool thousands of EVs into a single grid resource to provide frequency regulation services.",
            "Solar-canopy charging stations combine photovoltaic panels with EV chargers, generating renewable energy on-site.",
            "Microgrids integrating EVs, solar, and battery storage can operate independently from the main grid during outages.",
            "The duck curve — a grid imbalance caused by midday solar generation — can be smoothed by smart EV charging scheduling.",

            # ─── Government Policy & Incentives ─────────────────────────────────────────
            "The US Bipartisan Infrastructure Law allocated $7.5 billion to build a national EV charging network of 500,000 chargers by 2030.",
            "The European Union mandates Fast DC charging stations every 60 km on major TEN-T transport corridors by 2025.",
            "Many governments offer tax credits, rebates, and incentives for both EV purchases and home charging equipment installation.",
            "California's CARB regulations and Zero Emission Vehicle (ZEV) mandate have driven significant charging infrastructure growth.",
            "The US federal tax credit offers up to $7,500 for new EV purchases and up to $4,000 for used EVs under the Inflation Reduction Act.",
            "The EU has set a de facto ban on new internal combustion engine car sales from 2035, accelerating EV adoption across member states.",
            "China provides substantial central and local government subsidies for EV purchases and charging infrastructure deployment.",
            "The UK offers grants for home EV charger installations through the Electric Vehicle Homecharge Scheme (EVHS).",
            "Zero Emission Zones (ZEZs) in cities like London, Amsterdam, and Oslo restrict ICE vehicle access, incentivizing EV adoption.",
            "Corporate fleet electrification mandates in several countries require government and public sector fleets to transition to EVs by set dates.",
            "Low Emission Zones (LEZ) and congestion charge exemptions for EVs in major cities provide financial incentives for EV ownership.",

            # ─── EV Market & Adoption ───────────────────────────────────────────────────
            "Global EV sales exceeded 14 million units in 2023 representing over 18% of all new car sales worldwide.",
            "China is the world's largest EV market accounting for more than 60% of global EV sales in recent years.",
            "The total cost of ownership for EVs is increasingly competitive with ICE vehicles when factoring in fuel and maintenance savings.",
            "Range anxiety — concern about running out of charge — is cited as a top barrier to EV adoption alongside charging availability.",
            "Popular consumer EVs include the Tesla Model 3, Tesla Model Y, BYD Atto 3, Hyundai IONIQ 6, and Volkswagen ID.4.",
            "Electric two-wheelers and three-wheelers dominate EV adoption in Southeast Asia and India due to lower costs and urban utility.",
            "Electric buses are the fastest-growing EV segment in public transit with China operating over 600,000 electric buses.",
            "Electric commercial vans and light-duty trucks are seeing rapid adoption for last-mile delivery by companies like Amazon and FedEx.",
            "BYD surpassed Tesla as the world's top-selling EV manufacturer by volume in 2023.",
            "The used EV market is growing rapidly as lease returns and trade-ins increase availability of affordable second-hand electric vehicles.",
            "EV penetration rates exceed 80% of new car sales in Norway, making it the global leader in EV adoption per capita.",
            "Ride-hailing platforms like Uber and Lyft are incentivizing driver fleet electrification through bonus programs and charging partnerships.",

            # ─── Safety & Standards ─────────────────────────────────────────────────────
            "EV charging stations must comply with IEC 61851 for AC charging and IEC 62196 for connector safety standards internationally.",
            "Ground Fault Circuit Interrupter (GFCI) protection is mandatory in EV charging equipment to prevent electric shock hazards.",
            "Thermal runaway in lithium-ion batteries is a chain reaction of heat generation that can lead to fire — mitigated by advanced BMS design.",
            "UL 2594 is the safety standard for EV supply equipment (EVSE) in North America covering electrical, mechanical, and environmental safety.",
            "IP54 and IP55 ratings are the minimum ingress protection standards for outdoor EV charging equipment against dust and water.",
            "EV charging stations include automatic power cutoff if a fault, ground fault, or cable damage is detected.",
            "Fire suppression systems are increasingly installed in underground parking garages to address EV battery fire risks.",
            "EMF (electromagnetic field) emissions from EV wireless charging are regulated to stay within safe human exposure limits.",

            # ─── Environmental Impact ────────────────────────────────────────────────────
            "EVs charged on renewable energy grids produce near-zero lifecycle carbon emissions compared to ICE vehicles.",
            "Even on coal-heavy grids, EVs typically produce 30-40% lower lifecycle CO2 emissions than equivalent gasoline vehicles.",
            "Battery production for EVs is energy-intensive, but lifecycle analyses show EVs break even on carbon emissions after 1-3 years of use.",
            "Lithium, cobalt, nickel, and manganese mining for EV batteries raises environmental and ethical sourcing concerns.",
            "Battery recycling programs recover lithium, cobalt, and nickel from end-of-life packs, reducing the need for virgin material extraction.",
            "The carbon intensity of EV charging is improving globally as electricity grids incorporate increasing shares of renewable energy.",
            "Regenerative braking in EVs recovers kinetic energy during deceleration and converts it back into stored electrical energy.",

            # ─── Emerging Technologies ──────────────────────────────────────────────────
            "Solid-state batteries are expected to reach commercial EV production by 2027-2030, promising 500+ mile ranges and 10-minute charging.",
            "Autonomous EV charging robots can locate and plug into parked vehicles without any driver involvement.",
            "AI-powered charging management systems predict demand, optimize pricing, and pre-position mobile charging units dynamically.",
            "Hydrogen fuel cell electric vehicles (FCEVs) complement battery EVs for long-haul trucking and applications where charging time is critical.",
            "Blockchain-based platforms are being developed to enable peer-to-peer EV energy trading and transparent carbon credit tracking.",
            "Augmented reality (AR) navigation tools guide EV drivers to available chargers in real time and display live queue and pricing information.",
            "Mobile EV charging trucks and vans provide on-demand charging to stranded or fleet vehicles without fixed infrastructure.",
            "EV charging integration with autonomous vehicles will enable self-driving cars to navigate to chargers and recharge without human input.",
        ]
        
        db = Chroma.from_texts(
            texts=docs,
            embedding=embeddings,
            metadatas=[{"source": "EV Facts"}] * len(docs)
        )
        return db
    except Exception as e:
        st.error(f"Failed to initialize RAG database: {e}")
        return None

db = setup_rag()

# Agent Tools Definition
@tool
def search_ev_knowledge(query: str) -> str:
    """Search for factual information about EV charging infrastructure, connector standards, and EV charging speeds."""
    if db is None:
        return "Knowledge database offline."
    try:
        results = db.similarity_search(query, k=2)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def check_station_existence(latitude: float, longitude: float, radius_km: float = 10.0) -> str:
    """Checks if a charging station actually exists in reality within a given radius (in km) around coordinates.
    Always run this check first when the user queries specific coordinates.
    """
    try:
        nearby = StationDatabase.find_nearby_stations(latitude, longitude, radius_km)
        if not nearby:
            return (
                f"Reality Check: No active charging stations found in our real-world records within {radius_km} km of coordinates ({latitude:.4f}, {longitude:.4f}). "
                "However, this location can be evaluated for new site suitability. Tell the user you can predict if it's a suitable location using ML models via `predict_fast_dc`."
            )
        
        # Plot closest 5 on the dashboard map
        df_nearby = pd.DataFrame(nearby[:5])
        st.session_state.map_data = df_nearby[["latitude", "longitude", "name", "ports", "is_fast_dc"]]
        
        closest = nearby[0]
        st.session_state.last_location = {
            "latitude": float(closest["latitude"]),
            "longitude": float(closest["longitude"]),
            "name": closest["name"],
            "ports": int(closest["ports"]),
            "power_kw": float(closest["power_kw"]) if not pd.isna(closest["power_kw"]) else 150.0,
            "is_fast_dc": bool(closest["is_fast_dc"]),
            "country_code": closest["country_code"]
        }
        
        result_str = (
            f"Reality Check: Found {len(nearby)} real station(s) within {radius_km} km. Here are the details of the closest one:\n"
            f"- **Name:** {closest['name']}\n"
            f"- **Distance:** {closest['distance_km']:.2f} km\n"
            f"- **Ports:** {closest['ports']}\n"
            f"- **Power Rating:** {closest['power_kw']} kW ({closest['power_class']})\n"
            f"- **Type:** {'Fast DC Charger (Level 3)' if closest['is_fast_dc'] else 'AC Charger (Level 2)'}\n\n"
            f"I have plotted the closest {len(df_nearby)} stations on your interactive map. "
            f"You can now run a live simulation of the electrical grid load, vehicle queue, and carbon savings for this station by asking me to run a simulation!"
        )
        return result_str
    except Exception as e:
        return f"Error checking station existence: {str(e)}"


@tool
def predict_fast_dc(country_code: str, latitude: float, longitude: float, ports: int) -> str:
    """Predicts if a new location is likely suitable for a Fast DC charging station (Level 3) vs AC charging based on coordinates and ports.
    After predicting, suggest that you can explain the ML decision drivers using explain_ml_prediction.
    """
    if rf_model is None:
        return "ML Models not loaded."
    try:
        input_data = pd.DataFrame(
            [[country_code, latitude, longitude, ports]],
            columns=['country_code', 'latitude', 'longitude', 'ports']
        )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                input_data['country_code'] = le.transform(input_data['country_code'])
        except Exception:
            input_data['country_code'] = -1
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input_data[['latitude', 'longitude', 'ports']] = scaler.transform(input_data[['latitude', 'longitude', 'ports']])
            
        # Feature Engineering (MUST MATCH TRAINING EXACTLY)
        input_data['latitude_x_longitude'] = input_data['latitude'] * input_data['longitude']
        input_data['ports_x_latitude'] = input_data['ports'] * input_data['latitude']
        input_data['ports_x_longitude'] = input_data['ports'] * input_data['longitude']
        input_data['latitude_squared'] = input_data['latitude'] ** 2
        input_data['longitude_squared'] = input_data['longitude'] ** 2
        input_data['ports_squared'] = input_data['ports'] ** 2
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_prob = rf_model.predict_proba(input_data)[:, 1]
            xgb_prob = xgb_model.predict_proba(input_data)[:, 1]
            
        ensemble_prob = (rf_prob[0] + xgb_prob[0]) / 2
        result = "POSITIVE" if ensemble_prob >= threshold else "NEGATIVE"
        
        # Update map to show the predicted virtual station
        pred_label = f"Predicted Station: {'Fast DC suitable' if result == 'POSITIVE' else 'AC only suitable'}"
        new_row = pd.DataFrame([[latitude, longitude, pred_label, ports, result == "POSITIVE"]], 
                               columns=["latitude", "longitude", "name", "ports", "is_fast_dc"])
        st.session_state.map_data = pd.concat([st.session_state.map_data, new_row], ignore_index=True)
        
        # Save last location state
        st.session_state.last_location = {
            "latitude": latitude,
            "longitude": longitude,
            "name": f"Predicted: {result}",
            "ports": ports,
            "power_kw": 150.0 if result == "POSITIVE" else 22.0,
            "is_fast_dc": result == "POSITIVE",
            "country_code": country_code
        }
        
        return (
            f"Prediction: **{result}** suitability for Fast DC Charger.\n"
            f"- **Ensemble Confidence Score:** {ensemble_prob:.2f} (Threshold: {threshold:.2f})\n\n"
            f"I have plotted this proposed site on the interactive map. "
            f"If you want to know which features (coordinates or ports) influenced this ML prediction most, ask me to explain it!"
        )
        
    except Exception as e:
        return f"Prediction Error: {str(e)}"


@tool
def explain_ml_prediction(country_code: str, latitude: float, longitude: float, ports: int, confidence_score: float, prediction: str) -> str:
    """Explains why the ML ensemble model predicted a location as suitable or unsuitable for Fast DC charging.
    Use this when the user asks 'Why?' or requests feature explanation/weights for the prediction.
    """
    try:
        report = ExplainabilityEngine.explain_prediction(
            rf_model, xgb_model, scaler, le,
            country_code, latitude, longitude, ports,
            confidence_score, prediction
        )
        
        if isinstance(report, str):
            return report
            
        summary = (
            f"### 🧠 Explainable AI (XAI) Model Decision Analysis\n"
            f"The model predicted **{report['prediction']}** for Fast DC charger status (Confidence: **{report['confidence']:.2f}**).\n\n"
            f"**Key Decision Drivers (Local Feature Contributions):**\n"
        )
        for f in report["factors"]:
            impact_emoji = "🟢" if f["impact"] == "Positive" else "🔴" if f["impact"] == "Negative" else "🟡"
            summary += f"- {impact_emoji} **{f['factor']}** ({f['impact']} Impact): {f['desc']}\n"
            
        summary += "\n**Top Global Feature Weights in Ensemble:**\n"
        sorted_weights = sorted(report["global_importances"].items(), key=lambda x: x[1], reverse=True)[:3]
        for name, val in sorted_weights:
            summary += f"- `{name}`: {val:.3f}\n"
            
        return summary
    except Exception as e:
        return f"Error generating explainability report: {str(e)}"


@tool
def simulate_station_load(ports: int, power_kw: float, is_fast_dc: bool, country_code: str = "US") -> str:
    """Runs a real-time engineering simulation for a station's power grid impact, queue waiting times, and carbon savings based on current time.
    Use this when the user asks to run a simulation or wants to know grid load/waiting queues.
    """
    try:
        # 1. Queue simulation (M/M/c)
        q_res = QueuingModel.simulate_occupancy_and_queue(ports, is_fast_dc)
        occupied = q_res["occupied_ports"]
        
        # 2. Grid simulation
        grid_res = GridSimulator.simulate_grid_impact(ports, occupied, power_kw)
        
        # 3. Carbon savings
        co2_res = CarbonCalculator.calculate_co2_savings(country_code, power_kw, 1.0)
        
        # Map colors to css
        alert_bg = "rgba(102, 187, 106, 0.12)"
        alert_border = "#66bb6a"
        if grid_res["color"] == "red":
            alert_bg = "rgba(239, 83, 80, 0.12)"
            alert_border = "#ef5350"
        elif grid_res["color"] == "orange":
            alert_bg = "rgba(255, 167, 38, 0.12)"
            alert_border = "#ffa726"
            
        # Store in session state for Streamlit dashboard rendering
        st.session_state.last_simulation = {
            "occupied_ports": occupied,
            "total_ports": ports,
            "queue_length": q_res["queue_length"],
            "wait_time_minutes": q_res["wait_time_minutes"],
            "power_draw_kw": grid_res["power_draw_kw"],
            "transformer_load_percent": grid_res["transformer_load_percent"],
            "power_factor": grid_res["power_factor"],
            "grid_status": grid_res["status"],
            "grid_suggestion": grid_res["suggestion"],
            "grid_alert_bg": alert_bg,
            "grid_alert_border": alert_border,
            "net_saved_kg": co2_res["net_saved_kg"],
            "tree_days": co2_res["tree_days_absorbed"]
        }
        
        summary = (
            f"### ⚡ EV Infrastructure Real-Time Simulation Results\n\n"
            f"#### ⏳ Queuing Theory Metrics (M/M/c Model)\n"
            f"- **Occupancy Status:** {occupied} / {ports} ports busy right now (Traffic Ratio: {q_res['traffic_density_ratio']})\n"
            f"- **Current Queue Length:** {q_res['queue_length']} vehicle(s) waiting\n"
            f"- **Estimated Driver Wait Time:** **{q_res['wait_time_minutes']} minutes**\n\n"
            f"#### 🔌 Distribution Transformer Grid Impact\n"
            f"- **Instantaneous Power Draw:** {grid_res['power_draw_kw']} kW (Substation Capacity: {grid_res['transformer_capacity_kw']} kW)\n"
            f"- **Transformer Load Level:** {grid_res['transformer_load_percent']}% ({grid_res['status']})\n"
            f"- **Grid Power Factor:** {grid_res['power_factor']} (Inverter efficiency: {grid_res['power_factor']*100:.1f}%)\n"
            f"- **System Advice:** *{grid_res['suggestion']}*\n\n"
            f"#### 🌱 Environmental ESG Metrics\n"
            f"- **Net CO2 Savings:** **{co2_res['net_saved_kg']} kg** per hour of charging (gasoline vs. local {country_code} grid mix)\n"
            f"- **Carbon Absorption Equivalence:** Equivalent to **{co2_res['tree_days_absorbed']} days** of a mature tree absorbing carbon.\n"
        )
        return summary
    except Exception as e:
        return f"Error running simulation: {str(e)}"


# Agent Initialization
try:
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_retries=2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use a more robust agent creation
        agent_executor = create_react_agent(
            llm, 
            tools=[search_ev_knowledge, check_station_existence, predict_fast_dc, explain_ml_prediction, simulate_station_load]
        )
except Exception as e:
    st.error(f"Failed to initialize AI Agent: {e}")
    st.stop()
    
# System Prompt
SYSTEM_PROMPT = (
    "You are an EV Charging Station Intelligent Assistant. Your purpose is to help users with topics strictly related to electric vehicles (EVs) and EV charging infrastructure.\n\n"
    "You carry out the following workflows:\n"
    "1. Answer factual questions about EV charging using the `search_ev_knowledge` tool.\n"
    "2. Check if a station actually exists in reality near specific coordinates using `check_station_existence`. Always check this first if the user provides coordinates.\n"
    "3. Predict if a location has a Fast DC charging station using the `predict_fast_dc` tool (for new location planning).\n"
    "4. Explain the ML model's prediction decision drivers using the `explain_ml_prediction` tool.\n"
    "5. Run a real-time engineering simulation for a station's power grid impact, queuing wait times, and carbon offset using the `simulate_station_load` tool.\n\n"
    "STRICT GUARDRAILS:\n"
    "- If the user's question is NOT related to EVs, charging stations, or related infrastructure, "
    "you MUST gracefully decline and state that you only assist with topics related to electric vehicles and charging.\n"
    "- When coordinates are checked, if a station exists, explain it and suggest simulating the grid and queue load using `simulate_station_load`.\n"
    "- If no station exists, suggest predicting suitability using `predict_fast_dc` and explaining the prediction using `explain_ml_prediction`.\n"
    "- If the user asks for coordinates check or prediction, you must collect country_code, latitude, longitude, and number of ports. "
    "Ask for these one at a time if they are missing.\n"
    "Stay strictly within the EV domain and provide concise, accurate help."
)

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
# Render Sidebar Manager
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size: 1.5rem;">⚡ EV Agent Panel</div>', unsafe_allow_html=True)
    st.markdown("**Core LLM:** `llama-3.3-70b-versatile`")
    st.markdown("**ML Models:** RandomForest + XGBoost")
    st.markdown("**Simulation Core:** M/M/c Queuing + Local Grid Model")
    st.markdown("---")
    
    # Render quick stats of last lookup location if exists
    if "last_location" in st.session_state:
        loc = st.session_state.last_location
        st.subheader("📍 Target Station Profile")
        st.markdown(f"**Name:** {loc['name']}")
        st.markdown(f"**Coords:** `{loc['latitude']:.4f}, {loc['longitude']:.4f}`")
        st.markdown(f"**Ports:** {loc['ports']}")
        st.markdown(f"**Charging:** {'Fast DC (DC)' if loc['is_fast_dc'] else 'Standard AC (AC)'}")
        st.markdown(f"**Country:** {loc['country_code']}")
        
        # Trigger quick simulation button directly in sidebar
        if st.button("🚀 Fast Run Simulation"):
            sim_res = simulate_station_load.invoke({
                "ports": loc["ports"],
                "power_kw": loc["power_kw"],
                "is_fast_dc": loc["is_fast_dc"],
                "country_code": loc["country_code"]
            })
            # Inject simulated message as AI message
            st.session_state.messages.append(AIMessage(content=sim_res))
            st.rerun()
            
    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        if "last_simulation" in st.session_state:
            del st.session_state.last_simulation
        if "last_location" in st.session_state:
            del st.session_state.last_location
        # Reset map to sample
        try:
            df_full = StationDatabase.get_df()
            sample_df = df_full.sample(min(100, len(df_full)))
            st.session_state.map_data = sample_df[["latitude", "longitude", "name", "ports", "is_fast_dc"]]
        except Exception:
            pass
        st.rerun()
    st.info("Rate Limiter: 1 request per 5 seconds.")

# Main Layout split into two columns: Left for AI Chat, Right for Interactive Map & Metrics
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown('<h1 class="main-title">⚡ AI EV Charging Station Intelligence</h1>', unsafe_allow_html=True)
    st.markdown("Predict station suitability, verify real-world existence, and simulate real-time queue times & power grid loads.")
    st.markdown("---")
    
    # Render Main Chat Interface History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if isinstance(msg, SystemMessage):
                continue
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content if isinstance(msg.content, str) else str(msg.content))
                
    # Dynamic User Input Hook 
    if prompt := st.chat_input("Ask about EV standards, check coordinates, or run a grid load simulation..."):
        allowed, wait_time = check_rate_limit()
        if not allowed:
            st.warning(f"⏳ Please wait {wait_time} seconds before your next interaction.")
        else:
            st.session_state.last_request_time = time.time()
            
            user_msg = HumanMessage(content=prompt)
            st.session_state.messages.append(user_msg)
            
            # Display user message instantly
            st.rerun()

# Run agent response generation outside of column blocks to avoid layout glitches, 
# then append and rerun to update dashboard.
if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
    with col1:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing EV infrastructure context..."):
                try:
                    response = agent_executor.invoke({"messages": st.session_state.messages})
                    
                    new_msgs = response["messages"][len(st.session_state.messages):]
                    for m in new_msgs:
                        st.session_state.messages.append(m)
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent Processing Error: {str(e)}")

# Right Column: Maps and Live Simulation Dashboard
with col2:
    st.subheader("🗺️ Live Geospatial & Grid Simulator")
    
    # Initialize Map Data if missing
    if "map_data" not in st.session_state:
        try:
            df_full = StationDatabase.get_df()
            sample_df = df_full.sample(min(150, len(df_full)))
            st.session_state.map_data = sample_df[["latitude", "longitude", "name", "ports", "is_fast_dc"]]
        except Exception:
            st.session_state.map_data = pd.DataFrame(columns=["latitude", "longitude", "name", "ports", "is_fast_dc"])
            
    # Render Streamlit Map
    st.map(st.session_state.map_data, latitude="latitude", longitude="longitude")
    st.info("💡 **Agent Integration:** When you ask the agent to check coordinates or predict suitability, the map updates dynamically in real-time.")
    
    # Render Last Simulation metrics if triggered
    if "last_simulation" in st.session_state:
        sim = st.session_state.last_simulation
        st.markdown("---")
        st.subheader("📊 Dynamic Station Load Indicators")
        
        # Grid load card
        st.markdown(
            f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 10px 0; color: #00e676;">🔌 Distribution Substation Load</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px;">
                    <span>Active Power Draw:</span>
                    <span style="font-weight: 600;">{sim['power_draw_kw']} kW</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px;">
                    <span>Transformer Load Ratio:</span>
                    <span style="font-weight: 600;">{sim['transformer_load_percent']}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 14px;">
                    <span>Power Factor (cos φ):</span>
                    <span style="font-weight: 600; color: #00b0ff;">{sim['power_factor']}</span>
                </div>
                <div style="background-color: {sim['grid_alert_bg']}; padding: 10px; border-radius: 8px; border-left: 4px solid {sim['grid_alert_border']}; font-size: 13px; color: #ffffff;">
                    <b>{sim['grid_status']}</b><br/>{sim['grid_suggestion']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Queue card
        st.markdown(
            f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 10px 0; color: #00b0ff;">⏳ Queue & Service Indicators (M/M/c)</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px;">
                    <span>Chargers Busy:</span>
                    <span style="font-weight: 600; color: #00e676;">{sim['occupied_ports']} / {sim['total_ports']} ports</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px;">
                    <span>Vehicles Waiting in Queue:</span>
                    <span style="font-weight: 600; color: #ffeb3b;">{sim['queue_length']} vehicle(s)</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 14px;">
                    <span>Estimated Driver Wait Time:</span>
                    <span style="font-weight: 600; color: #ef5350;">{sim['wait_time_minutes']} minutes</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # ESG impact card
        st.markdown(
            f"""
            <div style="background-color: rgba(46, 125, 50, 0.1); padding: 16px; border-radius: 12px; border-left: 5px solid #2e7d32; margin-top: 15px; border: 1px solid rgba(46, 125, 50, 0.25);">
                <h4 style="margin: 0 0 8px 0; color: #81c784; font-size: 16px;">🌱 Real-time ESG Carbon Savings Certificate</h4>
                <p style="margin: 0; font-size: 14px; color: #e8f5e9;">
                    Charging operations at this site have displaced <b>{sim['net_saved_kg']} kg of CO2</b> emissions per hour compared to internal combustion engines (ICE).
                </p>
                <p style="margin: 6px 0 0 0; font-size: 13px; color: #a5d6a7; font-style: italic;">
                    Absorption equivalence: Equivalent to <b>{sim['tree_days']} tree-days</b> of carbon sequestration.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
