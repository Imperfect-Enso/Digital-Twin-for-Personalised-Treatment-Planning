from sqlalchemy import (
    create_engine, Column, Integer, Float,
    String, DateTime, Text, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "digital_twin.db"

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


# -----------------------------------------------------------------------------
# ORM Models
# -----------------------------------------------------------------------------

class PatientProfile(Base):
    """One row per patient — stores demographic and clinical profile."""
    __tablename__ = "patient_profiles"

    id            = Column(Integer, primary_key=True, index=True)
    patient_id    = Column(Integer, unique=True, index=True)
    name          = Column(String)
    age           = Column(Integer)
    gender        = Column(String)
    med_condition = Column(String)
    weight        = Column(Float)
    height        = Column(Float)
    created_at    = Column(DateTime, default=datetime.utcnow)


class VitalHistory(Base):
    """12 rows per patient — one per monthly timestep."""
    __tablename__ = "vital_history"

    id             = Column(Integer, primary_key=True, index=True)
    patient_id     = Column(Integer, index=True)
    timestep       = Column(Integer)
    blood_pressure = Column(Float)
    glucose        = Column(Float)
    heart_rate     = Column(Float)
    inflammation   = Column(Float)


class PrognosisLog(Base):
    """Audit log — every prognosis request is recorded here."""
    __tablename__ = "prognosis_logs"

    id             = Column(Integer, primary_key=True, index=True)
    patient_id     = Column(Integer, nullable=True)
    severity_score = Column(Float)
    risk_level     = Column(String)
    interpretation = Column(Text)
    months_of_data = Column(Integer)
    created_at     = Column(DateTime, default=datetime.utcnow)


class SimulationResult(Base):
    """Stores the outcome of every treatment comparison run."""
    __tablename__ = "simulation_results"

    id                 = Column(Integer, primary_key=True, index=True)
    patient_id         = Column(Integer, nullable=True)
    best_treatment     = Column(String)
    baseline_severity  = Column(Float)
    projected_severity = Column(Float)
    improvement        = Column(Float)
    final_score        = Column(Float)
    full_results       = Column(Text)
    created_at         = Column(DateTime, default=datetime.utcnow)


# -----------------------------------------------------------------------------
# Preset Patient Data
# -----------------------------------------------------------------------------

PRESET_PATIENTS = [
    {
        "patient_id": 1, "name": "Arjun Sharma", "age": 45, "gender": "M",
        "med_condition": "Type 2 Diabetes", "weight": 82.0, "height": 174.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 121.6, "glucose": 191.5, "heart_rate": 64.1, "inflammation": 1.299},
            {"timestep": 1,  "blood_pressure": 119.4, "glucose": 187.4, "heart_rate": 79.7, "inflammation": 1.7568},
            {"timestep": 2,  "blood_pressure": 125.3, "glucose": 183.5, "heart_rate": 68.2, "inflammation": 1.361},
            {"timestep": 3,  "blood_pressure": 119.9, "glucose": 183.2, "heart_rate": 72.1, "inflammation": 1.5816},
            {"timestep": 4,  "blood_pressure": 125.1, "glucose": 179.0, "heart_rate": 84.2, "inflammation": 1.183},
            {"timestep": 5,  "blood_pressure": 119.2, "glucose": 173.7, "heart_rate": 85.2, "inflammation": 1.3133},
            {"timestep": 6,  "blood_pressure": 123.3, "glucose": 172.1, "heart_rate": 76.7, "inflammation": 1.1634},
            {"timestep": 7,  "blood_pressure": 123.3, "glucose": 172.5, "heart_rate": 63.7, "inflammation": 1.1784},
            {"timestep": 8,  "blood_pressure": 113.5, "glucose": 163.2, "heart_rate": 65.3, "inflammation": 1.1595},
            {"timestep": 9,  "blood_pressure": 125.8, "glucose": 167.3, "heart_rate": 90.8, "inflammation": 1.264},
            {"timestep": 10, "blood_pressure": 115.7, "glucose": 157.3, "heart_rate": 73.7, "inflammation": 1.2246},
            {"timestep": 11, "blood_pressure": 120.5, "glucose": 158.4, "heart_rate": 88.4, "inflammation": 1.3597},
        ]
    },
    {
        "patient_id": 2, "name": "Priya Patel", "age": 52, "gender": "F",
        "med_condition": "Hypertension", "weight": 68.0, "height": 161.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 160.4, "glucose": 91.2,  "heart_rate": 69.1, "inflammation": 1.1297},
            {"timestep": 1,  "blood_pressure": 158.2, "glucose": 83.2,  "heart_rate": 63.7, "inflammation": 1.0161},
            {"timestep": 2,  "blood_pressure": 157.4, "glucose": 85.4,  "heart_rate": 87.6, "inflammation": 1.0838},
            {"timestep": 3,  "blood_pressure": 154.9, "glucose": 93.1,  "heart_rate": 87.0, "inflammation": 1.0799},
            {"timestep": 4,  "blood_pressure": 154.9, "glucose": 103.9, "heart_rate": 78.3, "inflammation": 1.1138},
            {"timestep": 5,  "blood_pressure": 152.0, "glucose": 98.0,  "heart_rate": 78.4, "inflammation": 0.8807},
            {"timestep": 6,  "blood_pressure": 152.4, "glucose": 87.1,  "heart_rate": 86.7, "inflammation": 1.0059},
            {"timestep": 7,  "blood_pressure": 148.8, "glucose": 93.9,  "heart_rate": 70.5, "inflammation": 1.0418},
            {"timestep": 8,  "blood_pressure": 149.3, "glucose": 92.3,  "heart_rate": 69.4, "inflammation": 1.036},
            {"timestep": 9,  "blood_pressure": 146.3, "glucose": 86.7,  "heart_rate": 81.9, "inflammation": 0.9216},
            {"timestep": 10, "blood_pressure": 145.6, "glucose": 97.2,  "heart_rate": 68.2, "inflammation": 1.0627},
            {"timestep": 11, "blood_pressure": 144.1, "glucose": 100.4, "heart_rate": 84.0, "inflammation": 1.1211},
        ]
    },
    {
        "patient_id": 3, "name": "Rahul Verma", "age": 61, "gender": "M",
        "med_condition": "Type 2 Diabetes & Hypertension", "weight": 91.0, "height": 170.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 169.6, "glucose": 216.5, "heart_rate": 91.4, "inflammation": 1.4531},
            {"timestep": 1,  "blood_pressure": 164.7, "glucose": 217.0, "heart_rate": 78.3, "inflammation": 1.2904},
            {"timestep": 2,  "blood_pressure": 165.7, "glucose": 216.7, "heart_rate": 70.0, "inflammation": 1.4674},
            {"timestep": 3,  "blood_pressure": 161.1, "glucose": 208.7, "heart_rate": 79.9, "inflammation": 1.7408},
            {"timestep": 4,  "blood_pressure": 160.8, "glucose": 210.2, "heart_rate": 75.8, "inflammation": 1.3762},
            {"timestep": 5,  "blood_pressure": 157.7, "glucose": 205.5, "heart_rate": 72.5, "inflammation": 1.2635},
            {"timestep": 6,  "blood_pressure": 155.6, "glucose": 201.8, "heart_rate": 85.4, "inflammation": 1.5524},
            {"timestep": 7,  "blood_pressure": 157.0, "glucose": 198.1, "heart_rate": 77.9, "inflammation": 1.1497},
            {"timestep": 8,  "blood_pressure": 153.5, "glucose": 192.3, "heart_rate": 90.3, "inflammation": 1.6641},
            {"timestep": 9,  "blood_pressure": 153.2, "glucose": 192.9, "heart_rate": 65.1, "inflammation": 1.7551},
            {"timestep": 10, "blood_pressure": 151.1, "glucose": 186.7, "heart_rate": 82.1, "inflammation": 1.2425},
            {"timestep": 11, "blood_pressure": 147.1, "glucose": 186.2, "heart_rate": 91.7, "inflammation": 1.665},
        ]
    },
    {
        "patient_id": 4, "name": "Sneha Iyer", "age": 38, "gender": "F",
        "med_condition": "Type 2 Diabetes", "weight": 74.0, "height": 158.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 119.9, "glucose": 184.3, "heart_rate": 79.1, "inflammation": 1.1775},
            {"timestep": 1,  "blood_pressure": 122.0, "glucose": 179.8, "heart_rate": 66.0, "inflammation": 1.2279},
            {"timestep": 2,  "blood_pressure": 115.6, "glucose": 180.8, "heart_rate": 67.3, "inflammation": 1.1878},
            {"timestep": 3,  "blood_pressure": 112.5, "glucose": 176.5, "heart_rate": 81.4, "inflammation": 1.2283},
            {"timestep": 4,  "blood_pressure": 113.8, "glucose": 174.6, "heart_rate": 70.8, "inflammation": 1.2843},
            {"timestep": 5,  "blood_pressure": 113.1, "glucose": 164.9, "heart_rate": 85.0, "inflammation": 1.3291},
            {"timestep": 6,  "blood_pressure": 116.6, "glucose": 160.8, "heart_rate": 68.5, "inflammation": 1.3386},
            {"timestep": 7,  "blood_pressure": 125.0, "glucose": 156.6, "heart_rate": 82.9, "inflammation": 1.3484},
            {"timestep": 8,  "blood_pressure": 123.6, "glucose": 160.3, "heart_rate": 65.1, "inflammation": 1.2432},
            {"timestep": 9,  "blood_pressure": 114.0, "glucose": 150.1, "heart_rate": 71.5, "inflammation": 1.2003},
            {"timestep": 10, "blood_pressure": 117.1, "glucose": 149.5, "heart_rate": 72.0, "inflammation": 1.3832},
            {"timestep": 11, "blood_pressure": 119.6, "glucose": 145.3, "heart_rate": 65.6, "inflammation": 1.3637},
        ]
    },
    {
        "patient_id": 5, "name": "Vikram Singh", "age": 70, "gender": "M",
        "med_condition": "Kidney Disease", "weight": 70.0, "height": 168.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 116.3, "glucose": 82.5,  "heart_rate": 83.8, "inflammation": 0.8839},
            {"timestep": 1,  "blood_pressure": 127.1, "glucose": 89.5,  "heart_rate": 88.9, "inflammation": 1.0318},
            {"timestep": 2,  "blood_pressure": 114.3, "glucose": 100.7, "heart_rate": 79.9, "inflammation": 0.8745},
            {"timestep": 3,  "blood_pressure": 114.7, "glucose": 97.9,  "heart_rate": 79.2, "inflammation": 1.0105},
            {"timestep": 4,  "blood_pressure": 118.8, "glucose": 97.7,  "heart_rate": 82.1, "inflammation": 0.9657},
            {"timestep": 5,  "blood_pressure": 112.7, "glucose": 87.8,  "heart_rate": 77.3, "inflammation": 1.1457},
            {"timestep": 6,  "blood_pressure": 124.5, "glucose": 97.7,  "heart_rate": 84.6, "inflammation": 1.0435},
            {"timestep": 7,  "blood_pressure": 112.5, "glucose": 82.3,  "heart_rate": 78.0, "inflammation": 1.0686},
            {"timestep": 8,  "blood_pressure": 125.7, "glucose": 84.7,  "heart_rate": 86.8, "inflammation": 1.0946},
            {"timestep": 9,  "blood_pressure": 116.2, "glucose": 88.2,  "heart_rate": 90.6, "inflammation": 0.8505},
            {"timestep": 10, "blood_pressure": 126.6, "glucose": 93.7,  "heart_rate": 63.1, "inflammation": 1.1109},
            {"timestep": 11, "blood_pressure": 113.9, "glucose": 88.9,  "heart_rate": 87.3, "inflammation": 1.0208},
        ]
    },
    {
        "patient_id": 6, "name": "Anjali Reddy", "age": 55, "gender": "F",
        "med_condition": "Hyperlipidemia", "weight": 83.0, "height": 163.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 126.0, "glucose": 98.4,  "heart_rate": 73.5, "inflammation": 0.9433},
            {"timestep": 1,  "blood_pressure": 114.3, "glucose": 83.1,  "heart_rate": 77.8, "inflammation": 1.0713},
            {"timestep": 2,  "blood_pressure": 126.8, "glucose": 104.0, "heart_rate": 83.9, "inflammation": 1.105},
            {"timestep": 3,  "blood_pressure": 124.3, "glucose": 92.4,  "heart_rate": 76.7, "inflammation": 1.0344},
            {"timestep": 4,  "blood_pressure": 127.1, "glucose": 99.0,  "heart_rate": 67.9, "inflammation": 0.8718},
            {"timestep": 5,  "blood_pressure": 119.4, "glucose": 102.8, "heart_rate": 68.6, "inflammation": 0.9538},
            {"timestep": 6,  "blood_pressure": 115.3, "glucose": 84.1,  "heart_rate": 75.0, "inflammation": 0.8924},
            {"timestep": 7,  "blood_pressure": 112.5, "glucose": 89.8,  "heart_rate": 88.0, "inflammation": 0.9468},
            {"timestep": 8,  "blood_pressure": 118.3, "glucose": 87.7,  "heart_rate": 63.9, "inflammation": 0.8696},
            {"timestep": 9,  "blood_pressure": 120.4, "glucose": 92.9,  "heart_rate": 91.2, "inflammation": 1.1188},
            {"timestep": 10, "blood_pressure": 120.7, "glucose": 85.6,  "heart_rate": 76.3, "inflammation": 1.0702},
            {"timestep": 11, "blood_pressure": 116.6, "glucose": 100.0, "heart_rate": 81.5, "inflammation": 0.922},
        ]
    },
    {
        "patient_id": 7, "name": "Deepak Gupta", "age": 48, "gender": "M",
        "med_condition": "Obesity & Type 2 Diabetes", "weight": 99.0, "height": 172.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 120.6, "glucose": 205.7, "heart_rate": 71.7, "inflammation": 1.4329},
            {"timestep": 1,  "blood_pressure": 126.1, "glucose": 207.9, "heart_rate": 71.9, "inflammation": 1.2704},
            {"timestep": 2,  "blood_pressure": 118.2, "glucose": 204.7, "heart_rate": 72.8, "inflammation": 1.3784},
            {"timestep": 3,  "blood_pressure": 125.2, "glucose": 196.5, "heart_rate": 86.8, "inflammation": 1.1402},
            {"timestep": 4,  "blood_pressure": 117.6, "glucose": 198.6, "heart_rate": 82.3, "inflammation": 1.7441},
            {"timestep": 5,  "blood_pressure": 116.4, "glucose": 191.4, "heart_rate": 89.0, "inflammation": 1.4012},
            {"timestep": 6,  "blood_pressure": 126.6, "glucose": 194.0, "heart_rate": 82.3, "inflammation": 1.1296},
            {"timestep": 7,  "blood_pressure": 113.7, "glucose": 183.8, "heart_rate": 91.0, "inflammation": 1.2818},
            {"timestep": 8,  "blood_pressure": 125.0, "glucose": 185.2, "heart_rate": 68.6, "inflammation": 1.5218},
            {"timestep": 9,  "blood_pressure": 119.2, "glucose": 173.8, "heart_rate": 89.7, "inflammation": 1.359},
            {"timestep": 10, "blood_pressure": 117.0, "glucose": 179.0, "heart_rate": 72.2, "inflammation": 1.3291},
            {"timestep": 11, "blood_pressure": 126.9, "glucose": 174.1, "heart_rate": 90.7, "inflammation": 1.4011},
        ]
    },
    {
        "patient_id": 8, "name": "Meera Nair", "age": 63, "gender": "F",
        "med_condition": "Hypertension", "weight": 71.0, "height": 155.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 153.6, "glucose": 99.8,  "heart_rate": 85.7, "inflammation": 1.1134},
            {"timestep": 1,  "blood_pressure": 150.4, "glucose": 99.3,  "heart_rate": 72.4, "inflammation": 0.9662},
            {"timestep": 2,  "blood_pressure": 148.7, "glucose": 104.7, "heart_rate": 83.6, "inflammation": 0.8828},
            {"timestep": 3,  "blood_pressure": 148.3, "glucose": 91.6,  "heart_rate": 82.3, "inflammation": 1.0256},
            {"timestep": 4,  "blood_pressure": 146.3, "glucose": 88.8,  "heart_rate": 70.2, "inflammation": 1.1381},
            {"timestep": 5,  "blood_pressure": 144.5, "glucose": 102.3, "heart_rate": 77.7, "inflammation": 0.9888},
            {"timestep": 6,  "blood_pressure": 144.4, "glucose": 97.0,  "heart_rate": 87.4, "inflammation": 0.8586},
            {"timestep": 7,  "blood_pressure": 139.1, "glucose": 88.8,  "heart_rate": 90.0, "inflammation": 0.9229},
            {"timestep": 8,  "blood_pressure": 137.8, "glucose": 92.3,  "heart_rate": 80.3, "inflammation": 0.8876},
            {"timestep": 9,  "blood_pressure": 135.6, "glucose": 90.5,  "heart_rate": 81.9, "inflammation": 0.967},
            {"timestep": 10, "blood_pressure": 135.9, "glucose": 89.5,  "heart_rate": 86.1, "inflammation": 1.0428},
            {"timestep": 11, "blood_pressure": 154.7, "glucose": 87.8,  "heart_rate": 86.3, "inflammation": 0.8553},
        ]
    },
    {
        "patient_id": 9, "name": "Sanjay Kumar", "age": 57, "gender": "M",
        "med_condition": "Type 2 Diabetes & Hypertension", "weight": 88.0, "height": 176.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 169.6, "glucose": 226.0, "heart_rate": 67.7, "inflammation": 1.5151},
            {"timestep": 1,  "blood_pressure": 167.4, "glucose": 224.3, "heart_rate": 73.4, "inflammation": 1.3084},
            {"timestep": 2,  "blood_pressure": 165.9, "glucose": 228.0, "heart_rate": 63.2, "inflammation": 1.5287},
            {"timestep": 3,  "blood_pressure": 164.2, "glucose": 222.7, "heart_rate": 63.4, "inflammation": 1.6031},
            {"timestep": 4,  "blood_pressure": 161.2, "glucose": 220.1, "heart_rate": 67.2, "inflammation": 1.6156},
            {"timestep": 5,  "blood_pressure": 159.4, "glucose": 216.8, "heart_rate": 68.7, "inflammation": 1.2989},
            {"timestep": 6,  "blood_pressure": 159.1, "glucose": 213.8, "heart_rate": 89.2, "inflammation": 1.714},
            {"timestep": 7,  "blood_pressure": 158.5, "glucose": 202.1, "heart_rate": 87.0, "inflammation": 1.6893},
            {"timestep": 8,  "blood_pressure": 155.7, "glucose": 201.0, "heart_rate": 84.0, "inflammation": 1.8025},
            {"timestep": 9,  "blood_pressure": 154.8, "glucose": 199.1, "heart_rate": 80.4, "inflammation": 1.4064},
            {"timestep": 10, "blood_pressure": 151.7, "glucose": 193.1, "heart_rate": 76.0, "inflammation": 1.419},
            {"timestep": 11, "blood_pressure": 150.2, "glucose": 192.7, "heart_rate": 76.3, "inflammation": 1.4804},
        ]
    },
    {
        "patient_id": 10, "name": "Kavya Menon", "age": 42, "gender": "F",
        "med_condition": "Type 2 Diabetes", "weight": 65.0, "height": 160.0,
        "vitals": [
            {"timestep": 0,  "blood_pressure": 113.9, "glucose": 174.1, "heart_rate": 81.5, "inflammation": 1.2004},
            {"timestep": 1,  "blood_pressure": 119.3, "glucose": 177.3, "heart_rate": 87.9, "inflammation": 1.4041},
            {"timestep": 2,  "blood_pressure": 116.1, "glucose": 174.1, "heart_rate": 85.9, "inflammation": 1.4224},
            {"timestep": 3,  "blood_pressure": 123.4, "glucose": 166.0, "heart_rate": 77.6, "inflammation": 1.4217},
            {"timestep": 4,  "blood_pressure": 119.7, "glucose": 161.6, "heart_rate": 82.7, "inflammation": 1.274},
            {"timestep": 5,  "blood_pressure": 113.9, "glucose": 163.3, "heart_rate": 79.8, "inflammation": 1.1924},
            {"timestep": 6,  "blood_pressure": 121.9, "glucose": 161.9, "heart_rate": 73.4, "inflammation": 1.2008},
            {"timestep": 7,  "blood_pressure": 123.0, "glucose": 150.3, "heart_rate": 80.0, "inflammation": 1.359},
            {"timestep": 8,  "blood_pressure": 111.8, "glucose": 145.0, "heart_rate": 66.8, "inflammation": 1.2588},
            {"timestep": 9,  "blood_pressure": 111.8, "glucose": 141.7, "heart_rate": 80.1, "inflammation": 1.2961},
            {"timestep": 10, "blood_pressure": 115.0, "glucose": 142.7, "heart_rate": 88.3, "inflammation": 1.2603},
            {"timestep": 11, "blood_pressure": 112.5, "glucose": 137.1, "heart_rate": 90.2, "inflammation": 1.0206},
        ]
    },
]


# -----------------------------------------------------------------------------
# Database Initialisation
# -----------------------------------------------------------------------------

def init_db():
    """
    Creates all tables and seeds patient data on first run.
    Safe to call on every startup — skips seeding if data already exists.
    """
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if db.query(PatientProfile).count() == 0:
            for p in PRESET_PATIENTS:
                profile = PatientProfile(
                    patient_id    = p["patient_id"],
                    name          = p["name"],
                    age           = p["age"],
                    gender        = p["gender"],
                    med_condition = p["med_condition"],
                    weight        = p["weight"],
                    height        = p["height"]
                )
                db.add(profile)

                for v in p["vitals"]:
                    vital = VitalHistory(
                        patient_id     = p["patient_id"],
                        timestep       = v["timestep"],
                        blood_pressure = v["blood_pressure"],
                        glucose        = v["glucose"],
                        heart_rate     = v["heart_rate"],
                        inflammation   = v["inflammation"]
                    )
                    db.add(vital)

            db.commit()
            print(f"Seeded 10 patients into database")

        print(f"Database ready at: {DB_PATH}")

    finally:
        db.close()


def get_db():
    """FastAPI dependency — yields a database session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
