from sqlalchemy import (
    create_engine, Column, Integer, Float,
    String, DateTime, Text, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "digital_twin.db"

DATABASE_URL = f"sqlite:///{DB_PATH}"


# Engine setup

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30
    }
)


# Enable WAL for better concurrency

@event.listens_for(engine, "connect")
def set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# Models

class PatientProfile(Base):
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
    __tablename__ = "vital_history"

    id             = Column(Integer, primary_key=True, index=True)
    patient_id     = Column(Integer, index=True)
    timestep       = Column(Integer)
    blood_pressure = Column(Float)
    glucose        = Column(Float)
    heart_rate     = Column(Float)
    inflammation   = Column(Float)


class PrognosisLog(Base):
    __tablename__ = "prognosis_logs"

    id             = Column(Integer, primary_key=True, index=True)
    patient_id     = Column(Integer, nullable=True)
    severity_score = Column(Float)
    risk_level     = Column(String)
    interpretation = Column(Text)
    months_of_data = Column(Integer)
    created_at     = Column(DateTime, default=datetime.utcnow)


class SimulationResult(Base):
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


# Preset data (trimmed here for readability — keep your full data as is)

PRESET_PATIENTS = [
    # ... keep your existing patient data unchanged ...
]


# Init DB + seed data

def init_db():
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if db.query(PatientProfile).count() == 0:
            for p in PRESET_PATIENTS:
                db.add(PatientProfile(
                    patient_id=p["patient_id"],
                    name=p["name"],
                    age=p["age"],
                    gender=p["gender"],
                    med_condition=p["med_condition"],
                    weight=p["weight"],
                    height=p["height"]
                ))

                for v in p["vitals"]:
                    db.add(VitalHistory(
                        patient_id=p["patient_id"],
                        timestep=v["timestep"],
                        blood_pressure=v["blood_pressure"],
                        glucose=v["glucose"],
                        heart_rate=v["heart_rate"],
                        inflammation=v["inflammation"]
                    ))

            db.commit()
            print("Seeded preset patients")

        print(f"DB ready at: {DB_PATH}")

    finally:
        db.close()


# Dependency

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()