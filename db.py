from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base  # ensure models are imported so metadata knows all tables
import os

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+mysqlconnector://noteflow:NoteFlow123!@localhost/noteflow")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
