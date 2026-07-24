from sqlalchemy import Column, Integer, String, Float, DateTime, Text
import datetime
from database import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    prediction = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    normal_probability = Column(Float, nullable=False)
    pneumonia_probability = Column(Float, nullable=False)
    llm_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)