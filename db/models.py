"""
Database models for the Energy Generation Dashboard.
"""
from sqlalchemy import create_engine, Column, String, Enum, DateTime, DECIMAL, Integer, Date, Time, text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TblPlants(Base):
    __tablename__ = 'tbl_plants'
    plant_id = Column(String(50), primary_key=True)
    plant_name = Column(String(255))
    client_name = Column(String(255))
    type = Column(Enum('solar', 'wind'), nullable=False)

class TblGeneration(Base):
    __tablename__ = 'tbl_generation'
    id = Column(Integer, primary_key=True, autoincrement=True)
    plant_id = Column(String(50), nullable=False)
    plant_name = Column(String(255))
    client_name = Column(String(255))
    type = Column(Enum('solar', 'wind'), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    datetime = Column(DateTime, server_default=text("(TIMESTAMP(date, time))"))
    generation = Column(DECIMAL(10, 2))
    active_power = Column(DECIMAL(10, 2))
    
    __table_args__ = (
        UniqueConstraint('plant_id', 'date', 'time', 'type', name='uq_gen'),
    )

class TblConsumption(Base):
    __tablename__ = 'tbl_consumption'
    id = Column(Integer, primary_key=True, autoincrement=True)
    cons_unit = Column(String(100), nullable=False)
    client_name = Column(String(255))
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    datetime = Column(DateTime, server_default=text("(TIMESTAMP(date, time))"))
    consumption = Column(DECIMAL(10, 2))
    
    __table_args__ = (
        UniqueConstraint('cons_unit', 'date', 'time', name='uq_cons'),
    )

class ConsumptionMapping(Base):
    __tablename__ = 'consumption_mapping'
    id = Column(Integer, primary_key=True, autoincrement=True)
    client_name = Column(String(255))
    cons_unit = Column(String(100))
    location_name = Column(String(255))
    percentage = Column(DECIMAL(5, 2))
    
    __table_args__ = (
        UniqueConstraint('client_name', 'cons_unit', name='uq_cons_pct'),
    )

class SettlementData(Base):
    __tablename__ = 'settlement_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    plant_id = Column(String(50), nullable=False)
    client_name = Column(String(255))
    cons_unit = Column(String(100))
    type = Column(Enum('solar', 'wind'), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    datetime = Column(DateTime, server_default=text("(TIMESTAMP(date, time))"))
    generation = Column(DECIMAL(10, 2))
    consumption = Column(DECIMAL(10, 2))
    surplus_demand = Column(DECIMAL(10, 2))
    surplus_deficit = Column(DECIMAL(10, 2))
    
    __table_args__ = (
        UniqueConstraint('plant_id', 'date', 'time', 'type', name='uq_settle'),
    )