"""
Database-based data fetching functions for the Energy Generation Dashboard.
All data is stored in 15-minute intervals in the database.
"""

import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Optional, Dict
import traceback
from functools import wraps

from sqlalchemy import and_, func

from db.db_setup import SessionLocal
from db.models import TblPlants, TblGeneration, TblConsumption, ConsumptionMapping, SettlementData
from backend.logs.logger_setup import setup_logger
from backend.config.tod_config import get_tod_slots

# Configure logging
logger = setup_logger('db_data', 'db_data.log')



def retry_on_exception(max_retries=3, retry_delay=1):
    """Decorator to retry a function on exception"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay)
            logger.error(f"Function {func.__name__} failed after {max_retries} attempts")
            raise last_exception
        return wrapper
    return decorator

def get_db_session():
    """Get database session"""
    return SessionLocal()

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_generation_data_db(plant_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get generation data from database for a plant and date range.
    
    Args:
        plant_id: Plant identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns: datetime, generation
    """
    try:
        session = get_db_session()
        
        # Convert string dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Query generation data
        query = session.query(
            TblGeneration.datetime,
            TblGeneration.generation
        ).filter(
            and_(
                TblGeneration.plant_id == plant_id,
                TblGeneration.date >= start_dt,
                TblGeneration.date <= end_dt
            )
        ).order_by(TblGeneration.datetime)
        
        # Execute query and convert to DataFrame
        result = query.all()
        session.close()
        
        if not result:
            logger.warning(f"No generation data found for plant {plant_id} from {start_date} to {end_date}")
            return pd.DataFrame(columns=['datetime', 'generation'])
        
        df = pd.DataFrame(result, columns=['datetime', 'generation'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['generation'] = pd.to_numeric(df['generation'], errors='coerce').fillna(0)
        
        logger.info(f"Retrieved {len(df)} generation records for plant {plant_id}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to get generation data from DB: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['datetime', 'generation'])

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_consumption_data_by_client(client_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get consumption data from database for a client and date range.
    Uses only client_name, datetime, consumption columns as specified.
    
    Args:
        client_name: Client name identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns: datetime, consumption
    """
    try:
        session = get_db_session()
        
        # Convert string dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Query consumption data using only client_name, datetime, consumption
        query = session.query(
            TblConsumption.datetime,
            TblConsumption.consumption
        ).filter(
            and_(
                TblConsumption.client_name == client_name,
                TblConsumption.date >= start_dt,
                TblConsumption.date <= end_dt
            )
        ).order_by(TblConsumption.datetime)
        
        # Execute query and convert to DataFrame
        result = query.all()
        session.close()
        
        if not result:
            logger.warning(f"No consumption data found for client {client_name} from {start_date} to {end_date}")
            return pd.DataFrame(columns=['datetime', 'consumption'])
        
        df = pd.DataFrame(result, columns=['datetime', 'consumption'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce').fillna(0)
        
        logger.info(f"Retrieved {len(df)} consumption records for client {client_name}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to get consumption data from DB: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['datetime', 'consumption'])

# Keep the old function for backward compatibility but mark as deprecated
@st.cache_data(ttl=3600)
@retry_on_exception()
def get_consumption_data_db(cons_unit: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    DEPRECATED: Get consumption data from database for a consumption unit and date range.
    Use get_consumption_data_by_client() instead.
    
    Args:
        cons_unit: Consumption unit identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns: datetime, consumption
    """
    try:
        session = get_db_session()
        
        # Convert string dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Query consumption data
        query = session.query(
            TblConsumption.datetime,
            TblConsumption.consumption
        ).filter(
            and_(
                TblConsumption.cons_unit == cons_unit,
                TblConsumption.date >= start_dt,
                TblConsumption.date <= end_dt
            )
        ).order_by(TblConsumption.datetime)
        
        # Execute query and convert to DataFrame
        result = query.all()
        session.close()
        
        if not result:
            logger.warning(f"No consumption data found for unit {cons_unit} from {start_date} to {end_date}")
            return pd.DataFrame(columns=['datetime', 'consumption'])
        
        df = pd.DataFrame(result, columns=['datetime', 'consumption'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce').fillna(0)
        
        logger.info(f"Retrieved {len(df)} consumption records for unit {cons_unit}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to get consumption data from DB: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['datetime', 'consumption'])

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_settlement_data_db(plant_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get settlement data from database for a plant and date range.
    
    Args:
        plant_id: Plant identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns: datetime, generation, consumption, surplus
    """
    try:
        session = get_db_session()
        
        # Convert string dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Query settlement data
        query = session.query(
            SettlementData.datetime,
            SettlementData.generation,
            SettlementData.consumption,
            SettlementData.surplus_demand,
            SettlementData.surplus_deficit
        ).filter(
            and_(
                SettlementData.plant_id == plant_id,
                SettlementData.date >= start_dt,
                SettlementData.date <= end_dt
            )
        ).order_by(SettlementData.datetime)
        
        # Execute query and convert to DataFrame
        result = query.all()
        session.close()
        
        if not result:
            logger.warning(f"No settlement data found for plant {plant_id} from {start_date} to {end_date}")
            return pd.DataFrame(columns=['datetime', 'generation', 'consumption', 'surplus_demand', 'surplus_deficit'])
        
        df = pd.DataFrame(result, columns=['datetime', 'generation', 'consumption', 'surplus_demand', 'surplus_deficit'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Convert numeric columns
        for col in ['generation', 'consumption', 'surplus_demand', 'surplus_deficit']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        logger.info(f"Retrieved {len(df)} settlement records for plant {plant_id}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to get settlement data from DB: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['datetime', 'generation', 'consumption', 'surplus_demand', 'surplus_deficit'])

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_plants_from_db() -> Dict:
    """
    Get all available plants from database.
    
    Returns:
        Dictionary with plant information organized by client and type
    """
    try:
        session = get_db_session()
        
        # Get unique plants from plants table
        plants_query = session.query(
            TblPlants.plant_id,
            TblPlants.plant_name,
            TblPlants.client_name,
            TblPlants.type
        ).distinct().all()
        
        session.close()
        
        # Organize plants by client and type
        plants_dict = {}
        
        for plant_id, plant_name, client_name, plant_type in plants_query:
            if client_name not in plants_dict:
                plants_dict[client_name] = {'solar': [], 'wind': []}
            
            plant_info = {
                'plant_id': plant_id,
                'name': plant_name or plant_id,
                'client': client_name
            }
            
            if plant_type in ['solar', 'wind']:
                plants_dict[client_name][plant_type].append(plant_info)
        
        logger.info(f"Retrieved {len(plants_query)} plants from database")
        return plants_dict
        
    except Exception as e:
        logger.error(f"Failed to get plants from DB: {e}")
        logger.error(traceback.format_exc())
        return {}

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_daily_aggregated_generation_db(plant_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get daily aggregated generation data from database.
    
    Args:
        plant_id: Plant identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns: date, generation
    """
    try:
        session = get_db_session()
        
        # Convert string dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Query daily aggregated generation data
        query = session.query(
            TblGeneration.date,
            func.sum(TblGeneration.generation).label('generation')
        ).filter(
            and_(
                TblGeneration.plant_id == plant_id,
                TblGeneration.date >= start_dt,
                TblGeneration.date <= end_dt
            )
        ).group_by(TblGeneration.date).order_by(TblGeneration.date)
        
        # Execute query and convert to DataFrame
        result = query.all()
        session.close()
        
        if not result:
            logger.warning(f"No daily generation data found for plant {plant_id}")
            return pd.DataFrame(columns=['date', 'generation'])
        
        df = pd.DataFrame(result, columns=['date', 'generation'])
        df['date'] = pd.to_datetime(df['date'])
        df['generation'] = pd.to_numeric(df['generation'], errors='coerce').fillna(0)
        
        logger.info(f"Retrieved {len(df)} daily generation records for plant {plant_id}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to get daily generation data from DB: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['date', 'generation'])

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_daily_aggregated_consumption_db(cons_unit: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get daily aggregated consumption data from database.
    
    Args:
        cons_unit: Consumption unit identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns: date, consumption
    """
    try:
        session = get_db_session()
        
        # Convert string dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Query daily aggregated consumption data
        query = session.query(
            TblConsumption.date,
            func.sum(TblConsumption.consumption).label('consumption')
        ).filter(
            and_(
                TblConsumption.cons_unit == cons_unit,
                TblConsumption.date >= start_dt,
                TblConsumption.date <= end_dt
            )
        ).group_by(TblConsumption.date).order_by(TblConsumption.date)
        
        # Execute query and convert to DataFrame
        result = query.all()
        session.close()
        
        if not result:
            logger.warning(f"No daily consumption data found for unit {cons_unit}")
            return pd.DataFrame(columns=['date', 'consumption'])
        
        df = pd.DataFrame(result, columns=['date', 'consumption'])
        df['date'] = pd.to_datetime(df['date'])
        df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce').fillna(0)
        
        logger.info(f"Retrieved {len(df)} daily consumption records for unit {cons_unit}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to get daily consumption data from DB: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['date', 'consumption'])

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_tod_aggregated_data_db(plant_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get Time-of-Day aggregated data from database.
    
    Args:
        plant_id: Plant identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with ToD analysis
    """
    try:
        # Get settlement data first
        settlement_df = get_settlement_data_db(plant_id, start_date, end_date)
        
        if settlement_df.empty:
            return pd.DataFrame()
        
        # Get ToD slots configuration
        tod_slots = get_tod_slots()
        
        def assign_tod_bin(hour):
            """Assign ToD bin based on hour"""
            for slot_name, slot_info in tod_slots.items():
                start_hour = slot_info['start_hour']
                end_hour = slot_info['end_hour']
                
                if start_hour <= end_hour:
                    if start_hour <= hour < end_hour:
                        return slot_name
                else:  # Crosses midnight
                    if hour >= start_hour or hour < end_hour:
                        return slot_name
            return 'Unknown'
        
        # Add ToD bin column
        settlement_df['hour'] = settlement_df['datetime'].dt.hour
        settlement_df['date'] = settlement_df['datetime'].dt.date
        settlement_df['tod_bin'] = settlement_df['hour'].apply(assign_tod_bin)
        
        # Check if this is a multi-day analysis
        start_date_obj = pd.to_datetime(start_date).date()
        end_date_obj = pd.to_datetime(end_date).date()
        is_multi_day = start_date_obj != end_date_obj
        
        if is_multi_day:
            # For multi-day analysis, we need to provide both:
            # 1. Daily breakdown (for daily comparison charts)
            # 2. Total aggregation across all days (for ToD comparison charts)
            
            # First, create daily breakdown by date and ToD bin
            daily_tod_aggregated = settlement_df.groupby(['date', 'tod_bin']).agg({
                'generation': 'sum',
                'consumption': 'sum',
                'surplus_demand': 'sum',
                'surplus_deficit': 'sum'
            }).reset_index()
            
            # Count intervals per date and ToD bin for daily data
            daily_interval_counts = settlement_df.groupby(['date', 'tod_bin']).size().reset_index(name='interval_count')
            daily_tod_aggregated = daily_tod_aggregated.merge(daily_interval_counts, on=['date', 'tod_bin'])
            
            # For multi-day ToD comparison, aggregate across all days by ToD bin only
            # This gives us the total for each ToD bin across all selected days
            total_tod_aggregated = settlement_df.groupby('tod_bin').agg({
                'generation': 'sum',
                'consumption': 'sum',
                'surplus_demand': 'sum',
                'surplus_deficit': 'sum'
            }).reset_index()
            
            # Count total intervals per ToD bin across all days
            total_interval_counts = settlement_df.groupby('tod_bin').size().reset_index(name='total_interval_count')
            total_tod_aggregated = total_tod_aggregated.merge(total_interval_counts, on='tod_bin')
            
            # Add date column to total aggregated data to maintain consistency
            # Use the start_date as representative date for total aggregation
            total_tod_aggregated['date'] = start_date_obj
            total_tod_aggregated['interval_count'] = total_tod_aggregated['total_interval_count']
            
            # Combine both datasets - daily breakdown + total aggregation
            # The visualization layer can choose which one to use based on the plot type
            tod_aggregated = pd.concat([daily_tod_aggregated, total_tod_aggregated], ignore_index=True)
            
        else:
            # For single-day analysis, aggregate by ToD bin only
            tod_aggregated = settlement_df.groupby('tod_bin').agg({
                'generation': 'sum',
                'consumption': 'sum',
                'surplus_demand': 'sum',
                'surplus_deficit': 'sum'
            }).reset_index()
            
            # Calculate normalized values (per 15-minute interval)
            # Count intervals per ToD bin
            interval_counts = settlement_df.groupby('tod_bin').size().reset_index(name='interval_count')
            tod_aggregated = tod_aggregated.merge(interval_counts, on='tod_bin')
            
            # Add date column for consistency
            tod_aggregated['date'] = start_date_obj
        
        # Normalize to per-interval values - this represents average per 15-minute interval
        for col in ['generation', 'consumption', 'surplus_demand', 'surplus_deficit']:
            tod_aggregated[f'{col}_normalized'] = tod_aggregated[col] / tod_aggregated['interval_count']
        
        # Add total columns (non-normalized) for cases where we need actual totals
        for col in ['generation', 'consumption', 'surplus_demand', 'surplus_deficit']:
            tod_aggregated[f'{col}_total'] = tod_aggregated[col]
        
        logger.info(f"Generated ToD aggregated data for plant {plant_id}")
        return tod_aggregated
        
    except Exception as e:
        logger.error(f"Failed to get ToD aggregated data from DB: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_combined_plants_data_db(client_name: str, plant_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get combined data for all plants of a specific type for a client.
    
    Args:
        client_name: Client name
        plant_type: Plant type ('solar' or 'wind')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with combined generation data
    """
    try:
        session = get_db_session()
        
        # Convert string dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Query combined generation data for all plants of the type
        query = session.query(
            TblGeneration.datetime,
            func.sum(TblGeneration.generation).label('generation')
        ).filter(
            and_(
                TblGeneration.client_name == client_name,
                TblGeneration.type == plant_type,
                TblGeneration.date >= start_dt,
                TblGeneration.date <= end_dt
            )
        ).group_by(TblGeneration.datetime).order_by(TblGeneration.datetime)
        
        # Execute query and convert to DataFrame
        result = query.all()
        session.close()
        
        if not result:
            logger.warning(f"No combined {plant_type} data found for client {client_name}")
            return pd.DataFrame(columns=['datetime', 'generation'])
        
        df = pd.DataFrame(result, columns=['datetime', 'generation'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['generation'] = pd.to_numeric(df['generation'], errors='coerce').fillna(0)
        
        logger.info(f"Retrieved {len(df)} combined {plant_type} records for client {client_name}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to get combined plants data from DB: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['datetime', 'generation'])

def get_plant_id_from_name(plant_name) -> Optional[str]:
    """
    Get plant ID from plant name by querying the database.
    
    Args:
        plant_name: Plant name or plant object
        
    Returns:
        Plant ID if found, None otherwise
    """
    try:
        # Handle case where plant_name is actually a plant object/dict
        if isinstance(plant_name, dict):
            if 'plant_id' in plant_name:
                return plant_name['plant_id']
            elif 'name' in plant_name:
                actual_plant_name = plant_name['name']
            else:
                logger.warning(f"Invalid plant object: {plant_name}")
                return None
        else:
            actual_plant_name = plant_name
        
        session = get_db_session()
        
        # Query to find plant_id by plant_name
        result = session.query(TblGeneration.plant_id).filter(
            TblGeneration.plant_name == actual_plant_name
        ).first()
        
        session.close()
        
        if result:
            return result[0]
        else:
            logger.warning(f"Plant ID not found for plant name: {actual_plant_name}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to get plant ID from name: {e}")
        return None

def get_consumption_unit_from_plant(plant_name) -> Optional[str]:
    """
    Get consumption unit associated with a plant.
    This is a mapping function that needs to be implemented based on your business logic.
    
    Args:
        plant_name: Plant name or plant object
        
    Returns:
        Consumption unit identifier
    """
    try:
        session = get_db_session()
        
        # Handle case where plant_name is actually a plant object/dict
        if isinstance(plant_name, dict):
            if 'plant_id' in plant_name:
                plant_id = plant_name['plant_id']
            elif 'name' in plant_name:
                plant_id = get_plant_id_from_name(plant_name['name'])
            else:
                logger.warning(f"Invalid plant object: {plant_name}")
                return None
        else:
            # Handle case where plant_name is a string
            plant_id = get_plant_id_from_name(plant_name)
        
        if not plant_id:
            logger.warning(f"Could not determine plant_id for: {plant_name}")
            return None
        
        # Query to find consumption unit associated with the plant
        # This assumes there's a relationship between plant and consumption unit
        result = session.query(SettlementData.cons_unit).filter(
            SettlementData.plant_id == plant_id
        ).first()
        
        session.close()
        
        if result:
            return result[0]
        else:
            logger.warning(f"Consumption unit not found for plant: {plant_name}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to get consumption unit from plant: {e}")
        return None