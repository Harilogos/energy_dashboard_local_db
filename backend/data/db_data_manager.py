"""
Database-based data manager that replaces all API-based data functions.
This module provides all the data functions needed by the dashboard using database queries.
"""

import pandas as pd
import streamlit as st
import traceback
from functools import wraps


from backend.data.db_data import (
    get_generation_data_db,
    get_consumption_data_db,
    get_consumption_data_by_client,
    get_settlement_data_db,
    get_plants_from_db,
    get_daily_aggregated_generation_db,
    get_daily_aggregated_consumption_db,
    get_tod_aggregated_data_db,
    get_combined_plants_data_db,
    get_plant_id_from_name,
    get_consumption_unit_from_plant
)
from backend.utils.client_mapping import (
    get_client_name_from_plant_name,
    get_plant_id_from_plant_name,
    validate_client_plant_mapping
)
from backend.logs.logger_setup import setup_logger
from backend.config.tod_config import get_tod_slots, get_tod_slot

# Configure logging
logger = setup_logger('db_data_manager', 'db_data_manager.log')

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

# Plant management functions
@st.cache_data(ttl=3600)
@retry_on_exception()
def get_plants():
    """
    Get all available plants from database.
    
    Returns:
        Dictionary with plant information organized by client and type
    """
    try:
        plants_dict = get_plants_from_db()
        logger.info(f"Retrieved plants data from database")
        return plants_dict
    except Exception as e:
        logger.error(f"Failed to get plants: {e}")
        return {}

def get_plant_display_name(plant_obj):
    """
    Get display name for a plant object.
    
    Args:
        plant_obj: Plant object with 'name' attribute or string
        
    Returns:
        Display name for the plant
    """
    if isinstance(plant_obj, dict):
        return plant_obj.get('name', str(plant_obj))
    elif hasattr(plant_obj, 'name'):
        return plant_obj.name
    else:
        return str(plant_obj)

def get_plant_id(plant_name):
    """
    Get plant ID from plant name using client.json mapping.
    
    Args:
        plant_name: Name of the plant or plant object with 'plant_id' key
        
    Returns:
        Plant ID if found, plant_name otherwise
    """
    try:
        # Handle case where plant_name is actually a plant object/dict
        if isinstance(plant_name, dict):
            if 'plant_id' in plant_name:
                return plant_name['plant_id']
            elif 'name' in plant_name:
                plant_id = get_plant_id_from_plant_name(plant_name['name'])
                return plant_id if plant_id else plant_name['name']
        
        # Handle case where plant_name is a string - use client.json mapping
        plant_id = get_plant_id_from_plant_name(plant_name)
        return plant_id if plant_id else plant_name
    except Exception as e:
        logger.error(f"Failed to get plant ID for {plant_name}: {e}")
        return plant_name

def is_solar_plant(plant_name):
    """
    Check if a plant is a solar plant.
    
    Args:
        plant_name: Name of the plant or plant object
        
    Returns:
        True if solar plant, False otherwise
    """
    try:
        # Handle case where plant_name is actually a plant object/dict
        if isinstance(plant_name, dict):
            actual_plant_name = plant_name.get('name', str(plant_name))
        else:
            actual_plant_name = plant_name
        
        plants = get_plants()
        for client_name, client_plants in plants.items():
            for plant in client_plants.get('solar', []):
                if plant.get('name') == actual_plant_name:
                    return True
        return False
    except Exception as e:
        logger.error(f"Failed to check if {plant_name} is solar: {e}")
        return False

# Generation data functions
@st.cache_data(ttl=3600)
@retry_on_exception()
def get_generation_consumption_comparison(plant_name, date):
    """
    Get generation vs consumption comparison for a single day.
    
    Args:
        plant_name: Name of the plant
        date: Date for comparison
        
    Returns:
        Tuple of (generation_df, consumption_df)
    """
    try:
        plant_id = get_plant_id(plant_name)
        cons_unit = get_consumption_unit_from_plant(plant_name)
        
        if not plant_id or not cons_unit:
            logger.warning(f"Could not find plant_id or cons_unit for {plant_name}")
            return pd.DataFrame(), pd.DataFrame()
        
        date_str = date.strftime('%Y-%m-%d')
        
        # Get generation data
        generation_df = get_generation_data_db(plant_id, date_str, date_str)
        
        # Get consumption data
        consumption_df = get_consumption_data_db(cons_unit, date_str, date_str)
        
        # Rename columns to match expected format
        if not generation_df.empty:
            generation_df = generation_df.rename(columns={'datetime': 'time', 'generation': 'Generation'})
        
        if not consumption_df.empty:
            consumption_df = consumption_df.rename(columns={'datetime': 'time', 'consumption': 'Consumption'})
        
        logger.info(f"Retrieved generation-consumption comparison for {plant_name} on {date_str}")
        return generation_df, consumption_df
        
    except Exception as e:
        logger.error(f"Failed to get generation-consumption comparison: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_daily_generation_consumption_comparison(selected_plant, start_date, end_date):
    """
    Get daily aggregated generation vs consumption comparison.
    
    Args:
        selected_plant: Name of the selected plant
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with daily comparison data
    """
    try:
        plant_id = get_plant_id(selected_plant)
        cons_unit = get_consumption_unit_from_plant(selected_plant)
        
        if not plant_id or not cons_unit:
            logger.warning(f"Could not find plant_id or cons_unit for {selected_plant}")
            return pd.DataFrame()
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Get daily aggregated data
        generation_df = get_daily_aggregated_generation_db(plant_id, start_str, end_str)
        consumption_df = get_daily_aggregated_consumption_db(cons_unit, start_str, end_str)
        
        if generation_df.empty or consumption_df.empty:
            logger.warning(f"No data found for {selected_plant} from {start_str} to {end_str}")
            return pd.DataFrame()
        
        # Merge the dataframes
        merged_df = pd.merge(generation_df, consumption_df, on='date', how='outer')
        merged_df = merged_df.fillna(0)
        
        # Rename columns to match expected format
        # Handle both lowercase and capitalized column names
        column_mapping = {}
        if 'generation' in merged_df.columns:
            column_mapping['generation'] = 'generation_kwh'
        if 'Generation' in merged_df.columns:
            column_mapping['Generation'] = 'generation_kwh'
        if 'consumption' in merged_df.columns:
            column_mapping['consumption'] = 'consumption_kwh'
        if 'Consumption' in merged_df.columns:
            column_mapping['Consumption'] = 'consumption_kwh'
        
        if column_mapping:
            merged_df = merged_df.rename(columns=column_mapping)
        
        logger.info(f"Retrieved daily comparison for {selected_plant} from {start_str} to {end_str}")
        return merged_df
        
    except Exception as e:
        logger.error(f"Failed to get daily generation-consumption comparison: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_generation_only_data(plant_name, start_date, end_date=None):
    """
    Get generation-only data for a plant.
    
    Args:
        plant_name: Name of the plant
        start_date: Start date
        end_date: End date (optional, defaults to start_date)
        
    Returns:
        DataFrame with generation data
    """
    try:
        if end_date is None:
            end_date = start_date
        
        plant_id = get_plant_id(plant_name)
        if not plant_id:
            logger.warning(f"Could not find plant_id for {plant_name}")
            return pd.DataFrame()
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Check if single day or multi-day
        if start_date == end_date:
            # Single day - return 15-minute data
            generation_df = get_generation_data_db(plant_id, start_str, end_str)
            if not generation_df.empty:
                generation_df = generation_df.rename(columns={'datetime': 'time', 'generation': 'generation_kwh'})
                # Add hour column for single day plotting
                generation_df['hour'] = generation_df['time'].dt.hour
        else:
            # Multi-day - return daily aggregated data
            generation_df = get_daily_aggregated_generation_db(plant_id, start_str, end_str)
            if not generation_df.empty:
                generation_df = generation_df.rename(columns={'date': 'time', 'generation': 'generation_kwh'})
        
        logger.info(f"Retrieved generation data for {plant_name} from {start_str} to {end_str}")
        return generation_df
        
    except Exception as e:
        logger.error(f"Failed to get generation data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# Consumption data functions
@st.cache_data(ttl=3600)
@retry_on_exception()
def get_consumption_data_from_csv(plant_name, start_date, end_date=None):
    """
    Get consumption data for a plant using client_name mapping from client.json.
    Uses only client_name, datetime, consumption columns as specified.
    
    Args:
        plant_name: Name of the plant
        start_date: Start date
        end_date: End date (optional)
        
    Returns:
        DataFrame with consumption data (columns: time, Consumption)
    """
    try:
        if end_date is None:
            end_date = start_date
        
        # Get client name from plant name using client.json mapping
        client_name = get_client_name_from_plant_name(plant_name)
        if not client_name:
            logger.warning(f"Could not find client name for plant {plant_name}")
            return pd.DataFrame()
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Get consumption data using client name
        consumption_df = get_consumption_data_by_client(client_name, start_str, end_str)
        
        if not consumption_df.empty:
            # Rename columns to match expected format
            consumption_df = consumption_df.rename(columns={'datetime': 'time', 'consumption': 'Consumption'})
        
        logger.info(f"Retrieved consumption data for {plant_name} (client: {client_name}) from {start_str} to {end_str}")
        return consumption_df
        
    except Exception as e:
        logger.error(f"Failed to get consumption data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_daily_consumption_data(plant_name, start_date, end_date):
    """
    Get daily aggregated consumption data.
    
    Args:
        plant_name: Name of the plant
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with daily consumption data
    """
    try:
        cons_unit = get_consumption_unit_from_plant(plant_name)
        if not cons_unit:
            logger.warning(f"Could not find consumption unit for {plant_name}")
            return pd.DataFrame()
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Get daily aggregated consumption data
        consumption_df = get_daily_aggregated_consumption_db(cons_unit, start_str, end_str)
        
        if not consumption_df.empty:
            consumption_df = consumption_df.rename(columns={'date': 'time', 'consumption': 'Consumption'})
        
        logger.info(f"Retrieved daily consumption data for {plant_name} from {start_str} to {end_str}")
        return consumption_df
        
    except Exception as e:
        logger.error(f"Failed to get daily consumption data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# Combined wind and solar functions
@st.cache_data(ttl=3600)
@retry_on_exception()
def get_combined_wind_solar_generation(client_name, start_date, end_date):
    """
    Get combined wind and solar generation data for a client.
    
    Args:
        client_name: Name of the client
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with combined generation data
    """
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Get solar data
        solar_df = get_combined_plants_data_db(client_name, 'solar', start_str, end_str)
        
        # Get wind data
        wind_df = get_combined_plants_data_db(client_name, 'wind', start_str, end_str)
        
        if solar_df.empty and wind_df.empty:
            logger.warning(f"No combined data found for client {client_name}")
            return pd.DataFrame()
        
        # Check if single day or multi-day
        if start_date == end_date:
            # Single day - use datetime
            time_col = 'datetime'
        else:
            # Multi-day - aggregate by date
            if not solar_df.empty:
                solar_df['date'] = solar_df['datetime'].dt.date
                solar_df = solar_df.groupby('date')['generation'].sum().reset_index()
                solar_df['date'] = pd.to_datetime(solar_df['date'])
            
            if not wind_df.empty:
                wind_df['date'] = wind_df['datetime'].dt.date
                wind_df = wind_df.groupby('date')['generation'].sum().reset_index()
                wind_df['date'] = pd.to_datetime(wind_df['date'])
            
            time_col = 'date'
        
        # Merge solar and wind data
        if not solar_df.empty and not wind_df.empty:
            merged_df = pd.merge(solar_df, wind_df, on=time_col, how='outer', suffixes=('_solar', '_wind'))
            merged_df = merged_df.fillna(0)
            merged_df = merged_df.rename(columns={
                'generation_solar': 'Solar Generation',
                'generation_wind': 'Wind Generation',
                time_col: 'time'
            })
        elif not solar_df.empty:
            merged_df = solar_df.copy()
            merged_df['Wind Generation'] = 0
            merged_df = merged_df.rename(columns={'generation': 'Solar Generation', time_col: 'time'})
        elif not wind_df.empty:
            merged_df = wind_df.copy()
            merged_df['Solar Generation'] = 0
            merged_df = merged_df.rename(columns={'generation': 'Wind Generation', time_col: 'time'})
        else:
            return pd.DataFrame()
        
        logger.info(f"Retrieved combined wind-solar data for {client_name} from {start_str} to {end_str}")
        return merged_df
        
    except Exception as e:
        logger.error(f"Failed to get combined wind-solar data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# Time-of-Day (ToD) functions
@st.cache_data(ttl=3600)
@retry_on_exception()
def get_tod_binned_data(plant_name, start_date, end_date=None):
    """
    Get Time-of-Day binned data for a plant.
    
    Args:
        plant_name: Name of the plant
        start_date: Start date
        end_date: End date (optional)
        
    Returns:
        DataFrame with ToD binned data
    """
    try:
        if end_date is None:
            end_date = start_date
        
        plant_id = get_plant_id(plant_name)
        if not plant_id:
            logger.warning(f"Could not find plant_id for {plant_name}")
            return pd.DataFrame()
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Get ToD aggregated data
        tod_df = get_tod_aggregated_data_db(plant_id, start_str, end_str)
        
        if tod_df.empty:
            logger.warning(f"No ToD data found for {plant_name}")
            return pd.DataFrame()
        
        # Check if this is multi-day data
        is_multi_day = start_date != end_date
        
        if is_multi_day:
            # For multi-day analysis, we have both daily breakdown and total aggregation
            # Filter to get only the total aggregation (where date equals start_date)
            start_date_obj = pd.to_datetime(start_str).date()
            total_aggregation = tod_df[tod_df['date'] == start_date_obj].copy()
            
            if not total_aggregation.empty:
                # Use total values for multi-day ToD comparison
                total_aggregation = total_aggregation.rename(columns={
                    'generation_total': 'generation_kwh',
                    'consumption_total': 'consumption_kwh',
                    'surplus_total': 'surplus'
                })
                
                logger.info(f"Using total aggregation for multi-day ToD data for {plant_name}")
                logger.info(f"Total generation across all days: {total_aggregation['generation_kwh'].sum():.2f} kWh")
                logger.info(f"Total consumption across all days: {total_aggregation['consumption_kwh'].sum():.2f} kWh")
                
                return total_aggregation
            else:
                # Fallback to normalized values if total aggregation is not available
                logger.warning(f"Total aggregation not found, using normalized values for {plant_name}")
                tod_df = tod_df.rename(columns={
                    'generation_normalized': 'generation_kwh',
                    'consumption_normalized': 'consumption_kwh',
                    'surplus_normalized': 'surplus'
                })
        else:
            # For single-day analysis, use normalized values (which represent totals for single day)
            tod_df = tod_df.rename(columns={
                'generation_normalized': 'generation_kwh',
                'consumption_normalized': 'consumption_kwh',
                'surplus_normalized': 'surplus'
            })
        
        # Log the data for debugging
        logger.info(f"Retrieved ToD data for {plant_name} from {start_str} to {end_str}")
        logger.info(f"ToD data columns: {tod_df.columns.tolist()}")
        
        if 'generation_kwh' in tod_df.columns:
            logger.info(f"ToD generation sum: {tod_df['generation_kwh'].sum():.2f} kWh")
            # Log individual ToD bin values
            for tod_bin in tod_df['tod_bin'].unique():
                bin_sum = tod_df[tod_df['tod_bin'] == tod_bin]['generation_kwh'].sum()
                logger.info(f"ToD bin '{tod_bin}' generation sum: {bin_sum:.2f} kWh")
        
        return tod_df
        
    except Exception as e:
        logger.error(f"Failed to get ToD data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# Power cost analysis functions
@st.cache_data(ttl=3600)
@retry_on_exception()
def calculate_power_cost_metrics(plant_name, start_date, end_date, grid_rate_per_kwh):
    """
    Calculate power cost metrics for a plant.
    
    Args:
        plant_name: Name of the plant
        start_date: Start date
        end_date: End date
        grid_rate_per_kwh: Grid electricity rate per kWh
        
    Returns:
        DataFrame with cost analysis
    """
    try:
        plant_id = get_plant_id(plant_name)
        if not plant_id:
            logger.warning(f"Could not find plant_id for {plant_name}")
            return pd.DataFrame()
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Try to get settlement data first
        settlement_df = get_settlement_data_db(plant_id, start_str, end_str)
        
        # Add surplus column for backward compatibility if it doesn't exist
        if not settlement_df.empty and 'surplus' not in settlement_df.columns:
            settlement_df['surplus'] = settlement_df['generation'] - settlement_df['consumption']
        
        if settlement_df.empty:
            logger.warning(f"No settlement data found for {plant_name}, trying to calculate from generation and consumption data")
            
            # Fallback: Get generation and consumption data separately and combine them
            generation_df = get_generation_data_db(plant_id, start_str, end_str)
            
            # Get consumption unit for the plant
            cons_unit = get_consumption_unit_from_plant(plant_name)
            if cons_unit:
                consumption_df = get_consumption_data_db(cons_unit, start_str, end_str)
            else:
                consumption_df = pd.DataFrame()
            
            if generation_df.empty and consumption_df.empty:
                logger.warning(f"No generation or consumption data found for {plant_name}")
                return pd.DataFrame()
            
            # Create a combined dataframe
            if not generation_df.empty and not consumption_df.empty:
                # Merge on datetime
                combined_df = pd.merge(
                    generation_df[['datetime', 'generation']],
                    consumption_df[['datetime', 'consumption']],
                    on='datetime',
                    how='outer'
                ).fillna(0)
            elif not generation_df.empty:
                combined_df = generation_df[['datetime', 'generation']].copy()
                combined_df['consumption'] = 0
            else:
                combined_df = consumption_df[['datetime', 'consumption']].copy()
                combined_df['generation'] = 0
            
            # Calculate surplus (as surplus_demand and surplus_deficit)
            combined_df['surplus_demand'] = (combined_df['generation'] - combined_df['consumption']).clip(lower=0)
            combined_df['surplus_deficit'] = (combined_df['consumption'] - combined_df['generation']).clip(lower=0)
            # For backward compatibility
            combined_df['surplus'] = combined_df['generation'] - combined_df['consumption']
            settlement_df = combined_df
        
        # Calculate cost metrics
        settlement_df['grid_cost'] = settlement_df['consumption'] * grid_rate_per_kwh
        settlement_df['actual_cost'] = (settlement_df['consumption'] - settlement_df['generation']).clip(lower=0) * grid_rate_per_kwh
        settlement_df['savings'] = settlement_df['grid_cost'] - settlement_df['actual_cost']
        settlement_df['savings_percentage'] = (settlement_df['savings'] / settlement_df['grid_cost'] * 100).fillna(0)
        
        # Create columns expected by display components
        settlement_df['consumption_kwh'] = settlement_df['consumption']
        settlement_df['generation_kwh'] = settlement_df['generation']
        settlement_df['net_consumption_kwh'] = (settlement_df['consumption'] - settlement_df['generation']).clip(lower=0)
        
        # Handle datetime column for both single day and multi-day scenarios
        if 'datetime' in settlement_df.columns:
            # Convert datetime to pandas datetime if it's not already
            settlement_df['datetime'] = pd.to_datetime(settlement_df['datetime'])
            
            # Determine if this is single day or multi-day data
            if start_date == end_date:
                # Single day: keep time information
                settlement_df['time'] = settlement_df['datetime']
                settlement_df['date'] = settlement_df['datetime'].dt.date
            else:
                # Multi-day: create both time and date columns
                settlement_df['time'] = settlement_df['datetime']
                settlement_df['date'] = settlement_df['datetime'].dt.date
        
        # Remove the original datetime column if it exists to avoid confusion
        if 'datetime' in settlement_df.columns:
            settlement_df = settlement_df.drop(columns=['datetime'])
        
        logger.info(f"Calculated power cost metrics for {plant_name} from {start_str} to {end_str}")
        return settlement_df
        
    except Exception as e:
        logger.error(f"Failed to calculate power cost metrics: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_power_cost_summary(cost_df):
    """
    Get summary metrics from power cost DataFrame.
    
    Args:
        cost_df: DataFrame with cost analysis
        
    Returns:
        Dictionary with summary metrics
    """
    try:
        if cost_df.empty:
            return {
                'total_grid_cost': 0,
                'total_actual_cost': 0,
                'total_savings': 0,
                'savings_percentage': 0
            }
        
        summary = {
            'total_grid_cost': cost_df['grid_cost'].sum(),
            'total_actual_cost': cost_df['actual_cost'].sum(),
            'total_savings': cost_df['savings'].sum(),
            'savings_percentage': (cost_df['savings'].sum() / cost_df['grid_cost'].sum() * 100) if cost_df['grid_cost'].sum() > 0 else 0
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get power cost summary: {e}")
        return {
            'total_grid_cost': 0,
            'total_actual_cost': 0,
            'total_savings': 0,
            'savings_percentage': 0
        }

# Utility functions for data processing
def compare_generation_consumption(generation_df, consumption_df):
    """
    Compare generation and consumption data.
    
    Args:
        generation_df: DataFrame with generation data
        consumption_df: DataFrame with consumption data
        
    Returns:
        Merged DataFrame with comparison
    """
    try:
        if generation_df.empty or consumption_df.empty:
            return pd.DataFrame()
        
        # Merge on time column
        merged_df = pd.merge(generation_df, consumption_df, on='time', how='outer')
        merged_df = merged_df.fillna(0)
        
        # Calculate surplus/deficit
        merged_df['surplus'] = merged_df['Generation'] - merged_df['Consumption']
        
        # Rename columns to match expected format
        merged_df = merged_df.rename(columns={'Generation': 'generation_kwh', 'Consumption': 'consumption_kwh'})
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Failed to compare generation and consumption: {e}")
        return pd.DataFrame()

def standardize_dataframe_columns(df):
    """
    Standardize DataFrame column names.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized columns
    """
    try:
        if df.empty:
            return df
        
        # Standard column mappings
        column_mappings = {
            'datetime': 'time',
            'Date': 'time',
            'generation': 'Generation',
            'consumption': 'Consumption',
            'surplus': 'Surplus'
        }
        
        # Apply mappings
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to standardize DataFrame columns: {e}")
        return df

# Additional utility functions for compatibility
def get_consumption_data_by_timeframe(plant_name, start_date, end_date=None):
    """
    Get consumption data by timeframe (wrapper for compatibility).
    """
    return get_consumption_data_from_csv(plant_name, start_date, end_date)

def get_generation_consumption_by_timeframe(plant_name, start_date, end_date=None):
    """
    Get generation and consumption data by timeframe.
    """
    try:
        if end_date is None:
            end_date = start_date
        
        if start_date == end_date:
            # Single day
            generation_df, consumption_df = get_generation_consumption_comparison(plant_name, start_date)
            return compare_generation_consumption(generation_df, consumption_df)
        else:
            # Multi-day
            return get_daily_generation_consumption_comparison(plant_name, start_date, end_date)
            
    except Exception as e:
        logger.error(f"Failed to get generation-consumption by timeframe: {e}")
        return pd.DataFrame()

def get_banking_data(plant_name, start_date, end_date=None, banking_type="daily", tod_based=False):
    """
    Get banking data for a plant (placeholder function).
    
    Args:
        plant_name: Name of the plant
        start_date: Start date
        end_date: End date (optional)
        banking_type: Type of banking data
        tod_based: Whether to use ToD-based logic
        
    Returns:
        DataFrame with banking data
    """
    try:
        # For now, return empty DataFrame as banking functionality needs to be implemented
        logger.warning("Banking functionality not yet implemented in database version")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Failed to get banking data: {e}")
        return pd.DataFrame()