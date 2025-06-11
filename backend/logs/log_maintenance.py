"""
Log maintenance utilities for the Energy Generation Dashboard.

This module provides utilities to check log file health, fix common logging issues,
and maintain log file integrity.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from backend.logs.logger_setup import setup_logger
from backend.logs.error_logger import setup_error_logging

# Setup logger for this module
logger = setup_logger('log_maintenance', 'log_maintenance.log')

def check_log_files_health():
    """
    Check the health of all log files and identify potential issues.
    
    Returns:
        dict: Dictionary containing log file health information
    """
    log_dir = Path('logs')
    health_report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'log_files': {},
        'issues': [],
        'recommendations': []
    }
    
    if not log_dir.exists():
        health_report['issues'].append("Log directory does not exist")
        health_report['recommendations'].append("Create logs directory")
        return health_report
    
    # Expected log files
    expected_logs = [
        'app.log',
        'error.log',
        'config.log',
        'db_data.log',
        'db_data_manager.log',
        'display_components.log',
        'visualization.log',
        'tod_config.log',
        'ui_components.log'
    ]
    
    for log_file in expected_logs:
        log_path = log_dir / log_file
        file_info = {
            'exists': log_path.exists(),
            'size': 0,
            'empty': True,
            'last_modified': None,
            'readable': False
        }
        
        if log_path.exists():
            try:
                stat = log_path.stat()
                file_info['size'] = stat.st_size
                file_info['empty'] = stat.st_size == 0
                file_info['last_modified'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                file_info['readable'] = os.access(log_path, os.R_OK)
                
                # Check if file is empty
                if file_info['empty']:
                    health_report['issues'].append(f"{log_file} is empty")
                    health_report['recommendations'].append(f"Check logger configuration for {log_file}")
                
                # Check if file is very large (>50MB)
                if file_info['size'] > 50 * 1024 * 1024:
                    health_report['issues'].append(f"{log_file} is very large ({file_info['size']} bytes)")
                    health_report['recommendations'].append(f"Consider rotating {log_file}")
                
            except Exception as e:
                file_info['error'] = str(e)
                health_report['issues'].append(f"Cannot access {log_file}: {e}")
        else:
            health_report['issues'].append(f"{log_file} does not exist")
            health_report['recommendations'].append(f"Initialize logger for {log_file}")
        
        health_report['log_files'][log_file] = file_info
    
    return health_report

def fix_empty_log_files():
    """
    Fix empty log files by initializing them with proper loggers.
    """
    logger.info("Starting log file maintenance...")
    
    # Initialize all loggers to ensure log files are created
    loggers_to_init = [
        ('app', 'app.log'),
        ('error', 'error.log'),
        ('config', 'config.log'),
        ('db_data', 'db_data.log'),
        ('db_data_manager', 'db_data_manager.log'),
        ('display_components', 'display_components.log'),
        ('visualization', 'visualization.log'),
        ('tod_config', 'tod_config.log'),
        ('ui_components', 'ui_components.log')
    ]
    
    for logger_name, log_file in loggers_to_init:
        try:
            # Create logger which will create the log file if it doesn't exist
            temp_logger = setup_logger(logger_name, log_file)
            temp_logger.info(f"Log file initialized by maintenance script at {datetime.now()}")
            logger.info(f"Initialized logger for {log_file}")
        except Exception as e:
            logger.error(f"Failed to initialize logger for {log_file}: {e}")
    
    # Setup error logging
    try:
        setup_error_logging()
        logger.info("Error logging setup completed")
    except Exception as e:
        logger.error(f"Failed to setup error logging: {e}")

def cleanup_old_logs(days_to_keep=30):
    """
    Clean up old log files and rotate large files.
    
    Args:
        days_to_keep (int): Number of days to keep log files
    """
    log_dir = Path('logs')
    if not log_dir.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    
    for log_file in log_dir.glob('*.log.*'):  # Rotated log files
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                logger.info(f"Deleted old log file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to delete old log file {log_file}: {e}")

def get_log_summary():
    """
    Get a summary of all log files and their status.
    
    Returns:
        dict: Summary of log file status
    """
    health_report = check_log_files_health()
    
    summary = {
        'total_files': len(health_report['log_files']),
        'existing_files': sum(1 for f in health_report['log_files'].values() if f['exists']),
        'empty_files': sum(1 for f in health_report['log_files'].values() if f.get('empty', True)),
        'total_issues': len(health_report['issues']),
        'total_size': sum(f.get('size', 0) for f in health_report['log_files'].values()),
        'last_check': health_report['timestamp']
    }
    
    return summary

def run_maintenance():
    """
    Run complete log maintenance routine.
    """
    logger.info("=== Starting Log Maintenance ===")
    
    # Check current health
    health_before = check_log_files_health()
    logger.info(f"Found {len(health_before['issues'])} issues before maintenance")
    
    # Fix empty log files
    fix_empty_log_files()
    
    # Clean up old files
    cleanup_old_logs()
    
    # Check health after maintenance
    health_after = check_log_files_health()
    logger.info(f"Found {len(health_after['issues'])} issues after maintenance")
    
    # Log summary
    summary = get_log_summary()
    logger.info(f"Maintenance complete. Files: {summary['existing_files']}/{summary['total_files']}, "
                f"Empty: {summary['empty_files']}, Total size: {summary['total_size']} bytes")
    
    logger.info("=== Log Maintenance Complete ===")
    
    return {
        'before': health_before,
        'after': health_after,
        'summary': summary
    }

if __name__ == "__main__":
    # Run maintenance when script is executed directly
    result = run_maintenance()
    
    print("Log Maintenance Report")
    print("=" * 50)
    print(f"Issues before: {len(result['before']['issues'])}")
    print(f"Issues after: {len(result['after']['issues'])}")
    print(f"Files existing: {result['summary']['existing_files']}/{result['summary']['total_files']}")
    print(f"Empty files: {result['summary']['empty_files']}")
    print(f"Total log size: {result['summary']['total_size']} bytes")
    
    if result['after']['issues']:
        print("\nRemaining Issues:")
        for issue in result['after']['issues']:
            print(f"  - {issue}")
    
    if result['after']['recommendations']:
        print("\nRecommendations:")
        for rec in result['after']['recommendations']:
            print(f"  - {rec}")