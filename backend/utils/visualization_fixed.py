"""
Fixed version of the create_tod_generation_plot function
"""

def create_tod_generation_plot(df, plant_name, start_date, end_date=None):
    """
    Create a stacked bar chart of generation data with custom ToD bins
    based on the configuration settings.

    Args:
        df (DataFrame): ToD binned data with generation
        plant_name (str): Name of the plant
        start_date (datetime): Start date
        end_date (datetime, optional): End date. If None, only start_date is used.

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    try:
        # Import pandas and numpy for DataFrame operations
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        
        # Check if date information is available for multi-day plots
        if end_date is not None and start_date != end_date and 'date' not in df.columns:
            # For multi-day plots without date information, create a simple bar chart
            logger.warning("Date information missing for multi-day ToD generation plot. Creating simple bar chart.")
            
            # Create a simple bar chart with the aggregated data
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot the data as a simple bar chart
            bars = ax.bar(
                range(len(df)),
                df['generation_kwh'],
                width=0.6,
                color=COLORS.get("generation", "#4CAF50"),
                alpha=0.8
            )
            
            # Add data labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height/2,
                    f'{height:.1f}',
                    ha='center',
                    va='center',
                    fontsize=9,
                    rotation=90,
                    color='white'
                )
            
            # Set x-axis ticks and labels
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([str(bin) for bin in df['tod_bin']], rotation=45, ha='right')
            
            # Set title and labels
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            ax.set_title(f"ToD Generation for {plant_name} ({date_range})", fontsize=16, pad=20)
            ax.set_ylabel("Generation (kWh)", fontsize=12)
            ax.set_xlabel("Time of Day", fontsize=12)
            
            # Add grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            
            # Format y-axis with K for thousands
            ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
            
            plt.tight_layout()
            return fig
            
        # DEBUGGING: Log visualization data for Y-axis consistency verification
        total_generation = df['generation_kwh'].sum()
        max_gen_value = df['generation_kwh'].max()
        logger.info(f"ToD Generation Only Plot - Total Gen: {total_generation:.2f} kWh, Max Gen Value: {max_gen_value:.2f} kWh")
        logger.info(f"ToD Generation Only Plot - Data points: {len(df)}, ToD bins: {df['tod_bin'].tolist()}")

        # SOLUTION: Normalize ToD values to show average per 15-minute interval for comparable Y-axis scales
        # Define intervals per ToD bin (assuming 15-minute intervals)
        # FIXED: Use actual ToD bin labels from configuration
        from backend.config.tod_config import get_tod_bin_labels
        tod_bin_labels = get_tod_bin_labels("full")
        
        tod_intervals = {
            tod_bin_labels[0]: 16,        # 6 AM - 10 AM (Peak): 4 hours × 4 intervals/hour
            tod_bin_labels[1]: 32,        # 10 AM - 6 PM (Off-Peak): 8 hours × 4 intervals/hour
            tod_bin_labels[2]: 16,        # 6 PM - 10 PM (Peak): 4 hours × 4 intervals/hour
            tod_bin_labels[3]: 32         # 10 PM - 6 AM (Off-Peak): 8 hours × 4 intervals/hour
        }

        # Create normalized dataframe for visualization
        df_normalized = df.copy()
        for idx, row in df_normalized.iterrows():
            tod_bin = row['tod_bin']
            if tod_bin in tod_intervals:
                intervals = tod_intervals[tod_bin]
                # Normalize to average per 15-minute interval
                df_normalized.at[idx, 'generation_kwh'] = row['generation_kwh'] / intervals

        # Log normalized values for verification
        logger.info(f"ToD Generation Only Plot NORMALIZED - Max Gen Value: {df_normalized['generation_kwh'].max():.2f} kWh per 15-min interval")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Define ToD categories and colors
        tod_categories = {
            0: tod_bin_labels[0],  # '6 AM - 10 AM (Peak)'
            1: tod_bin_labels[1],  # '10 AM - 6 PM (Off-Peak)'
            2: tod_bin_labels[2],  # '6 PM - 10 PM (Peak)'
            3: tod_bin_labels[3]   # '10 PM - 6 AM (Off-Peak)'
        }
        
        tod_colors = {
            tod_bin_labels[0]: '#FF5722',  # Morning peak
            tod_bin_labels[1]: '#FFC107',  # Daytime off-peak
            tod_bin_labels[2]: '#E91E63',  # Evening peak
            tod_bin_labels[3]: '#3F51B5'   # Nighttime off-peak
        }
        
        is_single_day = end_date is None or start_date == end_date
        
        # Handle multi-day case
        if not is_single_day and 'date' in df.columns:
            # Sort the dataframe by date
            df = df.sort_values('date')
            
            # Get unique dates
            unique_dates = df['date'].unique()
            
            # Create a new dataframe with data organized by date and ToD category
            plot_data = []
            
            # Process each date
            for date_val in unique_dates:
                date_df = df[df['date'] == date_val]
                
                # Initialize data for this date
                date_data = {
                    'date': date_val,
                    'date_str': date_val.strftime('%Y-%m-%d')
                }
                
                # Initialize all categories with zero
                for cat in tod_categories.values():
                    date_data[cat] = 0
                
                # Fill in the values from our data
                for _, row in date_df.iterrows():
                    tod_bin = row['tod_bin']
                    gen_kwh = row['generation_kwh']
                    
                    # Map ToD bin name to expected format
                    mapped_tod_bin = map_tod_bin_name(tod_bin)
                    
                    # Check if the mapped tod_bin is in our categories
                    if mapped_tod_bin in tod_categories.values():
                        date_data[mapped_tod_bin] = gen_kwh
                    # Check if the original tod_bin is in our categories
                    elif tod_bin in tod_categories.values():
                        date_data[tod_bin] = gen_kwh
                
                plot_data.append(date_data)
            
            # Convert to DataFrame for plotting
            plot_df = pd.DataFrame(plot_data)
            
            # Set up x positions for the bars
            x = np.arange(len(plot_df))
            width = 0.6
            
            # Create the stacked bar chart - one stacked bar for each date
            bottom = np.zeros(len(plot_df))
            
            # Plot each category as a segment of the stacked bars
            for cat in tod_categories.values():
                if cat in plot_df.columns:
                    ax.bar(
                        x,
                        plot_df[cat],
                        bottom=bottom,
                        label=cat,
                        color=tod_colors[cat],
                        alpha=0.8,
                        width=width
                    )
                    bottom += plot_df[cat].values
            
            # Set x-axis labels
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df['date_str'], rotation=45, ha='right')
            
        else:
            # Single day or no date column - create simple bar chart
            bars = ax.bar(
                range(len(df)),
                df['generation_kwh'],
                width=0.6,
                color=[tod_colors.get(map_tod_bin_name(bin), '#4CAF50') for bin in df['tod_bin']],
                alpha=0.8
            )
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height/2,
                    f'{height:.1f}',
                    ha='center',
                    va='center',
                    fontsize=9,
                    rotation=90,
                    color='white'
                )
            
            # Set x-axis ticks and labels
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([str(bin) for bin in df['tod_bin']], rotation=45, ha='right')
        
        # Get plant display name
        from backend.data.db_data_manager import get_plant_display_name
        plant_display_name = get_plant_display_name(plant_name)
        
        # Set title and labels
        if is_single_day:
            date_str = start_date.strftime('%Y-%m-%d')
            ax.set_title(f"ToD Generation for {plant_display_name} on {date_str}", fontsize=16, pad=20)
        else:
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            ax.set_title(f"ToD Generation for {plant_display_name} ({date_range})", fontsize=16, pad=20)
        
        ax.set_ylabel("Generation (kWh)", fontsize=12)
        ax.set_xlabel("Time of Day" if is_single_day else "Date", fontsize=12)
        
        # Format y-axis with K for thousands
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
        
        # Add grid and legend
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error creating ToD generation plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig