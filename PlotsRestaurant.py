# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:14:45 2025

@author: geots
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def plot_HVAC_results_by_day(data_frame, day_to_plot):
    # 1. Filter the data for the specific day requested
    data_day = data_frame[data_frame['Day'] == day_to_plot]

    if data_day.empty:
        print(f"No data found for Day {day_to_plot}")
        return

    # 2. Mapping variables from the filtered data
    T = data_day['Hour']
    Temp_r1 = data_day['Temp_Room1']
    Temp_r2 = data_day['Temp_Room2']
    h_r1 = data_day['Power_Heater1']
    h_r2 = data_day['Power_Heater2']
    v = data_day['Ventilation_On']
    Hum = data_day['Humidity']
    price = data_day['Price']
    Occ_r1 = data_day['Occupancy_R1']
    Occ_r2 = data_day['Occupancy_R2']

    # 3. Create the plots
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Room Temperatures
    axes[0].plot(T, Temp_r1, label='Room 1 Temp', marker='o')
    axes[0].plot(T, Temp_r2, label='Room 2 Temp', marker='s')
    axes[0].axhline(18, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(20, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"Room Temperatures - Day {day_to_plot}")
    axes[0].legend()
    axes[0].grid(True)

    # Heater consumption
    axes[1].bar(T, h_r1, width=0.4, label='Room 1 Heater', alpha=0.7)
    axes[1].bar(T, h_r2, width=0.4, bottom=h_r1, label='Room 2 Heater', alpha=0.7)
    axes[1].set_ylabel("Heater Power (kW)")
    axes[1].set_title(f"Heater Consumption - Day {day_to_plot}")
    axes[1].legend()
    axes[1].grid(True)

    # --- Graph 2: Ventilation and Humidity (MODIFIED) ---
    # Left Axis: Humidity
    ax2_hum = axes[2]
    ax2_hum.plot(T, Hum, label='Humidity (%)', color='tab:orange', marker='o')
    ax2_hum.axhline(45, color='gray', linestyle='--', alpha=0.5)
    ax2_hum.axhline(60, color='gray', linestyle='--', alpha=0.5)
    ax2_hum.set_ylabel("Humidity (%)", color='tab:orange')
    ax2_hum.tick_params(axis='y', labelcolor='tab:orange')
    ax2_hum.set_ylim(0, 100)  # Optional: keeps humidity on a 0-100 scale

    # Right Axis: Ventilation
    ax2_vent = ax2_hum.twinx()
    ax2_vent.step(T, v, where='mid', label='Ventilation ON', color='tab:blue', linewidth=2)
    ax2_vent.set_ylabel("Ventilation (0=OFF, 1=ON)", color='tab:blue', fontsize=8)
    ax2_vent.tick_params(axis='y', labelcolor='tab:blue')
    ax2_vent.set_ylim(-0.1, 1.5)  # Keeps the binary line from touching the top/bottom frame

    # Combined Legend handling
    lines1, labels1 = ax2_hum.get_legend_handles_labels()
    lines2, labels2 = ax2_vent.get_legend_handles_labels()
    ax2_hum.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax2_hum.set_title(f"Ventilation and Humidity - Day {day_to_plot}")
    ax2_hum.grid(True)

    # Electricity price and occupancy
    axes[3].plot(T, price, label='TOU Price (€/kWh)', color='tab:red', marker='x')
    axes[3].bar(T, Occ_r1, label='Occupancy Room 1', alpha=0.5)
    axes[3].bar(T, Occ_r2, bottom=Occ_r1, label='Occupancy Room 2', alpha=0.5)
    axes[3].set_ylabel("Price / Occupancy")
    axes[3].set_xlabel("Time (hours)")
    axes[3].set_title(f"Price and Occupancy - Day {day_to_plot}")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()


# Load your file
df = pd.read_csv('HVAC_Optimization_Results.csv')

# Call the function for any specific day
plot_HVAC_results_by_day(df, day_to_plot= 84)