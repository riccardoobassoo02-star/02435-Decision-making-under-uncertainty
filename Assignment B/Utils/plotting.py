import matplotlib.pyplot as plt



def plot_experiment(day, day_log, price_series=None, temp_thresholds=None, humidity_threshold=None):
    """Plot (for an experiment=day) two-room view with power vs temperature (twin axes) and thresholds.
    Inputs: 
    - rep: replication index (for title)
    - day: day index (for title)
    - day_log: dictionary containing the logged data for the day (hour, P1, P2, T1, T2, H, V)
    - price_series: series of electricity prices to plot on the lower graph
    - temp_thresholds: tuple of (min_comfort_threshold, OK_threshold) to plot as horizontal lines on the temperature graphs
    - humidity_threshold: humidity threshold to plot as a horizontal line on the humidity graph
    """
    hours = day_log["hour"]
    P1 = day_log["P1"] # here-and-now decision p1 (uppercase because it's a constant for the environment)
    P2 = day_log["P2"] # here-and-now decision p2 (uppercase because it's a constant for the environment)
    T1 = day_log["T1"]
    T2 = day_log["T2"]
    H = day_log["H"]
    V = day_log["V"]   # here-and-now decision v (uppercase because it's a constant for the environment)

    fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    fig.suptitle(f"Day {day+1}")

    # Room 1: Power and temperature
    ax1 = axs[0]
    ax1.plot(hours, P1, '-o', color='red', label='Heating Power R1')
    ax1.set_ylabel('Power R1 [kW]', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)

    ax1b = ax1.twinx()
    ax1b.plot(hours, T1, '-s', color='green', label='Temp R1')
    ax1b.set_ylabel('Temp R1 [°C]', color='green')
    ax1b.tick_params(axis='y', labelcolor='green')
    if temp_thresholds is not None:
        min_thr, ok_thr = temp_thresholds
        ax1b.axhline(min_thr, color='cyan', linestyle='--', label='Min comfort temp')
        ax1b.axhline(ok_thr, color='magenta', linestyle='--', label='OK temp')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(handles1 + handles1b, labels1 + labels1b, loc='upper right')
    ax1.set_title('Room 1: Power and Temperature')

    # Room 2: Power and temperature
    ax2 = axs[1]
    ax2.plot(hours, P2, '-o', color='red', label='Heating Power R2')
    ax2.set_ylabel('Power R2 [kW]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True)

    ax2b = ax2.twinx()
    ax2b.plot(hours, T2, '-s', color='green', label='Temp R2')
    ax2b.set_ylabel('Temp R2 [°C]', color='green')
    ax2b.tick_params(axis='y', labelcolor='green')
    if temp_thresholds is not None:
        min_thr, ok_thr = temp_thresholds
        ax2b.axhline(min_thr, color='cyan', linestyle='--', label='Min comfort temp')
        ax2b.axhline(ok_thr, color='magenta', linestyle='--', label='OK temp')

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(handles2 + handles2b, labels2 + labels2b, loc='upper right')
    ax2.set_title('Room 2: Power and Temperature')

    # Lower plot: humidity, ventilation and electricity price
    ax3 = axs[2]
    ax3.plot(hours, H, '-o', color='green', label='Humidity')
    ax3.set_ylabel('Humidity [%]', color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.grid(True)
    if humidity_threshold is not None:
        ax3.axhline(humidity_threshold, color='gray', linestyle='--', label='Humidity Threshold')

    if price_series is not None:
        ax3b = ax3.twinx()
        ax3b.plot(hours, price_series, '-o', color='blue', label='Electricity Price')
        ax3b.set_ylabel('Price [€/kWh]', color='blue')
        ax3b.tick_params(axis='y', labelcolor='blue')

        # Add ventilation to the same twin axis or separate
        ax3c = ax3.twinx()
        ax3c.plot(hours, V, '-s', color='red', label='Ventilation')
        ax3c.set_ylabel('Ventilation', color='red')
        ax3c.tick_params(axis='y', labelcolor='red')
        ax3c.set_ylim(-0.1, 1.1)
        # Position ax3c to the right of ax3b
        ax3c.spines["right"].set_position(("axes", 1.1))

    handles3, labels3 = ax3.get_legend_handles_labels()
    if price_series is not None:
        handles3b, labels3b = ax3b.get_legend_handles_labels()
        handles3c, labels3c = ax3c.get_legend_handles_labels()
        ax3.legend(handles3 + handles3b + handles3c, labels3 + labels3b + labels3c, loc='upper right')
    else:
        ax3.legend(handles3, labels3, loc='upper right')

    ax3.set_xlabel('Hour')
    ax3.set_title('Humidity, Ventilation and Electricity Price')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

