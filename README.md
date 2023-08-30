### Python implementation of simulation framework used for production of experimantal results published in the paper ```"Online distributed waveform-synchronization for acoustic sensor networks with dynamic topology"``` written by Aleksej Chinaev, Niklas Knaepper and Gerald Enzner

The simulation framework uses Python scripts and a notebook to present the online distributed waveform-based synchronization of wireless acoustic sensor networks with dynamic topology:
- a Python-Notebook with a demo of the simulation framework (distributed_synchro_dynTop.ipynb)
- the step-by-step Python main scripts (simulation_1_draw_nodeGroups.py, simulation_2_simulate_WASNs.py, simulation_3_evaluate.py, simulation_4_summarize_and_plot.py)
- the Python modules required for the simulation framework (audio_reader.py, delay_buffer.py, online_resampler.py, plot_utils.py, sro_estimation.py, topology_tools.py)

The main Python scripts are developed for the following tasks:
- Draw groups of nodes for which topologies are generated in the simulation (simulation_1_draw_nodeGroups.py)
- Simulates WASNs for previously drawn node groups (simulation_2_simulate_WASNs.py)
- Evaluate simulation results such as RMSE, AMSC and SSNR (simulation_3_evaluate.py)
- Load and sort evaluation results to produce the final plots (simulation_4_summarize_and_plot.py)

The download links for a data base required for the simulation framework will be included soon...
