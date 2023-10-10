![Example WASN](banner.png)

# Python implementation of simulation framework used for production of experimantal results published in the paper ```"Online distributed waveform-synchronization for acoustic sensor networks with dynamic topology"``` written by Aleksej Chinaev, Niklas Knaepper and Gerald Enzner

The simulation framework uses Python scripts and a notebook to present the online distributed waveform-based synchronization of wireless acoustic sensor networks with dynamic topology: A Jupyter-Notebook with a demo of the simulation framework
- distributed_synchro_dynTop.ipynb

and the step-by-step Python main scripts, developed for the following tasks:
- Draw groups of nodes for which topologies are generated in the simulation (simulation_1_draw_nodeGroups.py)
- Simulate WASN operation for previously drawn node groups (simulation_2_simulate_WASNs.py)
- Evaluate simulation results such as RMSE, AMSC and SSNR (simulation_3_evaluate.py)
- Load and sort evaluation results to produce the final plots (simulation_4_summarize_and_plot.py)


## 1 List of dependencies

- numpy 1.24.4
- numpy-stl 3.0.1
- tqdm 4.62.3
- scipy 1.8.0
- matplotlib 3.7.2
- paderbox 0.0.3
- paderwasn 0.0.0
- lazy-dataset 0.0.12


## 2 WASN signal database

Please download the database of simulated node signals and relevant metadata from:
[https://cloud.uol.de/s/dq5SCkLg7yPHgC3](https://cloud.uol.de/s/dq5SCkLg7yPHgC3) and place the files inside the `data/` directory.

## 3 Running the complete simulation

Please note: For the given set of 50 initial WASN topologies in `node_topologies.pkl`, the complete set of simulation and evaluation results is about **66GB**. Please make sure to have enough disk space available or, alternatively, utilize a smaller set of topologies (see 3.1).

### 3.1 Drawing random topologies

If you wish to run the simulation with a set of random topologies, different from that used in the publication, use the script `simulation_1_draw_topologies.py`. You may adjust the following parameters:
- `TARGET_FILE`: The exported file containing the generated set of topologies.
- `t_range_switch`: Interval from which a random time-point of network modification will be drawn for each initial WASN topology.
- `N_topologies`: Number of random topologies to draw.
- `N_range_init`: Interval from which the number of nodes in each initial WASN is drawn.

You can skip this step and use the provided file `node_topologies.pkl`, which allows to reproduce the results shown in the publication.

### 3.2 Simulating WASN operation

The script `simulation_2_simulate_WASNs.py` simulates WASN operation for every topology in the previously drawn set, including the network modification of the specified type. By default, this is done using the provided set of random topologies. To run the simulation based on your custom set of topologies, please set `FLAG_CUSTOM_SIMULATION=True` and optionally adjust the paths in `TOPOLOGIES_FILE` and `SIM_TARGET_DATA_ROOT`.

In order to generate a complete set of simulation results, **this script needs to be executed 4 times in total**, once for each of the four possible modification scenarios configured via `WASN_MODIFICATION`.

The following provides an overview over the adjustable parameters:

- `FLAG_CUSTOM_SIMULATION`: Flag to conveniently toggle between default and custom paths.
- `SIM_TARGET_ROOT`: Root directory where all simulation (and evaluation) results will be stored.
- `TOPOLOGIES_FILE`: Path to the file containing the generated set of topologies. Usually, this will be inside the directory specified in SIM_TARGET_ROOT.
- `DATA_ROOT`: Directory of the core database, containing node audio signals and other metadata.
- `WASN_MODIFICATION`: Specifies the type of modification that is simulated. Can be one of the following: "join", "leave", "unlink", "leave_root".
- `sig_len_sec`: Simulation length in seconds.

### 3.3 Evaluation of simulation results

`simulation_3_evaluate.py` is used to evaluate the results of the previously simulated WASN operation. Specifically, the following evaluation metrics are computed:
- RMSE of SRO estimation
- Settling time of SRO estimation
- SSNR of synchronized and asynchronous signals
- AMSC of synchronized and asynchronous signals

As before, set `FLAG_CUSTOM_SIMULATION=True` and optionally adjust the path in `RESULTS_DATA_ROOT` if you used a custom set of topologies for the simulation or otherwise changed the directory in which the simulation results are stored.

In order to generate a complete set of evaluation results, **this script needs to be executed 5 times in total**, once for each of the four possible modification scenarios configured via `WASN_MODIFICATION` while `EVAL_BEFORE_SEGMENT=False`, and one additional time to evaluate the signal segments before any network modification via `EVAL_BEFORE_SEGMENT=True` and `WASN_MODIFICATION` set to any of the four possible values, for example `WASN_MODIFICATION='join'`.

The following provides an overview over the adjustable parameters:
- `FLAG_CUSTOM_SIMULATION`: Flag to conveniently toggle between default and custom paths.
- `RESULTS_DATA_ROOT` Root directory of results. Should be the same as SIM_TARGET_ROOT in simulation_2_simulate_WASNs.py.
- `EVAL_BEFORE_SEGMENT` Toggles evaluation of simulation results before WASN modification versus results after WASN modification.
- `WASN_MODIFICATION` The type of modification for which simulation results are evaluated. Can be one of the following: "join", "leave", "unlink", "leave_root".
- `N_PROCS_MAX` Maximum number of parallel processes spawned. Can be used to limit memory requirements.

### 3.4 Plotting evaluation results

Run `simulation_4_summarize_and_plot.py` to generate plots summarizing the previously computed evaluation results. This script produces four plots corresponding to the following figures in the publication:
- Figure 13: **SRO-RMSE** values for persistent nodes
within last 10 seconds before Tc (left) and within
first 10 seconds after Tc (middle) and for newly
joined nodes within last 10 signal seconds (right).
- Figure 14: Settling times **Ts** of SRO estimation of
the initial WASN split by nodes depths ∈ {1, 2, 3}
(left) and of the newly joined nodes (right).
- Figure 15 (a): Synchronisation performance in terms of **AMSC** for persistent nodes within last 10 seconds before Tc (left) and first 10 seconds
after Tc (middle) and for newly integrated nodes
within last 10 signal seconds (right).
- Figure 15 (b): Synchronisation performance in terms of **SSNR** for persistent nodes within last 10 seconds before Tc (left) and first 10 seconds
after Tc (middle) and for newly integrated nodes
within last 10 signal seconds (right).


## 4 Exploring an exemplaray WASN and its modification

The jupyter-notebook `distributed_synchro_dynTop.ipynb` enables to explore a specific WASN topology and arbitrary modifications applied to it.

### 4.1 Demo configuration

Confirm that `DATA_ROOT` points to the location of the core database. You can adjust the duration of WASN simulation in `sig_lec_sec`.

#### 4.1.1 Configuration of WASN topology and network change

Every topology is managed by an instance of the `TopologyManager` class, which automatically constructs the optimal topology for any given set of nodes and further exposes an abstract interface for all possible modifications, while handling the algorithmic details, as described in the publication, under the hood. Note that `TopologyManager` only manages the topology and its modification, but not the WASN simulation itself.

The current topology, be it after initialization or a specific modification applied, can always be accessed via the `nodes_levels` attribute.

Use `node_ids_start` to define the list of all nodes in the initial WASN, where nodes are referred to by their numerical id. `node_ids_start` may contain any node id ranging from 0 to 12.

The timepoint of network modification can be defined using `Tc_sec`.

With `wasn_modification`, you can toggle between one of four pre-defined network modifications, each corresponding to one of the four fundamental types of modification. Valid values are 1, 2, 3 and 4:
1. Appearance of a new node
2. Failure of a communication link 
3. Failure of a non-reference sensor node 
4. Failure of the reference sensor node

Refer to the following summary for details on how to change specific modifications:

1. `TopMng.add_nodes(node_coords)` Adds one or more nodes to the network. `node_coords` is a dictionary, where the keys are node ids (int) and values specify the coordinates (list). Node coordinates can be accessed via the `node_coord()` function.
   - Example: `TopMng.add_nodes({4: node_coord(4)})`
2. `TopMng.set_node_links(links, state)` enables or disables links between nodes. `links` is a list where each element is itself a list of two node ids, representing the link between them. Use `state=True` to enable and `state=False` to disable links.
   - Example: `TopMng.set_node_links([[6, 9]], False)`

3. `TopMng.remove_nodes(nodes)` Removes one or more nodes from the network. `nodes` is a list of numerical node ids. Note: This method can be used to remove non-reference, as well as reference nodes.
   - Example: `TopMng.remove_nodes([1])`

### 4.2 Loading of Audio- and Metadata
Execute this cell to load all node audio signals and relevant metadata

### 4.3 Dynamic WASN simulation and evaluation results
Execute this cell to run the actual WASN simulation. During the simulation, the network switches to the modified WASN topology at the specified time point.

#### 4.3.1 Visualization of SRO trajectories
The upper plot displays each nodes estimated SRO over time, where the dashed lines indicate the true (relative) SRO for each node.
Similarily, the lower plot displays the corresponding residual SRO estimates, which should always converge to zero.
The black, dashed line in both plots indicates the time-point of network modification `Tc_sec`.

#### 4.3.2 Evaluation results
Execute these cells to run the evaluation for both before, and after the network modification.

#### 4.3.3 Visualization of evaluation results
These plots summarize the WASN performance before and after the network modification using boxplots via the following metrics:
- RMSE of SRO estimation
- AMSC of asynchronous and synchronized signals
- SSNR of asynchronous and synchronized signals
