'''
Load and sort evaluation results to produce the final plots.
'''

#%%
import os
import copy
import matplotlib.pyplot as plt
import pickle

from modules.topology_tools import TopologyManager

#%%
'''
INIT
'''

# Plot evaluation data for topologies from topologies_my.pkl
# SIM_DATA_ROOT_BASE = 'results/simulation/'
# EVAL_DATA_ROOT_BASE = 'results/evaluation/'

# Plot evaluation data for 50 topologies saved in results/2023_03_24/topologies.pkl used in [1], see GitHub.
SIM_DATA_ROOT_BASE = 'results/2023_03_24/simulation/'
EVAL_DATA_ROOT_BASE = 'results/2023_03_24/evaluation/'
PLOT_TARGET_DIR = 'results/figures/'# directory with trailing "/" or empty string to skip exports

if PLOT_TARGET_DIR:
    if not os.path.isdir(PLOT_TARGET_DIR):
        os.mkdir(PLOT_TARGET_DIR)
    import tikzplotlib

#%%
'''
PREPARE SORTED RESULTS
'''

result_block = {
    'amsc': [],
    'amsc_async': [], # only relevant for "before"
    'rmse': [],
    'ssnr': [],
    'ssnr_async': [], # only relevant for "before"
    'ts': [], #only relevant for "before" and "after-joinedOnly"
    'ts_perLevel': {}, #only relevant for "before"
}
eval_results = {
    'before': copy.deepcopy(result_block),
    'after': {
        'join': copy.deepcopy(result_block),
        'leave': copy.deepcopy(result_block),
        'leave_root': copy.deepcopy(result_block),
        'unlink': copy.deepcopy(result_block),
    },
    'after_joinedOnly': copy.deepcopy(result_block),
}


'''
COLLECT EVAL DATA (BEFORE)
'''

print('Collecting eval results for "before"...')
eval_before_data_dir = EVAL_DATA_ROOT_BASE+'before/'
for nn, file in enumerate(os.listdir(os.fsencode(eval_before_data_dir))):

    filename = os.fsdecode(file)
    with open(eval_before_data_dir+filename, 'rb') as f:
        eval_res = pickle.load(f)

    # Collect results: rmse, amsc, ssnr
    ignore_idcs = [0]
    eval_results['before']['amsc'].extend([a for idx, a in enumerate(eval_res['eval_res_before']['amsc']) if idx not in ignore_idcs])
    eval_results['before']['amsc_async'].extend([a for idx, a in enumerate(eval_res['eval_res_before']['amsc_async']) if idx not in ignore_idcs])
    eval_results['before']['rmse'].extend([a for idx, a in enumerate(eval_res['eval_res_before']['rmse']) if idx not in ignore_idcs])
    eval_results['before']['ssnr'].extend([a for idx, a in enumerate(eval_res['eval_res_before']['ssnr']) if idx not in ignore_idcs])
    eval_results['before']['ssnr_async'].extend([a for idx, a in enumerate(eval_res['eval_res_before']['ssnr_async']) if idx not in ignore_idcs])

    # Collect and sort results: Ts 
    sim_data_dir = SIM_DATA_ROOT_BASE+'join/' # could be any of the scenarios, as the required data is identical between them
    sim_filename = eval_res['filename']
    with open(sim_data_dir+sim_filename, 'rb') as f:
        sim_res = pickle.load(f)
    nodes_select_before = TopologyManager.get_unique_node_list(sim_res['nodes_levels_before'])
    node_level_positions_before = TopologyManager.get_node_level_positions(sim_res['nodes_levels_before'])
    for idx, tc in enumerate(eval_res['eval_res_before']['Tc']):
        if idx == 0: continue
        nid = nodes_select_before[idx]
        level = node_level_positions_before[nid]
        if not level in eval_results['before']['ts_perLevel']:
            eval_results['before']['ts_perLevel'][level] = []
        eval_results['before']['ts_perLevel'][level].append(tc)

'''
COLLECT EVAL DATA (AFTER)
'''

for scenario in ['join', 'leave', 'leave_root', 'unlink']:

    print('Collecting eval results for "after" - ', scenario, '...')

    eval_data_dir = EVAL_DATA_ROOT_BASE+'after/'+scenario+'/'
    directory = os.fsencode(eval_data_dir)

    for nn, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        with open(eval_data_dir+filename, 'rb') as f:
            eval_res = pickle.load(f)

        # Root node (always perfect) should be ignored
        ignore_idcs = [0]

        if scenario == 'join':
            # Add idx corresponding to newly joined node to ignore list
            sim_data_dir = SIM_DATA_ROOT_BASE+scenario+'/'
            sim_filename = eval_res['filename']
            with open(sim_data_dir+sim_filename, 'rb') as f:
                sim_res = pickle.load(f)
            node_ids_ever = sim_res['node_ids_ever']
            nodes_select_after = TopologyManager.get_unique_node_list(sim_res['nodes_levels_after'])
            idx_joined = nodes_select_after.index('node_'+str(sim_res['node_id_changed']))
            ignore_idcs.append(idx_joined)
            # Collect separate results for newly joined nodes
            eval_results['after_joinedOnly']['amsc'].append(eval_res['eval_res_after_joined']['amsc'][0])
            eval_results['after_joinedOnly']['rmse'].append(eval_res['eval_res_after_joined']['rmse'][0])
            eval_results['after_joinedOnly']['ssnr'].append(eval_res['eval_res_after_joined']['ssnr'][0])
            eval_results['after_joinedOnly']['ts'].append(eval_res['eval_res_after_joined']['Tc'][0])

        # Collect results
        eval_results['after'][scenario]['amsc'].extend([a for idx, a in enumerate(eval_res['eval_res_after']['amsc']) if idx not in ignore_idcs])
        eval_results['after'][scenario]['rmse'].extend([a for idx, a in enumerate(eval_res['eval_res_after']['rmse']) if idx not in ignore_idcs])
        eval_results['after'][scenario]['ssnr'].extend([a for idx, a in enumerate(eval_res['eval_res_after']['ssnr']) if idx not in ignore_idcs])




#%%
'''
PLOT: RMSE
'''

f, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 4, 1]})
ax[0].boxplot(eval_results['before']['rmse'], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[0].set_yscale('log')
ax[0].set_xticks((1,), ('before',))
ax[0].set_ylim((1e-3, 2))
ax[0].set_ylabel('SRO-RMSE [ppm]')

ax[1].boxplot([eval_results['after']['join']['rmse'],
               eval_results['after']['unlink']['rmse'],
               eval_results['after']['leave']['rmse'],
               eval_results['after']['leave_root']['rmse']], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[1].set_yscale('log')
ax[1].set_ylim((1e-3, 2))
ax[1].set_xticks((1, 2, 3, 4), ('(a)', '(b)', '(c)', '(d)'))
ax[1].set_yticks((), ())


ax[2].boxplot(eval_results['after_joinedOnly']['rmse'], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[2].set_yscale('log')
ax[2].set_ylim((1e-3, 2))
ax[2].set_xticks((1,), ('joined',))
ax[2].set_yticks((), ())

if PLOT_TARGET_DIR:
    plt.savefig(PLOT_TARGET_DIR+'rmse.jpg')
    tikzplotlib.save(PLOT_TARGET_DIR+'rmse.tex')


#%%

'''
PLOT: AMSC
'''

f, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 4, 1]})

ax[0].boxplot([eval_results['before']['amsc_async'],
               eval_results['before']['amsc']], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[0].set_xticks((1, 2), ('Async', 'Before'))
ax[0].set_ylim((0, 1.1))
ax[0].set_ylabel('AMSC')

ax[1].boxplot([eval_results['after']['join']['amsc'],
               eval_results['after']['unlink']['amsc'],
               eval_results['after']['leave']['amsc'],
               eval_results['after']['leave_root']['amsc']], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[1].set_xticks((1, 2, 3, 4), ('(a)', '(b)', '(c)', '(d)'))
ax[1].set_yticks((), ())
ax[1].set_ylim((0, 1.1))

ax[2].boxplot(eval_results['after_joinedOnly']['amsc'], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[2].set_xticks((1,), ('joined',))
ax[2].set_yticks((), ())
ax[2].set_ylim((0, 1.1))

if PLOT_TARGET_DIR:
    plt.savefig(PLOT_TARGET_DIR+'amsc.jpg')
    tikzplotlib.save(PLOT_TARGET_DIR+'amsc.tex')


# %%
'''
PLOT: SSNR
'''

f, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 4,1]})

ax[0].boxplot([eval_results['before']['ssnr_async'],
               eval_results['before']['ssnr']], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[0].set_xticks((1, 2), ('async', 'sync'))
ax[0].set_ylim((-5, 30))
ax[0].set_ylabel('SSNR [dB]')

ax[1].boxplot([eval_results['after']['join']['ssnr'],
               eval_results['after']['unlink']['ssnr'],
               eval_results['after']['leave']['ssnr'],
               eval_results['after']['leave_root']['ssnr']], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[1].set_xticks((1, 2, 3, 4), ('(a)', '(b)', '(c)', '(d)'))
ax[1].set_yticks((), ())
ax[1].set_ylim((-5, 30))


ax[2].boxplot(eval_results['after_joinedOnly']['ssnr'], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[2].set_xticks((1,), ('joined',))
ax[2].set_yticks((), ())
ax[2].set_ylim((-5, 30))

if PLOT_TARGET_DIR:
    plt.savefig(PLOT_TARGET_DIR+'ssnr.jpg')
    tikzplotlib.save(PLOT_TARGET_DIR+'ssnr.tex')

# %%
'''
PLOT: Settling times (Tc)
'''

f, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
ax[0].boxplot([eval_results['before']['ts_perLevel'][1],
               eval_results['before']['ts_perLevel'][2],
               eval_results['before']['ts_perLevel'][3],],
                notch=False,
                vert=True,
                widths=0.8)

ax[0].set_xticks((1, 2, 3), ('L1', 'L2', 'L3'))
ax[0].set_ylim((0, 150))
ax[0].set_ylabel('T_s [s]')

ax[1].boxplot(eval_results['after_joinedOnly']['ts'], 
                notch=False,
                vert=True,
                widths=0.8
)
ax[1].set_xticks((1,), ('joined',))
ax[1].set_ylim((0, 150))

if PLOT_TARGET_DIR:
    plt.savefig(PLOT_TARGET_DIR+'ts.jpg')
    tikzplotlib.save(PLOT_TARGET_DIR+'ts.tex')


# %%


plt.show()