'''
Functions for plotting topology, signals, diaries, etc.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits import mplot3d
from stl import mesh

def plot_scene_diary(scene_diary, max_len):

    figure = plt.figure(figsize=(12, 2))
    scenes = []
    for scene in sorted(scene_diary, key=lambda x: x['onset']):
        scene_id = scene['scene']
        onset = scene['onset']
        if onset > max_len:
            break
        if scene_id not in scenes:
            scenes.append(scene_id)

    rows = {scene: i for i, scene in enumerate(scenes)}
    num_rows = len(rows)
    for scene in sorted(scene_diary, key=lambda x: x['onset']):
        scene_id = scene['scene']
        onset = scene['onset']
        offset = np.minimum(scene['offset'], max_len)
        if onset > max_len:
            break
        ymin = (rows[scene_id]+.1)/num_rows
        ymax = (rows[scene_id]+.9)/num_rows
        plt.axvspan(onset, offset, ymin, ymax)

    plt.title('Fig.2 (b) Scene diary')
    plt.xlabel('Time [s]')
    plt.xlim(0, max_len)
    plt.ylabel('Scene')
    plt.yticks((np.arange(num_rows) + .5) / num_rows , scenes)
    plt.grid()

    #plt.savefig('scene_diary.svg')

def plot_positions_and_topology(example, room_model, max_len, nodes_levels=None, export_dir=""):

    room_mesh = mesh.Mesh.from_file(room_model)
    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot(111, projection='3d')#mplot3d.Axes3D(figure)
    ax.view_init(azim=-90, elev=90)

    poly = mplot3d.art3d.Poly3DCollection(room_mesh.vectors)
    poly.set_alpha(0.4)
    poly.set_facecolor('lightgray')
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])
    scale = room_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.add_collection3d(poly)

    node_coord = lambda nid: example['nodes'][str(nid)]['position']['coordinates']

   
    for node_id, params in example['nodes'].items():
        is_root = node_id == nodes_levels[0][0][0]
        line_width = 3 if is_root else 1.5
        pos = node_coord(node_id)#position_handler.get_node_pos(params['pos_id'])['coordinates']
        ax.scatter(pos[0], pos[1], 5, s=750, color='w', edgecolors='k', linewidths=line_width)
        ax.text(pos[0], pos[1], 5, node_id.split("_")[-1], fontsize=12, zorder=99, fontweight=800, horizontalalignment='center', verticalalignment='center')


    src_positions = []
    src_position_ids = []
    for src in sorted(example['src_diary'], key=lambda x: x['onset']):
        pos_id = src['pos_id']
        onset = src['onset']
        if onset > max_len:
            break
        if src['pos_id'] not in src_position_ids:
            src_position_ids.append(src['pos_id'])
            src_positions.append(src['position']['coordinates'])

    for idx, pos_id in enumerate(src_position_ids):
        pos = src_positions[idx]
        pos_shifted = [pos[0]+0.3, pos[1]-0.3, pos[2]] # for more visual clarity in plot
        ax.scatter(pos_shifted[0], pos_shifted[1], 15, s=100, color='b')
        ax.text(pos_shifted[0]-.35, pos_shifted[1]-.35, 15, pos_id, fontsize=12, fontweight=600)

    # Plot topolgy: Graph edges
    if nodes_levels is not None:
        colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
        zorder=100
        opacity = 1
        for lid, level in enumerate(nodes_levels):
            zorder -= 1
            for bid, branch in enumerate(level):
                col = colors[lid%len(colors)]
                root_pos = node_coord(branch[0])#position_handler.get_node_pos(example['nodes'][branch[0]]['pos_id'])['coordinates']
                for idx, node in enumerate(branch):
                    if idx == 0: continue
                    pos = node_coord(node)#position_handler.get_node_pos(example['nodes'][node]['pos_id'])['coordinates']
                    # arrow-head: (ax.arrow() is buggy)
                    alpha = np.pi/4 # arrow-dash angle
                    z = 0.15 # arrow-dash len
                    pos2d = np.array(pos[:2])
                    root_pos2d = np.array(root_pos[:2])
                    R_top = np.array([[np.cos(alpha), -np.sin(alpha)], 
                            [np.sin(alpha), np.cos(alpha)]])
                    R_bottom = np.array([[np.cos(-alpha), -np.sin(-alpha)], 
                            [np.sin(-alpha), np.cos(-alpha)]])
                    diff_vec_normed = (root_pos2d - pos2d)/np.linalg.norm(root_pos2d - pos2d)
                    xadd = np.array([1, 0]).dot(diff_vec_normed)*0.35
                    yadd = np.array([0, 1]).dot(diff_vec_normed)*0.35
                    root_pos2d = root_pos2d - np.array([xadd, yadd])
                    pos2d = pos2d + np.array([xadd, yadd])
                    dash1 = R_top.dot(diff_vec_normed*z)
                    dash2 = R_bottom.dot(diff_vec_normed*z)
                    plt.plot([root_pos2d[0], pos2d[0]], [root_pos2d[1], pos2d[1]], color=col, linewidth=3, zorder=zorder, alpha=opacity)
                    plt.plot([pos2d[0], pos2d[0]+dash1[0]], [pos2d[1], pos2d[1]+dash1[1]], color=col, linewidth=3, zorder=zorder, alpha=opacity) 
                    plt.plot([pos2d[0], pos2d[0]+dash2[0]], [pos2d[1], pos2d[1]+dash2[1]], color=col, linewidth=3, zorder=zorder, alpha=opacity)


    plt.tight_layout()

    if export_dir:
        plt.savefig(export_dir)

def plot_pos_hist(src_diary, max_len):
    
    figure = plt.figure(figsize=(12.5, 2))
    src_positions = []
    sources = []
    for src in sorted(src_diary, key=lambda x: x['onset']):
        pos_id = src['pos_id']
        onset = src['onset']
        if onset > max_len:
            break
        if 'speaker_id' in src.keys():
            src_id = f'speaker_{src["speaker_id"]}'
        else:
            src_id = 'loudspeaker'
        if src['pos_id'] not in src_positions:
            src_positions.append(src['pos_id'])
        if src_id not in sources:
            sources.append(src_id)

    colors = list(mcolors.TABLEAU_COLORS.values())
    #colors = {src: colors[i] for i, src in enumerate(sources)}
    rows = {src: i for i, src in enumerate(src_positions)}
    num_rows = len(src_positions)
    handles = []
    labels = []
    for src in sorted(src_diary, key=lambda x: x['onset']):
        pos_id = src['pos_id']
        onset = src['onset']
        offset = np.minimum(src['offset'], max_len)
        if onset > max_len:
            break
        if 'speaker_id' in src.keys():
            src_id = f'speaker_{src["speaker_id"]}'
        else:
            src_id = 'loudspeaker'
        ymin = (rows[pos_id]+.1)/num_rows
        ymax = (rows[pos_id]+.9)/num_rows
        handle = plt.axvspan(onset, offset, ymin, ymax, label='source', facecolor=colors[0])
        if src_id not in labels:
            handles.append(handle)
            labels.append(src_id)
    plt.xlim(0, max_len)
    plt.ylabel('Position ID')
    plt.yticks((np.arange(num_rows) + .5) / num_rows , src_positions)
    plt.grid()
    #plt.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(sources))
    #plt.legend(handles, labels,loc='upper center', ncol=len(sources))
