'''
Implementation of the class for managing topology of the sensor nodes.

Offers abstract methods for modifying given topology by adding or removing nodes or node-links, where topologies
are generated based on a minimum-spanning-tree (MST).

In addition, this class exposes various static methods as helper functions for all topology related tasks.
'''

import random
import numpy as np

class TopologyManager:

    def __init__(self, node_coords):

        N = len(node_coords.keys())
        self.node_coords = node_coords #dict, numeric ids
        self.adj_matr_eucl = self.get_eucl_graph(node_coords)
        self.node_link_blacklist = [ # simulation specific. avoid links through walls
            *[[8, i] for i in range(13) if i not in [7, 9]],
            *[[7, i] for i in range(13) if i not in [8, 9]],
            *[[10, i] for i in range(13) if i not in [9]],
            *[[11, i] for i in range(13) if i not in [9, 12]],
            *[[12, i] for i in range(13) if i not in [9, 11]],
        ]
        self.adj_matr_mask = self._node_links_blacklist_filter(np.ones((N, N)))        
        if not self.mst_exists(self.adj_matr_mask):
            raise Exception('No MST exists under given node-link masking! (node_link_blacklist)')
        self.nodes_levels = self._generate_topology()


    def set_node_links(self, links, on=True):
        '''
        Link or unlink nodes to allow or disallow connections in topology. Topology is rearranged afterwards
        Input:
            links (list): Link-pairs in form of lists. Reference by integer node ids, i.e. [nid1, nid8]
            on (bool): True to (re-)link, False to unlink
        '''
        node_ids = self.get_node_ids()
        val = 1 if on else 0
        for link in links:
            i = node_ids.index(link[0])
            j = node_ids.index(link[1])
            self.adj_matr_mask[i, j] = val
            self.adj_matr_mask[j, i] = val
        self.adj_matr_mask = self._node_links_blacklist_filter(self.adj_matr_mask)

        # verify
        if on == False and self.mst_exists(self.adj_matr_mask) == False:
            raise Exception('No MST exists under given node-link masking!')

        # rearrange
        self.nodes_levels = self._generate_topology()        
     
    def remove_nodes(self, node_ids):
        '''
        Remove nodes. Topology is re-arranged afterwards.
        Input:
            node_ids (list): int node ids of nodes to be removed
        '''
        node_ids_old = self.get_node_ids()
        for nid in node_ids:
            self.node_coords.pop(nid, None)
        self.adj_matr_eucl = self.get_eucl_graph(self.node_coords)
        node_ids_new = self.get_node_ids()

        # update link mask
        N = self.get_size()
        adj_matr_mask_old = self.adj_matr_mask
        idcs_new_old = [node_ids_old.index(nid) for nid in node_ids_new if nid in node_ids_old] #idcs of all new ids w.r.t. old id-list
        self.adj_matr_mask = np.ones((N, N))
        for idx_new, nid in enumerate(node_ids_new):
            if nid in node_ids_old:
                idx_old = node_ids_old.index(nid)
                self.adj_matr_mask[idx_new, :] = adj_matr_mask_old[idx_old, idcs_new_old]

        # verify
        if not self.mst_exists(self.adj_matr_mask):
            raise Exception('No MST exists under given node-link masking!')

        # rearrange topology
        self.nodes_levels = self._generate_topology()

    def add_nodes(self, new_node_coords):
        '''
        Add nodes. Topology is re-arranged afterwards.
        Input: 
            node_ids (dict): dict of new node coords. 
        '''
        # insert new coordinates and prepare base graph
        node_ids_old = self.get_node_ids()
        for nid in new_node_coords.keys():
            self.node_coords[nid] = new_node_coords[nid]
        node_ids_new = self.get_node_ids()
        #node_ids_base = [nid if nid in node_ids_old else None for nid in node_ids_new]
        #adj_matrix_base = self.graph_from_topology(self.nodes_levels, node_ids_base)

        # update euclidean graph
        self.adj_matr_eucl = self.get_eucl_graph(self.node_coords)

        # update graph mask
        N = self.get_size()
        node_idcs_old_new = [node_ids_new.index(nid) for nid in node_ids_old]
        adj_matr_mask_old = self.adj_matr_mask
        self.adj_matr_mask = np.ones((N, N))
        for idx in range(len(node_ids_old)):
            idx_new = node_ids_new.index(node_ids_old[idx])
            self.adj_matr_mask[idx_new, node_idcs_old_new] = adj_matr_mask_old[idx, :]
        # -- apply blacklist filter
        self.adj_matr_mask = self._node_links_blacklist_filter(self.adj_matr_mask)

        #rearrange topology
        #self.nodes_levels = self._generate_topology(ttype='ROT', adj_matr_mst_base=adj_matrix_base)
        self.nodes_levels = self._generate_topology()

    def get_size(self):
        '''
        Returns number of nodes currently in topology
        '''
        return len(self.node_coords.keys())

    def get_node_ids(self, form='int', ordered=False):
        '''
        Returns list of topologies node ids
        Input:
            form (str): In which form the node ids are returned, either 'int' or 'str' ("node_x")
            ordered (bool): If true, ids are ordered according to topology, i.e. root node at index 0 etc.
        '''
        node_ids_str = self.get_unique_node_list(self.nodes_levels) if ordered else None
        if form == 'int':
            if not ordered:
                return [node_id for node_id in self.node_coords.keys()]
            else:
                return [int(node_id.split('_')[1]) for node_id in node_ids_str]
        elif form == 'string':
            if not ordered:
                return ['node_'+str(node_id) for node_id in self.node_coords.keys()]
            else:
                return node_ids_str

    def _node_links_blacklist_filter(self, adj_matr_mask):
        '''
        Manipulates node link mask according to blacklisted node links
        '''
        node_ids = self.get_node_ids()
        for unlink in self.node_link_blacklist:
            nid1, nid2 = unlink[0], unlink[1]
            if nid1 in node_ids and nid2 in node_ids:
                i, j = node_ids.index(nid1), node_ids.index(nid2)
                adj_matr_mask[i, j] = adj_matr_mask[j, i] = 0   

        return adj_matr_mask

    def _generate_topology(self, ttype='ROT', adj_matr_mst_base=None):
        '''
        Generates topology by first finding the MST and then selecting a suitable
        global reference (root) node.
        '''
        # Adjust base matrix according to masked links
        if adj_matr_mst_base is not None:
            adj_matr_mst_base = adj_matr_mst_base*self.adj_matr_mask
        # find mst
        adj_matr_mst, start_node_best_idx = self.find_mst(np.copy(self.adj_matr_eucl), 
                                    find_best=True, 
                                    force_POT=(ttype=='POT'), 
                                    adj_matrix_mst_start=adj_matr_mst_base, 
                                    adj_matrix_mask=self.adj_matr_mask)
        # Select root node
        if ttype == 'POT':
            node_idx_root = start_node_best_idx #for now, just select first match...
        elif ttype == 'ROT' or ttype == 'SOT':
            try: # if possible, keep root unchanged
                node_id_root = self.nodes_levels[0][0][0]
                node_idx_root = self.get_node_ids('string').index(node_id_root)
            except:
                node_idx_root = np.argmin(np.mean(self.adj_matr_eucl, axis=1))
                if isinstance(node_idx_root, list):
                    node_idx_root = node_idx_root[0]
        else:
            raise ValueError('Unsupported topology type. Please use ROT, POT or SOT')     
        # translate to topology description
        nodes_levels = self.topology_from_graph(adj_matr_mst, node_idx_root, self.get_node_ids())
        return nodes_levels

    @staticmethod
    def mst_exists(adj_matr_mask):
        '''
        Checks if there exists at least one mininum spanning tree (MST) given the graph 
        provided via the binary, symmetric adjacency matrix (That is, if there are no disjointed subgraphs)
        '''
        N = np.shape(adj_matr_mask)[0]
        reached = [0]
        i = 0
        while i < len(reached) and i <= N:
            row_idx = reached[i]
            idcs = [x[0] for x in np.argwhere(adj_matr_mask[row_idx,:])]
            if len(idcs) == 0: break # fully isolated node
            for idx in idcs:
                if idx not in reached:
                    reached.append(idx)
            i += 1
        return len(reached) == N

    @staticmethod
    def get_eucl_graph(node_coords):
        '''
        Generetes fully connected graph with eucl. distance as weights in form of weighted adjacency matrix.
        Input: 
            node_coords (dict): dict of coordinates {int_node_id: [x, y, z], int_node_id2: [...], ...}                          
        Output:
            adj_matrix (N,N): weighted, symmetric adjacency matrix
        '''
        N = len(node_coords)
        adj_matrix = np.zeros((N, N))
        for i, nid1 in enumerate(node_coords.keys()):
            for j, nid2 in enumerate(node_coords.keys()):
                if i <= j: continue
                pos_diff = np.array(node_coords[nid1]) - np.array(node_coords[nid2])
                dist = np.sqrt(sum([x**2 for x in pos_diff]))
                adj_matrix[i, j] = dist
                adj_matrix[j, i] = dist
        
        return adj_matrix

    @staticmethod
    def topology_from_graph(adj_matrix, node_idx_root, node_ids):
        '''
        Translates undirected graph to topology-description
        Input:
            adj_matrix (N,N): symmetric, binary adjacency matrix
            root_node_idx (int): index of designated root node
            node_ids (list): list of int node ids for idx->id mapping
        Out:
            node_levels (list): Topology description based on string node ids
        '''
        level0 = [[node_idx_root] + np.nonzero(adj_matrix[node_idx_root,:])[0].tolist()] #contains only one branch
        node_levels = [level0]
        count = 0
        while True:
            level = node_levels[-1]
            level_next = []
            for branch in level:
                for i, nidx in enumerate(branch):
                    if i == 0: continue
                    leaves = np.nonzero(adj_matrix[nidx,:])[0]
                    leaves = leaves[leaves != branch[0]].tolist()#ignore own root
                    if len(leaves) > 0: #valid branch for new level
                        level_next.append([nidx] + leaves)
            if len(level_next) > 0:
                node_levels.append(level_next)
            else:
                break
            count += 1
            if count == 1000:
                raise Exception('Got stuck translating from graph to topology.')
        # translate node_level entries to string representation using original node ids
        for lid, level in enumerate(node_levels):
            for bid, branch in enumerate(level):
                for nid, node_idx in enumerate(branch):
                    node_levels[lid][bid][nid] = 'node_'+str(node_ids[node_idx])

        return node_levels

    @staticmethod
    def graph_from_topology(node_levels, node_ids):
        '''
        Translates topology description to undirected graph
        Input:
            node_levels(list): Topology description based on string node ids
            node_ids(list): list of string node ids for idx<->id mapping. Specifically, it serves as a map
                            for row/col-index of the adj-matrix to node-id.
                            List may contan None entries, resulting in zero rows/cols in the adj-matrix
                            at those the corresponding indices.
        Out:
            adj_matrix (N,N): Undirected graph; symmetric, binary adjacency matrix
        '''
        N = len(node_ids)
        adj_matrix = np.zeros((N, N))
        for level in node_levels:
            for branch in level:
                node_id_root = int(branch[0].split('_')[1])
                node_idx_root = node_ids.index(node_id_root)
                for nid, node_id_str in enumerate(branch):
                    if nid == 0: continue
                    node_id = int(node_id_str.split('_')[1])
                    node_idx = node_ids.index(node_id)
                    adj_matrix[node_idx_root, node_idx] = 1
                    adj_matrix[node_idx, node_idx_root] = 1

        return adj_matrix

    @staticmethod
    def verify_topology(nodes_levels, nodes_whitelist):
        '''
        Verifies topology description (No loops, no self-reference, etc.). Does not consider masked node links. 
        If a custom topology is set by assigning the nodes_levels attribute directly, previously masked node links can
        be used!
        Input:
            nodes_levels (list): Topology description
            nodes_whitelist (list): List of allowed node_id strings. 
        Output:
            is_valid (bool): True if nodes_levels is a valid topology description, False otherwise
        '''
        
        if not isinstance(nodes_levels, list) or len(nodes_levels) == 0:
            print('Error: nodes_levels must be non-empty list!')
            return False

        prev_level_leaf_nodes = []
        leaf_nodes_encountered = [] # filled up on the go
        for lid, level in enumerate(nodes_levels):
            if not isinstance(level, list) or len(level) == 0:
                print('Error: Levels must be non-empty lists!')
                return False
            if lid == 0 and len(level) > 1:
                print('Error: First level must only contain one branch (tree-root)')
                return False
            level_all_nodes = []
            level_leaf_nodes = []
            for _, branch in enumerate(level):
                if not isinstance(branch, list) or len(branch) < 2:
                    print('Error: Branches must be lists with at least two entries.')
                    return False
                for nid, node_id in enumerate(branch):
                    if not isinstance(node_id, str):
                        print('Error: Nodes must only be referenced with strings.')
                        return False
                    if not node_id in nodes_whitelist:
                        print('Error: Referenced non-existing node. Available nodes:', nodes_whitelist)
                        return False
                    level_all_nodes.append(node_id)
                    if nid > 0 and node_id in leaf_nodes_encountered:
                        print('Error: Nodes can only be referenced as leafs once.')
                        return False              
                    if nid == 0 and lid > 0:
                        if not node_id in prev_level_leaf_nodes:
                            print('Error: Branch root-nodes must appear as leafs in the preceding level.')
                            print(node_id, prev_level_leaf_nodes)
                            return False
                    else:
                        level_leaf_nodes.append(node_id)
            leaf_nodes_encountered.extend(level_leaf_nodes)
            prev_level_leaf_nodes = level_leaf_nodes
            # nodes must only be referenced once within level
            if not TopologyManager.is_unique_list(level_all_nodes):
                print('Error: Nodes can only be referenced once within a level.')
                return False

        return True

    @staticmethod
    def find_mst(adj_matrix, force_POT=False, find_best=True, adj_matrix_mst_start=None, adj_matrix_mask=None):
        '''
        WARNING: Only find_best=True guarantees that no masked node-links are used!

        Finds Minimum Spanning Tree (MST) via Prim's Algorithm given fully-connected graph with
        weighted edges
        Input:
            adj_matrix (N,N): weighted adjacency-matrix of fully connected, undirected graph
            force_POT (bool): If True, MST will be restricted to one continueous path (for path-out-topology)
            find_best (bool): If True, algorithm will be executed N times for each possible start-node to find 
                            the optimal MST. 
                            If False, algorithm is executed once with a random start-node
            adj_matrix_mst_start (N,N): binary adjacency-matrix of undirected graph that serves as 
                            a basis to start the algorithm from
            adj_matrix_mask (N,N): Binary mask for adjacency matrix of fully connected graph. 
                            Can be used to indicate missing node connections (=0)
        Out:
            adj_matrix_mst (N,N): binary adjacency-matrix of undirected MST
            start_node_best_idx (int): idx of start node from which the mst was found
        '''

        compl = lambda A, N: [n for n in range(N) if n not in A]
        N = adj_matrix.shape[0]

        # apply graph masking (set arbitrarily large distance on undesired connections to de-prioritize)
        if adj_matrix_mask is not None:
            adj_matrix[adj_matrix_mask == 0] = np.amax(adj_matrix)*N 

        start_node_idcs = list(range(N)) if find_best else [random.choice(list(range(N)))]
        start_node_idcs = [2] #TODO remove

        adj_matrix_mst_best = np.zeros((N, N))
        start_node_best_idx = -1
        dist_total_prev = 0

        for n, start_node_idx in enumerate(start_node_idcs):
            # Init graph
            adj_matrix_mst = np.zeros((N, N))
            nodes_in_mst = [start_node_idx]
            if adj_matrix_mst_start is not None:
                start_node_idx = None
                adj_matrix_mst = adj_matrix_mst_start
                nodes_in_mst = [x[0] for x in np.argwhere(np.sum(adj_matrix_mst, axis=1) > 0)]
            # Find MST...
            for _ in range(N-1):
                if len(compl(nodes_in_mst,N)) == 0:
                    break # may end early if start graph is provided
                if force_POT:
                    min_dist = np.amin(adj_matrix[nodes_in_mst[-1],:][compl(nodes_in_mst,N)])
                else:
                    min_dist = np.min(adj_matrix[nodes_in_mst,:][:,compl(nodes_in_mst,N)])
                if isinstance(min_dist, list): # should practically never happen...
                    min_dist = min_dist[0]
                idcs_all = np.argwhere(adj_matrix == min_dist) # both mirrored entries will be found
                # In rare cases there may exist two independent but identical node-distances, one of which must be ignored!
                found_idcs = False
                for idcs_ in idcs_all:
                    if force_POT:
                        if idcs_[0] == nodes_in_mst[-1] and idcs_[1] not in nodes_in_mst:
                            idcs = idcs_
                            found_idcs = True
                    else:
                        if idcs_[0] in nodes_in_mst and idcs_[1] not in nodes_in_mst:
                            idcs = idcs_
                            found_idcs = True
                if not found_idcs:
                    raise Exception('Could not locate valid coordinates for minimum distance.')
                # apply changes
                adj_matrix_mst[idcs[0], idcs[1]] = 1
                adj_matrix_mst[idcs[1], idcs[0]] = 1
                nodes_in_mst.append(idcs[1])
            if len(nodes_in_mst) < N:
                raise Exception('Not all nodes were added to MST!')
            # Evaluate MST
            dist_total = np.sum(adj_matrix * adj_matrix_mst)/2
            if n == 0 or dist_total < dist_total_prev:
                adj_matrix_mst_best = adj_matrix_mst
                start_node_best_idx = start_node_idx
                dist_total_prev = dist_total
        return adj_matrix_mst_best, start_node_best_idx

    @staticmethod
    def get_unique_node_list(nodes_levels):
        '''
        Generates ordered list of unique nodes in topology
        Input: 
            nodes_levels (list): Topology description
        Output:
            nodes (list): List of unique nodes
        '''
        nodes = []
        for level in nodes_levels:
            for branch in level:
                for node in branch:
                    if node not in nodes:
                        nodes.append(node)
        return nodes

    @staticmethod
    def get_nodes_roots(nodes_levels):
        '''
        Returns dict that lists every nodes local root (reference node)
        '''
        nodes_roots = {}
        for level in nodes_levels:
            for branch in level:
                node_id_root = branch[0]
                for nid, node_id in enumerate(branch):
                    if nid == 0: continue
                    nodes_roots[node_id] = node_id_root    
        return nodes_roots   

    @staticmethod
    def get_node_level_positions(nodes_levels):
        '''
        Returns dict that lists every nodes level-position. 
        Useful to predict the relative offset between synchronized signals.
        {'node_id': int(level), ...}
        '''
        root_node = nodes_levels[0][0][0]
        node_level_positions = {root_node: 0}
        for lid, level in enumerate(nodes_levels):
            for branch in level:
                for node in branch:
                    if node not in node_level_positions.keys():
                        node_level_positions[node] = lid+1
        return node_level_positions

    @staticmethod
    def is_unique_list(l):   
        for elem in l:
            if l.count(elem) > 1:
                return False
        return True
