import os

import numpy as np
import torch
from torch import Tensor

from functools import partial

from jaxtyping import Float
from typing import Dict


DEFAULT_GRAPH_PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ims")

class EAPGraph:

    def __init__(self, cfg, upstream_nodes=None, downstream_nodes=None, edges=None):
        self.cfg = cfg

        self.valid_upstream_node_types = ["resid_pre", "mlp", "head"]
        self.valid_downstream_node_types = ["resid_post", "mlp", "head"]

        self.valid_upstream_hook_types = ["hook_resid_pre", "hook_result", "hook_mlp_out"]
        self.valid_downstream_hook_types = ["hook_q", "hook_k", "hook_v", "hook_pre", "hook_resid_post"]

        # TODO valid_upstream_hook_types and upstream_component_ordering can be merged into one data structure
        self.upstream_component_ordering = {
            "hook_resid_pre": 0,
            "hook_result": 1,
            "hook_mlp_out": 2,
        }
        self.downstream_component_ordering = {
            "hook_q": 0,
            "hook_k": 1,
            "hook_v": 2,
            "hook_pre": 3,
            "hook_resid_post": 4
        }

        self.element_size = torch.empty((0), device=self.cfg.device, dtype=self.cfg.dtype).element_size()

        self.upstream_nodes = []
        self.downstream_nodes = []

        self.upstream_node_index: Dict[str, int] = {}
        self.downstream_node_index: Dict[str, int] = {}

        self.upstream_hook_slice: Dict[str, slice] = {}
        self.downstream_hook_slice: Dict[str, slice] = {}

        self.upstream_nodes_before_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_attn_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_mlp_layer: Dict[int, slice] = {}

        # If a list of edges is passed we only take the nodes that are connected by these edges
        if edges is not None:
            upstream_nodes = [edge[0] for edge in edges]
            downstream_nodes = [edge[1] for edge in edges]

        self.setup_graph_from_nodes(upstream_nodes, downstream_nodes)

        # We will create these tensors when needed
        self.eap_scores: Float[Tensor, "n_upstream_nodes n_downstream_nodes"] = None
        self.adj_matrix: Float[Tensor, "n_upstream_nodes n_downstream_nodes"] = None
        
        self.last_eap_score: Float[Tensor, "n_upstream_nodes n_downstream_nodes"] = None
        self.last_u_score: Float[Tensor, "n_upstream_nodes n_downstream_nodes"] = None

    def setup_graph_from_nodes(self, upstream_nodes=None, downstream_nodes=None):
        # if no nodes are specified, we assume that all of them will be used
        if upstream_nodes is None:
            upstream_nodes = self.valid_upstream_node_types.copy()
        
        if downstream_nodes is None:
            downstream_nodes = self.valid_downstream_node_types.copy()

        # we can assume that the two lists of hooks are sorted by layer number
        self.upstream_hooks, self.downstream_hooks = self.get_hooks_from_nodes(upstream_nodes, downstream_nodes)

        upstream_node_index = 0

        for hook_name in self.upstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]
            
            # we store the slice of all upstream nodes previous to this layer
            if layer not in self.upstream_nodes_before_layer:
                # we must check previous layers too because we might have skipped some
                for earlier_layer in range(0, layer + 1):
                    if earlier_layer not in self.upstream_nodes_before_layer:
                        self.upstream_nodes_before_layer[earlier_layer] = slice(0, upstream_node_index)
                        self.upstream_nodes_before_attn_layer[layer] = slice(0, upstream_node_index)
                        self.upstream_nodes_before_mlp_layer[layer] = slice(0, upstream_node_index)

            if hook_type == "hook_resid_pre":
                self.upstream_nodes.append(f"resid_pre.{layer}")
                self.upstream_node_index[f"resid_pre.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(upstream_node_index, upstream_node_index + 1)
                self.upstream_nodes_before_attn_layer[layer] = slice(0, upstream_node_index + 1)
                upstream_node_index += 1

            elif hook_type == "hook_result":
                for head_idx in range(self.cfg.n_heads):
                    self.upstream_nodes.append(f"head.{layer}.{head_idx}")
                    self.upstream_node_index[f"head.{layer}.{head_idx}"] = upstream_node_index + head_idx
                self.upstream_hook_slice[hook_name] = slice(upstream_node_index, upstream_node_index + self.cfg.n_heads)
                self.upstream_nodes_before_mlp_layer[layer] = slice(0, upstream_node_index + self.cfg.n_heads)
                upstream_node_index += self.cfg.n_heads 

            elif hook_type == "hook_mlp_out":
                self.upstream_nodes.append(f"mlp.{layer}")
                self.upstream_node_index[f"mlp.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(upstream_node_index, upstream_node_index + 1)
                upstream_node_index += 1

            else:
                assert False, "Invalid upstream hook type"

        # if there are no more upstream nodes after a certain layer we still have
        # to save that into the slice dictionaries
        for layer in range(0, self.cfg.n_layers):
            if layer not in self.upstream_nodes_before_layer:
                self.upstream_nodes_before_layer[layer] = slice(0, upstream_node_index)
                self.upstream_nodes_before_attn_layer[layer] = slice(0, upstream_node_index)
                self.upstream_nodes_before_mlp_layer[layer] = slice(0, upstream_node_index)

        downstream_node_index = 0

        for hook_name in self.downstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]

            if hook_type == "hook_q" or hook_type == "hook_k" or hook_type == "hook_v":
                letter = hook_type.split("_")[1].lower()
                for head_idx in range(self.cfg.n_heads):
                    self.downstream_nodes.append(f"head.{layer}.{head_idx}.{letter}")
                    self.downstream_node_index[f"head.{layer}.{head_idx}.{letter}"] = downstream_node_index + head_idx
                self.downstream_hook_slice[hook_name] = slice(downstream_node_index, downstream_node_index + self.cfg.n_heads)
                downstream_node_index += self.cfg.n_heads 

            elif hook_type == "hook_pre":
                self.downstream_nodes.append(f"mlp.{layer}")
                self.downstream_node_index[f"mlp.{layer}"] = downstream_node_index
                self.downstream_hook_slice[hook_name] = slice(downstream_node_index, downstream_node_index + 1)
                downstream_node_index += 1

            elif hook_type == "hook_resid_post":
                self.downstream_nodes.append(f"resid_post.{layer}")
                self.downstream_node_index[f"resid_post.{layer}"] = downstream_node_index
                self.downstream_hook_slice[hook_name] = slice(downstream_node_index, downstream_node_index + 1)
                downstream_node_index += 1

            else:
                assert False, "Invalid downstream hook type"

        self.n_upstream_nodes = len(self.upstream_nodes)
        self.n_downstream_nodes = len(self.downstream_nodes)
        self.last_eap_score = torch.zeros(
            (self.n_upstream_nodes, self.n_downstream_nodes),
            device=self.cfg.device
        )
        self.last_u_score = torch.zeros(
            (self.n_upstream_nodes, self.n_downstream_nodes),
            device=self.cfg.device
        )

        activations_tensor_in_gb = self.n_upstream_nodes * self.cfg.d_model * self.element_size / 2**30 
        print(f"Saving activations requires {activations_tensor_in_gb:.4f} GB of memory per token")

    # Given a set of upstream nodes and downstream nodes, this function returns the corresponding hooks
    # to access the activations of these nodes. We return the list of hooks sorted by layer number.
    def get_hooks_from_nodes(self, upstream_nodes, downstream_nodes):

        # we first check that the types of the nodes passed are valid
        for node in upstream_nodes:
            node_type = node.split(".")[0] # 'resid_pre', 'mlp' or 'head'
            assert node_type in self.valid_upstream_node_types, "Invalid upstream node"

        for node in downstream_nodes:
            node_type = node.split(".")[0] # 'resid_post', 'mlp' or 'head'
            assert node_type in self.valid_downstream_node_types, "Invalid downstream node"

        upstream_hooks = []
        downstream_hooks = []

        for node in upstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            node_type = node.split(".")[0] # 'resid_pre', 'mlp' or 'head'
            assert node_type in self.valid_upstream_node_types, "Invalid upstream node"
            if not node_is_layer_specific:
                # we are in the case of a global node that applies to all layers
                hook_type = "hook_resid_pre" if node_type == "resid_pre" else "hook_mlp_out" if node_type == "mlp" else "attn.hook_result"
                for layer in range(self.cfg.n_layers):
                    upstream_hooks.append(f"blocks.{layer}.{hook_type}")
            else:
                assert node.split(".")[1].isdigit(), "Layer number must be an integer"
                layer = int(node.split(".")[1])
                hook_type = "hook_resid_pre" if node_type == "resid_pre" else "hook_mlp_out" if node_type == "mlp" else "attn.hook_result"
                upstream_hooks.append(f"blocks.{layer}.{hook_type}")

        for node in downstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            if not node_is_layer_specific:
                # we are in the case of a global node that applies to all layers
                if node == "head":
                    for layer in range(self.cfg.n_layers):
                        for letter in "qkv":
                            downstream_hooks.append(f"blocks.{layer}.attn.hook_{letter}")
                elif node == "resid_post" or node == "mlp":
                    hook_type = "hook_resid_post" if node == "resid_post" else "hook_pre"
                    for layer in range(self.cfg.n_layers):
                        downstream_hooks.append(f"blocks.{layer}.mlp.{hook_type}")
                else:
                    raise NotImplementedError("Invalid downstream node")
            else:
                # we are in the case of a node specified for a single layer
                assert node.split(".")[1].isdigit(), "Layer number must be an integer"
                layer = int(node.split(".")[1])

                if node.startswith("resid_post") or node.startswith("mlp"):
                    hook_type = "hook_resid_post" if node.startswith("resid_post") else "hook_pre"
                    downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                elif node.startswith("head"):
                    all_heads = len(node.split(".")) <= 2 # head.10 means taking all heads at layer 10
                    head_idx = None if all_heads else int(node.split(".")[2]) # we don't use this variable because we have to add the same hook whether we want one head or all
                    letters = ["q", "k", "v"]
                    if len(node.split(".")) == 4:
                        # a specific input channel is specified so we modify the hook name accordingly
                        letter_specified = node.split(".")[3]
                        assert letter_specified in letters, "Invalid letter specified"
                        letters = [letter_specified]
                    for letter in letters:
                        downstream_hooks.append(f"blocks.{layer}.hook_{letter}")
                else:
                    raise NotImplementedError("Invalid downstream node")

        upstream_hooks = list(set(upstream_hooks))
        downstream_hooks = list(set(downstream_hooks))

        def get_hook_level(hook, component_ordering):
            # Function for differentiating the order of computation in between layers, e.g. attn_layer2 is before mlp_layer2
            num_components_per_layer = len(component_ordering)
            layer = int(hook.split(".")[1])
            hook_type = hook.split(".")[-1]
            component_order = component_ordering[hook_type]
            level = layer * num_components_per_layer + component_order
            return level

        get_upstream_hook_level = partial(get_hook_level, component_ordering=self.upstream_component_ordering)
        get_downstream_hook_level = partial(get_hook_level, component_ordering=self.downstream_component_ordering)

        # we sort the hooks by the order in which they appear in the computation
        upstream_hooks = sorted(upstream_hooks, key=get_upstream_hook_level)
        downstream_hooks = sorted(downstream_hooks, key=get_downstream_hook_level)

        return upstream_hooks, downstream_hooks
    
    def get_slice_previous_upstream_nodes(self, downstream_hook):
        layer = downstream_hook.layer()
        hook_type = downstream_hook.name.split(".")[-1]
        # if hook_type == "hook_resid_post":
        #     return self.upstream_nodes_before_layer[layer + 1]
        if hook_type == "hook_pre":
            return self.upstream_nodes_before_mlp_layer[layer]
        elif hook_type in ["hook_q", "hook_k", "hook_v", "hook_resid_post"]:
            return self.upstream_nodes_before_layer[layer]

    def get_hook_slice(self, hook_name):
        if hook_name in self.upstream_hook_slice:
            return self.upstream_hook_slice[hook_name]
        elif hook_name in self.downstream_hook_slice:
            return self.downstream_hook_slice[hook_name]

    def reset_scores(self):
        self.eap_scores = torch.zeros(
            (self.n_upstream_nodes, self.n_downstream_nodes),
            device=self.cfg.device
        )
        
    def get_nodes_num(self):
        return len(self.upstream_nodes) + len(self.downstream_nodes)

    def top_edges(
        self,
        n=1000,
        threshold=None,
        abs_scores=True,
        cross_layer=True,
        prune_method="top_edges"
    ):
        assert self.eap_scores is not None, "EAP scores have not been computed yet"
        
        # cross-layer or in-layer
        if not cross_layer:  # only reserve the scores between nodes in the same layer
            for up_stream_node, up_stream_idx in self.upstream_node_index.items():
                for down_stream_node, down_stream_idx in self.downstream_node_index.items():
                    if up_stream_node.split(".")[1] != down_stream_node.split(".")[1]:
                        self.eap_scores[up_stream_idx, down_stream_idx] = 0.0
    
        if prune_method == "top_edges":
            # get indices of maximum values in 2d tensor
            if abs_scores:
                top_scores, top_indices = torch.topk(self.eap_scores.flatten().abs(), k=n, dim=0)
            else:
                top_scores, top_indices = torch.topk(self.eap_scores.flatten(), k=n, dim=0)

            top_edges = []
            for i, (abs_score, index) in enumerate(zip(top_scores, top_indices)):
                if threshold is not None and abs_score < threshold:
                    break
                upstream_node_idx, downstream_node_idx = np.unravel_index(index, self.eap_scores.shape)
                score = self.eap_scores[upstream_node_idx, downstream_node_idx]
                # print(index, self.eap_scores.shape, upstream_node_idx, downstream_node_idx, score)

                top_edges.append((self.upstream_nodes[upstream_node_idx], self.downstream_nodes[downstream_node_idx], score.item()))
        else:
            upstream_nodes_scores = {}
            downstream_node_scores = {}
            for up_stream_node, up_stream_idx in self.upstream_node_index.items():
                upstream_nodes_scores[up_stream_node] = self.eap_scores[up_stream_idx, :].abs().sum()
            for down_stream_node, down_stream_idx in self.downstream_node_index.items():
                downstream_node_scores[down_stream_node] = self.eap_scores[:, down_stream_idx].abs().sum()
            node_scores = {**upstream_nodes_scores, **downstream_node_scores}
            top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
            # for each node, get the edge with the highest score that connects to it
            top_edges = []
            while len(top_edges) < n:
                for node, _ in top_nodes:
                    if node in upstream_nodes_scores:
                        node_idx = self.upstream_node_index[node]
                        _, downstream_node_idx = torch.max(self.eap_scores[node_idx, :].abs(), dim=0)
                        score = self.eap_scores[node_idx, downstream_node_idx]
                        top_edges.append((node, self.downstream_nodes[downstream_node_idx], score.item()))
                        self.eap_scores[node_idx, downstream_node_idx] = 0.0
                    else:
                        node_idx = self.downstream_node_index[node]
                        _, upstream_node_idx = torch.max(self.eap_scores[:, node_idx].abs(), dim=0)
                        # print("upstream_node_idx:", upstream_node_idx, "node_idx:", node_idx)
                        score = self.eap_scores[upstream_node_idx, node_idx]
                        top_edges.append((self.upstream_nodes[upstream_node_idx], node, score.item()))
                        self.eap_scores[upstream_node_idx, node_idx] = 0.0
                    if len(top_edges) >= n:
                        break

        return top_edges

    def subgraph_top_edges(
        self,
        threshold=None,
        abs_scores=True
    ):

        assert self.eap_scores is not None, "EAP scores have not been computed yet"

        top_edges = self.top_edges(threshold=threshold, abs_scores=abs_scores)

        upstream_nodes = [edge[0] for edge in top_edges]
        downstream_nodes = [edge[1] for edge in top_edges]
        subgraph = EAPGraph(upstream_nodes, downstream_nodes)

        return subgraph

    def show(
        self,
        edges=None,
        key_nodes=[],
        key_edges=[],
        threshold=None,
        abs_scores=True,
        fname: str="eap_graph.png",
        fdir=None,
    ):
        import pygraphviz as pgv

        minimum_penwidth = 0.2
        if not edges:
            edges = self.top_edges(threshold=threshold, abs_scores=abs_scores)

        g = pgv.AGraph(
            name='root',
            strict=True,
            directed=True
        )

        g.graph_attr.update(ranksep='0.1', nodesep='0.1', compound=True)
        g.node_attr.update(fixedsize='true', width='1.5', height='.5')

        def find_layer_node(node):
            if node == f'resid_post.{self.cfg.n_layers - 1}':
                return self.cfg.n_layers
            else:
                return int(node.split(".")[1])

        layer_to_subgraph = {}
        layer_to_subgraph[-1] = g.add_subgraph(name=f'cluster_-1', rank='same', color='invis')
        layer_to_subgraph[-1].add_node(f'-1_invis', style='invis')

        min_layer = 999
        max_layer = -1
        layers = list(range(0, 32))

        for edge in edges:
            parent_node = edge[0]
            child_node = edge[1]
            min_layer = min(min_layer, find_layer_node(parent_node))
            max_layer = max(max_layer, find_layer_node(child_node))

        layers = list(range(min_layer, max_layer + 1))
        prev_layer = None

        for layer in layers:
            layer_to_subgraph[layer] = g.add_subgraph(name=f'cluster_{layer}', rank='same', color='invis')
            layer_to_subgraph[layer].add_node(f'{layer}_invis', style='invis')

            if prev_layer is not None:
                g.add_edge(f'{prev_layer}_invis', f'{layer}_invis', style='invis', weight=1000)

            prev_layer = layer
                
        # normalize edge score to be between 0 and 1
        max_score = max([abs(edge[2]) for edge in edges])
        edges = [(edge[0], edge[1], edge[2] / max_score) for edge in edges]
        
        # Adding nodes and edges between nodes
        for edge in edges:
            parent_node, child_node, edge_score = edge

            parent_name = parent_node
            child_name = child_node

            child_name = child_name.replace(".q", "").replace(".k", "").replace(".v", "")
            
            for node_name in [parent_name, child_name]:

                node_layer = find_layer_node(node_name)

                # yellow green
                # node_color = '#ffd700' if node_name.startswith("head") else '#32cd32' if node_name.startswith("mlp") else '#2ca02c' if node_name.startswith("resid") else '#d62728'
                # Marsh
                # node_color = '#a1ead6' if node_name.startswith("head") else '#88ddf1' if node_name.startswith("mlp") else '#2ca02c' if node_name.startswith("resid") else '#d62728'
                # coconut
                node_color = '#f5daab' if node_name.startswith("head") else '#8ad088' if node_name.startswith("mlp") else '#2ca02c' if node_name.startswith("resid") else '#d62728'
                if key_nodes and node_name not in key_nodes:
                    node_color = '#ffffff'

                layer_to_subgraph[node_layer].add_node(
                    node_name,
                    fillcolor=node_color,
                    color="black",
                    style="filled, rounded",
                    shape="box",
                    fontname="Helvetica",
                )
                
            # edge_width = max(minimum_penwidth, edge_score*4000)
            edge_width = edge_score * 4

            edge_color = '#0091E4'
            if key_edges and (parent_name, child_name) not in key_edges:
                edge_color = '#d3d3d3'
                
            g.add_edge(
                parent_name,
                child_name,
                penwidth=edge_width,
                color=edge_color,
                weight=10,
                minlen='0.5',
            )

        print(f"Saving graph")
        if fdir is not None:
            save_path = os.path.join(fdir, fname)
            if not os.path.exists(fdir):
                os.makedirs(fdir, exist_ok=True)
        else:
            save_path = os.path.join(DEFAULT_GRAPH_PLOT_DIR, fname)
            if not os.path.exists(DEFAULT_GRAPH_PLOT_DIR):
                os.makedirs(DEFAULT_GRAPH_PLOT_DIR, exist_ok=True)
        
        if not fname.endswith(".gv"): # turn the .gv file into a .png file
            g.draw(path=save_path, prog='dot')

        return g, edges


if __name__ == "__main__":
    
    import jsonlines
    import numpy as np
    from transformer_lens import HookedTransformer
    
    
    """
    After circuit-tuning, we draw the graph of every log step.
    """
    def draw_graph_for_every_log_step():
        
        model = HookedTransformer.from_pretrained(
            'gpt2-small',
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device="cuda"
        )
        
        model.set_use_hook_mlp_in(True)
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        
        upstream_nodes=["mlp", "head"]
        downstream_nodes=["mlp", "head"]
        graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
        
        log_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/checkpoints-new/gpt2-small-sv-epochs_3-bsz_16-lr_1e-3-Opt_SGD-warm_up_100-top_n_1000-threshold_0-metric-logit_diff-random_ratio_0-randn_0-ablation_mean-loss_weighted_p_1-prune_method_top_nodes/log.jsonl"
        topn=1000
        select = 35
        graph_dir = f"/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eap/ims/eap_graph_svd_topn_{topn}_select_{select}"
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir, exist_ok=True)
        
        steps = []
        edges_list = []
        with jsonlines.open(log_path, "r") as f:
            for line in f:
                if "edges" in line:
                    step = line["step"]
                    while step in steps:
                        step += 1500
                    steps.append(step)
                    edges = line["edges"]
                    edges_list.append({"edges": edges, "name": f"eap_graph_sv_to_svd_topn_{topn}_{step}.pdf"})
        # if not edges_list:
        #     with jsonlines.open(log_path, "r") as f:
        #         for line in f:
        #             if "edge_info" in line:
        #                 step=4400
        #                 edges = line["edge_info"]
        #                 edges_list.append({"edges": edges, "name": f"eap_graph_svd_topn_{topn}_{step}.pdf"})
        
        # sort and select for each edges item
        for item in edges_list:
            item["edges"] = sorted(item["edges"], key=lambda x: x[-1], reverse=True)[:select]
        
        # get log scores
        for item in edges_list:
            for edge in item["edges"]:
                edge[-1] = np.log(edge[-1] + 1)
        
        # # min-max normalization  
        # scores = [edge[-1] for item in edges_list for edge in item["edges"]]
        # max_score = max(scores)
        # min_score = min(scores)
        # avg_score = sum(scores) / len(scores)
        # print("max_score:", max_score, "min_score:", min_score, "avg_score:", avg_score)
        # input()
        # for item in edges_list:
        #     for edge in item["edges"]:
        #         edge[-1] = (edge[-1] - min_score) / (max_score - min_score)
        
        normalized_edges_list = []
        for i, item in enumerate(edges_list):
            _, edges = graph.show(edges=item["edges"], abs_scores=True, fname=item["name"], fdir=graph_dir)
            normalized_edges_list.append({"edges": edges, "name": item["name"]})
        
        edge_score_path = os.path.join(graph_dir, "edge_scores.jsonl")
        with jsonlines.open(edge_score_path, "w") as f:
            for item in normalized_edges_list:
                f.write(item)
                
    
    def analyze_hebbian_learning_sv_svd():
        
        sv_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eap/ims/sv_topn_50_sel_35_edge_info.jsonl"
        with jsonlines.open(sv_path, "r") as f:
            lines = [line for line in f]
        sv_edges = lines[0]["edges"]
        
        edge_info_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eap/ims/eap_graph_svd_topn_50_select_35/edge_scores.jsonl"
        steps = [2000, 3400, 4000]
        key_format = "eap_graph_svd_topn_50_{step}.pdf"
        
        with jsonlines.open(edge_info_path, "r") as f:
            lines = [line for line in f]
            
        edges_list = [sv_edges]
        for step in steps:
            key = key_format.format(step=step)
            for line in lines:
                if line["name"] == key:
                    edges_list.append(line["edges"])
        
        edge_dicts = []
        for edges in edges_list:
            edge_dict = {}
            for edge in edges:
                edge_dict[f"{edge[0]} -> {edge[1]}"] = round(edge[2], 4)
            edge_dicts.append(edge_dict)
            
        edges_0_keys = set(edge_dicts[0].keys())
        edges_1_keys = set(edge_dicts[1].keys())
        edges_2_keys = set(edge_dicts[2].keys())
        edges_3_keys = set(edge_dicts[3].keys())
        
        all_edges = edges_0_keys.union(edges_1_keys, edges_2_keys, edges_3_keys)
        # common_edges = edges_0_keys.intersection(edges_1_keys, edges_2_keys)
        for edge in all_edges:
            for edge_dict in edge_dicts:
                if edge not in edge_dict:
                    edge_dict[edge] = 0
                    
        strengthened_edges = []
        weakened_edges = []
        for common_edge in all_edges:
            score_0 = edge_dicts[0][common_edge]
            score_1 = edge_dicts[1][common_edge]
            score_2 = edge_dicts[2][common_edge]
            score_3 = edge_dicts[3][common_edge]
            if score_0 <= score_1 <= score_2 <= score_3:
                strengthened_edges.append([common_edge, [score_0, score_1, score_2, score_3]])
                # print(common_edge, f"{score_0}->{score_1}->{score_2}")
            elif score_0 >= score_1 >= score_2 >= score_3:
                weakened_edges.append([common_edge, [score_0, score_1, score_2, score_3]])
        strengthened_edges.sort(key=lambda x: x[1][-1], reverse=True)
        weakened_edges.sort(key=lambda x: x[1][1], reverse=True)
        
        print("Strengthened edges")
        for edge in strengthened_edges:
            print(f"{edge[0]}: {edge[1][0]}->{edge[1][1]}->{edge[1][2]}->{edge[1][3]}")
        print("Weakended edges")
        for edge in weakened_edges:
            print(f"{edge[0]}: {edge[1][0]}->{edge[1][1]}->{edge[1][2]}->{edge[1][3]}")
    
    def analyze_hebbian_learning_svd():
        
        edge_info_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/eap/ims/eap_graph_svd_topn_50_select_35/edge_scores.jsonl"
        steps = [2000, 3400, 4000]
        key_format = "eap_graph_svd_topn_50_{step}.pdf"
        
        with jsonlines.open(edge_info_path, "r") as f:
            lines = [line for line in f]
            
        edges_list = []
        for step in steps:
            key = key_format.format(step=step)
            for line in lines:
                if line["name"] == key:
                    edges_list.append(line["edges"])
        
        edge_dicts = []
        for edges in edges_list:
            edge_dict = {}
            for edge in edges:
                edge_dict[f"{edge[0]} -> {edge[1]}"] = round(edge[2], 4)
            edge_dicts.append(edge_dict)
            
        edges_0_keys = set(edge_dicts[0].keys())
        edges_1_keys = set(edge_dicts[1].keys())
        edges_2_keys = set(edge_dicts[2].keys())
        
        all_edges = edges_0_keys.union(edges_1_keys, edges_2_keys)
        # common_edges = edges_0_keys.intersection(edges_1_keys, edges_2_keys)
        for edge in all_edges:
            for edge_dict in edge_dicts:
                if edge not in edge_dict:
                    edge_dict[edge] = 0
                    
        strengthened_edges = []
        weakened_edges = []
        for common_edge in all_edges:
            score_0 = edge_dicts[0][common_edge]
            score_1 = edge_dicts[1][common_edge]
            score_2 = edge_dicts[2][common_edge]
            if score_0 <= score_1 <= score_2:
                strengthened_edges.append([common_edge, [score_0, score_1, score_2]])
                # print(common_edge, f"{score_0}->{score_1}->{score_2}")
            elif score_0 >= score_1 >= score_2:
                weakened_edges.append([common_edge, [score_0, score_1, score_2]])
        strengthened_edges.sort(key=lambda x: x[1][-1], reverse=True)
        weakened_edges.sort(key=lambda x: x[1][0], reverse=True)
        
        print("Strengthened edges")
        for edge in strengthened_edges:
            print(f"{edge[0]}: {edge[1][0]}->{edge[1][1]}->{edge[1][2]}")
        print("Weakended edges")
        for edge in weakened_edges:
            print(f"{edge[0]}: {edge[1][0]}->{edge[1][1]}->{edge[1][2]}")

    def check_graph():
        model = HookedTransformer.from_pretrained(
            'gpt2-small',
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device="cuda"
        )
        
        model.set_use_hook_mlp_in(True)
        model.set_use_split_qkv_input(True)
        model.set_use_attn_result(True)
        
        upstream_nodes=["mlp", "head"]
        downstream_nodes=["mlp", "head"]
        graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
        
        print(graph.upstream_nodes)
        print(graph.downstream_nodes)
        print(graph.upstream_node_index)
        print(graph.upstream_nodes_before_layer[0])
    
    
    draw_graph_for_every_log_step()
    # analyze_hebbian_learning_sv_svd()
    # check_graph()