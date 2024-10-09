#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

# IMPORTS

import argparse
import os
import sys
import random
import statistics
import textwrap
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Dict, List
from itertools import combinations
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    random_layout,
    draw_networkx_nodes,
    draw_networkx_edges

)
import matplotlib
import matplotlib.pyplot as plt

random.seed(9001)

matplotlib.use("Agg")

__author__ = "Giulia Di Gennaro"
__copyright__ = "Université Paris Cité"
__credits__ = ["Dr. Amine Ghouzlane"]
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "your@email.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


############################################################
############# Identification des -mers uniques #############
############################################################


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, "r") as f:
        while True:
            f.readline()  # Ignorer l'identifiant
            sequence = f.readline().strip()  # Lire la séquence
            f.readline()  # Ignorer le '+'
            f.readline()  # Ignorer la qualité
            if not sequence:
                break
            yield sequence


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i + kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = defaultdict(int)
    for sequence in read_fastq(fastq_file):
        for kmer in cut_kmer(sequence, kmer_size):
            kmer_dict[kmer] += 1
    return kmer_dict


############################################################
############ Construction de l'arbre de Bruijn #############
############################################################


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()
    for kmer, count in kmer_dict.items():
        prefix = kmer[:-1]
        suffix = kmer[1:]
        graph.add_edge(prefix, suffix, weight=count)
    return graph


############################################################
############ Parcours du graphe de Bruijn ##################
############################################################


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    return [node for node in graph.nodes if graph.in_degree(node) == 0]


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    return [node for node in graph.nodes if graph.out_degree(node) == 0]


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for start in starting_nodes:
        for end in ending_nodes:
            for path in all_simple_paths(graph, start, end):
                contig = path[0]
                for node in path[1:]:
                    contig += node[-1]  # Ajoute uniquement le dernier caractère du k-mer
                contigs.append((contig, len(contig)))
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w') as f:
        for i, (contig, length) in enumerate(contigs_list):
            # Writes the FASTA header
            f.write(f">contig_{i} len={length}\n")
            # Uses textwrap.fill to cut the contig lines to fit only 80 characters
            f.write(textwrap.fill(contig, width=80) + "\n")


############################################################
########## Simplification du graphe de Bruijn ##############
############################################################


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (DiGraph) A directed graph object
    """
    new_graph = graph.copy()

    for path in path_list:
        if delete_entry_node == True and delete_sink_node == True:
            new_graph.remove_nodes_from(path)
        elif delete_entry_node == True and delete_sink_node == False:
            new_graph.remove_nodes_from(path[:-1])
        elif delete_entry_node == False and delete_sink_node == True:
            new_graph.remove_nodes_from(path[1:])
        else:
            new_graph.remove_nodes_from(path[1:-1])
    return new_graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (DiGraph) A directed graph object
    """
    if len(weight_avg_list) > 1:
        # Calculate standard deviation only if we have more than one path
        std_poids = statistics.stdev(weight_avg_list)
    else:
        std_poids = 0  # No variation if only one path

    if len(path_length) > 1:
        std_long = statistics.stdev(path_length)
    else:
        std_long = 0  # No variation if only one path

    if std_poids > 0:
        best_path_index = weight_avg_list.index(max(weight_avg_list))
    elif std_long > 0:
        best_path_index = path_length.index(max(path_length))
    else:
        best_path_index = random.randint(0, len(path_list) - 1)  # Random choice if no variation

    best_path = path_list[best_path_index]

    for path in path_list:
        if path != best_path:
            graph = remove_paths(graph, [path], delete_entry_node, delete_sink_node)

    return graph


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (DiGraph) A directed graph object
    """
    paths = list(all_simple_paths(graph, ancestor_node, descendant_node))
    path_lengths = []
    path_weights = []
    for path in paths:
        path_lengths.append(len(path))
        path_weights.append(path_average_weight(graph, path))

    graph = select_best_path(
        graph, paths, path_lengths, path_weights, delete_entry_node=False, delete_sink_node=False)
    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (DiGraph) A directed graph object
    :return: (DiGraph) A directed graph object
    """
    bubble = False
    list_nodes_graph = list(graph.nodes())

    for node in list_nodes_graph:
        list_predecessors = list(graph.predecessors(node))
        if len(list_predecessors) > 1:
            for pred1, pred2 in combinations(list_predecessors, 2):
                noeud_ancetre = lowest_common_ancestor(graph, pred1, pred2)
                if noeud_ancetre is not None:
                    bubble = True
                    graph = solve_bubble(graph, noeud_ancetre, node)
                    break
            if bubble:
                break
    if bubble:
        graph = simplify_bubbles(graph)
    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    total_weight = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
    return total_weight / (len(path) - 1)


################################################
########## Détetction des pointes ##############
################################################


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (DiGraph) A directed graph object
    """
    path_list = [list(all_simple_paths(graph, start, node)) for node in graph.nodes for start in starting_nodes if len(list(graph.predecessors(node))) > 1]
    
    if path_list:
        path_length = [len(path) for paths in path_list for path in paths]
        weight_avg = [path_average_weight(graph, path) for paths in path_list for path in paths]
        graph = select_best_path(graph, [path for paths in path_list for path in paths], path_length, weight_avg, delete_entry_node=True, delete_sink_node=False)
    
    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (DiGraph) A directed graph object
    """
    path_list = [list(all_simple_paths(graph, node, end)) for node in graph.nodes for end in ending_nodes if len(list(graph.successors(node))) > 1]
    
    if path_list:
        path_length = [len(path) for paths in path_list for path in paths]
        weight_avg = [path_average_weight(graph, path) for paths in path_list for path in paths]
        graph = select_best_path(graph, [path for paths in path_list for path in paths], path_length, weight_avg, delete_entry_node=False, delete_sink_node=True)
    
    return graph


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    # fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = random_layout(graph)
    draw_networkx_nodes(graph, pos, node_size=6)
    draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================


def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
