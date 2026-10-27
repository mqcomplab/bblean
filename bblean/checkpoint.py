"""Iterative checkpoints for large BitBirch trees.

The leaf list is doubly linked. A recursive pickle follows that list and can hit
Python's recursion limit even when the clustering tree itself is shallow. Store
nodes separately and replace graph edges with integer references instead.

This is a trusted, version-specific Python checkpoint, not a portable interchange
format. Never load a checkpoint from an untrusted source. The caller must commit
its input position alongside this file and verify its checksum on restart.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

from bblean.bitbirch import BitBirch, _BFNode, _BITBIRCH_INSTANCES

_MAGIC = b"BBLEAN-FLAT-NODES-1\n"


class _NodePickler(pickle.Pickler):
    def __init__(self, stream, node_ids):
        super().__init__(stream, protocol=5)
        self.node_ids = node_ids

    def persistent_id(self, obj):
        if isinstance(obj, _BFNode):
            return ("node", self.node_ids[id(obj)])
        return None


class _NodeUnpickler(pickle.Unpickler):
    def __init__(self, stream, nodes):
        super().__init__(stream)
        self.nodes = nodes

    def persistent_load(self, identifier):
        kind, index = identifier
        if kind != "node" or not isinstance(index, int) or not 0 <= index < len(self.nodes):
            raise pickle.UnpicklingError("Invalid node reference")
        return self.nodes[index]


def save_checkpoint(tree: BitBirch, path: str | Path) -> dict:
    """Atomically save a tree without recursively visiting its linked nodes.

    The recursion limit is unchanged. Pickle memo storage is bounded to one node
    record, in addition to an O(number_of_nodes) graph index. Array serialization
    can still require temporary memory; monitor large checkpoint operations.
    """
    if type(tree) is not BitBirch:
        raise TypeError("Checkpoint format currently supports BitBirch, not subclasses")
    nodes = []
    node_ids = {}

    def add(node):
        if node is not None and id(node) not in node_ids:
            node_ids[id(node)] = len(nodes)
            nodes.append(node)

    add(tree._root)
    add(tree._dummy_leaf)
    # Iterating a list that grows is deliberate: visit every node exactly once.
    for node in nodes:
        add(node._prev_leaf)
        add(node._next_leaf)
        for cluster in node._subclusters:
            add(cluster.child)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_MAGIC)
            pickle.dump(len(nodes), stream, protocol=5)
            _NodePickler(stream, node_ids).dump(tree.__dict__)
            for node in nodes:
                state = {slot: getattr(node, slot) for slot in _BFNode.__slots__}
                _NodePickler(stream, node_ids).dump(state)
            stream.write(b"END\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return {"format": "bblean-flat-nodes-v1", "nodes": len(nodes),
            "fitted_rows": tree.num_fitted_fps,
            "leaf_clusters": sum(len(node._subclusters) for node in nodes if node.is_leaf)}


def load_checkpoint(path: str | Path) -> BitBirch:
    """Restore a trusted flat-node checkpoint, including links and fit progress."""
    with open(path, "rb") as stream:
        if stream.read(len(_MAGIC)) != _MAGIC:
            raise ValueError("Not a flat-node BBLean checkpoint")
        count = pickle.load(stream)
        if not isinstance(count, int) or count < 1:
            raise ValueError("Invalid checkpoint node count")
        nodes = [_BFNode.__new__(_BFNode) for _ in range(count)]
        tree_state = _NodeUnpickler(stream, nodes).load()
        for node in nodes:
            state = _NodeUnpickler(stream, nodes).load()
            if set(state) != set(_BFNode.__slots__):
                raise ValueError("Checkpoint node schema mismatch")
            for slot, value in state.items():
                setattr(node, slot, value)
        if stream.read(4) != b"END\n" or stream.read(1):
            raise ValueError("Truncated checkpoint or trailing data")
    tree = BitBirch.__new__(BitBirch)
    tree.__dict__.update(tree_state)
    _BITBIRCH_INSTANCES.add(tree)
    return tree
