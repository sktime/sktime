import pandas as pd

__all__ = [
    "loc_series_idxs",
    "get_bottom_level_idxs",
    "get_bottom_series",
    "get_total_level_idxs",
    "get_total_series",
    "get_middle_level_series",
    "is_ancestor",
    "filter_descendants",
    "_get_index_level_aggregators",
    "_promote_hierarchical_indexes",
    "_promote_hierarchical_indexes_and_keep_timeindex",
]


def loc_series_idxs(y, idxs):
    return y.loc[y.index.droplevel(-1).isin(idxs)]


def get_bottom_level_idxs(y):
    idx = y.index
    idx = idx.droplevel(-1).unique()
    return idx[idx.get_level_values(-1) != "__total"]


def get_bottom_series(y):
    bottom_idx = get_bottom_level_idxs(y)
    return loc_series_idxs(y, bottom_idx)


def get_total_level_idxs(y):
    nlevels = y.index.droplevel(-1).nlevels
    if nlevels == 1:
        return pd.Index(["__total"])
    return pd.Index([tuple(["__total"] * nlevels)])


def get_total_series(y):
    total_idx = get_total_level_idxs(y)
    return loc_series_idxs(y, total_idx)


def get_middle_level_series(y, middle_level):
    idx = y.index
    idx = idx.droplevel(-1).unique()
    idx_middle_level = idx.get_level_values(middle_level)
    idx_below_middle_level = idx.get_level_values(middle_level + 1)
    is_middle_level_mask = (idx_below_middle_level == "__total") & (
        idx_middle_level != "__total"
    )
    return idx[is_middle_level_mask]


def is_ancestor(agg_node, node):
    """Return True if agg_node is an ancestor of node."""
    return all(a == b or a == "__total" for a, b in zip(agg_node, node))


def filter_descendants(X, aggregator_node):
    """
    Get descendants of aggregator_node from X.

    Returns a sub-DataFrame/Series of X containing only rows whose
    droplevel(-1) is a descendant of 'aggregator_node' at the given 'middle_level'.

    aggregator_node is a tuple like ('CA', 'CA_1', '__total', '__total')
    or ('regionA', 'storeA', '__total', '__total'), etc.
    """
    # We'll operate on the "higher-level" portion of the index
    # (i.e., the index after dropping the time or final level).
    idx_upper = X.index.droplevel(-1)  # e.g. (region, store, cat, dept)
    nodes = idx_upper.unique()

    # We only want to keep those which are descendants of aggregator_node,
    # i.e., aggregator_node is an ancestor of that node.
    # Because aggregator_node is "at" middle_level,
    # but it may also have __total at deeper levels.
    descendant_mask = [is_ancestor(aggregator_node, n) for n in nodes]
    descendant_nodes = nodes[descendant_mask]
    descendant_nodes = descendant_nodes.unique()

    # Now filter X by these nodes
    return X.loc[idx_upper.isin(descendant_nodes)]


def _is_hierarchical_dataframe(X):
    if not X.index.nlevels > 1:
        return False
    has_total = False
    for i in range(X.index.nlevels - 1):
        if "__total" in X.index.get_level_values(i):
            has_total = True
            break
    return has_total or X.index.nlevels > 2


def _get_index_level_aggregators(X, index_level):
    """
    Get middle level aggregators from the index of X.

    Identify aggregator nodes at the specified index, i.e.
    those for which the next level is '__total'.

    Example:
      If middle_level=1, then for a node= (region, store, cat, dept),
      we check node[middle_level+1] == '__total'.
      The node itself must not be __total at middle_level
      (otherwise it's above or not a well-defined middle node).
    """
    idx = X.index.droplevel(-1).unique()

    # Positive indexed middle level
    if index_level < 0:
        index_level = idx.nlevels + index_level

    # +1 since zero-indexed
    if index_level + 1 > idx.nlevels:
        # If the user picks a middle_level at the last level, there's no "next" level
        return []

    # The aggregator condition:
    #   * the next level's value is '__total'
    #   * the middle level's value != '__total'
    this_level_vals = idx.get_level_values(index_level)

    if index_level == 0:
        # Then we want the total``
        return idx[this_level_vals == "__total"]

    is_agg = this_level_vals != "__total"
    # If not last level
    if index_level + 1 != idx.nlevels:
        next_level_vals = idx.get_level_values(index_level + 1)
        is_agg = (next_level_vals == "__total") & (this_level_vals != "__total")
    return idx[is_agg]


def _promote_hierarchical_indexes(idx_tuple: tuple):
    """
    Walk the index one level up, promoting to the upper level in the hier.

    As an example, if a tuple is ("region", "store", "product"),
    the function will return ("region", "store", "__total").

    If the tuple is ("region", "store", "__total"),
    the function will return ("region", "__total", "__total").

    If the tuple is ("__total", "__total", "__total"),
    the function will return ("__total", "__total", "__total").

    Parameters
    ----------
    idx_tuple : tuple
        The index tuple to promote.

    Returns
    -------
    promoted_idx : tuple
        The promoted index tuple.
    """
    idx_as_list = list(idx_tuple)
    # Find first idx where "__total"
    total_idx = idx_tuple.index("__total") if "__total" in idx_tuple else None

    if total_idx is None:
        idx_as_list[-1] = "__total"
        return tuple(idx_as_list)

    if total_idx == 0:
        return idx_tuple

    idx_as_list[total_idx - 1] = "__total"
    return tuple(idx_as_list)


def _promote_hierarchical_indexes_and_keep_timeindex(idx_tuple: tuple):
    return (*_promote_hierarchical_indexes(idx_tuple[:-1]), idx_tuple[-1])


def _recursively_propagate_topdown(X):
    # Initialize the transformed data
    _X = X.copy()

    # Recursively apply the ratios from top levels to bottom levels
    for level in range(0, X.index.nlevels - 1):
        # Filter series at the current level
        # Get the ratio for the level above
        parent_ratio = _X.loc[
            X.index.map(_promote_hierarchical_indexes_and_keep_timeindex)
        ]
        parent_ratio.index = _X.index

        # Apply the ratio to the current level
        idx_current_level = _get_index_level_aggregators(_X, index_level=level)

        idx_current_level = loc_series_idxs(_X, idx_current_level).index
        _X.loc[idx_current_level] *= parent_ratio.loc[idx_current_level].values

    return _X


def _get_series_for_each_hierarchical_level(idx):
    """
    Return a list of series for each hierarchical level.

    This function returns a list with length H, where
    H is the height of the hierarchical tree that defines
    the hierarchical index. Each element of the list is
    an index or multiindex that represents the series.

    Parameters
    ----------
    idx : pd.Index
        The hierarchical index.

    Returns
    -------
    tree_level_nodes : list
        A list of series for each hierarchical level
    """
    tree_level_nodes = []
    for level in range(idx.nlevels):
        current_level_values = idx.get_level_values(level)
        previous_level_values = idx.get_level_values(level - 1) if level > 0 else None
        next_level_values = (
            idx.get_level_values(level + 1) if level < idx.nlevels - 1 else None
        )

        # First level
        if previous_level_values is None:
            total_series = idx[current_level_values == "__total"]
            tree_level_nodes.append(total_series)

        if next_level_values is None:
            bottom_series = idx[current_level_values != "__total"]
            tree_level_nodes.append(bottom_series)

        else:
            middle_series = idx[
                (current_level_values != "__total") & (next_level_values == "__total")
            ]
            tree_level_nodes.append(middle_series)
    return tree_level_nodes


def _split_middle_levels(y, middle_level):
    """Return two series: middle level and above, and middle level and below."""
    idx = y.index
    idx = idx.droplevel(-1).unique()
    hierarchical_level_nodes = _get_series_for_each_hierarchical_level(idx)

    idx_middle_level = loc_series_idxs(y, hierarchical_level_nodes[middle_level])
    if middle_level >= len(hierarchical_level_nodes):
        idx_below_middle_level = None
    else:
        idx_below_middle_level = loc_series_idxs(
            y, hierarchical_level_nodes[middle_level + 1]
        )

    if middle_level == 0:
        idx_above_middle_level = None
    else:
        idx_above_middle_level = loc_series_idxs(
            y, hierarchical_level_nodes[middle_level - 1]
        )

    return idx_above_middle_level, idx_middle_level, idx_below_middle_level
