import pandas as pd


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


def split_middle_levels(y, middle_level):
    """Return two series: middle level and above, and middle level and below."""
    idx = y.index
    idx = idx.droplevel(-1).unique()
    idx_middle_level = idx.get_level_values(middle_level)
    idx_below_middle_level = idx.get_level_values(middle_level + 1)
    is_middle_or_above_mask = idx_below_middle_level == "__total"
    is_middle_or_below_mask = idx_middle_level != "__total"

    return loc_series_idxs(y, idx[is_middle_or_above_mask]), loc_series_idxs(
        y, idx[is_middle_or_below_mask]
    )


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


def get_middle_level_aggregators(X, middle_level):
    """
    Get middle level aggregators from the index of X.

    Identify aggregator nodes at the specified middle_level, i.e.
    those for which the next level is '__total'.

    Example:
      If middle_level=1, then for a node= (region, store, cat, dept),
      we check node[middle_level+1] == '__total'.
      The node itself must not be __total at middle_level
      (otherwise it's above or not a well-defined middle node).
    """
    idx = X.index.droplevel(-1).unique()
    if middle_level + 1 >= idx.nlevels:
        # If the user picks a middle_level at the last level, there's no "next" level
        return []

    # The aggregator condition:
    #   * the next level's value is '__total'
    #   * the middle level's value != '__total'
    next_level_vals = idx.get_level_values(middle_level + 1)
    this_level_vals = idx.get_level_values(middle_level)

    is_agg = (next_level_vals == "__total") & (this_level_vals != "__total")
    return idx[is_agg]
