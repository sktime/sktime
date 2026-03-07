"""Shared matching routine for windowed detection metrics."""


def _count_windowed_matches(targets, candidates, margin):
    """Count one-to-one windowed matches between targets and candidates.

    Uses a greedy two-pointer scan over *sorted* lists to count how many
    elements in ``targets`` have at least one element in ``candidates``
    whose absolute difference is ``<= margin``.

    The scan enforces a one-to-one matching: each candidate can be used
    to match at most one target. When a candidate matches a target it is
    consumed (advanced past in the scan) and will not be considered for
    subsequent targets. If multiple candidates fall within the margin of
    the same target, only the first unmatched candidate encountered in
    the scan is used. Likewise, a single candidate that lies within the
    margin of multiple targets will match only the earliest such target
    reached by the scan.

    Parameters
    ----------
    targets : array-like of int or float
        Sorted sequence of target iloc positions (ascending).
    candidates : array-like of int or float
        Sorted sequence of candidate iloc positions (ascending).
    margin : int or float
        Maximum absolute iloc difference for a match.

    Returns
    -------
    int
        Number of targets that obtain a one-to-one match with a candidate
        within the given margin.
    """
    matched_count = 0
    cand_index = 0

    for target in targets:
        # Advance cand_index while candidate < (target - margin)
        while cand_index < len(candidates) and candidates[cand_index] < target - margin:
            cand_index += 1
        # If current candidate is within margin, it's a match
        if (
            cand_index < len(candidates)
            and candidates[cand_index] <= target + margin
        ):
            matched_count += 1
            # ensure one-to-one matching
            cand_index += 1

    return matched_count
