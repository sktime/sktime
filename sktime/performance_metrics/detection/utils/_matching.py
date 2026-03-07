"""Shared matching routine for windowed detection metrics."""


def _count_windowed_matches(targets, candidates, margin):
    """Count targets that have at least one candidate within a margin.

    Uses a two-pointer scan over sorted lists to count how many elements
    in ``targets`` have at least one element in ``candidates`` whose
    absolute difference is ``<= margin``.

    Parameters
    ----------
    targets : list of int or float
        Sorted list of target iloc positions.
    candidates : list of int or float
        Sorted list of candidate iloc positions.
    margin : int
        Maximum absolute iloc difference for a match.

    Returns
    -------
    int
        Number of targets matched by at least one candidate.
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
            and abs(candidates[cand_index] - target) <= margin
        ):
            matched_count += 1

    return matched_count
