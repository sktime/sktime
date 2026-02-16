"""utility functions for benchmarking module."""

import re


def _check_id_format(id_format: str, id: str) -> None:
    """Check if given input ID follows regex specified in id_format."""
    if id_format is not None:
        if not isinstance(id_format, str):
            raise TypeError(f"id_format must be str but receive {type(id_format)}")
        entity_id_re = re.compile(id_format)
        match = entity_id_re.search(id)
        if not match:
            raise ValueError(
                f"Attempted to register malformed entity ID: [id={id}]. "
                f"All IDs must be of the form {entity_id_re.pattern}."
            )
