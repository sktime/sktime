"""Notebook formatter.

Formats jupyter notebooks as follows:

* changes execution count to consecutive numbers, starting with 1
* sets json indent to 1 space
* adds a newline at the end, if not present

Does not change cell content or output.
"""


def format_nb_json(notebook_path, indent=1):
    """Format a jupyter notebook.

    Changes notebook at ``notebook_path`` in place, as follows:

    Formats jupyter notebooks as follows:

    * changes execution count to consecutive numbers, starting with 1
    * sets json indent to 1 space
    * adds a newline at the end, if not present

    Does not change cell content or output.

    Parameters
    ----------
    notebook_path : str
        Path to the notebook file. Accessed via ``open``.
    indent: int, optional, default=1
        Indentation to use for json output. Used in ``json.dump``.
    """
    import json

    # Read the notebook
    with open(notebook_path, encoding="utf-8") as f:
        notebook = json.load(f)

    # Update execution counts
    execution_count = 1
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = execution_count
            execution_count += 1
            for output in cell.get("outputs", []):
                if "execution_count" in output:
                    output["execution_count"] = cell["execution_count"]

    # Write the updated notebook back
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=indent, ensure_ascii=False)
        f.write("\n")  # Add a newline at the end of the file
