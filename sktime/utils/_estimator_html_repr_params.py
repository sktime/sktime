"""Parameter HTML representation for estimator display.

Based on scikit-learn's params module (BSD-3-Clause).
"""

import html
import reprlib
from collections import UserDict



def _read_params(name, value, non_default_params):
    """Categorize parameters as 'default' or 'user-set' and format values."""
    name = html.escape(name)
    r = reprlib.Repr()
    r.maxlist = 2
    r.maxtuple = 1
    r.maxstring = 50
    cleaned_value = html.escape(r.repr(value))

    param_type = "user-set" if name in non_default_params else "default"

    return {"param_type": param_type, "param_name": name, "param_value": cleaned_value}


def _params_html_repr(params):
    """Generate HTML representation of estimator parameters.

    Creates an HTML table with parameter names and values, wrapped in a
    collapsible details element. Parameters are styled differently based
    on whether they are default or user-set values.
    """
    PARAMS_TABLE_TEMPLATE = """
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>
                    {rows}
                  </tbody>
                </table>
            </details>
        </div>
    """

    # Long onclick needed for copy button (sklearn 1.7+ compatible)
    copy_onclick = (
        "copyToClipboard('{param_name}', this.parentElement.nextElementSibling)"
    )
    PARAM_ROW_TEMPLATE = f"""
        <tr class="{{param_type}}">
            <td><i class="copy-paste-icon" onclick="{copy_onclick}"></i></td>
            <td class="param">{{param_display}}</td>
            <td class="value">{{param_value}}</td>
        </tr>
    """

    rows = []
    for row in params:
        param = _read_params(row, params[row], params.non_default)
        param_display = param["param_name"]
        rows.append(PARAM_ROW_TEMPLATE.format(**param, param_display=param_display))

    return PARAMS_TABLE_TEMPLATE.format(rows="\n".join(rows))


class ParamsDict(UserDict):
    """Dictionary-like class to store and provide an HTML representation.

    It builds an HTML structure to be used with Jupyter notebooks or similar
    environments. It allows storing metadata to track non-default parameters.

    Parameters
    ----------
    params : dict, default=None
        The original dictionary of parameters and their values.

    non_default : tuple, default=()
        The list of non-default parameters.

    estimator_class : type, default=None
        The class of the estimator. Reserved for future doc link support.

    doc_link : str, default=""
        The base URL to the online documentation. Reserved for future use.
    """

    def __init__(
        self, *, params=None, non_default=(), estimator_class=None, doc_link=""
    ):
        super().__init__(params or {})
        self.non_default = non_default
        self.estimator_class = estimator_class
        self.doc_link = doc_link

    def _repr_html_inner(self):
        """Return HTML representation of the parameters table."""
        return _params_html_repr(self)
