"""Functionality to represent instance of BaseObject as html."""
# based on the sklearn module of the same name

import html
import importlib
import uuid
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template

from packaging.version import parse as parse_version

__author__ = ["RNKuhns", "mateuszkasprowicz"]


class _VisualBlock:
    """HTML Representation of BaseObject.

    Parameters
    ----------
    kind : {'serial', 'parallel', 'single'}
        kind of HTML block

    estimators : list of ``BaseObject``s or ``_VisualBlock`s or a single ``BaseObject``
        If ``kind != 'single'``, then ``estimators`` is a list of ``BaseObjects``.
        If ``kind == 'single'``, then ``estimators`` is a single ``BaseObject``.

    names : list of str, default=None
        If ``kind != 'single'``, then ``names`` corresponds to ``BaseObjects``.
        If ``kind == 'single'``, then ``names`` is a single string corresponding to
        the single ``BaseObject``.

    name_details : list of str, str, or None, default=None
        If ``kind != 'single'``, then ``name_details`` corresponds to ``names``.
        If ``kind == 'single'``, then ``name_details`` is a single string
        corresponding to the single ``BaseObject``.

    dash_wrapped : bool, default=True
        If true, wrapped HTML element will be wrapped with a dashed border.
        Only active when ``kind != 'single'``.
    """

    def __init__(
        self, kind, estimators, *, names=None, name_details=None, dash_wrapped=True
    ):
        self.kind = kind
        self.estimators = estimators
        self.dash_wrapped = dash_wrapped

        if self.kind in ("parallel", "serial"):
            if names is None:
                names = (None,) * len(estimators)
            if name_details is None:
                name_details = (None,) * len(estimators)

        self.names = names
        self.name_details = name_details

    def _sk_visual_block_(self):
        return self


def _write_label_html(
    out,
    name,
    name_details,
    outer_class="sk-label-container",
    inner_class="sk-label",
    checked=False,
    doc_link="",
):
    """Write labeled html with or without a dropdown with named details."""
    out.write(f'<div class={outer_class!r}><div class="{inner_class} sk-toggleable">')
    name = html.escape(name)

    if name_details is not None:
        name_details = html.escape(str(name_details))
        label_class = "sk-toggleable__label sk-toggleable__label-arrow"

        checked_str = "checked" if checked else ""
        est_id = uuid.uuid4()
        if doc_link:
            doc_label = "<span>Online documentation</span>"
            if name is not None:
                doc_label = f"<span>Documentation for {name}</span>"
            doc_link = (
                f'<a class="sk-estimator-doc-link"'
                f' rel="noreferrer" target="_blank" href="{doc_link}">?{doc_label}</a>'
            )

        out.write(
            '<input class="sk-toggleable__control sk-hidden--visually" '
            f'id={est_id!r} type="checkbox" {checked_str}>'
            f"<label for={est_id!r} class={label_class!r}>{name}{doc_link}</label>"
            f'<div class="sk-toggleable__content"><pre>{name_details}'
            "</pre></div>"
        )
    else:
        out.write(f"<label>{name}</label>")
    out.write("</div></div>")  # outer_class inner_class


def _get_visual_block(base_object):
    """Generate information about how to display a BaseObject."""
    if hasattr(base_object, "_sk_visual_block_"):
        return base_object._sk_visual_block_()

    if isinstance(base_object, str):
        return _VisualBlock(
            "single", base_object, names=base_object, name_details=base_object
        )
    elif base_object is None:
        return _VisualBlock("single", base_object, names="None", name_details="None")

    # check if estimator looks like a meta estimator (wraps estimators)
    if hasattr(base_object, "get_params") and not isclass(base_object):
        base_objects = [
            (key, est)
            for key, est in base_object.get_params(deep=False).items()
            if hasattr(est, "get_params") and hasattr(est, "fit") and not isclass(est)
        ]
        if base_objects:
            return _VisualBlock(
                "parallel",
                [est for _, est in base_objects],
                names=[f"{key}: {est.__class__.__name__}" for key, est in base_objects],
                name_details=[str(est) for _, est in base_objects],
            )

    return _VisualBlock(
        "single",
        base_object,
        names=base_object.__class__.__name__,
        name_details=str(base_object),
    )


def _write_base_object_html(
    out, base_object, base_object_label, base_object_label_details, first_call=False
):
    """Write BaseObject to html in serial, parallel, or by itself (single)."""
    est_block = _get_visual_block(base_object)

    if hasattr(base_object, "_get_doc_link"):
        doc_link = base_object._get_doc_link()
    else:
        doc_link = ""

    if est_block.kind in ("serial", "parallel"):
        dashed_wrapped = first_call or est_block.dash_wrapped
        dash_cls = " sk-dashed-wrapped" if dashed_wrapped else ""
        out.write(f'<div class="sk-item{dash_cls}">')

        if base_object_label:
            _write_label_html(
                out, base_object_label, base_object_label_details, doc_link=doc_link
            )

        kind = est_block.kind
        out.write(f'<div class="sk-{kind}">')
        est_infos = zip(est_block.estimators, est_block.names, est_block.name_details)

        for est, name, name_details in est_infos:
            if kind == "serial":
                _write_base_object_html(out, est, name, name_details)
            else:  # parallel
                out.write('<div class="sk-parallel-item">')
                # wrap element in a serial visualblock
                serial_block = _VisualBlock("serial", [est], dash_wrapped=False)
                _write_base_object_html(out, serial_block, name, name_details)
                out.write("</div>")  # sk-parallel-item

        out.write("</div></div>")
    elif est_block.kind == "single":
        _write_label_html(
            out,
            est_block.names,
            est_block.name_details,
            outer_class="sk-item",
            inner_class="sk-estimator",
            checked=first_call,
            doc_link=doc_link,
        )


with open(
    Path(__file__).parent / "_estimator_html_repr.css", encoding="utf-8"
) as style_file:
    # use the style defined in the css file
    _STYLE = style_file.read()


def _object_html_repr(base_object):
    """Build a HTML representation of a BaseObject.

    Parameters
    ----------
    base_object : base object
        The BaseObject or inheriting class to visualize.

    Returns
    -------
    html: str
        HTML representation of BaseObject.
    """
    with closing(StringIO()) as out:
        container_id = "sk-" + str(uuid.uuid4())
        style_template = Template(_STYLE)
        style_with_id = style_template.substitute(id=container_id)
        base_object_str = str(base_object)

        # The fallback message is shown by default and loading the CSS sets
        # div.sk-text-repr-fallback to display: none to hide the fallback message.
        #
        # If the notebook is trusted, the CSS is loaded which hides the fallback
        # message. If the notebook is not trusted, then the CSS is not loaded and the
        # fallback message is shown by default.
        #
        # The reverse logic applies to HTML repr div.sk-container.
        # div.sk-container is hidden by default and the loading the CSS displays it.
        fallback_msg = (
            "Please rerun this cell to show the HTML repr or trust the notebook."
        )
        out.write(
            f"<style>{style_with_id}</style>"
            f'<div id={container_id!r} class="sk-top-container">'
            '<div class="sk-text-repr-fallback">'
            f"<pre>{html.escape(base_object_str)}</pre><b>{fallback_msg}</b>"
            "</div>"
            '<div class="sk-container" hidden>'
        )
        _write_base_object_html(
            out,
            base_object,
            base_object.__class__.__name__,
            base_object_str,
            first_call=True,
        )
        out.write("</div></div>")

        html_output = out.getvalue()
        return html_output


def _get_reduced_path(input_path_string):
    """Remove submodules starting with an underscore to get a reduced path string."""
    substrings = input_path_string.split(".")

    index_to_remove = None
    for i, substring in enumerate(substrings):
        if substring.startswith("_"):
            index_to_remove = i
            break

    if index_to_remove is not None:
        substrings = substrings[:index_to_remove] + substrings[-1:]

    result_string = ".".join(substrings)

    return result_string


class _HTMLDocumentationLinkMixin:
    """Mixin class allowing to generate a link to the API documentation.

    This mixin relies on three attributes:
    - `_doc_link_module`: it corresponds to the root module (e.g. `sktime`). Using this
      mixin, the default value is `sktime`.

    The method :meth:`_get_doc_link` generates the link to the API documentation for a
    given estimator.
    """

    _doc_link_module = "sktime"

    @classmethod
    def _generate_doc_link(cls):
        module = importlib.import_module(cls._doc_link_module)
        version = parse_version(module.__version__).base_version
        modpath = str(cls)[8:-2]
        path = _get_reduced_path(modpath)

        return f"https://www.sktime.net/en/v{version}/api_reference/auto_generated/{path}.html"

    def _get_doc_link(self):
        """Generate a link to the API documentation for a given base object.

        For compatibility with sklearn it's an instance method.

        Returns
        -------
        url : str
            The URL to the API documentation for this estimator. If the estimator does
            not belong to module `_doc_link_module`, the empty string (i.e. `""`) is
            returned.
        """
        if self.__class__.__module__.split(".")[0] != self._doc_link_module:
            return ""

        # TODO: add check if link is well-structured
        # TODO: add fallback to stable version (?)
        return self.__class__._generate_doc_link()

    @property
    def _repr_html_(self):
        """HTML representation of BaseObject.

        This is redundant with the logic of `_repr_mimebundle_`. The latter
        should be favorted in the long term, `_repr_html_` is only
        implemented for consumers who do not interpret `_repr_mimbundle_`.
        """
        if self.get_config()["display"] != "diagram":
            raise AttributeError(
                "_repr_html_ is only defined when the "
                "'display' configuration option is set to "
                "'diagram'"
            )
        return self._repr_html_inner

    def _repr_html_inner(self):
        """Return HTML representation of class.

        This function is returned by the @property `_repr_html_` to make
        `hasattr(BaseObject, "_repr_html_") return `True` or `False` depending
        on `self.get_config()["display"]`.
        """
        return _object_html_repr(self)

    def _repr_mimebundle_(self, **kwargs):
        """Mime bundle used by jupyter kernels to display instances of BaseObject."""
        output = {"text/plain": repr(self)}
        if self.get_config()["display"] == "diagram":
            output["text/html"] = _object_html_repr(self)
        return output
