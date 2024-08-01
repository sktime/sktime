"""Functionality to represent instance of BaseObject as html."""

import html
import uuid
from contextlib import closing
from io import StringIO
from string import Template

__author__ = ["RNKuhns"]


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
):
    """Write labeled html with or without a dropdown with named details."""
    out.write(f'<div class={outer_class!r}><div class="{inner_class} sk-toggleable">')
    name = html.escape(name)

    if name_details is not None:
        name_details = html.escape(str(name_details))
        label_class = "sk-toggleable__label sk-toggleable__label-arrow"

        checked_str = "checked" if checked else ""
        est_id = uuid.uuid4()
        out.write(
            '<input class="sk-toggleable__control sk-hidden--visually" '
            f'id={est_id!r} type="checkbox" {checked_str}>'
            f"<label for={est_id!r} class={label_class!r}>{name}</label>"
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

    # check if base_object looks like a meta base_object wraps base_object
    if hasattr(base_object, "get_params"):
        base_objects = []
        for key, value in base_object.get_params().items():
            # Only look at the BaseObjects in the first layer
            if "__" not in key and hasattr(value, "get_params"):
                base_objects.append(value)
        if len(base_objects):
            return _VisualBlock("parallel", base_objects, names=None)

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

    if est_block.kind in ("serial", "parallel"):
        dashed_wrapped = first_call or est_block.dash_wrapped
        dash_cls = " sk-dashed-wrapped" if dashed_wrapped else ""
        out.write(f'<div class="sk-item{dash_cls}">')

        if base_object_label:
            _write_label_html(out, base_object_label, base_object_label_details)

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
        )


_STYLE = """
#$id {
  color: black;
  background-color: white;
}
#$id pre{
  padding: 0;
}
#$id div.sk-toggleable {
  background-color: white;
}
#$id label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.3em;
  box-sizing: border-box;
  text-align: center;
}
#$id label.sk-toggleable__label-arrow:before {
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: #696969;
}
#$id label.sk-toggleable__label-arrow:hover:before {
  color: black;
}
#$id div.sk-estimator:hover label.sk-toggleable__label-arrow:before {
  color: black;
}
#$id div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  background-color: #f0f8ff;
}
#$id div.sk-toggleable__content pre {
  margin: 0.2em;
  color: black;
  border-radius: 0.25em;
  background-color: #f0f8ff;
}
#$id input.sk-toggleable__control:checked~div.sk-toggleable__content {
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}
#$id input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}
#$id div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: #d4ebff;
}
#$id div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: #d4ebff;
}
#$id input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}
#$id div.sk-estimator {
  font-family: monospace;
  background-color: #f0f8ff;
  border: 1px dotted black;
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
}
#$id div.sk-estimator:hover {
  background-color: #d4ebff;
}
#$id div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 1px solid gray;
  flex-grow: 1;
}
#$id div.sk-label:hover label.sk-toggleable__label {
  background-color: #d4ebff;
}
#$id div.sk-serial::before {
  content: "";
  position: absolute;
  border-left: 1px solid gray;
  box-sizing: border-box;
  top: 2em;
  bottom: 0;
  left: 50%;
}
#$id div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: white;
  padding-right: 0.2em;
  padding-left: 0.2em;
}
#$id div.sk-item {
  z-index: 1;
}
#$id div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: white;
}
#$id div.sk-parallel::before {
  content: "";
  position: absolute;
  border-left: 1px solid gray;
  box-sizing: border-box;
  top: 2em;
  bottom: 0;
  left: 50%;
}
#$id div.sk-parallel-item {
  display: flex;
  flex-direction: column;
  position: relative;
  background-color: white;
}
#$id div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}
#$id div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}
#$id div.sk-parallel-item:only-child::after {
  width: 0;
}
#$id div.sk-dashed-wrapped {
  border: 1px dashed gray;
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: white;
  position: relative;
}
#$id div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  background-color: white;
  display: inline-block;
  line-height: 1.2em;
}
#$id div.sk-label-container {
  position: relative;
  z-index: 2;
  text-align: center;
}
#$id div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}
#$id div.sk-text-repr-fallback {
  display: none;
}
""".replace("  ", "").replace("\n", "")  # noqa


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
