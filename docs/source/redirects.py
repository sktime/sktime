"""Sphinx extension to generate redirect pages for moved estimators.

When a class moves to a different module, its old doc URL becomes a 404.
This extension writes a small HTML redirect page at the old URL path so
users are sent to the correct page automatically.

To handle a new move, add an entry to MOVED_ESTIMATORS below.
"""

import os

# old dotted path -> new dotted path
MOVED_ESTIMATORS = {
    "sktime.forecasting.model_selection.ExpandingWindowSplitter": "sktime.split.ExpandingWindowSplitter",
    "sktime.forecasting.model_selection.SlidingWindowSplitter": "sktime.split.SlidingWindowSplitter",
    "sktime.forecasting.model_selection.temporal_train_test_split": "sktime.split.temporal_train_test_split",
}

_REDIRECT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="0; url={new_url}" />
  <title>Redirecting to {new_path}</title>
  <style>
    body {{ font-family: sans-serif; margin: 3em; color: #333; }}
    .notice {{
      background: #fff3cd;
      border: 1px solid #ffc107;
      border-radius: 4px;
      padding: 1em 1.5em;
      max-width: 700px;
    }}
    a {{ color: #0066cc; }}
  </style>
</head>
<body>
  <div class="notice">
    <strong>This page has moved.</strong>
    <p>
      <code>{old_path}</code> has been relocated to
      <code>{new_path}</code>.
    </p>
    <p>
      You are being redirected automatically. If it does not work,
      <a href="{new_url}">click here</a>.
    </p>
  </div>
</body>
</html>
"""


def _generate_redirects(app):
    # docs/_extra/ is on html_extra_path, so files placed there are copied
    # verbatim into the HTML output at the same relative path.
    extra_root = os.path.normpath(os.path.join(app.srcdir, "..", "_extra"))

    for old, new in MOVED_ESTIMATORS.items():
        out_path = os.path.join(
            extra_root, "api_reference", "auto_generated", f"{old}.html"
        )
        new_url = f"{new}.html"
        html = _REDIRECT_TEMPLATE.format(old_path=old, new_path=new, new_url=new_url)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)


def setup(app):
    app.connect("builder-inited", _generate_redirects)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
