<a href="https://sktime.org"><img src="https://github.com/sktime/sktime/blob/main/docs/source/images/sktime-logo.jpg?raw=true)" width="175" align="right" /></a>

# Welcome to sktime

> A unified interface for machine learning with time series

:rocket: **Version 0.16.1 out now!** [Check out the release notes here](https://www.sktime.org/en/latest/changelog.html).

sktime is a library for time series analysis in Python. It provides a unified interface for multiple time series learning tasks. Currently, this includes time series classification, regression, clustering, annotation and forecasting. It comes with [time series algorithms](https://www.sktime.org/en/stable/estimator_overview.html) and [scikit-learn] compatible tools to build, tune and validate time series models.

[scikit-learn]: https://scikit-learn.org/stable/

| Overview | |
|---|---|
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/sktime/sktime/wheels.yml?logo=github)](https://github.com/sktime/sktime/actions/workflows/wheels.yml) [![!codecov](https://img.shields.io/codecov/c/github/sktime/sktime?label=codecov&logo=codecov)](https://codecov.io/gh/sktime/sktime) [![readthedocs](https://img.shields.io/readthedocs/sktime?logo=readthedocs)](https://www.sktime.org/en/latest/?badge=latest) [![platform](https://img.shields.io/conda/pn/conda-forge/sktime)](https://github.com/sktime/sktime) |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/sktime?color=orange)](https://pypi.org/project/sktime/) [![!conda](https://img.shields.io/conda/vn/conda-forge/sktime)](https://anaconda.org/conda-forge/sktime) [![!python-versions](https://img.shields.io/pypi/pyversions/sktime)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples) |
| **Downloads**| [![Downloads](https://static.pepy.tech/personalized-badge/sktime?period=week&units=international_system&left_color=grey&right_color=blue&left_text=weekly%20(pypi))](https://pepy.tech/project/sktime) [![Downloads](https://static.pepy.tech/personalized-badge/sktime?period=month&units=international_system&left_color=grey&right_color=blue&left_text=monthly%20(pypi))](https://pepy.tech/project/sktime) [![Downloads](https://static.pepy.tech/personalized-badge/sktime?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/sktime) |
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.com/invite/gqSab2K) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/sktime/) [![!twitter](https://img.shields.io/static/v1?logo=twitter&label=Twitter&message=news&color=lightblue)](https://twitter.com/sktime_toolbox) [![!youtube](https://img.shields.io/static/v1?logo=youtube&label=YouTube&message=tutorials&color=red)](https://www.youtube.com/playlist?list=PLKs3UgGjlWHqNzu0LEOeLKvnjvvest2d0) |
| **Citation** | [![!zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg)](https://doi.org/10.5281/zenodo.3749000) |

## :books: Documentation

| Documentation              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| :star: **[Tutorials]**        | New to sktime? Here's everything you need to know!              |
| :clipboard: **[Binder Notebooks]** | Example notebooks to play with in your browser.              |
| :woman_technologist: **[User Guides]**      | How to use sktime and its features.                             |
| :scissors: **[Extension Templates]** | How to build your own estimator using sktime's API.            |
| :control_knobs: **[API Reference]**      | The detailed reference for sktime's API.                        |
| :tv: **[Video Tutorial]**            | Our video tutorial from 2021 PyData Global.      |
| :hammer_and_wrench: **[Changelog]**          | Changes and version history.                                   |
| :deciduous_tree: **[Roadmap]**          | sktime's software and community development plan.                                   |
| :pencil: **[Related Software]**          | A list of related software. |

[tutorials]: https://www.sktime.org/en/latest/tutorials.html
[binder notebooks]: https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples
[user guides]: https://www.sktime.org/en/latest/user_guide.html
[video tutorial]: https://github.com/sktime/sktime-tutorial-pydata-global-2021
[api reference]: https://www.sktime.org/en/latest/api_reference.html
[changelog]: https://www.sktime.org/en/latest/changelog.html
[roadmap]: https://www.sktime.org/en/latest/roadmap.html
[related software]: https://www.sktime.org/en/latest/related_software.html

## :speech_balloon: Where to ask questions

Questions and feedback are extremely welcome! Please understand that we won't be able to provide individual support via email. We also believe that help is much more valuable if it's shared publicly, so that more people can benefit from it.

| Type                            | Platforms                               |
| ------------------------------- | --------------------------------------- |
| :bug: **Bug Reports**              | [GitHub Issue Tracker]                  |
| :sparkles: **Feature Requests & Ideas** | [GitHub Issue Tracker]                       |
| :woman_technologist: **Usage Questions**          | [GitHub Discussions] · [Stack Overflow] |
| :speech_balloon: **General Discussion**        | [GitHub Discussions] |
| :factory: **Contribution & Development** | [Slack], contributors channel · [Discord] |
| :globe_with_meridians: **Community collaboration session** | [Discord] - Fridays 1pm UTC, dev/meet-ups channel |

[github issue tracker]: https://github.com/sktime/sktime/issues
[github discussions]: https://github.com/sktime/sktime/discussions
[stack overflow]: https://stackoverflow.com/questions/tagged/sktime
[discord]: https://discord.com/invite/gqSab2K

## :dizzy: Features
Our aim is to make the time series analysis ecosystem more interoperable and usable as a whole. sktime provides a __unified interface for distinct but related time series learning tasks__. It features [__dedicated time series algorithms__](https://www.sktime.org/en/stable/estimator_overview.html) and __tools for composite model building__ including pipelining, ensembling, tuning and reduction that enables users to apply an algorithm for one task to another.

sktime also provides **interfaces to related libraries**, for example [scikit-learn], [statsmodels], [tsfresh], [PyOD] and [fbprophet], among others.

For **deep learning**, see our companion package: [sktime-dl](https://github.com/sktime/sktime-dl).

[statsmodels]: https://www.statsmodels.org/stable/index.html
[tsfresh]: https://tsfresh.readthedocs.io/en/latest/
[pyod]: https://pyod.readthedocs.io/en/latest/
[fbprophet]: https://facebook.github.io/prophet/

| Module | Status | Links |
|---|---|---|
| **[Forecasting]** | stable | [Tutorial](https://www.sktime.org/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.org/en/latest/api_reference.html#sktime-forecasting-time-series-forecasting) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py)  |
| **[Time Series Classification]** | stable | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.org/en/latest/api_reference.html#sktime-classification-time-series-classification) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py) |
| **[Time Series Regression]** | stable | [API Reference](https://www.sktime.org/en/latest/api_reference.html#sktime-classification-time-series-regression) |
| **[Transformations]** | stable | [API Reference](https://www.sktime.org/en/latest/api_reference.html#sktime-transformations-time-series-transformers) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **[Time Series Clustering]** | maturing | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py) |
| **[Time Series Distances/Kernels]** | experimental | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| **[Annotation]** | experimental | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/annotation.py) |

[forecasting]: https://github.com/sktime/sktime/tree/main/sktime/forecasting
[time series classification]: https://github.com/sktime/sktime/tree/main/sktime/classification
[time series regression]: https://github.com/sktime/sktime/tree/main/sktime/regression
[time series clustering]: https://github.com/sktime/sktime/tree/main/sktime/clustering
[annotation]: https://github.com/sktime/sktime/tree/main/sktime/annotation
[time series distances/kernels]: https://github.com/sktime/sktime/tree/main/sktime/dists_kernels
[transformations]: https://github.com/sktime/sktime/tree/main/sktime/transformations


## :hourglass_flowing_sand: Install sktime
For trouble shooting and detailed installation instructions, see the [documentation](https://www.sktime.org/en/latest/installation.html).

- **Operating system**: macOS X · Linux · Windows 8.1 or higher
- **Python version**: Python 3.7, 3.8, 3.9, 3.10, and 3.11 (only 64 bit)
- **Package managers**: [pip] · [conda] (via `conda-forge`)

[pip]: https://pip.pypa.io/en/stable/
[conda]: https://docs.conda.io/en/latest/

### pip
Using pip, sktime releases are available as source packages and binary wheels. You can see all available wheels [here](https://pypi.org/simple/sktime/).

```bash
pip install sktime
```

or, with maximum dependencies,

```bash
pip install sktime[all_extras]
```

### conda
You can also install sktime from `conda` via the `conda-forge` channel. For the feedstock including the build recipe and configuration, check out [this repository](https://github.com/conda-forge/sktime-feedstock).

```bash
conda install -c conda-forge sktime
```

or, with maximum dependencies,

```bash
conda install -c conda-forge sktime-all-extras
```

## :zap: Quickstart

### Forecasting

```python
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
mean_absolute_percentage_error(y_test, y_pred)
>>> 0.08661467738190656
```

### Time Series Classification

```python
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_arrow_head()
X_train, X_test, y_train, y_test = train_test_split(X, y)
classifier = TimeSeriesForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)
>>> 0.8679245283018868
```

## :wave: How to get involved

There are many ways to join the sktime community. We follow the [all-contributors](https://github.com/all-contributors/all-contributors) specification: all kinds of contributions are welcome - not just code.

| Documentation              |                                                                |
| -------------------------- | --------------------------------------------------------------        |
| :gift_heart: **[Contribute]**        | How to contribute to sktime.          |
| :school_satchel:  **[Mentoring]** | New to open source? Apply to our mentoring program! |
| :date: **[Meetings]** | Join our discussions, tutorials, workshops and sprints! |
| :woman_mechanic:  **[Developer Guides]**      | How to further develop sktime's code base.                             |
| :construction: **[Enhancement Proposals]** | Design a new feature for sktime. |
| :medal_sports: **[Contributors]** | A list of all contributors. |
| :raising_hand: **[Roles]** | An overview of our core community roles. |
| :money_with_wings: **[Donate]** | Fund sktime maintenance and development. |
| :classical_building: **[Governance]** | How and by whom decisions are made in sktime's community.   |

[contribute]: https://www.sktime.org/en/latest/get_involved/contributing.html
[donate]: https://opencollective.com/sktime
[extension templates]: https://github.com/sktime/sktime/tree/main/extension_templates
[developer guides]: https://www.sktime.org/en/latest/developer_guide.html
[contributors]: https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md
[governance]: https://www.sktime.org/en/latest/governance.html
[mentoring]: https://github.com/sktime/mentoring
[meetings]: https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC
[enhancement proposals]: https://github.com/sktime/enhancement-proposals
[roles]: https://www.sktime.org/en/latest/about/team.html

## :bulb: Project vision

* **by the community, for the community** -- developed by a friendly and collaborative community.
* the **right tool for the right task** -- helping users to diagnose their learning problem and suitable scientific model types.
* **embedded in state-of-art ecosystems** and **provider of interoperable interfaces** -- interoperable with [scikit-learn], [statsmodels], [tsfresh], and other community favourites.
* **rich model composition and reduction functionality** -- build tuning and feature extraction pipelines, solve forecasting tasks with [scikit-learn] regressors.
* **clean, descriptive specification syntax** -- based on modern object-oriented design principles for data science.
* **fair model assessment and benchmarking** -- build your models, inspect your models, check your models, avoid pitfalls.
* **easily extensible** -- easy extension templates to add your own algorithms compatible with sktime's API.
