<a href="https://sktime.org"><img src="https://github.com/alan-turing-institute/sktime/blob/main/docs/source/images/sktime-logo-no-text.jpg?raw=true)" width="175" align="right" /></a>

# Welcome to sktime

> A unified interface for machine learning with time series

:rocket: **Version 0.7.0 out now!** [Check out the release notes here.](https://github.com/alan-turing-institute/sktime/releases)

sktime is a library for time series analsyis in Python. It provides a unified interface for multiple time series learning tasks. Currently, this includes time series classification, regression, clustering, annotation and forecasting. It comes with time series algorithms and [scikit-learn] compatible tools to build, tune and validate time series models.

[scikit-learn]: https://scikit-learn.org/stable/

| Overview | |
|---|---|
| **CI/CD** | [![github-actions](https://img.shields.io/github/workflow/status/alan-turing-institute/sktime/build-and-test?logo=github)](https://github.com/alan-turing-institute/sktime/actions?query=workflow%3Abuild-and-test) [![!appveyor](https://img.shields.io/appveyor/ci/mloning/sktime/main?logo=appveyor)](https://ci.appveyor.com/project/mloning/sktime) [![!azure-devops](https://img.shields.io/azure-devops/build/mloning/30e41314-4c72-4751-9ffb-f7e8584fc7bd/1/main?logo=azure-pipelines)](https://dev.azure.com/mloning/sktime/_build) [![!codecov](https://img.shields.io/codecov/c/github/alan-turing-institute/sktime?label=codecov&logo=codecov)](https://codecov.io/gh/alan-turing-institute/sktime) [![readthedocs](https://img.shields.io/readthedocs/sktime?logo=readthedocs)](https://www.sktime.org/en/latest/?badge=latest) |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/sktime?color=orange)](https://pypi.org/project/sktime/) [![!conda](https://img.shields.io/conda/vn/conda-forge/sktime)](https://anaconda.org/conda-forge/sktime) [![!python-versions](https://img.shields.io/pypi/pyversions/sktime)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.com/invite/gqSab2K) [![!gitter](https://img.shields.io/static/v1?logo=gitter&label=gitter&message=chat&color=lightgreen)](https://gitter.im/sktime/community) [![!twitter](https://img.shields.io/twitter/follow/sktime_toolbox?label=%20Twitter&style=social)](https://twitter.com/sktime_toolbox) [![!youtube](https://img.shields.io/youtube/views/wqQKFu41FIw?label=watch&style=social)](https://www.youtube.com/watch?v=wqQKFu41FIw&t=14s) |
| **Citation** | [![!zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg)](https://doi.org/10.5281/zenodo.3749000) |

## :books: Documentation

| Documentation              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| :star: **[Tutorials]**        | New to sktime? Here's everything you need to know!              |
| :woman_technologist: **[User Guides]**      | How to use sktime and its features.                             |
| :scissors: **[Extension Templates]** | How to build your own estimator using sktime's API.            |
| :control_knobs: **[API Reference]**      | The detailed reference for sktime's API.                        |
| :tv: **[Video Tutorial]**            | Our video tutorial from the 2020 PyData Festival.      |
| :hammer_and_wrench: **[Changelog]**          | Changes and version history.                                   |
| :deciduous_tree: **[Roadmap]**          | sktime's software and community development plan.                                   |
| :pencil: **[Related Software]**          | A list of related software. |

[tutorials]: https://www.sktime.org/en/latest/tutorials.html
[user guides]: https://www.sktime.org/en/latest/user_guide.html
[video tutorial]: https://github.com/sktime/sktime-tutorial-pydata-amsterdam-2020
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
| :speech_balloon: **General Discussion**        | [GitHub Discussions] · [Gitter] · [Discord]          |

[github issue tracker]: https://github.com/alan-turing-institute/sktime/issues
[github discussions]: https://github.com/alan-turing-institute/sktime/discussions
[stack overflow]: https://stackoverflow.com/questions/tagged/sktime
[gitter]: https://gitter.im/sktime/community
[discord]: https://discord.com/invite/gqSab2K

## :stars: Features
Our aim is to make the time series analysis ecosystem more interoperable and usable as a whole. sktime provides a __unified interface for distinct but related time series learning tasks__. It features __dedicated time series algorithms__ and __tools for composite model building__ including pipelining, ensembling, tuning and reduction that enables users to apply an algorithm for one task to another.

sktime also provides **interfaces to related libraries**, for example [scikit-learn], [statsmodels], [tsfresh], [PyOD] and [fbprophet], among others.

For **deep learning**, see our companion package: [sktime-dl](https://github.com/sktime/sktime-dl).

[statsmodels]: https://www.statsmodels.org/stable/index.html
[tsfresh]: https://tsfresh.readthedocs.io/en/latest/
[pyod]: https://pyod.readthedocs.io/en/latest/
[fbprophet]: https://facebook.github.io/prophet/

| Module | Status | Links |
|---|---|---|
| **[Forecasting]** | stable | [Tutorial](https://www.sktime.org/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.org/en/latest/api_reference.html#sktime-forecasting-time-series-forecasting) · [Extension Template](https://github.com/alan-turing-institute/sktime/blob/main/extension_templates/forecasting.py)  |
| **[Time Series Classification]** | stable | [Tutorial](https://github.com/alan-turing-institute/sktime/blob/main/examples/02_classification_univariate.ipynb) · [API Reference](https://www.sktime.org/en/latest/api_reference.html#sktime-classification-time-series-classification) · [Extension Template](https://github.com/alan-turing-institute/sktime/blob/main/extension_templates/classification.py) |
| **[Time Series Regression]** | stable | [API Reference](https://www.sktime.org/en/latest/api_reference.html#sktime-classification-time-series-regression) |
| **[Transformations]** | maturing | [API Reference](https://www.sktime.org/en/latest/api_reference.html#sktime-transformations-time-series-transformers) |
| **[Time Series Clustering]** | experimental | [Extension Template](https://github.com/alan-turing-institute/sktime/blob/main/extension_templates/clustering.py) |
| **[Time Series Distances/Kernels]** | experimental | [Extension Template](https://github.com/alan-turing-institute/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| **[Annotation]** | experimental | [Extension Template](https://github.com/alan-turing-institute/sktime/blob/main/extension_templates/annotation.py) |

[forecasting]: https://github.com/alan-turing-institute/sktime/tree/main/sktime/forecasting
[time series classification]: https://github.com/alan-turing-institute/sktime/tree/main/sktime/classification
[time series regression]: https://github.com/alan-turing-institute/sktime/tree/main/sktime/regression
[time series clustering]: https://github.com/alan-turing-institute/sktime/tree/main/sktime/clustering
[annotation]: https://github.com/alan-turing-institute/sktime/tree/main/sktime/annotation
[time series distances/kernels]: https://github.com/alan-turing-institute/sktime/tree/main/sktime/dists_kernels
[transformations]: https://github.com/alan-turing-institute/sktime/tree/main/sktime/transformations


## :hourglass_flowing_sand: Install sktime
For trouble shooting and detailed installation instructions, see the [documentation](https://www.sktime.org/en/latest/installation.html).

- **Operating system**: macOS X · Linux · Windows 8.1 or higher
- **Python version**: Python 3.6, 3.7 and 3.8 (only 64 bit)
- **Package managers**: [pip] · [conda] (via `conda-forge`)

[pip]: https://pip.pypa.io/en/stable/
[conda]: https://docs.conda.io/en/latest/

### pip
Using pip, sktime releases are available as source packages and binary wheels. You can see all available wheels [here](https://pypi.org/simple/sktime/).

```bash
pip install sktime
```

### conda
You can also install sktime from `conda` via the `conda-forge` channel. For the feedstock including the build recipe and configuration, check out [this repository](https://github.com/conda-forge/sktime-feedstock).

```bash
conda install -c conda-forge sktime
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

X, y = load_arrow_head(return_X_y=True)
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
| :gift_heart: **[Contribute]**        | How to contribute to the sktime's project.          |
| :school_satchel:  **[Mentoring]** | New to open source? Apply to our mentoring program! |
| :date: **[Meetings]** | Join our discussions, tutorials, workshops and sprints! |
| :woman_mechanic:  **[Developer Guides]**      | How to further develop sktime's code base.                             |
| :construction: **[Enhancement Proposals]** | Design a new feature for sktime. |
| :medal_sports: **[Contributors]** | A list of all contributors. |
| :classical_building: **[Governance]** | How and by whom decisions are made in sktime's community.   |

[contribute]: https://github.com/alan-turing-institute/sktime/blob/main/CONTRIBUTING.md
[extension templates]: https://github.com/alan-turing-institute/sktime/tree/main/extension_templates
[developer guides]: https://www.sktime.org/en/latest/developer_guide.html
[contributors]: https://github.com/alan-turing-institute/sktime/blob/main/CONTRIBUTORS.md
[governance]: https://www.sktime.org/en/latest/governance.html
[mentoring]: https://github.com/sktime/mentoring
[meetings]: https://github.com/sktime/community-council
[enhancement proposals]: https://github.com/sktime/enhancement-proposals

## :bulb: Project vision

* **by the community, for the community** -- developed by a friendly and collaborative community.
* the **right tool for the right task** -- helping users to diagnose their learning problem and suitable scientific model types.
* **embedded in state-of-art ecosystems** and **provider of interoperable interfaces** -- interoperable with [scikit-learn], [statsmodels], [tsfresh], and other community favourites.
* **rich model composition and reduction functionality** -- build tuning and feature extraction pipelines, solve forecasting tasks with [scikit-learn] regressors.
* **clean, descriptive specification syntax** -- based on modern object-oriented design principles for data science.
* **fair model assessment and benchmarking** -- build your models, inspect your models, check your models, avoid pitfalls.
* **easily extensible** -- easy extension templates to add your own algorithms compatible with sktime's API.
