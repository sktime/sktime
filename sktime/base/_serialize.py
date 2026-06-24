# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utilities for serializing and deserializing objects.

IMPORTANT CAVEAT FOR DEVELOPERS:
Do not add estimator specific functionality to the `load` utility.
All estimator specific functionality should be in
the class methods `load_from_serial` and `load_from_path`.
"""

__author__ = ["fkiraly", "achieveordie"]


SERIALIZATION_FORMATS = {
    "pickle",
    "cloudpickle",
}


class _NativeArtifactBackend:
    """Base strategy for native artifact serialization."""

    backend = None

    def supports(self, obj):
        """Return whether backend supports the object."""
        raise NotImplementedError

    def dump(self, obj, path, *, estimator, name):
        """Dump object to path and return artifact metadata."""
        raise NotImplementedError

    def load(self, path, record, *, estimator, name):
        """Load object from path and artifact metadata."""
        raise NotImplementedError


class _TransformersArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for transformers models."""

    backend = "transformers"

    def supports(self, obj):
        """Return whether object looks like a transformers PreTrainedModel."""
        return any(
            cls.__name__ == "PreTrainedModel"
            and cls.__module__.startswith("transformers")
            for cls in type(obj).__mro__
        )

    def dump(self, obj, path, *, estimator, name):
        """Dump a transformers model using save_pretrained."""
        obj.save_pretrained(path, safe_serialization=True)
        cls = type(obj)
        return {
            "backend": self.backend,
            "class": f"{cls.__module__}.{cls.__qualname__}",
        }


_NATIVE_ARTIFACT_BACKENDS = [
    _TransformersArtifactBackend(),
]


def _get_native_artifact_backend(obj, *, name):
    """Return native artifact backend for object."""
    for backend in _NATIVE_ARTIFACT_BACKENDS:
        if backend.supports(obj):
            return backend

    raise TypeError(
        f"No native serialization backend is available for artifact {name!r} "
        f"of type {type(obj)!r}."
    )


class _NativeArtifactStore:
    """Store native serialization artifacts inside an sktime save bundle."""

    def __init__(self, artifact_root):
        self.artifact_root = artifact_root
        self.index = {}

    def dump(self, name, obj, *, estimator):
        """Dump a native artifact to the store."""
        artifact_path = self.artifact_root / name
        artifact_path.mkdir(parents=True)
        backend = _get_native_artifact_backend(obj, name=name)
        record = backend.dump(
            obj,
            artifact_path,
            estimator=estimator,
            name=name,
        )
        self.index[name] = {
            **record,
            "path": name,
        }
        return self.index[name]

    def write_index(self):
        """Write native artifact index."""
        import json

        self.artifact_root.mkdir(exist_ok=True)
        with open(self.artifact_root / "index.json", "w", encoding="utf-8") as file:
            json.dump(self.index, file, indent=2)


class _SerializationMixin:
    """Mixin containing serialization API for sktime base objects."""

    def __getstate__(self):
        """Get object state for serialization."""
        state = self.__dict__.copy()
        skip = self.get_tag("serialization:skip", ())
        native_artifacts = self.get_tag("serialization:native_artifacts", ())

        for name in (*skip, *native_artifacts):
            state.pop(name, None)

        return state

    def __setstate__(self, state):
        """Set object state after deserialization."""
        self.__dict__.update(state)

    def _write_native_artifacts(self, path):
        """Write native artifacts for self if it opts into them."""
        native_artifacts = self.get_tag("serialization:native_artifacts", ())

        if not native_artifacts:
            return

        store = _NativeArtifactStore(path / "_artifacts")

        for name in native_artifacts:
            artifact = getattr(self, name, None)
            if artifact is not None:
                store.dump(name, artifact, estimator=self)

        store.write_index()

    def save(self, path=None, serialization_format="pickle"):
        """Save serialized self to bytes-like object or to (.zip) file.

        Behaviour:

        * if ``path`` is None, returns an in-memory serialized self
        * if ``path`` is a file location, stores self at that location as a zip file

        saved files are zip files with following contents:

        * ``_metadata`` - contains class of self, i.e., ``type(self)``
        * ``_obj`` - serialized self. This class uses the default serialization
          (pickle).

        Parameters
        ----------
        path : None or file location (str or Path)
            if None, self is saved to an in-memory object
            if file location, self is saved to that file location. If:

            - path="estimator" then a zip file ``estimator.zip`` will be made at cwd.
            - path="/home/stored/estimator" then a zip file ``estimator.zip`` will be
            stored in ``/home/stored/``.

        serialization_format: str, default = "pickle"
            Module to use for serialization.
            The available options are "pickle" and "cloudpickle".
            Note that non-default formats might require
            installation of other soft dependencies.

        Returns
        -------
        if ``path`` is None - in-memory serialized self
        if ``path`` is file location - ZipFile with reference to the file
        """
        import pickle
        import shutil
        from pathlib import Path
        from zipfile import ZipFile

        from skbase.utils.dependencies import _check_soft_dependencies

        if serialization_format not in SERIALIZATION_FORMATS:
            raise ValueError(
                f"The provided `serialization_format`='{serialization_format}' "
                "is not yet supported. The possible formats are: "
                f"{SERIALIZATION_FORMATS}."
            )

        if path is not None and not isinstance(path, (str, Path)):
            raise TypeError(
                "`path` is expected to either be a string or a Path object "
                f"but found of type:{type(path)}."
            )
        if path is not None:
            path = Path(path) if isinstance(path, str) else path
            path.mkdir()

        if serialization_format == "cloudpickle":
            _check_soft_dependencies("cloudpickle", severity="error")
            import cloudpickle

            if path is None:
                return (type(self), cloudpickle.dumps(self))

            with open(path / "_metadata", "wb") as file:
                cloudpickle.dump(type(self), file)
            with open(path / "_obj", "wb") as file:
                cloudpickle.dump(self, file)

        elif serialization_format == "pickle":
            if path is None:
                return (type(self), pickle.dumps(self))

            with open(path / "_metadata", "wb") as file:
                pickle.dump(type(self), file)
            with open(path / "_obj", "wb") as file:
                pickle.dump(self, file)

        self._write_native_artifacts(path)
        shutil.make_archive(base_name=path, format="zip", root_dir=path)
        shutil.rmtree(path)
        return ZipFile(path.with_name(f"{path.stem}.zip"))

    @classmethod
    def load_from_serial(cls, serial):
        """Load object from serialized memory container.

        Parameters
        ----------
        serial : 1st element of output of ``cls.save(None)``

        Returns
        -------
        deserialized self resulting in output ``serial``, of ``cls.save(None)``
        """
        import pickle

        return pickle.loads(serial)

    @classmethod
    def load_from_path(cls, serial):
        """Load object from file location.

        Parameters
        ----------
        serial : result of ZipFile(path).open("object)

        Returns
        -------
        deserialized self resulting in output at ``path``, of ``cls.save(path)``
        """
        import pickle
        from zipfile import ZipFile

        with ZipFile(serial, "r") as file:
            return pickle.loads(file.open("_obj").read())


def load(serial):
    """Load an object either from in-memory object or from a file location.

    Parameters
    ----------
    serial : serialized container (tuple), str (path), or Path object (reference)
        if serial is a tuple (serialized container):
            Contains two elements, first in-memory metadata and second
            the related object.
        if serial is a string (path reference):
            The name of the file without the extension, for e.g: if the file
            is `estimator.zip`, `serial='estimator'`. It can also represent a
            path, for eg: if location is `home/stored/models/estimator.zip`
            then `serial='home/stored/models/estimator'`.
        if serial is a Path object (path reference):
            `serial` then points to the `.zip` file into which the
            object was stored using class method `.save()` of an estimator.

    Returns
    -------
    Deserialized self resulting in output `serial`, of `cls.save`

    Examples
    --------
    Example 1: saving an estimator in-memory and loading it back

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>>
    >>> # 1. fit the estimator
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster()
    >>> forecaster.fit(y, fh=[1, 2, 3])
    NaiveForecaster()
    >>>
    >>> # 2. save the fitted estimator
    >>> pkl = forecaster.save()
    >>>
    >>> # 3. load the saved estimator (can do this on empty kernel)
    >>> from sktime.base import load
    >>> forecaster_loaded = load(pkl)
    >>>
    >>> # 4. continue using the loaded estimator
    >>> y_pred = forecaster_loaded.predict()

    Example 2: saving a deep learning estimator on the hard drive and loading

    >>> import numpy as np
    >>> from sktime.classification.deep_learning import CNNClassifier
    >>>
    >>> # 1. fit the estimator
    >>> sample_X = np.random.randn(15, 24, 16) # doctest: +SKIP
    >>> sample_y = np.random.randint(0, 2, size=(15, )) # doctest: +SKIP
    >>> sample_test_X = np.random.randn(5, 24, 16) # doctest: +SKIP
    >>> cnn = CNNClassifier(n_epochs=1) # doctest: +SKIP
    >>> cnn.fit(sample_X, sample_y) # doctest: +SKIP
    >>>
    >>> # 2. save the fitted estimator
    >>> save_folder_location = "save_folder" # doctest: +SKIP
    >>> cnn.save(save_folder_location) # doctest: +SKIP
    >>>
    >>> # 3. load the saved estimator (can do this on empty kernel)
    >>> from sktime.base import load
    >>> save_folder_location = "save_folder" # doctest: +SKIP
    >>> loaded_cnn = load(save_folder_location) # doctest: +SKIP
    >>>
    >>> # 4. continue using the loaded estimator
    >>> pred = cnn.predict(X=sample_test_X) # doctest: +SKIP
    >>> loaded_pred = loaded_cnn.predict(X=sample_test_X) # doctest: +SKIP

    Example 3:  saving an estimator using cloudpickle's serialization functionality
                and loading it back
        Note: `cloudpickle` is a soft dependency and is not present
        with the base-installation.

    >>> from sktime.classification.feature_based import Catch22Classifier
    >>> from sktime.datasets import load_basic_motions  # doctest: +SKIP
    >>>
    >>> # 1. Fit the estimator
    >>> X_train, y_train = load_basic_motions(split="TRAIN")  # doctest: +SKIP
    >>> X_test, y_test = load_basic_motions(split="TEST")  # doctest: +SKIP
    >>> est = Catch22Classifier().fit(X_train, y_train)  # doctest: +SKIP
    >>>
    >>> # 2. save the fitted estimator
    >>> cpkl_serialized = est.save(serialization_format="cloudpickle")  # doctest: +SKIP
    >>>
    >>> # 3. load the saved estimator (possibly after sending it across a stream)
    >>> from sktime.base import load  # doctest: +SKIP
    >>> loaded_est = load(cpkl_serialized)  # doctest: +SKIP
    >>>
    >>> # 4. continue using the estimator as normal
    >>> pred = loaded_est.predict(X_test)    # doctest: +SKIP
    >>> loaded_pred = loaded_est.predict(X_test)  # doctest: +SKIP
    """
    import pickle
    from pathlib import Path
    from zipfile import ZipFile

    if isinstance(serial, tuple):
        if len(serial) != 2:
            raise ValueError(
                "`serial` should be a tuple of size 2 "
                f"found, a tuple of size: {len(serial)}"
            )
        cls, stored = serial
        return cls.load_from_serial(stored)

    elif isinstance(serial, (str, Path)):
        path = Path(serial + ".zip") if isinstance(serial, str) else serial
        if not path.exists():
            raise FileNotFoundError(f"The given save location: {serial}\nwas not found")
        with ZipFile(path) as file:
            cls = pickle.loads(file.open("_metadata", "r").read())
        return cls.load_from_path(path)
    else:
        raise TypeError(
            "serial must either be a serialized in-memory sktime object, "
            "a str, Path or ZipFile pointing to a file which is a serialized sktime "
            "object, created by save of an sktime object; but found serial "
            f"of type {serial}"
        )
