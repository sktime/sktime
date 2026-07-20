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

    def _load_class(self, record):
        """Load class from artifact record."""
        import importlib

        class_path = record["class"]
        parts = class_path.split(".")

        for i in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:i])
            qualname = parts[i:]

            try:
                obj = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue

            for attr in qualname:
                obj = getattr(obj, attr)
            return obj

        raise ModuleNotFoundError(f"Could not import class {class_path!r}.")

    def supports(self, obj):
        """Return whether backend supports the object."""
        raise NotImplementedError

    def save(self, obj, path, *, estimator, name):
        """Save object to path."""
        raise NotImplementedError

    def load(self, path, record, *, estimator, name):
        """Load object from path and artifact metadata."""
        raise NotImplementedError


class _PretrainedArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for save_pretrained/from_pretrained objects."""

    backend = "pretrained"

    def supports(self, obj):
        """Return whether object supports pretrained-style serialization."""
        return callable(getattr(obj, "save_pretrained", None)) and callable(
            getattr(type(obj), "from_pretrained", None)
        )

    def save(self, obj, path, *, estimator, name):
        """Save an object using save_pretrained."""
        obj.save_pretrained(path)

    def load(self, path, record, *, estimator, name):
        """Load an object using from_pretrained with estimator-provided kwargs."""
        cls = self._load_class(record)
        load_kwargs = {}
        get_load_kwargs = getattr(estimator, "_get_native_artifact_load_kwargs", None)
        if callable(get_load_kwargs):
            load_kwargs = get_load_kwargs(name)
        if cls.__module__.startswith("peft.") and "model" in load_kwargs:
            model = load_kwargs.pop("model")
            return cls.from_pretrained(model, path, **load_kwargs)
        return cls.from_pretrained(path, **load_kwargs)


class _KerasArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for Keras models."""

    backend = "keras"

    def supports(self, obj):
        """Return whether object looks like a Keras model."""
        return any(
            cls.__name__ == "Model" and "keras" in cls.__module__
            for cls in type(obj).__mro__
        )

    def save(self, obj, path, *, estimator, name):
        """Save a Keras model using the native .keras format."""
        obj.save(path / "model.keras")

    def load(self, path, record, *, estimator, name):
        """Load a Keras model using keras.models.load_model."""
        from tensorflow import keras

        custom_objects = None
        get_custom_objects = getattr(estimator, "get_custom_objects", None)
        if callable(get_custom_objects):
            custom_objects = get_custom_objects()

        model = keras.models.load_model(
            path / "model.keras",
            custom_objects=custom_objects,
        )

        if hasattr(model, "optimizer"):
            estimator.optimizer_ = model.optimizer
            estimator.optimizer = model.optimizer

        return model


class _LightningCheckpointArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for Lightning checkpoint models."""

    backend = "lightning_checkpoint"

    def supports(self, obj):
        """Return whether object supports Lightning checkpoint loading."""
        return callable(getattr(type(obj), "load_from_checkpoint", None)) and any(
            cls.__name__ == "LightningModule" and "lightning" in cls.__module__
            for cls in type(obj).__mro__
        )

    def save(self, obj, path, *, estimator, name):
        """Save a Lightning model checkpoint."""
        import lightning
        import torch

        checkpoint_path = path / "model.ckpt"
        checkpoint = {
            "state_dict": obj.state_dict(),
            obj.CHECKPOINT_HYPER_PARAMS_KEY: dict(obj.hparams),
            "pytorch-lightning_version": lightning.__version__,
        }
        obj.on_save_checkpoint(checkpoint)
        torch.save(checkpoint, checkpoint_path)

    def load(self, path, record, *, estimator, name):
        """Load a Lightning model checkpoint."""
        cls = self._load_class(record)
        checkpoint_path = path / "model.ckpt"
        return cls.load_from_checkpoint(checkpoint_path)


class _TorchStateDictArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for torch modules, using state dictionaries."""

    backend = "torch_state_dict"

    def supports(self, obj):
        """Return whether object is a torch module."""
        return any(
            cls.__name__ == "Module" and cls.__module__ == "torch.nn.modules.module"
            for cls in type(obj).__mro__
        )

    def save(self, obj, path, *, estimator, name):
        """Save a torch module's state dictionary on CPU."""
        import torch

        state_dict = obj.state_dict()
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.detach().cpu()

        torch.save(state_dict, path / "state_dict.pt")

    def load(self, path, record, *, estimator, name):
        """Construct a torch module and restore its state dictionary."""
        import torch

        create_artifact = getattr(estimator, "_create_torch_artifact", None)
        if not callable(create_artifact):
            raise TypeError(
                f"Estimator {type(estimator).__name__} must implement "
                "`_create_torch_artifact(name)` to load native torch artifact "
                f"{name!r}."
            )

        artifact = create_artifact(name)
        if not isinstance(artifact, torch.nn.Module):
            raise TypeError(
                "`_create_torch_artifact` must return a torch.nn.Module, but "
                f"returned {type(artifact)!r} for artifact {name!r}."
            )

        state_dict_path = path / "state_dict.pt"
        state_dict = torch.load(
            state_dict_path,
            map_location="cpu",
            weights_only=True,
        )
        artifact.load_state_dict(state_dict)
        return artifact


_NATIVE_ARTIFACT_BACKENDS = [
    _PretrainedArtifactBackend(),
    _KerasArtifactBackend(),
    _LightningCheckpointArtifactBackend(),
    _TorchStateDictArtifactBackend(),
]


def _save_native_artifact_backend(obj, *, name):
    """Return native artifact backend for saving object."""
    for backend in _NATIVE_ARTIFACT_BACKENDS:
        if backend.supports(obj):
            return backend

    raise TypeError(
        f"No native serialization backend is available for artifact {name!r} "
        f"of type {type(obj)!r}."
    )


def _load_native_artifact_backend(backend_name):
    """Return native artifact backend for loading by backend name."""
    for backend in _NATIVE_ARTIFACT_BACKENDS:
        if backend.backend == backend_name:
            return backend

    raise ValueError(
        f"No native artifact backend is available for backend {backend_name!r}."
    )


class _NativeArtifactStore:
    """Store native serialization artifacts inside an sktime save bundle."""

    def __init__(self, artifact_root):
        self.artifact_root = artifact_root
        self.index = {}

    def save(self, name, obj, *, estimator):
        """Save a native artifact to the store."""
        artifact_path = self.artifact_root / name
        artifact_path.mkdir(parents=True)
        backend = _save_native_artifact_backend(obj, name=name)
        backend.save(
            obj,
            artifact_path,
            estimator=estimator,
            name=name,
        )
        cls = type(obj)
        self.index[name] = {
            "backend": backend.backend,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "path": name,
        }
        return self.index[name]

    def save_index(self):
        """Save native artifact index."""
        if len(self.index) == 0:
            return

        import json

        self.artifact_root.mkdir(exist_ok=True)
        with open(self.artifact_root / "index.json", "w", encoding="utf-8") as file:
            json.dump(self.index, file, indent=2)


class _SerializationMixin:
    """Mixin containing serialization API for sktime base objects."""

    def _remove_attrs_from_pickle(self, path):
        """Temporarily remove attributes that are stored outside _obj."""
        skip = self.get_tag("serialization:skip", ())
        native_artifacts = self.get_tag("serialization:native_artifacts", ())
        attrs_to_remove = skip if path is None else (*skip, *native_artifacts)
        removed_attrs = {}

        for name in attrs_to_remove:
            if name in self.__dict__:
                removed_attrs[name] = self.__dict__.pop(name)

        return removed_attrs

    def _save_native_artifacts(self, path):
        """Save native artifacts for self if it opts into them."""
        native_artifacts = self.get_tag("serialization:native_artifacts", ())

        if not native_artifacts:
            return

        store = _NativeArtifactStore(path)

        for name in native_artifacts:
            artifact = getattr(self, name, None)
            if artifact is not None:
                store.save(name, artifact, estimator=self)

        store.save_index()

    def _load_native_artifacts(self, path):
        """Load native artifacts from an extracted save bundle."""
        import json

        index_path = path / "index.json"
        if not index_path.exists():
            return

        with open(index_path, encoding="utf-8") as file:
            index = json.load(file)

        for name, record in index.items():
            backend = _load_native_artifact_backend(record["backend"])
            artifact_path = path / record["path"]
            artifact = backend.load(
                artifact_path,
                record,
                estimator=self,
                name=name,
            )
            setattr(self, name, artifact)

    def _get_serializer(self, serialization_format):
        """Return serialization module for the given serialization format."""
        import pickle

        from skbase.utils.dependencies import _check_soft_dependencies

        if serialization_format not in SERIALIZATION_FORMATS:
            raise ValueError(
                f"The provided `serialization_format`='{serialization_format}' "
                "is not yet supported. The possible formats are: "
                f"{SERIALIZATION_FORMATS}."
            )

        if serialization_format == "cloudpickle":
            _check_soft_dependencies("cloudpickle", severity="error")
            import cloudpickle

            return cloudpickle

        return pickle

    def _get_save_path(self, path):
        """Check path and return path used for saving."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        if path is not None and not isinstance(path, (str, Path)):
            raise TypeError(
                "`path` is expected to either be a string or a Path object "
                f"but found of type:{type(path)}."
            )

        save_path = TemporaryDirectory().name if path is None else path
        save_path = Path(save_path) if isinstance(save_path, str) else save_path

        return save_path

    def save(self, path=None, serialization_format="pickle"):
        """Save serialized self to bytes-like object or to (.zip) file.

        Behaviour:

        * if ``path`` is None, returns an in-memory serialized self
        * if ``path`` is a file location, stores self at that location as a zip file

        saved files are zip files with following contents:

        * ``_metadata`` - contains class of self, i.e., ``type(self)``
        * ``_obj`` - serialized self
        * ``_artifacts/`` - optional framework-native model artifacts

        See :ref:`serialization_ref` for archive and artifact layouts.

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
            This setting applies to ``_metadata`` and ``_obj``; native artifact
            backends always use their framework-specific formats.

        Returns
        -------
        if ``path`` is None - in-memory serialized self
        if ``path`` is file location - ZipFile with reference to the file
        """
        import shutil
        from zipfile import ZipFile

        serializer = self._get_serializer(serialization_format)
        save_path = self._get_save_path(path)
        removed_attrs = self._remove_attrs_from_pickle(save_path)
        native_artifacts = self.get_tag("serialization:native_artifacts", ())

        try:
            if path is None and not native_artifacts:
                serial = serializer.dumps(self)
                return (type(self), serial)

            save_path.mkdir()
            with open(save_path / "_metadata", "wb") as file:
                serializer.dump(type(self), file)
            with open(save_path / "_obj", "wb") as file:
                serializer.dump(self, file)
        finally:
            self.__dict__.update(removed_attrs)

        native_save_path = save_path / "_artifacts"
        native_save_path.mkdir()
        self._save_native_artifacts(native_save_path)
        if not any(native_save_path.iterdir()):
            native_save_path.rmdir()

        shutil.make_archive(base_name=save_path, format="zip", root_dir=save_path)
        zip_path = save_path.with_name(f"{save_path.stem}.zip")
        shutil.rmtree(save_path)

        if path is None:
            with open(zip_path, "rb") as file:
                serial = file.read()
            zip_path.unlink()
            return (type(self), serial)

        return ZipFile(zip_path)

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
        from io import BytesIO
        from zipfile import is_zipfile

        if is_zipfile(BytesIO(serial)):
            return cls.load_from_path(BytesIO(serial))
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
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from zipfile import ZipFile

        with TemporaryDirectory() as tmpdir:
            with ZipFile(serial, "r") as file:
                file.extractall(tmpdir)

            path = Path(tmpdir)
            with open(path / "_obj", "rb") as file:
                obj = pickle.load(file)

            obj._load_native_artifacts(path / "_artifacts")
            return obj


def load(serial):
    """Load an object either from in-memory object or from a file location.

    Parameters
    ----------
    serial : serialized container (tuple), str (path), or Path object (reference)
        if serial is a tuple (serialized container):
            Contains two elements: the object's class and serialized bytes returned
            by ``save()``. The bytes can be a pickle stream or an in-memory ZIP
            archive containing native artifacts.
        if serial is a string (path reference):
            The name of the file without the extension, for e.g: if the file
            is `estimator.zip`, `serial='estimator'`. It can also represent a
            path, for eg: if location is `home/stored/models/estimator.zip`
            then `serial='home/stored/models/estimator'`.
        if serial is a Path object (path reference):
            `serial` then points to the `.zip` file into which the
            object was stored using class method `.save()` of an estimator.

        See :ref:`serialization_ref` for in-memory, archive, and native artifact
        layouts.

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
