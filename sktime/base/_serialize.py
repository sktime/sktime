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

    def _get_class(self, record):
        """Return class from artifact record."""
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
        meta = {}

        try:
            device = obj.device
        except Exception:
            device = None

        if device is not None:
            meta["device"] = str(device)

        return meta

    def load(self, path, record, *, estimator, name):
        """Load a transformers model using from_pretrained."""
        from warnings import warn

        cls = self._get_class(record)

        meta = record["meta"]
        device = meta.get("device")
        if device is None:
            return cls.from_pretrained(path)

        try:
            return cls.from_pretrained(path, device_map=device)
        except Exception as exc:
            warn(
                f"Could not load native artifact {name!r} on saved device "
                f"{device!r}. Falling back to CPU. Original error: {exc}",
                stacklevel=2,
            )
            return cls.from_pretrained(path, device_map="cpu")


class _TorchArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for torch modules."""

    backend = "torch"

    def _get_constructor_params(self, obj):
        """Get constructor parameters from object attributes."""
        import inspect

        init_params = {}
        tuple_params = []
        signature = inspect.signature(type(obj).__init__)

        for name, param in signature.parameters.items():
            if name == "self" or param.kind in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                continue

            if not hasattr(obj, name):
                continue

            value = getattr(obj, name)
            if isinstance(value, tuple):
                tuple_params.append(name)

            init_params[name] = value

        return init_params, tuple_params

    def _get_device(self, obj):
        """Return device for a torch module if it can be inferred."""
        return str(next(obj.parameters()).device)

    def supports(self, obj):
        """Return whether object looks like a torch.nn.Module."""
        return any(
            cls.__name__ == "Module" and cls.__module__ == "torch.nn.modules.module"
            for cls in type(obj).__mro__
        )

    def dump(self, obj, path, *, estimator, name):
        """Dump a torch module using state_dict."""
        import torch

        torch.save(obj.state_dict(), path / "state_dict.pt")
        init_params, tuple_params = self._get_constructor_params(obj)
        meta = {
            "init_params": init_params,
        }

        if tuple_params:
            meta["tuple_params"] = tuple_params

        device = self._get_device(obj)
        if device is not None:
            meta["device"] = device

        return meta

    def load(self, path, record, *, estimator, name):
        """Load a torch module from state_dict."""
        from warnings import warn

        import torch

        cls = self._get_class(record)

        meta = record["meta"]
        init_params = meta.get("init_params", {})
        for param in meta.get("tuple_params", ()):
            init_params[param] = tuple(init_params[param])

        try:
            obj = cls(**init_params)
        except Exception as exc:
            raise TypeError(
                f"Could not reconstruct native torch artifact {name!r} "
                f"of class {record['class']!r} from stored constructor "
                f"parameters {init_params!r}."
            ) from exc

        state_dict = torch.load(path / "state_dict.pt", map_location="cpu")
        obj.load_state_dict(state_dict)

        device = meta.get("device")
        if device is None:
            return obj

        try:
            return obj.to(device)
        except Exception as exc:
            warn(
                f"Could not load native artifact {name!r} on saved device "
                f"{device!r}. Falling back to CPU. Original error: {exc}",
                stacklevel=2,
            )
            return obj.to("cpu")


_NATIVE_ARTIFACT_BACKENDS = [
    _TransformersArtifactBackend(),
    _TorchArtifactBackend(),
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


def _get_native_artifact_backend_by_name(backend_name):
    """Return native artifact backend by backend name."""
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

    def dump(self, name, obj, *, estimator):
        """Dump a native artifact to the store."""
        artifact_path = self.artifact_root / name
        artifact_path.mkdir(parents=True)
        backend = _get_native_artifact_backend(obj, name=name)
        meta = backend.dump(
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
            "meta": meta,
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

    def _remove_attrs_from_pickle(self, path):
        """Temporarily remove attributes that are stored outside _obj."""
        # TODO: update this when native memory serialization is implemented
        skip = self.get_tag("serialization:skip", ())
        native_artifacts = self.get_tag("serialization:native_artifacts", ())
        attrs_to_remove = skip if path is None else (*skip, *native_artifacts)
        removed_attrs = {}

        for name in attrs_to_remove:
            if name in self.__dict__:
                removed_attrs[name] = self.__dict__.pop(name)

        return removed_attrs

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

    def _restore_native_artifacts(self, path):
        """Restore native artifacts from an extracted save bundle."""
        import json

        index_path = path / "_artifacts" / "index.json"
        if not index_path.exists():
            return

        with open(index_path, encoding="utf-8") as file:
            index = json.load(file)

        for name, record in index.items():
            backend = _get_native_artifact_backend_by_name(record["backend"])
            artifact_path = path / "_artifacts" / record["path"]
            artifact = backend.load(
                artifact_path,
                record,
                estimator=self,
                name=name,
            )
            setattr(self, name, artifact)

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

        removed_attrs = self._remove_attrs_from_pickle(path)

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

        self.__dict__.update(removed_attrs)
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
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from zipfile import ZipFile

        with TemporaryDirectory() as tmpdir:
            with ZipFile(serial, "r") as file:
                file.extractall(tmpdir)

            path = Path(tmpdir)
            with open(path / "_obj", "rb") as file:
                obj = pickle.load(file)

            obj._restore_native_artifacts(path)
            return obj


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
