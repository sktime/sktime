"""Mixin to bridge sklearn version differences."""


class _SklVersionBridgeMixin:
    """Mixin to handle differences in sklearn versions.

    This mixin provides a method to validate data that works across different
    versions of scikit-learn, specifically for versions 1.5 and lower.
    """

    def _sklearn_15_or_lower(self):
        """Check if the installed scikit-learn version is 1.5 or lower."""
        from sktime.utils.dependencies import _check_soft_dependencies

        return _check_soft_dependencies("scikit-learn<1.6", severity="none")

    def _validate_data_version_safe(self, **kwargs):
        """Validate data using the version-safe method."""
        if self._sklearn_15_or_lower():
            return self._validate_data(**kwargs)
        else:
            from sklearn.utils.validation import validate_data

            return validate_data(self, **kwargs)
