"""Mixin to bridge sklearn version differences."""


class _SklVersionBridgeMixin:
    """Mixin to handle differences in sklearn versions.

    This mixin provides a method to validate data that works across different
    versions of scikit-learn, specifically for versions 1.5 and lower.
    """

    def _sklearn_15_or_lower(self):
        """Check if the installed scikit-learn version is 1.5 or lower."""
        from skbase.utils.dependencies import _check_soft_dependencies

        return _check_soft_dependencies("scikit-learn<1.6", severity="none")

    def _validate_data_version_safe(self, **kwargs):
        """Validate data using the version-safe method."""
        from skbase.utils.dependencies import _check_soft_dependencies

        # from sklearn 1.6 onwards, "force_all_finite" changed to "ensure_all_finite"
        if "force_all_finite" in kwargs:
            val = kwargs.pop("force_all_finite")
            if _check_soft_dependencies("scikit-learn>=1.6", severity="none"):
                kwargs["ensure_all_finite"] = val
            else:
                kwargs["force_all_finite"] = val

        # from sklearn 1.5 onwards, the location of the validate_data function changed
        if _check_soft_dependencies("scikit-learn<1.6", severity="none"):
            return self._validate_data(**kwargs)
        else:
            from sklearn.utils.validation import validate_data

            return validate_data(self, **kwargs)
