# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Quick crafting methods to build an object from string and registry.

craft(spec)
    craft an object or estimator from string

deps(spec)
    retrieves all dependencies required to craft str, in PEP440 format

The ``craft`` function is a pair to ``str`` coercion, the two can be seen as
deserialization/serialization counterparts to each other.

That is,
spec = str(my_est)
new_est = craft(spec)

will have the same effect as new_est = spec.clone()
"""

__author__ = ["fkiraly"]

import ast
import re

from sktime.registry._lookup import all_estimators
from sktime.registry._lookup_sklearn import _all_sklearn_estimators


def _extract_class_names(spec):
    """Get all maximal alphanumeric substrings that start with a capital letter.

    Parameters
    ----------
    spec : str (any)

    Returns
    -------
    cls_name_list : list of str
        list of all maximal alphanumeric substrings starting with a capital in ``spec``
        excluding reserved expressions True, False (if they occur)
    """
    # class names are all UpperCamelCase alphanumeric strings
    # which is the same as maximal substrings starting with a capital
    pattern = r"\b([A-Z][A-Za-z0-9_]*)\b"
    cls_name_list = re.findall(pattern, spec)

    # we need to exclude expressions that look like classes per the regex
    # but aren't
    EXCLUDE_LIST = ["True", "False", "None"]
    cls_name_list = [x for x in cls_name_list if x not in EXCLUDE_LIST]

    return cls_name_list


def craft(spec, safe=False):
    """Instantiate an object from the specification string.

    The ``craft`` utility can be used to deserialize an estimator specification string,
    including composites such as pipelines.

    It takes a specification string and returns an ``sktime`` estimator, class,
    or object, corresponding to that string.

    Specification strings can be:

    * simple expressions such as ``"NaiveForecaster"`` or ``"NaiveForecaster(sp=2)"``
    * compositions such as ``"Deseasonalizer() * NaiveForecaster()"``
    * a longer block of code, closing with a return statement, e.g., the string block

    .. code-block:: python

        deseason = Deseasonalizer()
        naive = NaiveForecaster()
        return deseason * naive

    The ``craft`` utility is useful as a serialization / deserialization pair,
    together with ``str`` coercion (or commandline printing) of an
    unfitted estimator.

    ``craft`` recognizes estimators present in ``sktime`` and ``scikit-learn``,
    and base python (built-in types and functions).

    If ``safe=True`` mode is enabled, only simple propositional expressions are allowed.

    The accepted "safe" grammar is intentionally small:
    
    .. code-block:: text

        expression ::= NAME | NAME "(" arguments ")"

        arguments  ::= positional_argument | keyword_argument| arguments "," arguments

        positional_argument ::= expression | CONSTANT
        keyword_argument    ::= NAME "=" (expression | CONSTANT)

    Where

    .. code-block:: text

        NAME       ::= valid Python identifier, object name in ``sktime`` or ``sklearn``
        CONSTANT   ::= literal value (e.g., number, string, boolean)

    This permits simple constructor calls such as ``A(a=42)``,
    or nested constructor calls such as ``A(a=42, b=B("test"))``, and only such calls.

    In particular, does not permit attribute access, lambdas, comprehensions, operators,
    imports, assignments, function calls through arbitrary expressions, etc,
    which are "unsafe" in the sense of allowing arbitrary code injection.

    Parameters
    ----------
    spec : str, sktime/skbase compatible object specification
        i.e., a string that executes to construct an object if all imports were present
        imports inferred are of any classes in the scope of ``all_estimators``

        * option 1: a string that evaluates to an estimator
        * option 2: a sequence of assignments in valid python code,
          with the object to be defined preceded by a "return".
          assignments can use names of classes as if all imports were present.
          Option 2 is not available in ``safe=True`` mode.

    safe : bool, optional (default=False)
        whether to enforce safe expressions according to the safe specification rules.
    
        * if True, only allow safe expressions according to the safe specification
          rules, see above for the exact rules.
        * if False, allow all expressions (default behavior).

    Returns
    -------
    obj : skbase BaseObject descendant, constructed from ``spec``
        this will have the property that ``spec == str(obj)`` (up to formatting)

    Examples
    --------
    >>> from sktime.registry import craft

    Example 1: simple estimator

    * serialized as the string ``"NaiveForecaster(sp=2)"``
    * deserialized as the estimator ``NaiveForecaster(sp=2)``

    >>> spec = "NaiveForecaster(sp=2)"
    >>> est = craft(spec)
    >>> print(est)
    NaiveForecaster(sp=2)

    Example 2: composite estimator

    * serialized as the string ``"Deseasonalizer() * NaiveForecaster()"``
    * deserialized as the estimator ``Deseasonalizer() * NaiveForecaster()``,
      same as ``ForecastingPipeline([Deseasonalizer(), NaiveForecaster()])``

    >>> spec = "Deseasonalizer() * NaiveForecaster()"
    >>> est = craft(spec)

    Example 3: longer code block

    * serialized as a code block with assignments and return statement
    * deserialized as the estimator defined in the return statement

    >>> spec = '''
    ... deseason = Deseasonalizer()
    ... naive = NaiveForecaster()
    ... return deseason * naive
    ... '''
    >>> est = craft(spec)
    """
    # retrieve all estimators from sktime and sklearn for namespace resolution
    register_sktime = dict(all_estimators())  # noqa: F841
    register_sklearn = dict(_all_sklearn_estimators())  # noqa: F841
    register = {**register_sklearn, **register_sktime}

    # Parse the specification once.
    # Both safe and unsafe modes operate on the resulting AST.
    tree = ast.parse(spec, mode="exec")

    # safe mode: validate the AST before evaluation
    if safe and not _validate_ast(tree, register):
        raise ValueError(
            "Error in craft utility: unsafe or invalid specification supplied: "
            f"{spec}"
        )

    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        expr = tree.body[0].value
        expr_tree = ast.Expression(body=expr)
        ast.fix_missing_locations(expr_tree)

        try:
            obj = eval(
                compile(expr_tree, "<craft>", "eval"),
                {"__builtins__": {}},
                register,
            )
        except Exception as e:
            if safe:
                raise ValueError(
                    "Error in craft utility: failed to evaluate specification: "
                    f"{spec}"
                ) from e

        return obj

    elif safe:
        raise ValueError(
            "Error in craft utility: safe mode requires a single expression, "
            f"but got a block of code: {spec}"
        )

    else:
        # unsafe mode: attempt to execute the specification directly
        from textwrap import indent

        spec_fun = indent(spec, "    ")
        spec_fun = (
            """
def build_obj():
        """
            + spec_fun
        )

        exec(spec_fun, register, register)
        obj = eval("build_obj()", register, register)

    return obj


def _validate_ast(tree, register):
    """Validate that ``spec`` contains only safe constructor expressions.

    The accepted grammar is intentionally small:

        expression ::= NAME | NAME "(" arguments ")"

        arguments  ::= positional_argument | keyword_argument| arguments "," arguments

        positional_argument ::= expression | CONSTANT
        keyword_argument    ::= NAME "=" (expression | CONSTANT)

    This permits nested constructor calls such as:

        A(a=42, b=B("test"))

    and only such calls.
    In particular, does not permit attribute access, lambdas, comprehensions, operators,
    imports, assignments, function calls through arbitrary expressions, etc.

    Parameters
    ----------
    tree : ast.Expression or ast.AST
        The abstract syntax tree of the specification to validate.
    register : dict
        The registry of allowed names (estimators, methods, variables) for validation.

    Returns
    -------
    bool
        True if the AST is valid according to the safe spec rules, False otherwise.
    """
    reg = register  # for shorter expressions below

    if isinstance(tree, ast.Module):
        return (
            len(tree.body) == 1
            and isinstance(tree.body[0], ast.Expr)
            and _validate_ast(tree.body[0].value, reg)
        )

    # Only names explicitly supplied in the estimator registry are
    # available for variables. In particular, don't permit dunder names.
    if isinstance(tree, ast.Name):
        return tree.id in reg and not tree.id.startswith("__")

    # Constants are safe as values.
    # In particular, this allows strings, numbers, booleans, None, etc.
    if isinstance(tree, ast.Constant):
        return True

    # binary and unary operators (dunder operations)
    # are allowed, since this is how we can build pipelines
    allowed_binops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.MatMult,
        ast.BitOr,
        ast.BitAnd,
        ast.BitXor,
        ast.LShift,
        ast.RShift,
    )

    allowed_unaryops = (
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,
    )

    if isinstance(tree, ast.BinOp):
        if not isinstance(tree.op, allowed_binops):
            return False
        return _validate_ast(tree.left, reg) and _validate_ast(tree.right, reg)

    if isinstance(tree, ast.UnaryOp):
        if not isinstance(tree.op, allowed_unaryops):
            return False
        return _validate_ast(tree.operand, reg)

    if isinstance(tree, ast.Call):
        # Only direct calls such as A(...) are allowed.
        #
        # This deliberately rejects:
        #   obj.A(...)
        #   getattr(...)(...)
        #   (lambda: ...)(...)
        if not isinstance(tree.func, ast.Name):
            return False

        if not _validate_ast(tree.func, reg):
            return False

        for arg in tree.args:
            if not _validate_ast(arg, reg):
                return False

        for kw in tree.keywords:
            # **kwargs is represented by keyword.arg == None.
            if kw.arg is None:
                return False
            if not _validate_ast(kw.value, reg):
                return False

        return True

    # the above is an exhaustive list of what is allowed
    # hence, if we reach this point, the AST contains an unsupported node type
    return False


def deps(spec, include_test_deps=False):
    """Get PEP 440 dependency requirements for a craft spec.

    The ``deps`` utility returns a PEP 440 compatible requirement string
    required to construct the deserialization via ``craft``.

    In case the spec includes estimators with disjunctions in their requirement
    specifications, the first disjunctive requirement is returned, i.e.,
    any disjunctions are resolved by picking the first dependency in the disjunction.

    Parameters
    ----------
    spec : str, sktime/skbase compatible object specification
        i.e., a string that executes to construct an object if all imports were present.
        imports inferred are of any classes in the scope of ``all_estimators``

        * option 1: a string that evaluates to an estimator
        * option 2: a sequence of assignments in valid python code,
          with the object to be defined preceded by a "return".
          assignments can use names of classes as if all imports were present

    include_test_deps : bool, default=False
        whether to include dependencies tagged as
        "tests:python_dependencies" in addition to "python_dependencies"

    Returns
    -------
    reqs : list of str
        each str is PEP 440 compatible requirement string for craft(spec)
        if spec has no requirements, return is [], the length 0 list
    """
    register = dict(all_estimators())

    dep_strs = []

    for x in _extract_class_names(spec):
        if x not in register.keys():
            raise RuntimeError(
                f"class {x} is required to build spec, but was not found "
                "in all_estimators scope"
            )
        cls = register[x]

        new_deps = cls.get_class_tag("python_dependencies")

        def _resolve_disjunctions(dep):
            """Resolve disjunctions in dependencies by picking first."""
            if isinstance(dep, list):
                return dep[0]  # pick first dependency in disjunction
            return dep

        def _coerce_dep_strs(dep):
            """Coerce dependency tag value to list of strings."""
            if dep is None:
                return []
            elif isinstance(dep, str) and len(dep) > 0:
                return [dep]
            elif isinstance(dep, str) and len(dep) == 0:
                return []
            elif isinstance(dep, list):
                return [_resolve_disjunctions(d) for d in dep]
            else:
                return dep

        new_deps = cls.get_class_tag("python_dependencies")
        dep_strs += _coerce_dep_strs(new_deps)

        if include_test_deps:
            test_deps = cls.get_class_tag("tests:python_dependencies")
            dep_strs += _coerce_dep_strs(test_deps)

        reqs = list(set(dep_strs))

    return reqs


def imports(spec):
    """Get import code block for a craft spec.

    The ``imports`` utility returns a full python import block,
    as a string block, required for importing all estimators and objects occurring
    in the ``spec`` block.

    The ``imports`` utility recognizes ``sktime`` and ``scikit-learn`` objects.

    Parameters
    ----------
    spec : str, sktime/skbase compatible object specification
        i.e., a string that executes to construct an object if all imports were present
        imports inferred are of any classes in the scope of ``all_estimators``

        * option 1: a string that evaluates to an estimator
        * option 2: a sequence of assignments in valid python code,
          with the object to be defined preceded by a "return".
          assignments can use names of classes as if all imports were present

    Returns
    -------
    import_str : str
        python code consisting of all import statements required for spec
        imports cover object/estimator classes found as sub-strings of spec
    """
    register = dict(all_estimators())

    import_strs = []

    for x in _extract_class_names(spec):
        if x not in register.keys():
            raise RuntimeError(
                f"class {x} is required to build spec, but was not found "
                "in all_estimators scope"
            )
        cls = register[x]

        import_module = _get_public_import(cls.__module__)
        import_str = f"from {import_module} import {x}"
        import_strs += [import_str]

    if len(import_strs) == 0:
        imports_str = ""
    else:
        imports_str = "\n".join(sorted(import_strs))

    return imports_str


def _get_public_import(module_path: str) -> str:
    """Get the public import path from full import path.

    Removes everything from the first private submodule (starting with '_') onwards.
    """
    parts = module_path.split(".")
    for i, part in enumerate(parts):
        if part.startswith("_"):
            return ".".join(parts[:i])  # Keep only part before first private submodule
    return module_path  # Return the original path if no private submodules are found


def _r_deps(spec, include_test_deps=False):
    """Get R dependency requirements for a craft spec.

    The ``_r_deps`` utility returns a list of R requirement strings
    required to construct the deserialization via ``craft``.

    In case the spec includes estimators with disjunctions in their requirement
    specifications, the first disjunctive requirement is returned, i.e.,
    any disjunctions are resolved by picking the first dependency in the disjunction.

    Parameters
    ----------
    spec : str, sktime/skbase compatible object specification
        i.e., a string that executes to construct an object if all R
        dependencies were present.
        dependencies inferred are of any classes in the scope of ``all_estimators``

        * option 1: a string that evaluates to an estimator
        * option 2: a sequence of assignments in valid python code,
          with the object to be defined preceded by a "return".
          assignments can use names of classes as if all imports were present

    include_test_deps : bool, default=False
        whether to include dependencies tagged as
        "tests:r_dependencies" in addition to "r_dependencies"

    Returns
    -------
    reqs : list of str
        each str is R compatible requirement string for craft(spec)
        if spec has no requirements, return is [], the length 0 list
    """
    register = dict(all_estimators())

    dep_strs = []

    for x in _extract_class_names(spec):
        if x not in register.keys():
            raise RuntimeError(
                f"class {x} is required to build spec, but was not found "
                "in all_estimators scope"
            )
        cls = register[x]

        new_deps = cls.get_class_tag("r_dependencies")

        def _resolve_disjunctions(dep):
            """Resolve disjunctions in dependencies by picking first."""
            if isinstance(dep, list):
                return dep[0]  # pick first dependency in disjunction
            return dep

        def _coerce_dep_strs(dep):
            """Coerce dependency tag value to list of strings."""
            if dep is None:
                return []
            elif isinstance(dep, str) and len(dep) > 0:
                return [dep]
            elif isinstance(dep, str) and len(dep) == 0:
                return []
            elif isinstance(dep, list):
                return [_resolve_disjunctions(d) for d in dep]
            else:
                return dep

        new_deps = cls.get_class_tag("r_dependencies")
        dep_strs += _coerce_dep_strs(new_deps)

        if include_test_deps:
            test_deps = cls.get_class_tag("tests:r_dependencies")
            dep_strs += _coerce_dep_strs(test_deps)

        reqs = list(set(dep_strs))

    return reqs
