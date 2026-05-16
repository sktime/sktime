from pathlib import Path
import ast
import traceback

statements_need_to_pass = []
for path in Path("./sktime/transformations/").rglob("*.py"):
    if "tests" in path.parts:
        continue

    all_imports_to_test = []
    with open(path) as f:
        file_content = f.read()

    if path.name == "__init__.py":
        import_path = ".".join(path.parent.parts)
        is_init = True
    else:
        import_path = ".".join(path.with_suffix("").parts)
        is_init = False

    tree = ast.parse(file_content)
    class_or_functions = [
        node.name
        for node in tree.body  # tree.body == top-level only; nested are inside their parent
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    for name in class_or_functions:
        if name.startswith("_"):
            continue
        all_imports_to_test.append(
            f"from {import_path} import {name}; assert {name}.__name__ == '{name}'"
        )
        if is_init:
            continue
        import_sub_path = import_path.rsplit(".", 1)[0]
        all_imports_to_test.extend([
            f"from {import_sub_path} import {name}; assert {name}.__name__ == '{name}'",
            f"from {import_sub_path} import {name} as A; "
            f"from {import_path} import {name} as B; assert A is B",
        ])
    for code in all_imports_to_test:
        try:
            exec(compile(code, "<gen>", "exec"), {})
        except Exception:
            print("FAIL", code)
            traceback.print_exc()
        else:
            print("OK  ", code)
            statements_need_to_pass.append(code)

    # 1. leaf from-import (what pickle uses)
    # "from sktime.transformations.series.detrend.mstl import MSTL; assert MSTL.__name__ == 'MSTL'",
    # 2. submodule from parent (breaks with naive sys.modules tricks)
    # "from sktime.transformations.series.detrend import mstl; assert mstl.MSTL.__name__ == 'MSTL'",
    # 3. re-export from detrend/__init__.py (must be the *same* class)
    # "from sktime.transformations.series.detrend import MSTL as A; "
    # "from sktime.transformations.series.detrend.mstl import MSTL as B; assert A is B",
with open("bugs/imports/all_import_statements.txt", "w") as f:
    for statement in statements_need_to_pass:
        f.write(statement + "\n")
