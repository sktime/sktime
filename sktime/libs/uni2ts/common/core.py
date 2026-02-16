# noqa: D100
#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


# noqa: D103
def abstract_class_property(*names: str) -> Callable[[type[T], ...], type[T]]:
    """
    Check that each attribute is defined in the subclass.

    Parameters
    ----------
    names: str

    Returns
    -------
    type[T]
    """

    def _func(cls: type[T]) -> type[T]:
        original_init_subclass = cls.__init_subclass__

        def _init_subclass(_cls, **kwargs):
            # The default implementation of __init_subclass__ takes no
            # positional arguments, but a custom implementation does.
            # If the user has not reimplemented __init_subclass__ then
            # the first signature will fail and we try the second.
            try:
                original_init_subclass(_cls, **kwargs)
            except TypeError:
                original_init_subclass(**kwargs)

            # Check that each attribute is defined.
            for name in names:
                if not hasattr(_cls, name):
                    raise NotImplementedError(
                        f"{name} has not been defined for {_cls.__name__}"
                    )
                if getattr(_cls, name, NotImplemented) is NotImplemented:
                    raise NotImplementedError(
                        f"dataset_list has not been defined for {_cls.__name__}"
                    )

        cls.__init_subclass__ = classmethod(_init_subclass)
        return cls

    return _func
