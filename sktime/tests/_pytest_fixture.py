# __all__ = ["lock", "serial"]
# import contextlib
# import os

# import filelock
# import pytest


# @pytest.fixture(scope="session")
# def lock(tmp_path_factory):
#     base_temp = tmp_path_factory.getbasetemp()
#     lock_file = base_temp.parent / "serial.lock"
#     yield filelock.FileLock(lock_file=str(lock_file))
#     with contextlib.suppress(OSError):
#         os.remove(path=lock_file)


# @pytest.fixture()
# def serial(lock):
#     with lock.acquire(poll_interval=0.1):
#         yield
