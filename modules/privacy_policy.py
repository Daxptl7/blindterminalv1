"""Session policy shared by capture, processing, storage and queued speech."""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
import threading
import logging


@dataclass(frozen=True)
class ProcessingPolicy:
    private: bool = False
    allow_cloud: bool = True
    allow_persistence: bool = True


PUBLIC = ProcessingPolicy()
PRIVATE = ProcessingPolicy(True, False, False)
_policy = ContextVar("blindassist_policy", default=PUBLIC)
_session = ContextVar("blindassist_session", default=False)


class PrivateLogFilter(logging.Filter):
    """Defense in depth for third-party exception details during private work."""
    def filter(self, record):
        if current_policy().private:
            record.msg = "Private operation: %s"
            record.args = (record.levelname,)
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


def current_policy():
    return _policy.get()


def session_active():
    return _session.get()


@contextmanager
def processing_session(policy):
    token = _policy.set(policy)
    active = _session.set(True)
    try:
        yield policy
    finally:
        _session.reset(active)
        _policy.reset(token)


def session_thread(target, **kwargs):
    """Threads must inherit the caller's privacy policy explicitly."""
    context = copy_context()
    return threading.Thread(target=lambda: context.run(target), **kwargs)
