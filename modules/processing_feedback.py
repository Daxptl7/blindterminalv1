"""Bounded background processing with prompt cancellation and inherited policy."""
import threading
import time
from contextvars import ContextVar
from modules.privacy_policy import session_thread


class ProcessingCancelled(Exception):
    pass


# Abandoned third-party calls cannot create an unbounded backlog. They never
# own a camera or publish callbacks; their only output is a private result box.
_slots = threading.BoundedSemaphore(2)
_operation = ContextVar("reader_operation", default=(None, None))


def check_processing():
    stop_check, deadline = _operation.get()
    if stop_check and stop_check():
        raise ProcessingCancelled()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Processing deadline reached.")


def run_with_feedback(operation, speak=None, stop_check=None,
                      message="Still processing. Please wait a little longer.",
                      first_delay=8.0, interval=20.0, max_updates=2,
                      timeout=90.0):
    if stop_check and stop_check():
        raise ProcessingCancelled()
    if not _slots.acquire(blocking=False):
        raise RuntimeError("The reader is finishing previous work. Please try again shortly.")
    done = threading.Event()
    result, errors = [], []
    started = time.monotonic()

    def work():
        token = _operation.set((stop_check, started + timeout))
        try:
            check_processing()
            result.append(operation())
        except BaseException as error:
            errors.append(error)
        finally:
            _operation.reset(token)
            _slots.release()
            done.set()

    thread = session_thread(work, daemon=True, name="reader_operation")
    try:
        thread.start()
    except BaseException:
        _slots.release()
        raise
    next_update = started + first_delay
    updates = 0
    while not done.wait(0.05):
        if stop_check and stop_check():
            raise ProcessingCancelled()
        now = time.monotonic()
        if now - started >= timeout:
            raise TimeoutError("Reader processing timed out.")
        if speak and updates < max_updates and now >= next_update:
            speak(message)
            updates += 1
            next_update = now + interval
    if stop_check and stop_check():
        raise ProcessingCancelled()
    if errors:
        raise errors[0]
    return result[0]
