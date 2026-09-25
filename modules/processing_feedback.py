"""Run slow work with bounded spoken progress and no late speech callbacks."""
import threading


class ProcessingCancelled(Exception):
    pass


def run_with_feedback(operation, speak=None, stop_check=None,
                      message="Still processing. Please wait a little longer.",
                      first_delay=8.0, interval=20.0, max_updates=2):
    """Only the caller speaks; worker completion cannot enqueue stale prompts.

    In-flight third-party work must finish or reach its own timeout on cancel.
    Its result is discarded, and no further progress speech is generated.
    """
    if stop_check and stop_check():
        raise ProcessingCancelled()
    if speak is None:
        result = operation()
        if stop_check and stop_check():
            raise ProcessingCancelled()
        return result
    done = threading.Event()
    result = []
    errors = []

    def work():
        try:
            result.append(operation())
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    # Event waits wake immediately when work completes. No periodic speech
    # thread survives into document playback or the next mode.
    delay = first_delay
    updates = 0
    while not done.wait(delay):
        if not (stop_check and stop_check()) and updates < max_updates:
            speak(message)
            updates += 1
        delay = interval
    if stop_check and stop_check():
        raise ProcessingCancelled()
    if errors:
        raise errors[0]
    return result[0]
