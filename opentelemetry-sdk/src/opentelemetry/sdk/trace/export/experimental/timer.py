import abc
import threading
import typing


class TimerABC(abc.ABC):
    """
    An interface extracted from PeriodicTimer so alternative implementations can be used for testing.
    """

    @abc.abstractmethod
    def set_callback(self, cb) -> None:
        pass

    @abc.abstractmethod
    def start(self) -> None:
        pass

    @abc.abstractmethod
    def poke(self) -> None:
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        pass

    @abc.abstractmethod
    def started(self) -> None:
        pass

    @abc.abstractmethod
    def stopped(self) -> None:
        pass


class PeriodicTimer(TimerABC):
    """
    Executes the passed-in callback on a timer at the specified interval. The callback can be run sooner than the
    interval via the poke() method, which also resets the timer.
    """

    def __init__(
        self,
        interval_sec: int,
        callback: typing.Callable[[], None] = lambda: None,
        daemon: bool = True,
    ):
        self._callback = callback
        self._interval_sec = interval_sec
        self._stop = threading.Event()
        self._poke = threading.Event()
        self._thread = threading.Thread(target=self._work, daemon=daemon)

    def set_callback(self, callback: typing.Callable[[], None]) -> None:
        self._callback = callback

    def start(self) -> None:
        self._thread.start()

    def _work(self) -> None:
        while True:
            self._poke.wait(timeout=self._interval_sec)
            if self._stop.is_set():
                break
            self._callback()
            self._poke.clear()

    def poke(self) -> None:
        """
        This method schedules the callback to be executed immediately instead of waiting for the next timeout. It also
        resets the timer.
        """
        self._poke.set()

    def stop(self) -> None:
        self._stop.set()
        self.poke()  # in case we're waiting for a poke timeout
        self._thread.join()

    def started(self) -> bool:
        return self._thread.is_alive()

    def stopped(self) -> bool:
        return self._stop.is_set()


class ThreadlessTimer(TimerABC):
    """
    For testing/experimentation. Only executes the callback when you run poke().
    """

    def __init__(self):
        self._cb = lambda: None

    def set_callback(self, cb):
        self._cb = cb

    def start(self):
        pass

    def poke(self):
        self._cb()

    def stop(self):
        pass

    def started(self) -> None:
        pass

    def stopped(self) -> None:
        pass