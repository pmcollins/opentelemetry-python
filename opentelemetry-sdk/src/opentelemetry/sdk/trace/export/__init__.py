# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import collections
import logging
import os
import sys
import threading
import time
import typing
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from os import environ, linesep
from time import time_ns
from typing import Optional

from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    Context,
    attach,
    detach,
    set_value,
)
from opentelemetry.sdk.environment_variables import (
    OTEL_BSP_EXPORT_TIMEOUT,
    OTEL_BSP_MAX_EXPORT_BATCH_SIZE,
    OTEL_BSP_MAX_QUEUE_SIZE,
    OTEL_BSP_SCHEDULE_DELAY,
)
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.util._once import Once

_DEFAULT_SCHEDULE_DELAY_MILLIS = 5000
_DEFAULT_MAX_EXPORT_BATCH_SIZE = 512
_DEFAULT_EXPORT_TIMEOUT_MILLIS = 30000
_DEFAULT_MAX_QUEUE_SIZE = 2048
_ENV_VAR_INT_VALUE_ERROR_MESSAGE = (
    "Unable to parse value for %s as integer. Defaulting to %s."
)

logger = logging.getLogger(__name__)


class SpanExportResult(Enum):
    SUCCESS = 0
    FAILURE = 1


class SpanExporter:
    """Interface for exporting spans.

    Interface to be implemented by services that want to export spans recorded
    in their own format.

    To export data this MUST be registered to the :class`opentelemetry.sdk.trace.Tracer` using a
    `SimpleSpanProcessor` or a `BatchSpanProcessor`.
    """

    def export(
        self, spans: typing.Sequence[ReadableSpan]
    ) -> "SpanExportResult":
        """Exports a batch of telemetry data.

        Args:
            spans: The list of `opentelemetry.trace.Span` objects to be exported

        Returns:
            The result of the export
        """

    def shutdown(self) -> None:
        """Shuts down the exporter.

        Called when the SDK is shut down.
        """

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Hint to ensure that the export of any spans the exporter has received
        prior to the call to ForceFlush SHOULD be completed as soon as possible, preferably
        before returning from this method.
        """


class SimpleSpanProcessor(SpanProcessor):
    """Simple SpanProcessor implementation.

    SimpleSpanProcessor is an implementation of `SpanProcessor` that
    passes ended spans directly to the configured `SpanExporter`.
    """

    def __init__(self, span_exporter: SpanExporter):
        self.span_exporter = span_exporter

    def on_start(
        self, span: Span, parent_context: typing.Optional[Context] = None
    ) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        if not span.context.trace_flags.sampled:
            return
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            self.span_exporter.export((span,))
        # pylint: disable=broad-except
        except Exception:
            logger.exception("Exception while exporting Span.")
        detach(token)

    def shutdown(self) -> None:
        self.span_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # pylint: disable=unused-argument
        return True


class _FlushRequest:
    """Represents a request for the BatchSpanProcessor to flush spans."""

    __slots__ = ["event", "num_spans"]

    def __init__(self):
        self.event = threading.Event()
        self.num_spans = 0


_BSP_RESET_ONCE = Once()


class MaturityLevel(Enum):
    UNSPECIFIED = 0
    LEGACY = 1
    STABLE = 2
    BETA = 3

    @staticmethod
    def from_string(v):
        if str in MaturityLevel.__members__:
            return MaturityLevel[v]
        return MaturityLevel.UNSPECIFIED


def _resolve_maturity_level(maturity_level_override: MaturityLevel) -> MaturityLevel:
    if maturity_level_override is not None:
        return maturity_level_override
    env = os.environ.get('OTEL_PYTHON_MATURITY_LEVEL', '').upper()
    return MaturityLevel.from_string(env)


class BatchSpanProcessor(SpanProcessor):

    def __init__(
        self,
        span_exporter: SpanExporter,
        max_queue_size: int = None,
        schedule_delay_millis: float = None,
        max_export_batch_size: int = None,
        export_timeout_millis: float = None,
        maturity_level: MaturityLevel = None,
    ):
        if _resolve_maturity_level(maturity_level) == MaturityLevel.BETA:
            self.delegate = _BatchSpanProcessor2(
                span_exporter,
                max_queue_size,
                schedule_delay_millis,
                max_export_batch_size,
                export_timeout_millis
            )
        else:
            self.delegate = _BatchSpanProcessor1(
                span_exporter,
                max_queue_size,
                schedule_delay_millis,
                max_export_batch_size,
                export_timeout_millis
            )

    def on_start(
        self,
        span: "Span",
        parent_context: Optional[Context] = None,
    ) -> None:
        self.delegate.on_start(span, parent_context)

    def on_end(self, span: "ReadableSpan") -> None:
        self.delegate.on_end(span)

    def shutdown(self) -> None:
        self.delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.delegate.force_flush(timeout_millis)


class _BatchSpanProcessor1(SpanProcessor):
    """Batch span processor implementation.

    `BatchSpanProcessor` is an implementation of `SpanProcessor` that
    batches ended spans and pushes them to the configured `SpanExporter`.

    `BatchSpanProcessor` is configurable with the following environment
    variables which correspond to constructor parameters:

    - :envvar:`OTEL_BSP_SCHEDULE_DELAY`
    - :envvar:`OTEL_BSP_MAX_QUEUE_SIZE`
    - :envvar:`OTEL_BSP_MAX_EXPORT_BATCH_SIZE`
    - :envvar:`OTEL_BSP_EXPORT_TIMEOUT`
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        max_queue_size: int = None,
        schedule_delay_millis: float = None,
        max_export_batch_size: int = None,
        export_timeout_millis: float = None,
    ):
        if max_queue_size is None:
            max_queue_size = _BatchSpanProcessor1._default_max_queue_size()

        if schedule_delay_millis is None:
            schedule_delay_millis = (
                _BatchSpanProcessor1._default_schedule_delay_millis()
            )

        if max_export_batch_size is None:
            max_export_batch_size = (
                _BatchSpanProcessor1._default_max_export_batch_size()
            )

        if export_timeout_millis is None:
            export_timeout_millis = (
                _BatchSpanProcessor1._default_export_timeout_millis()
            )

        _BatchSpanProcessor1._validate_arguments(
            max_queue_size, schedule_delay_millis, max_export_batch_size
        )

        self.span_exporter = span_exporter
        self.queue = collections.deque(
            [], max_queue_size
        )  # type: typing.Deque[ReadableSpan]
        self.worker_thread = threading.Thread(
            name="OtelBatchSpanProcessor", target=self.worker, daemon=True
        )
        self.condition = threading.Condition(threading.Lock())
        self._flush_request = None  # type: typing.Optional[_FlushRequest]
        self.schedule_delay_millis = schedule_delay_millis
        self.max_export_batch_size = max_export_batch_size
        self.max_queue_size = max_queue_size
        self.export_timeout_millis = export_timeout_millis
        self.done = False
        # flag that indicates that spans are being dropped
        self._spans_dropped = False
        # precallocated list to send spans to exporter
        self.spans_list = [
                              None
                          ] * self.max_export_batch_size  # type: typing.List[typing.Optional[Span]]
        self.worker_thread.start()
        # Only available in *nix since py37.
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(
                after_in_child=self._at_fork_reinit
            )  # pylint: disable=protected-access
        self._pid = os.getpid()

    def on_start(
        self, span: Span, parent_context: typing.Optional[Context] = None
    ) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        if self.done:
            logger.warning("Already shutdown, dropping span.")
            return
        if not span.context.trace_flags.sampled:
            return
        if self._pid != os.getpid():
            _BSP_RESET_ONCE.do_once(self._at_fork_reinit)

        if len(self.queue) == self.max_queue_size:
            if not self._spans_dropped:
                logger.warning("Queue is full, likely spans will be dropped.")
                self._spans_dropped = True

        self.queue.appendleft(span)

        if len(self.queue) >= self.max_export_batch_size:
            with self.condition:
                self.condition.notify()

    def _at_fork_reinit(self):
        self.condition = threading.Condition(threading.Lock())
        self.queue.clear()

        # worker_thread is local to a process, only the thread that issued fork continues
        # to exist. A new worker thread must be started in child process.
        self.worker_thread = threading.Thread(
            name="OtelBatchSpanProcessor", target=self.worker, daemon=True
        )
        self.worker_thread.start()
        self._pid = os.getpid()

    def worker(self):
        timeout = self.schedule_delay_millis / 1e3
        flush_request = None  # type: typing.Optional[_FlushRequest]
        while not self.done:
            with self.condition:
                if self.done:
                    # done flag may have changed, avoid waiting
                    break
                flush_request = self._get_and_unset_flush_request()
                if (
                    len(self.queue) < self.max_export_batch_size
                    and flush_request is None
                ):

                    self.condition.wait(timeout)
                    flush_request = self._get_and_unset_flush_request()
                    if not self.queue:
                        # spurious notification, let's wait again, reset timeout
                        timeout = self.schedule_delay_millis / 1e3
                        self._notify_flush_request_finished(flush_request)
                        flush_request = None
                        continue
                    if self.done:
                        # missing spans will be sent when calling flush
                        break

            # subtract the duration of this export call to the next timeout
            start = time_ns()
            self._export(flush_request)
            end = time_ns()
            duration = (end - start) / 1e9
            timeout = self.schedule_delay_millis / 1e3 - duration

            self._notify_flush_request_finished(flush_request)
            flush_request = None

        # there might have been a new flush request while export was running
        # and before the done flag switched to true
        with self.condition:
            shutdown_flush_request = self._get_and_unset_flush_request()

        # be sure that all spans are sent
        self._drain_queue()
        self._notify_flush_request_finished(flush_request)
        self._notify_flush_request_finished(shutdown_flush_request)

    def _get_and_unset_flush_request(
        self,
    ) -> typing.Optional[_FlushRequest]:
        """Returns the current flush request and makes it invisible to the
        worker thread for subsequent calls.
        """
        flush_request = self._flush_request
        self._flush_request = None
        if flush_request is not None:
            flush_request.num_spans = len(self.queue)
        return flush_request

    @staticmethod
    def _notify_flush_request_finished(
        flush_request: typing.Optional[_FlushRequest],
    ):
        """Notifies the flush initiator(s) waiting on the given request/event
        that the flush operation was finished.
        """
        if flush_request is not None:
            flush_request.event.set()

    def _get_or_create_flush_request(self) -> _FlushRequest:
        """Either returns the current active flush event or creates a new one.

        The flush event will be visible and read by the worker thread before an
        export operation starts. Callers of a flush operation may wait on the
        returned event to be notified when the flush/export operation was
        finished.

        This method is not thread-safe, i.e. callers need to take care about
        synchronization/locking.
        """
        if self._flush_request is None:
            self._flush_request = _FlushRequest()
        return self._flush_request

    def _export(self, flush_request: typing.Optional[_FlushRequest]):
        """Exports spans considering the given flush_request.

        In case of a given flush_requests spans are exported in batches until
        the number of exported spans reached or exceeded the number of spans in
        the flush request.
        In no flush_request was given at most max_export_batch_size spans are
        exported.
        """
        if not flush_request:
            self._export_batch()
            return

        num_spans = flush_request.num_spans
        while self.queue:
            num_exported = self._export_batch()
            num_spans -= num_exported

            if num_spans <= 0:
                break

    def _export_batch(self) -> int:
        """Exports at most max_export_batch_size spans and returns the number of
        exported spans.
        """
        idx = 0
        # currently only a single thread acts as consumer, so queue.pop() will
        # not raise an exception
        while idx < self.max_export_batch_size and self.queue:
            self.spans_list[idx] = self.queue.pop()
            idx += 1
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            # Ignore type b/c the Optional[None]+slicing is too "clever"
            # for mypy
            self.span_exporter.export(self.spans_list[:idx])  # type: ignore
        except Exception:  # pylint: disable=broad-except
            logger.exception("Exception while exporting Span batch.")
        detach(token)

        # clean up list
        for index in range(idx):
            self.spans_list[index] = None
        return idx

    def _drain_queue(self):
        """Export all elements until queue is empty.

        Can only be called from the worker thread context because it invokes
        `export` that is not thread safe.
        """
        while self.queue:
            self._export_batch()

    def force_flush(self, timeout_millis: int = None) -> bool:

        if timeout_millis is None:
            timeout_millis = self.export_timeout_millis

        if self.done:
            logger.warning("Already shutdown, ignoring call to force_flush().")
            return True

        with self.condition:
            flush_request = self._get_or_create_flush_request()
            # signal the worker thread to flush and wait for it to finish
            self.condition.notify_all()

        # wait for token to be processed
        ret = flush_request.event.wait(timeout_millis / 1e3)
        if not ret:
            logger.warning("Timeout was exceeded in force_flush().")
        return ret

    def shutdown(self) -> None:
        # signal the worker thread to finish and then wait for it
        self.done = True
        with self.condition:
            self.condition.notify_all()
        self.worker_thread.join()
        self.span_exporter.shutdown()

    @staticmethod
    def _default_max_queue_size():
        try:
            return int(
                environ.get(OTEL_BSP_MAX_QUEUE_SIZE, _DEFAULT_MAX_QUEUE_SIZE)
            )
        except ValueError:
            logger.exception(
                _ENV_VAR_INT_VALUE_ERROR_MESSAGE,
                OTEL_BSP_MAX_QUEUE_SIZE,
                _DEFAULT_MAX_QUEUE_SIZE,
            )
            return _DEFAULT_MAX_QUEUE_SIZE

    @staticmethod
    def _default_schedule_delay_millis():
        try:
            return int(
                environ.get(
                    OTEL_BSP_SCHEDULE_DELAY, _DEFAULT_SCHEDULE_DELAY_MILLIS
                )
            )
        except ValueError:
            logger.exception(
                _ENV_VAR_INT_VALUE_ERROR_MESSAGE,
                OTEL_BSP_SCHEDULE_DELAY,
                _DEFAULT_SCHEDULE_DELAY_MILLIS,
            )
            return _DEFAULT_SCHEDULE_DELAY_MILLIS

    @staticmethod
    def _default_max_export_batch_size():
        try:
            return int(
                environ.get(
                    OTEL_BSP_MAX_EXPORT_BATCH_SIZE,
                    _DEFAULT_MAX_EXPORT_BATCH_SIZE,
                )
            )
        except ValueError:
            logger.exception(
                _ENV_VAR_INT_VALUE_ERROR_MESSAGE,
                OTEL_BSP_MAX_EXPORT_BATCH_SIZE,
                _DEFAULT_MAX_EXPORT_BATCH_SIZE,
            )
            return _DEFAULT_MAX_EXPORT_BATCH_SIZE

    @staticmethod
    def _default_export_timeout_millis():
        try:
            return int(
                environ.get(
                    OTEL_BSP_EXPORT_TIMEOUT, _DEFAULT_EXPORT_TIMEOUT_MILLIS
                )
            )
        except ValueError:
            logger.exception(
                _ENV_VAR_INT_VALUE_ERROR_MESSAGE,
                OTEL_BSP_EXPORT_TIMEOUT,
                _DEFAULT_EXPORT_TIMEOUT_MILLIS,
            )
            return _DEFAULT_EXPORT_TIMEOUT_MILLIS

    @staticmethod
    def _validate_arguments(
        max_queue_size, schedule_delay_millis, max_export_batch_size
    ):
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be a positive integer.")

        if schedule_delay_millis <= 0:
            raise ValueError("schedule_delay_millis must be positive.")

        if max_export_batch_size <= 0:
            raise ValueError(
                "max_export_batch_size must be a positive integer."
            )

        if max_export_batch_size > max_queue_size:
            raise ValueError(
                "max_export_batch_size must be less than or equal to max_queue_size."
            )


class ConsoleSpanExporter(SpanExporter):
    """Implementation of :class:`SpanExporter` that prints spans to the
    console.

    This class can be used for diagnostic purposes. It prints the exported
    spans to the console STDOUT.
    """

    def __init__(
        self,
        service_name: Optional[str] = None,
        out: typing.IO = sys.stdout,
        formatter: typing.Callable[
            [ReadableSpan], str
        ] = lambda span: span.to_json()
                         + linesep,
    ):
        self.out = out
        self.formatter = formatter
        self.service_name = service_name

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self.out.write(self.formatter(span))
        self.out.flush()
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


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


class ThreadingTimer(TimerABC):

    def __init__(self, interval_sec: float):
        self.interval_sec = interval_sec
        self.cb = lambda: None
        self.timer = None
        self.lock = threading.Lock()

    def set_callback(self, cb) -> None:
        with self.lock:
            self.cb = cb

    def start(self) -> None:
        with self.lock:
            self.timer = threading.Timer(self.interval_sec, self._work)
            self.timer.daemon = True
            self.timer.start()

    def _work(self):
        self.cb()
        self.start()

    def poke(self) -> None:
        with self.lock:
            self._stop_unsafe()
            threading.Thread(target=self._work, daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            self._stop_unsafe()

    def _stop_unsafe(self):
        if self.timer is None:
            return
        self.timer.cancel()
        self.timer = None


class ThreadlessTimer(TimerABC):
    """
    For testing. Executes the callback synchronously when you run poke().
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


class SpanAccumulator:
    """
    A thread-safe container designed to collect and batch spans. It accumulates spans until a specified batch size is
    reached, at which point the accumulated spans are moved into a FIFO queue. Provides methods to add spans, check if
    the accumulator is non-empty, and retrieve the earliest batch of spans from the queue.
    """

    def __init__(self, batch_size: int):
        self._batch_size = batch_size
        self._spans: typing.List[ReadableSpan] = []
        self._batches = collections.deque()  # fixme set max size
        self._lock = threading.Lock()

    def nonempty(self) -> bool:
        """
        Checks if the accumulator contains any spans or batches. It returns True if either the span list or the batch
        queue is non-empty, and False otherwise.
        """
        with self._lock:
            return len(self._spans) > 0 or len(self._batches) > 0

    def push(self, span: ReadableSpan) -> bool:
        """
        Adds a span to the accumulator. If the addition causes the number of spans to reach the
        specified batch size, the accumulated spans are moved into a FIFO queue as a new batch. Returns True if a new
        batch was created, otherwise returns False.
        """
        with self._lock:
            self._spans.append(span)
            if len(self._spans) < self._batch_size:
                return False
            self._batches.appendleft(self._spans)
            self._spans = []
            return True

    def batch(self) -> typing.List[ReadableSpan]:
        """
        Returns the earliest (rightmost, first in line) batch of spans from the FIFO queue. If the queue is empty,
        returns any remaining spans that haven't been batched.
        """
        try:
            return self._batches.pop()
        except IndexError:
            # if there are no batches left, return the current spans
            with self._lock:
                out = self._spans
                self._spans = []
                return out

    def reinsert(self, batch):
        """
        Returns a batch back into the queue at the front of the line.
        """
        with self._lock:
            self._batches.append(batch)


class _BatchSpanProcessor2(SpanProcessor):
    """
    A SpanProcessor that sends spans in batches on an interval or when a maximum number of spans has been reached,
    whichever comes first.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        max_queue_size: int = None,
        schedule_delay_millis: float = None,
        max_export_batch_size: int = None,
        export_timeout_millis: float = None,
        timer: typing.Optional[TimerABC] = None,
    ):
        self._exporter = span_exporter

        max_export_batch_size_final = _BatchSpanProcessor1._default_max_export_batch_size() \
            if max_export_batch_size is None \
            else max_export_batch_size
        self._accumulator = SpanAccumulator(max_export_batch_size_final)

        max_interval = _BatchSpanProcessor1._default_schedule_delay_millis() \
            if schedule_delay_millis is None \
            else schedule_delay_millis
        self._timer = timer or ThreadingTimer(max_interval / 1e3)

        self._timer.set_callback(self._export_single_batch)
        self._timer.start()

    def on_start(self, span: Span, parent_context: typing.Optional[Context] = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """
        This method must be extremely fast. It adds the span to the accumulator for later sending and pokes the timer
        if the number of spans waiting to be sent has reached the maximum batch size.
        """
        full = self._accumulator.push(span)
        if full:
            self._timer.poke()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Stops the timer, exports any spans in the accumulator then restarts the timer.
        """
        self._timer.stop()
        out = self._flush(timeout_millis)
        self._timer.start()
        return out

    def _flush(self, timeout_millis):
        self._exporter.force_flush(timeout_millis)  # typically a no-op
        time_remaining_millis = timeout_millis
        start = time.time()
        while self._accumulator.nonempty():
            result = self._export_single_batch_with_timeout(time_remaining_millis)
            if result != SpanExportResult.SUCCESS:
                return False
            elapsed_millis = (time.time() - start) * 1e3
            if elapsed_millis > timeout_millis:
                logger.warning('Timed out during force_flush.')
                return False
            time_remaining_millis -= elapsed_millis
        return True

    def shutdown(self) -> None:
        self._timer.stop()
        while self._accumulator.nonempty():
            self._export_single_batch()
        self._exporter.shutdown()

    def _export_single_batch_with_timeout(self, timeout_millis):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._export_single_batch)
            try:
                return future.result(timeout=(timeout_millis / 1e3))
            except futures.TimeoutError:
                logger.warning('Exporting single batch timed out.')
                return SpanExportResult.FAILURE

    def _export_single_batch(self) -> SpanExportResult:
        """
        Exports one batch. Any retry is handled by the exporter. If export fails, reinserts the batch into the
        accumulator. Used by both the timer and force_flush.
        """
        batch = self._accumulator.batch()
        if len(batch) == 0:
            return SpanExportResult.SUCCESS

        export_result = self._exporter.export(batch)
        if export_result != SpanExportResult.SUCCESS:
            self._accumulator.reinsert(batch)
        return export_result
