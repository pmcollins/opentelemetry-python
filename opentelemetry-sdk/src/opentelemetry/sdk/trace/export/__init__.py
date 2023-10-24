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

import logging
import os
import sys
import threading
import typing
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
from opentelemetry.sdk.util import BatchAccumulator
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


_BSP_RESET_ONCE = Once()


class BatchSpanProcessor(SpanProcessor):
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
            max_queue_size = BatchSpanProcessor._default_max_queue_size()

        if schedule_delay_millis is None:
            schedule_delay_millis = (
                BatchSpanProcessor._default_schedule_delay_millis()
            )

        if max_export_batch_size is None:
            max_export_batch_size = (
                BatchSpanProcessor._default_max_export_batch_size()
            )

        if export_timeout_millis is None:
            export_timeout_millis = (
                BatchSpanProcessor._default_export_timeout_millis()
            )

        BatchSpanProcessor._validate_arguments(
            max_queue_size, schedule_delay_millis, max_export_batch_size
        )

        self.accumulator = BatchAccumulator(max_export_batch_size)
        self.flush_lock = threading.Lock()

        self.span_exporter = span_exporter
        self.condition = threading.Condition(threading.Lock())
        self.schedule_delay_millis = schedule_delay_millis
        self.max_export_batch_size = max_export_batch_size
        self.max_queue_size = max_queue_size
        self.export_timeout_millis = export_timeout_millis

        self.worker_stopped: AtomicBool = AtomicBool(False)
        self.processor_shutdown: AtomicBool = AtomicBool(False)

        # flag that indicates that spans are being dropped
        self._spans_dropped = False

        self._start_worker()

        # Only available in *nix since py37.
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(
                after_in_child=self._at_fork_reinit
            )  # pylint: disable=protected-access
        self._pid = os.getpid()

    def _start_worker(self):
        self.worker_stopped.set(False)
        self.worker_thread = threading.Thread(
            name="OtelBatchSpanProcessor", target=self.worker, daemon=True
        )
        self.worker_thread.start()

    def _stop_worker(self):
        self.worker_stopped.set(True)
        with self.condition:
            self.condition.notify_all()
        self.worker_thread.join()

    def on_start(
        self, span: Span, parent_context: typing.Optional[Context] = None
    ) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        if self.processor_shutdown.get():
            logger.warning("Already shutdown, dropping span.")
            return
        if not span.context.trace_flags.sampled:
            return
        if self._pid != os.getpid():
            _BSP_RESET_ONCE.do_once(self._at_fork_reinit)

        full = self.accumulator.push(span)
        if full:
            with self.condition:
                self.condition.notify()

    def _at_fork_reinit(self):
        self.condition = threading.Condition(threading.Lock())

        # worker_thread is local to a process, only the thread that issued fork continues
        # to exist. A new worker thread must be started in child process.
        self.worker_thread = threading.Thread(
            name="OtelBatchSpanProcessor", target=self.worker, daemon=True
        )
        self.worker_thread.start()
        self._pid = os.getpid()

    def worker(self):
        timeout = self.schedule_delay_millis / 1e3
        while not self.worker_stopped.get():
            with self.condition:
                if self.worker_stopped.get():
                    break
                self.condition.wait(timeout)
            batch = self.accumulator.batch()
            start = time_ns()
            self._export(batch)
            end = time_ns()
            duration = (end - start) / 1e9
            timeout = self.schedule_delay_millis / 1e3 - duration

    def _export(self, batch):
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        # noinspection PyBroadException
        try:
            self.span_exporter.export(batch)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Exception while exporting Span batch.")
        detach(token)

    def force_flush(self, timeout_millis: int = None) -> bool:
        with self.flush_lock:
            start = time_ns()
            self._stop_worker()
            out = True
            while not self.accumulator.empty():
                batch = self.accumulator.batch()
                self._export(batch)
                if self._has_timed_out(start, timeout_millis):
                    logger.warning("Timeout was exceeded in force_flush().")
                    out = False
                    break
            self._start_worker()
            return out

    def shutdown(self) -> None:
        self.processor_shutdown.set(True)
        self._stop_worker()
        self.span_exporter.shutdown()

    @staticmethod
    def _has_timed_out(start_time_ns, timeout_millis):
        if timeout_millis is None:
            return False
        elapsed_millis = (time_ns() - start_time_ns) / 1e6
        return elapsed_millis > timeout_millis

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


class AtomicBool:
    def __init__(self, v: bool):
        self.lock = threading.Lock()
        self.v = v

    def set(self, v: bool):
        with self.lock:
            self.v = v

    def get(self) -> bool:
        with self.lock:
            return self.v


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
