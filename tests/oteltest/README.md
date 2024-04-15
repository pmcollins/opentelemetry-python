# oteltest

[![PyPI - Version](https://img.shields.io/pypi/v/oteltest.svg)](https://pypi.org/project/oteltest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oteltest.svg)](https://pypi.org/project/oteltest)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install oteltest
```

## Overview

The `oteltest` package contains utilities for testing OpenTelemetry Python.

### oteltest

The `oteltest` command runs black box tests against Python scripts that send telemetry.

#### Execution

Run `oteltest` as a shell command and provide a directory as an argument:

```shell
oteltest my_script_dir
```

in which case it will attempt to run all `oteltest`-eligible scripts in `my_script_dir`, non-recursively.

#### Operation

Running `oteltest` against a directory containing `my_script.py`

1) Starts an [otelsink](#otelsink) instance
2) Creates a new Python virtual environment with `requirements()`
3) In that environment, starts running `my_script.py` in a subprocess
4) Meanwhile, calls `OtelTest#on_script_start()` waiting until completion
5) Depending on the return value from `on_script_start()`, waits for `my_script.py` to complete or interrupts
6) Stops the OTLP listener
7) Calls `validate(telemetry)` with otelsink's received telemetry
8) Writes the telemetry to a `.json` file next to the script (script name but with ".{number}.json" instead of ".py")

#### Script Eligibility

For a Python script to be runnable by `oteltest`, it must both be executable and define an implementation of the
[OtelTest]() abstract base class. The script below has an implementation called `MyOtelTest`:

```python
import time

from opentelemetry import trace
from oteltest import OtelTest, Telemetry

SERVICE_NAME = "my-otel-test"
NUM_ADDS = 12

if __name__ == "__main__":
    tracer = trace.get_tracer("my-tracer")
    for i in range(NUM_ADDS):
        with tracer.start_as_current_span("my-span"):
            print(f"simple_loop.py: {i+1}/{NUM_ADDS}")
            time.sleep(0.5)


class MyOtelTest(OtelTest):
    def requirements(self):
        return "opentelemetry-distro", "opentelemetry-exporter-otlp-proto-grpc"

    def environment_variables(self):
        return {"OTEL_SERVICE_NAME": SERVICE_NAME}

    def wrapper_script(self):
        return "opentelemetry-instrument"

    def on_script_start(self):
        return None

    def on_script_end(self, stdout, stderr, returncode) -> None:
        pass

    def on_shutdown(self, telemetry: Telemetry):
        assert telemetry.num_spans() == NUM_ADDS
```

### otelsink

`otelsink` is a gRPC server that listens for OTel metrics, traces, and logs.

#### Operation

You can run otelink either from the command line by using the `otelsink` command (installed when you
`pip install oteltest`), or programatically.

Either way, `otelsink` runs a gRPC server listening on 0.0.0.0:4317.

#### Command Line

```
% otelsink
starting otelsink with print handler
```

#### Programmatic

```
from oteltest.sink import GrpcSink, PrintHandler

class MyHandler(RequestHandler):
    def handle_logs(self, request, context):
        print(f"received log request: {request}")

    def handle_metrics(self, request, context):
        print(f"received metrics request: {request}")

    def handle_trace(self, request, context):
        print(f"received trace request: {request}")


sink = GrpcSink(MyHandler())
sink.start()
sink.wait_for_termination()
```

## License

`oteltest` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
