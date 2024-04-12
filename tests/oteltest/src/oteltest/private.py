import glob
import importlib
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import typing
import venv
from pathlib import Path

from google.protobuf.json_format import MessageToDict

from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
)
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from oteltest import OtelTest, Telemetry
from oteltest.sink import GrpcSink, RequestHandler


def run(script_dir: str, wheel_file: str, venv_parent_dir: str):
    temp_dir = venv_parent_dir or tempfile.mkdtemp()
    print(f"- Using temp dir: {temp_dir}")

    sys.path.append(script_dir)

    for script in ls_scripts(script_dir):
        setup_script_environment(temp_dir, script_dir, script, wheel_file)


def ls_scripts(script_dir):
    original_dir = os.getcwd()
    os.chdir(script_dir)
    scripts = [script_name for script_name in glob.glob("*.py")]
    os.chdir(original_dir)
    return scripts


def setup_script_environment(tempdir, script_dir, script, wheel_file):
    handler = AccumulatingHandler()
    sink = GrpcSink(handler)
    sink.start()

    module_name = script[:-3]
    test_class = load_test_class_for_script(module_name)
    oteltest_instance: OtelTest = test_class()

    v = Venv(str(Path(tempdir) / module_name))
    v.create()

    pip_path = v.path_to_executable("pip")

    oteltest_dep = wheel_file or "oteltest"
    run_subprocess([pip_path, "install", oteltest_dep])

    for req in oteltest_instance.requirements():
        print(f"- Will install requirement: '{req}'")
        run_subprocess([pip_path, "install", req])

    run_python_script(script, script_dir, oteltest_instance, v)

    v.rm()

    save_telemetry_json(script_dir, module_name, handler.telemetry_to_json())

    oteltest_instance.on_shutdown(handler.telemetry)
    print(f"- {script} PASSED")


def save_telemetry_json(script_dir, module_name, json_str):
    path_str = str(Path(script_dir) / f"{module_name}.json")
    with open(path_str, "w") as file:
        file.write(json_str)


def run_python_script(script, script_dir, oteltest_instance: OtelTest, v):
    python_script_cmd = [
        v.path_to_executable("python"),
        str(Path(script_dir) / script),
    ]

    wrapper_script = oteltest_instance.wrapper_script()
    if wrapper_script is not None:
        python_script_cmd.insert(0, v.path_to_executable(wrapper_script))

    sprocess = subprocess.Popen(
        python_script_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=oteltest_instance.environment_variables(),
    )

    timeout = oteltest_instance.on_script_start()
    if timeout is None:
        print(
            f"- Will wait indefinitely for {script} to finish (on_script_start() returned None)"
        )
    else:
        print(
            f"- Will wait for up to {timeout} seconds for {script} to finish"
        )

    stdout, stderr, returncode = wait_for_subprocess(sprocess, script, timeout)
    print_result(stdout, stderr, returncode)
    oteltest_instance.on_script_end(stdout, stderr, returncode)


def wait_for_subprocess(
    process: subprocess.Popen, script, timeout
) -> typing.Tuple[str, str, int]:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, process.returncode
    except subprocess.TimeoutExpired as ex:
        print(f"- Script {script} was force quit")
        return decode(ex.stdout), decode(ex.stderr), process.returncode


def print_result(stdout, stderr, returncode):
    print(f"- Return Code: {returncode}")
    print("- Standard Output:")
    if stdout:
        print(stdout)
    print("- Standard Error:")
    if stderr:
        print(stderr)
    print("- End Subprocess -\n")


def run_subprocess(args):
    print(f"- Subprocess: {args}")
    result = subprocess.run(
        args,
        capture_output=True,
    )
    returncode = result.returncode
    stdout = result.stdout
    stderr = result.stderr
    print_result(returncode, stderr, stdout)


def decode(s):
    return s.decode("utf-8")


def load_test_class_for_script(module_name):
    module = importlib.import_module(module_name)
    for attr_name in dir(module):
        value = getattr(module, attr_name)
        if is_test_class(value):
            return value
    return None


def is_test_class(value):
    return (
        inspect.isclass(value)
        and issubclass(value, OtelTest)
        and value is not OtelTest
    )


class Venv:
    def __init__(self, venv_dir):
        self.venv_dir = venv_dir

    def create(self):
        venv.create(self.venv_dir, with_pip=True)

    def path_to_executable(self, executable_name: str):
        return f"{self.venv_dir}/bin/{executable_name}"

    def rm(self):
        shutil.rmtree(self.venv_dir)


class AccumulatingHandler(RequestHandler):
    def __init__(self):
        self.telemetry = Telemetry()

    def handle_logs(
        self, request: ExportLogsServiceRequest, context
    ):  # noqa: ARG002
        self.telemetry.add_log(
            MessageToDict(request), get_context_headers(context)
        )

    def handle_metrics(
        self, request: ExportMetricsServiceRequest, context
    ):  # noqa: ARG002
        self.telemetry.add_metric(
            MessageToDict(request), get_context_headers(context)
        )

    def handle_trace(
        self, request: ExportTraceServiceRequest, context
    ):  # noqa: ARG002
        self.telemetry.add_trace(
            MessageToDict(request), get_context_headers(context)
        )

    def telemetry_to_json(self):
        return self.telemetry.to_json()


def get_context_headers(context):
    return pbmetadata_to_dict(context.invocation_metadata())


def pbmetadata_to_dict(pbmetadata):
    out = {}
    for k, v in pbmetadata:
        out[k] = v
    return out