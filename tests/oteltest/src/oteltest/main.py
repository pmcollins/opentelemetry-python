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

import glob
import importlib
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
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
from oteltest.common import OtelTest, Telemetry
from oteltest.sink import GrpcSink, RequestHandler

import argparse


def main():
    parser = argparse.ArgumentParser(description="OpenTelemetry Python Tester")

    w_help = "Path to an optional wheel (.whl) file to `pip install` instead of `oteltest`"
    parser.add_argument(
        "-w", "--wheel-file", type=str, required=False, help=w_help
    )

    d_help = (
        "An optional override directory to hold per-script venv directories."
    )
    parser.add_argument(
        "-d", "--venv-parent-dir", type=str, required=False, help=d_help
    )

    parser.add_argument(
        "script_dir",
        type=str,
        help="The directory containing your oteltest scripts at the top level",
    )

    args = parser.parse_args()
    run(args.script_dir, args.wheel_file, args.venv_parent_dir)


def run(script_dir: str, wheel_file: str, venv_parent_dir: str):
    venv_dir = venv_parent_dir or tempfile.mkdtemp()
    print(f"using temp dir: {venv_dir}")

    sys.path.append(script_dir)

    for script in ls_scripts(script_dir):
        setup_script_environment(script, script_dir, venv_dir, wheel_file)


def ls_scripts(script_dir):
    original_dir = os.getcwd()
    os.chdir(script_dir)
    scripts = [script_name for script_name in glob.glob("*.py")]
    os.chdir(original_dir)
    return scripts


def setup_script_environment(script, script_dir, tempdir, wheel_file):
    handler = AccumulatingHandler()
    sink = GrpcSink(handler)
    sink.start()

    module_name = script[:-3]
    test_class = load_test_class_for_script(module_name)
    test_instance: OtelTest = test_class()

    v = Venv(str(Path(tempdir) / module_name))
    v.create()

    pip_path = v.path_to_executable("pip")

    oteltest_dep = wheel_file or "oteltest"
    run_subprocess([pip_path, "install", oteltest_dep])

    for req in test_instance.requirements():
        print(f"- Will install requirement: '{req}'")
        run_subprocess([pip_path, "install", req])

    run_python_script(script, script_dir, test_instance, v)

    v.rm()

    with open(str(Path(script_dir) / f"{module_name}.json"), "w") as file:
        file.write(handler.telemetry_to_json())

    test_instance.validate(handler.telemetry)
    print(f"- {script} PASSED")


def run_python_script(script, script_dir, test_instance: OtelTest, v):
    python_script_cmd = [
        v.path_to_executable("python"),
        str(Path(script_dir) / script),
    ]
    script = test_instance.wrapper_script()
    if script is not None:
        python_script_cmd.insert(0, v.path_to_executable(script))

    process = subprocess.Popen(
        python_script_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=test_instance.environment_variables(),
    )

    test_instance.run_client()

    # wait for the process to terminate
    stdout, stderr = process.communicate()

    print_result(process.returncode, stderr, stdout)


def run_subprocess(args, env_vars=None):
    print(f"- Subprocess: {args}")
    print(f"- Environment: {env_vars}")
    result = subprocess.run(
        args,
        capture_output=True,
        env=env_vars,
    )
    returncode = result.returncode
    stdout = result.stdout
    stderr = result.stderr
    print_result(returncode, stderr, stdout)


def print_result(returncode, stderr, stdout):
    print(f"- Return Code: {returncode}")
    print("- Standard Output:")
    if stdout:
        print(decode(stdout))
    print("- Standard Error:")
    if stderr:
        print(decode(stderr))
    print("- End Subprocess -\n")


def decode(stream):
    return stream.decode("utf-8").strip()


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


if __name__ == "__main__":
    main()
