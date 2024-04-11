import grpc  # type: ignore

from opentelemetry.proto.collector.logs.v1 import logs_service_pb2, logs_service_pb2_grpc  # type: ignore
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest  # type: ignore
from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2, metrics_service_pb2_grpc  # type: ignore
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest  # type: ignore
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2, trace_service_pb2_grpc  # type: ignore
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest  # type: ignore


class _LogsServiceServicer(logs_service_pb2_grpc.LogsServiceServicer):
    def __init__(self, handle_request):
        self.handle_request = handle_request

    def Export(self, request, context):  # noqa: N802
        self.handle_request(request, context)
        return logs_service_pb2.ExportLogsServiceResponse()


class _TraceServiceServicer(trace_service_pb2_grpc.TraceServiceServicer):
    def __init__(self, handle_request):
        self.handle_request = handle_request

    def Export(self, request, context):  # noqa: N802
        self.handle_request(request, context)
        return trace_service_pb2.ExportTraceServiceResponse()


class _MetricsServiceServicer(metrics_service_pb2_grpc.MetricsServiceServicer):
    def __init__(self, handle_request):
        self.handle_request = handle_request

    def Export(self, request, context):  # noqa: N802
        self.handle_request(request, context)
        return metrics_service_pb2.ExportMetricsServiceResponse()
