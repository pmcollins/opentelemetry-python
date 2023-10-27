import unittest

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace.export import MaturityLevel
from test_export import MySpanExporter, _create_start_and_end_span


class TestBetaBSP(unittest.TestCase):

    def test_beta_bsp(self):
        spans_exported = []
        exporter = MySpanExporter(spans_exported)
        bsp = export.BatchSpanProcessor(exporter, maturity_level=MaturityLevel.BETA)
        resource = Resource.create({})
        for i in range(1024):
            print('.', end='')
            if i % 128 == 0:
                print()
            _create_start_and_end_span('foo', bsp, resource)
        print()
        print('shutting down')
        bsp.shutdown()


