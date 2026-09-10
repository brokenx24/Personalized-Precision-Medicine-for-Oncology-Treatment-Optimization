import unittest
from STAGE_04_INTEGRATION.validation.validate_integration import run_integration_quality_gate

class TestValidation(unittest.TestCase):
    def test_quality_gate_function(self):
        self.assertTrue(callable(run_integration_quality_gate))
