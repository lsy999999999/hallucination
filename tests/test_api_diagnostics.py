import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hallulab.diagnostics import doctor
from hallulab.llm import OpenAIJSONClient, error_details, retryable, safe_error


class ServiceError(Exception):
    def __init__(self, status, code):
        super().__init__('Service rejected request')
        self.status_code = status
        self.body = {'error': {'code': code}}


class APIDiagnosticsTests(unittest.TestCase):
    def test_quota_is_distinguished_from_temporary_rate_limit(self):
        quota = ServiceError(429, 'insufficient_quota')
        rate = ServiceError(429, 'rate_limit_exceeded')
        self.assertFalse(retryable(quota))
        self.assertTrue(retryable(rate))
        self.assertIn('计费', error_details(quota)['hint'])
        self.assertIn('限流', error_details(rate)['hint'])

    def test_credentials_are_redacted_from_server_errors(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'local-fake-credential'}):
            message = safe_error(ValueError('local-fake-credential sk-fake_key Bearer fake-token'))
        for secret in ['local-fake-credential', 'sk-fake_key', 'fake-token']:
            self.assertNotIn(secret, message)

    def test_credit_balance_error_is_permanent_quota_failure(self):
        for code in ['credit_balance_exhausted', 'provider_specific_code']:
            with self.subTest(code=code):
                exc = ServiceError(429, code)
                exc.body['error']['type'] = 'insufficient_quota'
                self.assertFalse(retryable(exc))
                self.assertTrue(error_details(exc)['quota_exhausted'])
                self.assertIn('计费', error_details(exc)['hint'])
        self.assertFalse(retryable(ServiceError(429, 'credit_balance_exhausted')))

    def test_invalid_key_fails_after_one_request(self):
        create = Mock(side_effect=ServiceError(401, 'invalid_api_key'))
        client = OpenAIJSONClient.__new__(OpenAIJSONClient)
        client.client = SimpleNamespace(responses=SimpleNamespace(create=create))
        client.model = 'test-model'
        client.max_attempts = 3
        with contextlib.redirect_stdout(io.StringIO()), patch('hallulab.llm.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, '认证失败'):
                client.json('test', 'test')
        self.assertEqual(create.call_count, 1)
        sleep.assert_not_called()

    def test_doctor_without_probe_never_creates_client(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake-key'}, clear=True), \
                patch('hallulab.diagnostics.OpenAIJSONClient') as client, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(doctor())
        client.assert_not_called()
        self.assertIn('api.openai.com', output.getvalue())
        self.assertNotIn('fake-key', output.getvalue())


if __name__ == '__main__':
    unittest.main()
