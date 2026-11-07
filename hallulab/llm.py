"""Thin, retrying OpenAI Responses API JSON client."""
import json
import os
import re
import time

QUOTA_CODES = {'insufficient_quota', 'credit_balance_exhausted',
               'billing_hard_limit_reached', 'billing_not_active'}


def safe_error(exc):
    """Describe failures without echoing credentials supplied by a gateway."""
    message = str(exc)
    key = os.getenv('OPENAI_API_KEY')
    if key:
        message = message.replace(key, '<redacted>')
    message = re.sub(r'\bsk-[A-Za-z0-9_-]+', '<redacted>', message)
    message = re.sub(r'(?i)Bearer\s+[^\s\"\'<>]+', 'Bearer <redacted>', message)
    return message[:1500]


def error_details(exc):
    status = getattr(exc, 'status_code', None)
    body = getattr(exc, 'body', None)
    error = body.get('error', body) if isinstance(body, dict) else {}
    error = error if isinstance(error, dict) else {}
    code = error.get('code') or getattr(exc, 'code', None)
    code = str(code) if code is not None else None
    quota_exhausted = code in QUOTA_CODES or error.get('type') == 'insufficient_quota'
    if status == 401:
        hint = '认证失败：核对 Key 是否属于当前 Base URL；登录 token 不能直接当官方 API Key。'
    elif status == 402 or quota_exhausted:
        hint = '服务端报告额度/计费问题：核对当前接口所属平台、项目余额和消费上限。'
    elif status == 403:
        hint = '访问被拒绝：检查项目权限、模型权限或服务商的访问限制。'
    elif status == 404:
        hint = '检查 Base URL 路径、Responses 接口支持情况以及模型名/权限。'
    elif status == 429:
        hint = '请求被限流或配额受限；结合 error_code 判断，不能仅凭 429 认定余额不足。'
    elif status in {400, 422}:
        hint = '请求参数不被接受：检查模型、JSON 模式及 Responses API 的兼容性。'
    elif type(exc).__name__ in {'APIConnectionError', 'APITimeoutError'}:
        hint = '尚未收到 API HTTP 响应：检查 DNS、网络、代理及 Base URL。'
    else:
        hint = '检查错误详情和服务端状态。'
    return {'exception': type(exc).__name__, 'http_status': status,
            'error_code': code, 'quota_exhausted': quota_exhausted,
            'hint': hint, 'detail': safe_error(exc)}


def retryable(exc):
    details = error_details(exc)
    if details['quota_exhausted']:
        return False
    status = details['http_status']
    return status is None or status in {408, 409, 429} or status >= 500


def parse_json(text):
    s = (text or '').strip()
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s, flags=re.I)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        begin, end = s.find('{'), s.rfind('}')
        if begin < 0 or end < begin:
            raise ValueError(f'Model returned no JSON object: {s[:250]}')
        return json.loads(s[begin:end+1])


class OpenAIJSONClient:
    def __init__(self, model=None, max_attempts=3, timeout=60):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('pip install -r requirements.txt') from exc
        if not os.environ.get('OPENAI_API_KEY'):
            raise RuntimeError('OPENAI_API_KEY is missing. Set it in .env or export the variable.')
        # Keep retries here, rather than multiplying them by SDK retries.
        self.client = OpenAI(timeout=timeout, max_retries=0)
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-5.4-mini-2026-03-17')
        self.max_attempts = max_attempts

    def json(self, system, user, temperature=None):
        err = None
        for attempt in range(self.max_attempts):
            try:
                kw = {'model': self.model, 'instructions': system + '\nReturn valid JSON only.',
                      'input': 'Return a valid JSON object.\n\n' + user,
                      'text': {'format': {'type': 'json_object'}},
                      'max_output_tokens': 2200}
                if temperature is not None:
                    kw['temperature'] = temperature
                try:
                    response = self.client.responses.create(**kw)
                except Exception as e:
                    # For OpenAI-compatible gateways or models that do not accept JSON mode/temperature.
                    m = str(e).lower()
                    if any(x in m for x in ('unsupported_parameter','unsupported parameter','unknown parameter','invalid parameter')):
                        kw.pop('temperature', None)
                        kw.pop('text', None)
                        response = self.client.responses.create(**kw)
                    else:
                        raise
                result = parse_json(response.output_text)
                if not isinstance(result, dict):
                    raise ValueError('Expected JSON object')
                return result
            except Exception as exc:
                err = exc
                details = error_details(exc)
                print(f'[LLM attempt {attempt+1}/{self.max_attempts}] {details["exception"]}: {details["detail"]}')
                if not retryable(exc):
                    break
                if attempt+1 < self.max_attempts:
                    time.sleep(min(2**attempt, 4))
        details = error_details(err)
        raise RuntimeError(f'LLM failed: {details["hint"]} {details["detail"]}') from None
