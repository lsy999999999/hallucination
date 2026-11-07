"""Bounded, credential-safe probes of the configured Responses endpoint."""
import json
import os
from urllib.parse import urlsplit

from .llm import OpenAIJSONClient, error_details


def doctor(model=None, probe=False, backend=None):
    backend = backend or os.getenv('HALLULAB_BACKEND','api')
    print('Backend:', backend)
    if backend == 'codex':
        return doctor_codex(model,probe)
    if backend != 'api':
        print('Invalid HALLULAB_BACKEND; expected api or codex')
        return False
    # Validate URL before printing it: never print credentials or query strings.
    base = os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
    parsed = urlsplit(base)
    print('Base URL host:', parsed.hostname or '<invalid URL>')
    print('Base URL path:', parsed.path if not parsed.query and not parsed.username else '<hidden; URL contains credentials/query>')
    print('Model:', model or os.getenv('OPENAI_MODEL', 'gpt-5.4-mini-2026-03-17'))
    print('API key:', 'present (hidden)' if os.getenv('OPENAI_API_KEY') else 'missing')
    if parsed.hostname == 'api.openai.com':
        print('当前目标是官方 API。账号转 API 工具通常需要填写它自己的 OPENAI_BASE_URL。')
    if not probe:
        print('配置检查结束，未发送网络请求。添加 --probe 可发送一次最小 JSON 生成请求（可能计费）。')
        return True
    try:
        client = OpenAIJSONClient(model=model, max_attempts=1, timeout=20)
        # Exercise exactly the same endpoint/JSON mode used by the experiment.
        response = client.client.responses.create(
            model=client.model, instructions='Return a JSON object with ok=true.',
            input='Return JSON: {"ok":true}',
            text={'format': {'type': 'json_object'}}, max_output_tokens=2200)
        print('Responses API: HTTP request succeeded')
        print('Response status:', getattr(response, 'status', 'unknown'))
        from .llm import parse_json
        obj = parse_json(response.output_text)
        if obj.get('ok') is not True:
            raise ValueError('Probe response did not contain ok=true')
        print('JSON probe: PASS')
        return True
    except Exception as exc:
        print(json.dumps(error_details(exc), ensure_ascii=False, indent=2))
        return False


def doctor_codex(model=None, probe=False):
    from .codex_client import CodexJSONClient
    try:
        client = CodexJSONClient(model=model)
        print('Authentication: ChatGPT account')
        print('Model:', client.model)
        print('CLI:', client.version)
        print('使用账号中的 Codex 可用额度；不使用 OPENAI_API_KEY，也不转移 API 余额。')
        if not probe:
            print('登录状态检查通过，未发送生成请求。可添加 --probe 验证模型调用。')
            return True
        if client.json('Return JSON with ok=true.', 'Return {"ok":true}.').get('ok') is not True:
            raise ValueError('Probe response did not contain ok=true')
        print('Codex JSON probe: PASS')
        return True
    except Exception as exc:
        from .llm import safe_error
        print(safe_error(exc))
        return False
