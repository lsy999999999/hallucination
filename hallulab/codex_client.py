"""JSON inference through an installed, ChatGPT-authenticated Codex CLI.

This consumes the account's available Codex allowance, not API credit. Every
request starts a fresh, ephemeral session in an empty directory.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

from .llm import parse_json, safe_error


LOGIN_COMMAND = 'codex -c \'forced_login_method="chatgpt"\' login --device-auth'
PROTOCOL = 'codex-json-no-tools-v1'


def cli_environment():
    env = os.environ.copy()
    # Preserve the user's normal Codex authentication store. Never read/copy its tokens.
    for key in list(env):
        if key.startswith('OPENAI_') or key in {'CODEX_API_KEY', 'CODEX_ACCESS_TOKEN'}:
            env.pop(key)
    return env


def login_status(binary):
    result = subprocess.run([binary, 'login', 'status'], env=cli_environment(),
                            capture_output=True, text=True, timeout=15)
    message = (result.stdout + '\n' + result.stderr).lower()
    if result.returncode == 0 and 'chatgpt' in message:
        return 'chatgpt'
    if 'api key' in message or 'api_key' in message:
        return 'api_key'
    return 'not_authenticated'


class CodexJSONClient:
    backend = 'codex'

    def __init__(self, model=None, timeout=None, log_dir=None):
        self.binary = shutil.which(os.getenv('HALLULAB_CODEX_BIN', 'codex'))
        if not self.binary:
            raise RuntimeError('Codex CLI not found. Install Codex CLI, then log in with ChatGPT.')
        self.model = model or os.getenv('CODEX_MODEL') or os.getenv('OPENAI_MODEL') or 'gpt-6-luna'
        self.timeout = float(timeout or os.getenv('HALLULAB_CODEX_TIMEOUT', '180'))
        self.log_dir = Path(log_dir or os.getenv('HALLULAB_CODEX_LOG_DIR', 'results/codex_calls')).resolve()
        self.effort = os.getenv('CODEX_REASONING_EFFORT', 'low')
        self.call_ids = []
        if self.effort not in {'minimal', 'low', 'medium', 'high', 'xhigh'}:
            raise ValueError('Invalid CODEX_REASONING_EFFORT')
        self.auth = login_status(self.binary)
        if self.auth != 'chatgpt':
            raise RuntimeError(f'Codex 当前不是 ChatGPT 登录模式（{self.auth}）。'
                               f'请在你自己的终端运行：{LOGIN_COMMAND}。'
                               '此后端不会回退到 API Key。')
        version = subprocess.run([self.binary, '--version'], env=cli_environment(),
                                 capture_output=True, text=True, timeout=15)
        if version.returncode:
            raise RuntimeError('Cannot read Codex CLI version')
        self.version = version.stdout.strip()
        self.metadata = {'backend': self.backend, 'protocol': PROTOCOL,
                         'cli_version': self.version, 'auth': self.auth,
                         'reasoning_effort': self.effort, 'temperature_control': False}

    def command(self, workdir, output):
        cmd = [self.binary, '--no-daemon', 'exec', '--ignore-user-config', '--strict-config',
               '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only',
               '--cd', str(workdir), '--color', 'never', '--json',
               '--output-last-message', str(output), '--model', self.model]
        overrides = {
            'model_provider': 'openai', 'forced_login_method': 'chatgpt',
            'web_search': 'disabled', 'project_doc_max_bytes': 0,
            'model_reasoning_effort': self.effort,
            'developer_instructions': (
                'You are a JSON-only experimental question answering model. '
                'Use only the user message and model knowledge. '
                'Never use tools, search, files, skills, memory, or other agents. '
                'Follow the task instructions inside the user message and output one JSON object.'),
        }
        for key, value in overrides.items():
            cmd.extend(['-c', key + '=' + json.dumps(value)])
        # No config means no configured MCP servers. Disable built-in tool/context providers.
        for feature in ['shell_tool', 'unified_exec', 'apps', 'plugins', 'hooks',
                        'memories', 'multi_agent', 'browser_use', 'computer_use',
                        'view_image', 'image_generation', 'skill_search', 'code_mode_host',
                        'sleep_tool', 'goals', 'workspace_dependencies']:
            cmd.extend(['--disable', feature])
        cmd.extend(['--enable', 'skip_host_skill_discovery', '-'])
        return cmd

    def json(self, system, user, temperature=None):
        if temperature is not None:
            raise ValueError('Codex CLI 后端不能精确设置 temperature；detect 请使用 API 后端。')
        prompt = f'Task instructions:\n{system}\n\nTask input:\n{user}\n\nReturn one valid JSON object only.'
        call_id = uuid.uuid4().hex
        self.call_ids.append(call_id)
        audit = {**self.metadata, 'call_id': call_id, 'requested_model': self.model,
                 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                 'prompt': prompt, 'started_unix': time.time(), 'status': 'started'}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix='hallulab-inference-') as directory:
                workdir = Path(directory)
                output = workdir / 'answer.json'
                # stdin keeps experiment content out of command-line process listings.
                result = subprocess.run(self.command(workdir, output), input=prompt,
                                        env=cli_environment(), capture_output=True, text=True,
                                        timeout=self.timeout, cwd=workdir)
                events = []
                for line in result.stdout.splitlines():
                    if line.strip():
                        events.append(json.loads(line))
                audit['events'] = events
                audit['stderr'] = safe_error(result.stderr)
                audit['returncode'] = result.returncode
                for event in events:
                    item = event.get('item', {})
                    if event.get('type') in {'item.started', 'item.updated', 'item.completed'}:
                        if item.get('type') not in {'agent_message', 'reasoning'}:
                            raise RuntimeError('Codex emitted a tool/unknown item; rejecting contaminated inference.')
                    if event.get('type') == 'turn.completed':
                        audit['usage'] = event.get('usage')
                if result.returncode:
                    errors = [e for e in events if e.get('type') in {'error', 'turn.failed'}]
                    raise RuntimeError('Codex request failed: ' + safe_error(json.dumps(errors, ensure_ascii=False))
                                       + ' ' + safe_error(result.stderr))
                if not any(e.get('type') == 'turn.completed' for e in events):
                    raise RuntimeError('Codex did not complete the inference turn')
                if not output.exists():
                    raise RuntimeError('Codex did not write a final answer')
                answer = output.read_text(encoding='utf-8')
                obj = parse_json(answer)
                if not isinstance(obj, dict):
                    raise ValueError('Expected JSON object')
                audit['response'] = obj
                audit['status'] = 'ok'
                return obj
        except Exception as exc:
            audit['status'] = 'failed'
            audit['error'] = safe_error(exc)
            if isinstance(exc, subprocess.TimeoutExpired):
                raise RuntimeError(f'Codex request timed out after {self.timeout:g}s; see call audit {call_id}.') from None
            raise
        finally:
            audit['finished_unix'] = time.time()
            (self.log_dir / f'{call_id}.json').write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
