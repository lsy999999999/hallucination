import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hallulab.cli import load_client, make_parser, run_experiment
from hallulab.codex_client import CodexJSONClient, cli_environment
from hallulab.diagnostics import doctor


class CodexClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = patch.dict(os.environ, {'HALLULAB_BACKEND':'codex',
                              'OPENAI_API_KEY':'fake-secret','CODEX_API_KEY':'fake-other',
                              'OPENAI_BASE_URL':'https://invalid.example/v1',
                              'CODEX_MODEL':'test-model','CODEX_HOME':'/test/existing-home'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.which = patch('hallulab.codex_client.shutil.which',return_value='/test/codex')
        self.which.start()
        self.addCleanup(self.which.stop)

    def run_cli(self, cmd, **kwargs):
        if cmd[1:] == ['login','status']:
            return SimpleNamespace(returncode=0,stdout='Logged in using ChatGPT',stderr='')
        if cmd[1:] == ['--version']:
            return SimpleNamespace(returncode=0,stdout='codex-cli fixture',stderr='')
        self.assertNotIn('fake-secret', str(cmd))
        self.assertNotIn('OPENAI_API_KEY', kwargs['env'])
        self.assertEqual(kwargs['env']['CODEX_HOME'],'/test/existing-home')
        self.assertTrue(Path(kwargs['cwd']).is_dir())
        self.assertIn('--ignore-user-config',cmd)
        self.assertIn('--ephemeral',cmd)
        self.assertIn('forced_login_method="chatgpt"',cmd)
        self.assertIn('web_search="disabled"',cmd)
        self.assertNotIn('danger-full-access',cmd)
        output = Path(cmd[cmd.index('--output-last-message')+1])
        output.write_text('{"ok":true}')
        return SimpleNamespace(returncode=0,stderr='',stdout='\n'.join([
            json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'{"ok":true}'}}),
            json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':5}})]))

    def test_account_inference_and_audit(self):
        with patch('hallulab.codex_client.subprocess.run', side_effect=self.run_cli) as run:
            client = CodexJSONClient(log_dir=self.root)
            self.assertEqual(client.json('system','user'),{'ok':True})
            self.assertEqual(client.json('second','independent'),{'ok':True})
        self.assertEqual(run.call_count,4)
        audits=[json.loads(p.read_text()) for p in self.root.glob('*.json')]
        self.assertEqual(len(audits),2)
        self.assertTrue(all(a['status']=='ok' and a['auth']=='chatgpt' for a in audits))
        self.assertEqual(audits[0]['usage']['output_tokens'],5)
        self.assertEqual(len(set(client.call_ids)),2)

    def test_api_key_auth_is_rejected_without_model_request(self):
        reply=SimpleNamespace(returncode=0,stdout='API key configured',stderr='')
        with patch('hallulab.codex_client.subprocess.run',return_value=reply) as run:
            with self.assertRaisesRegex(RuntimeError,'不会回退到 API Key'):
                CodexJSONClient(log_dir=self.root)
        self.assertEqual(run.call_count,1)

    def test_cli_timeout_is_audited_without_retry(self):
        def run(cmd,**kw):
            if 'exec' in cmd:
                raise subprocess.TimeoutExpired(cmd,180)
            return self.run_cli(cmd,**kw)
        with patch('hallulab.codex_client.subprocess.run',side_effect=run) as runner:
            client=CodexJSONClient(log_dir=self.root)
            with self.assertRaisesRegex(RuntimeError,'timed out'):
                client.json('sys','user')
        self.assertEqual(runner.call_count,3)
        self.assertEqual(json.loads(next(self.root.glob('*.json')).read_text())['status'],'failed')

    def test_tool_use_invalidates_response(self):
        def run(cmd,**kw):
            result=self.run_cli(cmd,**kw)
            if 'exec' in cmd:
                event=json.dumps({'type':'item.started','item':{'type':'web_search'}})
                result.stdout=event+'\n'+result.stdout
            return result
        with patch('hallulab.codex_client.subprocess.run',side_effect=run):
            client=CodexJSONClient(log_dir=self.root)
            with self.assertRaisesRegex(RuntimeError,'contaminated'):
                client.json('sys','user')
        self.assertEqual(json.loads(next(self.root.glob('*.json')).read_text())['status'],'failed')

    def test_temperature_is_not_silently_ignored(self):
        with patch('hallulab.codex_client.subprocess.run',side_effect=self.run_cli) as run:
            client=CodexJSONClient(log_dir=self.root)
            with self.assertRaisesRegex(ValueError,'temperature'):
                client.json('sys','user',temperature=1.0)
        self.assertEqual(run.call_count,2)

    def test_factory_and_doctor_select_account_backend(self):
        with patch('hallulab.codex_client.subprocess.run',side_effect=self.run_cli), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIsInstance(load_client(None),CodexJSONClient)
            self.assertTrue(doctor())
        self.assertIn('ChatGPT account',output.getvalue())
        self.assertNotIn('fake-secret',output.getvalue())
        self.assertEqual(cli_environment()['CODEX_HOME'],'/test/existing-home')

    def test_resume_rejects_api_and_account_mix(self):
        corpus=self.root/'corpus.jsonl'
        questions=self.root/'questions.jsonl'
        corpus.write_text(json.dumps({'id':'test','text':'test'}))
        questions.write_text(json.dumps({'id':'q1','question':'test'}))
        output=self.root/'experiment.jsonl'
        output.with_suffix('.meta.json').write_text('{}')
        args=make_parser().parse_args(['run','--backend','codex','--corpus',str(corpus),
                                      '--questions',str(questions),'--out',str(output)])
        with patch('hallulab.codex_client.subprocess.run',side_effect=self.run_cli):
            with self.assertRaisesRegex(ValueError,'settings changed'):
                run_experiment(args)

    def test_paired_experiment_records_three_independent_account_calls(self):
        corpus=self.root/'corpus.jsonl'
        questions=self.root/'questions.jsonl'
        corpus.write_text(json.dumps({'id':'alpha_p001_c00','text':'Alpha began in 2020.',
                                     'paper_id':'alpha','page':1,'title':'Alpha'}))
        questions.write_text(json.dumps({'id':'q1','type':'single_fact',
                                        'question':'When did Alpha begin?'}))
        output=self.root/'paired.jsonl'
        args=make_parser().parse_args(['run','--backend','codex','--limit','1',
                                      '--corpus',str(corpus),'--questions',str(questions),
                                      '--out',str(output)])
        def run(cmd,**kw):
            result=self.run_cli(cmd,**kw)
            if 'exec' in cmd:
                prompt=kw['input']
                if 'entailment verifier' in prompt:
                    obj={'verdicts':[{'claim_id':'0','label':'SUPPORTED','reason':'fixture'}]}
                else:
                    obj={'answer':'Alpha began in 2020.','abstained':False,
                         'claims':[{'text':'Alpha began in 2020.',
                                    'citations':['alpha_p001_c00'] if 'Sources (ONLY' in prompt else []}]}
                Path(cmd[cmd.index('--output-last-message')+1]).write_text(json.dumps(obj))
            return result
        with patch.dict(os.environ,{'HALLULAB_CODEX_LOG_DIR':str(self.root/'calls')}), \
                patch('hallulab.codex_client.subprocess.run',side_effect=run) as runner, \
                contextlib.redirect_stdout(io.StringIO()):
            run_experiment(args)
            # Resume should consume no more inference calls.
            inference_calls=sum('exec' in call.args[0] for call in runner.call_args_list)
            run_experiment(args)
            self.assertEqual(sum('exec' in call.args[0] for call in runner.call_args_list),inference_calls)
        row=json.loads(output.read_text())
        self.assertEqual(inference_calls,3)
        self.assertEqual(row['A']['claims'][0]['citations'],[])
        self.assertEqual(row['B']['claims'],row['C']['claims'])
        self.assertEqual(row['backend'],'codex')
        self.assertEqual(len(row['codex_call_ids']['generation']),2)
        self.assertEqual(len(row['codex_call_ids']['verification']),1)
        self.assertEqual(len(list((self.root/'calls').glob('*.json'))),3)
        meta=json.loads(output.with_suffix('.meta.json').read_text())
        self.assertEqual(meta['generation_client']['auth'],'chatgpt')


if __name__=='__main__':
    unittest.main()
