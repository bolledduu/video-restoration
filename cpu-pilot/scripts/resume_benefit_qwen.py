"""Transport-only resumption: preserve an in-flight conversation across quota waits."""
import argparse
import json
import re
import runpy
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


def retry_seconds(response):
    try:
        return max(5., float(response.headers['retry-after'])) + 2
    except (KeyError, ValueError):
        message = response.json().get('error', {}).get('message', '')
        match = re.search(r'try again in\s+((?:[\d.]+[hms])+)', message)
        if match:
            return max(5., sum(float(n)*{'h':3600,'m':60,'s':1}[u]
                              for n,u in re.findall(r'([\d.]+)([hms])',match[1])))+2
        return 60.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.runtime_root.resolve()/'scripts'))
    import qwen_evidence_agent as api
    status_path = package/'results/benefit_transport_status.json'

    def status(state, **extra):
        status_path.write_text(json.dumps(dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2)+'\n')

    def request(session, method, path, **kwargs):
        generation_retries = 0
        while True:
            response = session.request(method, api.BASE+path, timeout=90, **kwargs)
            if response.status_code == 429:
                delay = retry_seconds(response)
                retry = (datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat()
                status('WAITING_FOR_QUOTA', retry_utc=retry, waiting_seconds=delay)
                print(f'Quota wait: {delay:.0f}s; retaining the current request. Retry UTC {retry}', flush=True)
                time.sleep(delay)
                continue
            if not response.ok:
                try:
                    error = response.json().get('error', {})
                except ValueError:
                    error = {}
                code = str(error.get('code', 'unknown'))
                message = str(error.get('message', ''))
                secret = session.headers.get('Authorization', '').removeprefix('Bearer ')
                if secret:
                    message = message.replace(secret, '[REDACTED]')
                with (package/'results/benefit_transport_errors.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), http_status=response.status_code, code=code, message=message[:800]))+'\n')
                if code == 'tool_use_failed' and generation_retries < 2:
                    generation_retries += 1
                    print('Provider rejected tool generation; retrying identical request; error recorded.', flush=True)
                    time.sleep(5)
                    continue
                status('STOPPED_HTTP_ERROR', http_status=response.status_code, code=code)
                raise RuntimeError(f'Groq HTTP {response.status_code}; response body omitted')
            status('RUNNING')
            return response.json()

    api.request = request
    sys.argv = [str(package/'scripts/benefit_qwen.py'), '--runtime-root', str(args.runtime_root)]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
        sys.argv = [str(package/'scripts/report_benefit_screen.py'), '--runtime-root', str(args.runtime_root)]
        runpy.run_path(sys.argv[0], run_name='__main__')
        status('COMPLETE')
    except BaseException:
        status('STOPPED')
        raise


if __name__ == '__main__':
    main()
