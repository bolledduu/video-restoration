import os, sys, time, winreg, requests, json
sys.path.insert(0, 'scripts')
from ava_evidence_environment import EvidenceEnvironment, schemas
from run_evidence_agent import blind_case, SYSTEM

key = os.environ.get('GROQ_API_KEY') or winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment'), 'GROQ_API_KEY')[0]

measured = json.loads(open('results/ava_smoke_v1/measurements.json').read())
case = measured['cases'][2] # Case 3: eaa58bfe0ca84011
env = EvidenceEnvironment(blind_case(case))

prompt = "Inspect the available candidates, compare restored candidates with capture, select the best candidate, and submit decisions covering all timestamps using submit_decisions."

messages = [
    {'role': 'system', 'content': SYSTEM},
    {'role': 'user', 'content': prompt}
]

for turn in range(12):
    has_inspected = set(env.data) <= env.inspected
    has_compared = (set(env.data) - {'captured'}) <= env.compared
    ready = has_inspected and has_compared
    tools = schemas()
    tool_choice = 'auto'

    if has_inspected and not has_compared and turn >= 6:
        messages.append({'role': 'user', 'content': 'Do not inspect local intervals. Call compare_with_capture for the restored candidates now.'})
    elif ready and turn >= 8:
        messages.append({'role': 'user', 'content': 'All candidate evidence has been inspected and compared. Please call submit_decisions now to submit the speaking decisions.'})
    if turn == 11:
        tools = [s for s in schemas() if s['function']['name'] == 'submit_decisions']
        tool_choice = 'required'
        if not ready:
            messages.append({'role': 'user', 'content': 'Budget limit reached. You must submit decisions now using candidate="captured".'})

    payload = {
        'model': 'openai/gpt-oss-20b',
        'messages': messages,
        'tools': tools,
        'tool_choice': tool_choice,
        'temperature': 0,
        'max_completion_tokens': 1024
    }

    data = None
    for attempt in range(5):
        r = requests.post('https://api.groq.com/openai/v1/chat/completions', headers={'Authorization': 'Bearer ' + key}, json=payload)
        if r.status_code == 429:
            retry_after = float(r.headers.get('retry-after', 5))
            print(f'Rate limited, waiting {retry_after:.1f}s...')
            time.sleep(retry_after + 1)
            continue
        if not r.ok:
            print('HTTP Error:', r.status_code, r.text[:200])
            break
        data = r.json()
        break

    if not data:
        print('Failed to get response after retries')
        break

    msg = data['choices'][0]['message']
    messages.append(msg)
    calls = msg.get('tool_calls') or []
    print(f'Turn {turn}: {len(calls)} calls -> {[c["function"]["name"] for c in calls]}')
    for c in calls:
        name = c['function']['name']
        args = json.loads(c['function']['arguments'])
        try:
            out = env.call(name, args)
            messages.append({'role': 'tool', 'tool_call_id': c['id'], 'content': json.dumps(out)})
        except Exception as e:
            print(f'Error calling {name}: {e}')
            messages.append({'role': 'tool', 'tool_call_id': c['id'], 'content': json.dumps({'error': str(e)})})
    if env.submission:
        print('SUCCESS! Submission received:', env.submission['candidate'], len(env.submission['decisions']), 'decisions')
        break
    time.sleep(1)
