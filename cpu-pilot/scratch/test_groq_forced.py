import os, sys, winreg, requests, json
sys.path.insert(0, 'scripts')
from ava_evidence_environment import schemas

key = os.environ.get('GROQ_API_KEY') or winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment'), 'GROQ_API_KEY')[0]
tools = [s for s in schemas() if s['function']['name'] == 'submit_decisions']
body = {
    'model': 'openai/gpt-oss-20b',
    'messages': [
        {'role': 'system', 'content': 'You must call submit_decisions to submit speaking decisions for 99 timestamps.'},
        {'role': 'user', 'content': 'Submit speaking decisions for candidate fastdvdnet covering timestamps 0 to 99.'}
    ],
    'tools': tools,
    'tool_choice': 'required',
    'max_completion_tokens': 2048
}
r = requests.post('https://api.groq.com/openai/v1/chat/completions', headers={'Authorization': 'Bearer ' + key}, json=body)
print('Status:', r.status_code)
if r.status_code == 200:
    msg = r.json()['choices'][0]['message']
    print('Reasoning length:', len(msg.get('reasoning', '')))
    print('Tool call name:', msg.get('tool_calls', [{}])[0].get('function', {}).get('name'))
    print('Tool call args preview:', msg.get('tool_calls', [{}])[0].get('function', {}).get('arguments', '')[:100])
else:
    print('Error:', r.text)
