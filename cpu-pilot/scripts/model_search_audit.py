"""Record public index searches; absence from a search is not proof of nonexistence."""
import datetime
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    urls=[
      'https://api.github.com/search/repositories?q='+urllib.parse.quote(q)
      for q in ['GAVN','"Audio-Assisted Face Video Restoration"','"DAVD-Net"']]
    urls += ['https://huggingface.co/api/models?search='+q for q in ['GAVN','DAVD-Net']]
    records=[]
    for url in urls:
        row={'url':url,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            with urllib.request.urlopen(url,timeout=30) as r:obj=json.load(r)
            if isinstance(obj,dict):
                row['total_count']=obj.get('total_count')
                row['results']=[{'repository':x['full_name'],'description':x.get('description')} for x in obj.get('items',[])]
            else:row['results']=[{'model_id':x['id']} for x in obj]
        except Exception as exc:row['error']=str(exc)
        records.append(row);print(json.dumps(row),flush=True)
    (ROOT/'results'/'model_search_audit.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
