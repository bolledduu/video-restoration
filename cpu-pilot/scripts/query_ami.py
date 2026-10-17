import urllib.request
import urllib.parse
import re

params = [
    ('sessionID', 'ES2002'),
    ('ES', 'a'),
    ('mediaType', 'DivXvideos'),
    ('mediaType', 'Headset-'),
    ('mediaType', 'Mix-Headset'),
]
data = urllib.parse.urlencode(params).encode('utf-8')
req = urllib.request.Request('https://homepages.inf.ed.ac.uk/cgi/simonk/AMI/estimate.cgi', data=data)
res = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')

idx = res.find("fillText('wgetsh'")
if idx != -1:
    end_idx = res.find("');", idx)
    content = res[idx:end_idx]
    # replace literal \n with actual newlines
    lines = content.replace('\\n', '\n').split('\n')
    for line in lines:
        if 'http' in line:
            print(line.strip())
else:
    print("Could not find fillText('wgetsh')")
