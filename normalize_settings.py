import json
from pathlib import Path
path = Path('ui/settings.json')
data = json.loads(path.read_text(encoding='utf-8'))
sections = ['last_settings', 'prev_settings']
for s in sections:
    data.setdefault(s, {})
    sec = data[s]
    sec.setdefault('輪郭強調(新)', False)
    sec.setdefault('輪郭強さ', '60')
    sec.setdefault('輪郭太さ', '30')
    if '輪郭強さ(0-100)' in sec:
        sec['輪郭強さ'] = sec.pop('輪郭強さ(0-100)')
    sec.setdefault('バイラテラルσ', 75)
    sec.setdefault('グラデーション強度', 0)
    sec.setdefault('use_super_sampling', False)
    data[s] = sec
path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
print('normalized settings.json')
