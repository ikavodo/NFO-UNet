"""Triage page for the PanAf500 candidate videos (filter_moving.py output): one row per video with its strip, yes / no /
unsure buttons, keyboard shortcuts, autosave in the browser and a CSV download. Open the HTML file in a browser
(it stays local: images and videos are relative links).

    PYTHONPATH=. python -m benchmark.panaf.triage_page [triage|static_moving|rest|all]   # -> images/panaf/<tag>/<tag>.html
tag 'all' = the three lists merged (each video once), for a second annotator. Type your initials in the page first:
answers autosave per annotator and the download is <tag>_occluded_<initials>.csv (video, split, occluded = yes/no/unsure).
Serve it with  python -m http.server 8765 --directory images/panaf  (videos are linked through images/panaf/videos).
"""
import csv
import html
import json
import os

ROOT = os.path.abspath(os.environ.get('PANAF_ROOT', '../PanAf20K/PanAf500'))
SRC = {'triage': 'candidates', 'static_moving': 'static_moving', 'rest': 'rest'}
ORDER = {'test': 0, 'validation': 1, 'train': 2}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PanAf Occlusion Triage</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;--yes:#1f7a4d;--no:#a33a2b;--uns:#9a6b00;--cur:#2f5fd0}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#33322f;--card:#1f1f1d;--yes:#4cc38a;--no:#f07a68;--uns:#e0b04a;--cur:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
header h1{font-size:16px;margin:0 0 4px}header p{margin:2px 0;color:var(--mut)}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
main{padding:8px 16px 80px;max-width:1100px;margin:auto}
.row{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;margin:8px 0}
.row.cur{outline:2px solid var(--cur)}.meta{display:flex;gap:10px;flex-wrap:wrap;align-items:baseline;margin-bottom:6px}
.meta b{font-family:ui-monospace,monospace}.meta span{color:var(--mut)}
.row img{max-width:100%;height:auto;display:block;border-radius:4px}
.ans{display:flex;gap:6px;margin-top:6px}.ans button.on[data-v=yes]{background:var(--yes);color:#fff;border-color:var(--yes)}
.ans button.on[data-v=no]{background:var(--no);color:#fff;border-color:var(--no)}.ans button.on[data-v=unsure]{background:var(--uns);color:#fff;border-color:var(--uns)}
.hide{display:none}
</style></head><body>
<header><h1>PanAf500 triage: is the moving ape fragmented by vegetation?</h1>
<p><b>yes</b> = during the moving stretch, foliage/branches in front of the ape split its visible body into pieces (for part of the strip is enough). <b>no</b> = ape clearly visible, or only cut by the frame edge / another ape. <b>unsure</b> = can't tell (click the id to open the video).</p>
<p>Keys: <b>y</b> / <b>n</b> / <b>u</b> answer and advance, <b>j</b> / <b>k</b> next / previous. Answers autosave in this browser.</p>
<div class="bar"><label>annotator <input id="who" size="6" placeholder="initials"></label><span id="prog"></span><button id="dl">Download CSV</button><button id="only">Show unanswered only</button></div></header>
<main id="list">ROWS</main>
<script>
const who=document.getElementById('who');try{who.value=localStorage.getItem('panaf-who')||''}catch(e){}
let KEY,A={};function load(){KEY='panaf-triage-TAG-'+who.value.trim();try{A=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){A={}}}load();
who.onchange=()=>{try{localStorage.setItem('panaf-who',who.value.trim())}catch(e){}load();paint()};
const rows=[...document.querySelectorAll('.row')];let cur=0,only=false;
function save(){try{localStorage.setItem(KEY,JSON.stringify(A))}catch(e){}}
function paint(){rows.forEach((r,i)=>{const v=A[r.dataset.id];r.querySelectorAll('.ans button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
 r.classList.toggle('cur',i===cur);r.classList.toggle('hide',only&&!!v&&i!==cur)});
 const n=rows.filter(r=>A[r.dataset.id]).length,y=rows.filter(r=>A[r.dataset.id]==='yes').length;
 document.getElementById('prog').textContent=`${n} / ${rows.length} answered, ${y} yes`}
function go(i){cur=Math.max(0,Math.min(rows.length-1,i));paint();rows[cur].scrollIntoView({block:'center'})}
function set(i,v){if(!who.value.trim()){alert('Type your initials in the annotator box first');who.focus();return}A[rows[i].dataset.id]=v;save();go(i+1)}
rows.forEach((r,i)=>r.querySelectorAll('.ans button').forEach(b=>b.onclick=()=>{cur=i;set(i,b.dataset.v)}));
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();
 if(k==='y')set(cur,'yes');else if(k==='n')set(cur,'no');else if(k==='u')set(cur,'unsure');else if(k==='j')go(cur+1);else if(k==='k')go(cur-1)});
document.getElementById('only').onclick=e=>{only=!only;e.target.textContent=only?'Show all':'Show unanswered only';paint()};
document.getElementById('dl').onclick=()=>{const lines=['video,split,occluded'].concat(rows.map(r=>`${r.dataset.id},${r.dataset.split},${A[r.dataset.id]||''}`));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')+'\\n'],{type:'text/csv'}));a.download='TAG_occluded_'+who.value.trim()+'.csv';a.click()};
paint();
</script></body></html>"""


def main(tag='triage'):
    """triage: moving-ape candidates; static_moving: apes labelled sitting/standing/hanging whose box moves; rest: all other
    videos; all: the three merged, each video once (for a second annotator)."""
    rows, seen = [], set()
    for t in (SRC if tag == 'all' else [tag]):
        for r in sorted(csv.DictReader(open(f'results/panaf/{SRC[t]}.csv')), key=lambda r: (ORDER[r['split']], r['video'])):
            if r['video'] not in seen:
                seen.add(r['video']); rows.append(dict(r, src=t))
    out = []
    for i, r in enumerate(rows, 1):
        v = html.escape(r['video'])
        out.append(f'<div class="row" data-id="{v}" data-split="{r["split"]}"><div class="meta"><b>{i}.</b>'
                   f'<b><a href="../videos/{v}.mp4" target="_blank">{v}</a></b><span>{r["split"]}</span>'
                   f'<span>ape {r["ape_id"]}: {html.escape(r["moving_behaviours"])}</span></div>'
                   f'<img loading="lazy" src="../{r["src"]}/strips/{v}.jpg" alt="{v} strip">'
                   '<div class="ans"><button data-v="yes">occluded (y)</button><button data-v="no">not occluded (n)</button>'
                   '<button data-v="unsure">unsure (u)</button></div></div>')
    os.makedirs(f'images/panaf/{tag}', exist_ok=True)
    if not os.path.lexists('images/panaf/videos'):
        os.symlink(os.path.join(ROOT, 'videos'), 'images/panaf/videos')
    page = PAGE.replace('ROWS', '\n'.join(out)).replace('TAG', 'v1' if tag == 'triage' else tag)
    if tag == 'triage':
        page = page.replace("'v1_occluded_'", "'triage_occluded_'")
    else:
        q = {'static_moving': 'apes labelled sitting / standing / hanging whose box moves: fragmented by vegetation?',
             'rest': 'remaining videos (strip spans the whole track): is the ape fragmented by vegetation at any point?',
             'all': 'all candidate videos (second annotator): is the ape fragmented by vegetation?'}[tag]
        page = page.replace('<title>PanAf Occlusion Triage</title>', f'<title>PanAf Triage {tag}</title>').replace(
            'is the moving ape fragmented by vegetation?', q)
    open(f'images/panaf/{tag}/{tag}.html', 'w').write(page)
    print(f'wrote images/panaf/{tag}/{tag}.html,', len(rows), 'videos (test first)')


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:2])
