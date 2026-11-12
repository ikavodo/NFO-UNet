"""Occlusion triage page for PanAf500 (one row per video: a strip of 6 frames around one ape, filter_moving.py output).
Four classes (user, 2026-11-12): occluded (> ~50% of the ape hidden), partial (25-50%, or frames that clearly disagree),
clear (< 25%), unknown (ape not really observable). Keyboard answers, a zoom view of single frames, autosave per annotator in the browser, CSV download.

    PYTHONPATH=. python -m benchmark.panaf.triage_page [all|triage|static_moving|rest]   # -> images/panaf/<tag>/<tag>.html
tag 'all' = the three lists merged (each video once, 500 videos). Type your initials first: answers autosave per
annotator and the download is <tag>_occlusion_<initials>.csv (video, split, occlusion). Rule: the class that fits most
of the six frames; partial when they clearly disagree (some clear, some heavily hidden).
Serve with  python -m http.server 8765 --bind 127.0.0.1 --directory images/panaf  (videos via the images/panaf/videos link).
"""
import csv
import html
import os

ORDER = {'test': 0, 'validation': 1, 'train': 2}
ROOT = os.path.abspath(os.environ.get('PANAF_ROOT', '../PanAf20K/PanAf500'))
SRC = {'triage': 'candidates', 'static_moving': 'static_moving', 'rest': 'rest'}
CLASSES = [('occluded', 'o', '1'), ('partial', 'p', '2'), ('clear', 'c', '3'), ('unknown', 'u', '4')]

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PanAf Occlusion TAG</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;--occluded:#a33a2b;--partial:#b86e00;--clear:#1f7a4d;--unknown:#5b5b8f;--cur:#2f5fd0}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#33322f;--card:#1f1f1d;--occluded:#f07a68;--partial:#e0a04a;--clear:#4cc38a;--unknown:#9a9ae0;--cur:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
header h1{font-size:16px;margin:0 0 4px}header p{margin:2px 0;color:var(--mut)}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px}
button,input{font:inherit;padding:4px 10px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px}
button{cursor:pointer}
main{padding:8px 16px 80px;max-width:1300px;margin:auto}
.row{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;margin:8px 0}
.row.cur{outline:2px solid var(--cur)}.meta{display:flex;gap:10px;flex-wrap:wrap;align-items:baseline;margin-bottom:6px}
.meta b{font-family:ui-monospace,monospace}.meta span{color:var(--mut)}
.strip{position:relative;cursor:zoom-in}.strip img{width:100%;height:auto;display:block;border-radius:4px}
.ans{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}
.ans button.on{color:#fff}.ans button.on[data-v=occluded]{background:var(--occluded);border-color:var(--occluded)}
.ans button.on[data-v=partial]{background:var(--partial);border-color:var(--partial)}.ans button.on[data-v=clear]{background:var(--clear);border-color:var(--clear)}
.ans button.on[data-v=unknown]{background:var(--unknown);border-color:var(--unknown)}

.hide{display:none}
#zoom{position:fixed;inset:0;z-index:5;background:rgba(0,0,0,.88);display:none;flex-direction:column;align-items:center;justify-content:center;color:#eee}
#zoom.on{display:flex}#zwrap{overflow:auto;max-width:96vw;max-height:84vh}
#zimg{background-repeat:no-repeat}
#zinfo{margin-top:8px;font-size:13px;text-align:center}
</style></head><body>
<header><h1>PanAf500 occlusion: QUESTION</h1>
<p>How much of the ape's body is hidden by vegetation or objects in front of it? Choose the class that fits most of the six frames. If the frames clearly disagree (some clear, some heavily hidden), choose <b>partial</b>. The last frame often shows the ape disappearing into cover, so judge mainly on the first five.
 <b>occluded</b> (o/1) = more than ~50% hidden. <b>partial</b> (p/2) = roughly 25-50%. <b>clear</b> (c/3) = less than 25%.
 <b>unknown</b> (u/4) = the ape is not really observable (too small, dark, blurred, or out of view). The frame edge and other apes do not count as occluders.</p>
<p>Keys: <b>o p c u</b> (or <b>1-4</b>) answer and advance; <b>j</b>/<b>k</b> next/previous video; <b>z</b> zoom in (or click a frame);
 in zoom: <b>&larr;</b>/<b>&rarr;</b> frame, <b>+</b>/<b>-</b> magnify, <b>z</b> or <b>Esc</b> zoom out (answer keys work inside zoom too). Click the id to open the video.</p>
<div class="bar"><label>annotator <input id="who" size="6" placeholder="initials"></label><span id="prog"></span>
<button id="dl">Download CSV</button><button id="only">Show unanswered only</button></div></header>
<main id="list">ROWS</main>
<div id="zoom"><div id="zwrap"><div id="zimg"></div></div><div id="zinfo"></div></div>
<script>
const NF=6,CL={o:'occluded',1:'occluded',p:'partial',2:'partial',c:'clear',3:'clear',u:'unknown',4:'unknown'};
const who=document.getElementById('who');try{who.value=localStorage.getItem('panaf-who')||''}catch(e){}
let KEY,A={};function load(){KEY='panaf-occ4-TAG-'+who.value.trim();try{A=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){A={}}}load();
who.onchange=()=>{try{localStorage.setItem('panaf-who',who.value.trim())}catch(e){}load();paint()};
const rows=[...document.querySelectorAll('.row')];let cur=0,only=false,zf=0,zs=1;
function save(){try{localStorage.setItem(KEY,JSON.stringify(A))}catch(e){}}
function paint(){rows.forEach((r,i)=>{const v=A[r.dataset.id];r.querySelectorAll('.ans button[data-v]').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
 r.classList.toggle('cur',i===cur);r.classList.toggle('hide',only&&!!v&&i!==cur)});
 const c={occluded:0,partial:0,clear:0,unknown:0};rows.forEach(r=>{const v=A[r.dataset.id];if(v)c[v]++});
 const n=rows.filter(r=>A[r.dataset.id]).length;
 document.getElementById('prog').textContent=`${n} / ${rows.length} answered (occluded ${c.occluded}, partial ${c.partial}, clear ${c.clear}, unknown ${c.unknown})`}
function go(i){cur=Math.max(0,Math.min(rows.length-1,i));paint();rows[cur].scrollIntoView({block:'center'});if(zoomOn())showZoom()}
function set(i,v){if(!who.value.trim()){alert('Type your initials in the annotator box first');who.focus();return}A[rows[i].dataset.id]=v;save();if(zoomOn())zf=0;go(i+1)}
const Z=document.getElementById('zoom'),ZI=document.getElementById('zimg');
function zoomOn(){return Z.classList.contains('on')}
function showZoom(){const r=rows[cur],src=r.querySelector('.strip img').getAttribute('src');
 const base=Math.min(window.innerHeight*0.8,window.innerWidth*0.9)*zs;
 ZI.style.width=base+'px';ZI.style.height=base+'px';ZI.style.backgroundImage=`url("${src}")`;
 ZI.style.backgroundSize=`${NF*100}% 100%`;ZI.style.backgroundPosition=`${zf/(NF-1)*100}% 0`;Z.classList.add('on');
 document.getElementById('zinfo').innerHTML=`<b>${r.dataset.id}</b> &middot; frame ${zf+1} / ${NF} &middot; x${zs} &middot; answer: ${A[r.dataset.id]||'-'} &middot; video ${cur+1} / ${rows.length}`}
function hideZoom(){Z.classList.remove('on')}
rows.forEach((r,i)=>{r.querySelectorAll('.ans button[data-v]').forEach(b=>b.onclick=()=>{cur=i;set(i,b.dataset.v)});
 const s=r.querySelector('.strip');s.onclick=e=>{const b=s.getBoundingClientRect();cur=i;zf=Math.min(NF-1,Math.floor((e.clientX-b.left)/b.width*NF));paint();showZoom()}});
Z.onclick=e=>{if(e.target===Z)hideZoom()};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();
 if(CL[k]){set(cur,CL[k]);return}
 if(k==='z'){zoomOn()?hideZoom():showZoom();return}
 if(k==='escape'){hideZoom();return}
 if(zoomOn()){if(k==='arrowright'){zf=Math.min(NF-1,zf+1);showZoom();e.preventDefault()}else if(k==='arrowleft'){zf=Math.max(0,zf-1);showZoom();e.preventDefault()}
  else if(k==='+'||k==='='){zs=Math.min(4,zs+0.5);showZoom()}else if(k==='-'){zs=Math.max(1,zs-0.5);showZoom()}
  else if(k==='j'){zf=0;go(cur+1)}else if(k==='k'){zf=0;go(cur-1)}return}
 if(k==='j')go(cur+1);else if(k==='k')go(cur-1)});
document.getElementById('only').onclick=e=>{only=!only;e.target.textContent=only?'Show all':'Show unanswered only';paint()};
document.getElementById('dl').onclick=()=>{const lines=['video,split,occlusion'].concat(rows.map(r=>`${r.dataset.id},${r.dataset.split},${A[r.dataset.id]||''}`));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')+'\\n'],{type:'text/csv'}));a.download='TAG_occlusion_'+who.value.trim()+'.csv';a.click()};
paint();
</script></body></html>"""

QUESTION = {'all': 'all 500 videos', 'triage': 'moving-ape candidates',
            'static_moving': 'apes labelled sitting / standing / hanging whose box moves', 'rest': 'remaining videos'}


def main(tag='all'):
    rows, seen = [], set()
    for t in (SRC if tag == 'all' else [tag]):
        for r in csv.DictReader(open(f'results/panaf/{SRC[t]}.csv')):
            if r['video'] not in seen:
                seen.add(r['video']); rows.append(dict(r, src=t))
    rows.sort(key=lambda r: (ORDER[r['split']], r['video']))
    btn = ''.join(f'<button data-v="{c}">{c} ({k}/{n})</button>' for c, k, n in CLASSES)
    out = []
    for i, r in enumerate(rows, 1):
        v = html.escape(r['video'])
        out.append(f'<div class="row" data-id="{v}" data-split="{r["split"]}"><div class="meta"><b>{i}.</b>'
                   f'<b><a href="../videos/{v}.mp4" target="_blank">{v}</a></b><span>{r["split"]}</span>'
                   f'<span>ape {r["ape_id"]}: {html.escape(r["moving_behaviours"])}</span></div>'
                   f'<div class="strip"><img loading="lazy" src="../{r["src"]}/strips/{v}.jpg" alt="{v} strip"></div>'
                   f'<div class="ans">{btn}</div></div>')
    os.makedirs(f'images/panaf/{tag}', exist_ok=True)
    if not os.path.lexists('images/panaf/videos'):
        os.symlink(os.path.join(ROOT, 'videos'), 'images/panaf/videos')
    page = PAGE.replace('ROWS', '\n'.join(out)).replace('QUESTION', QUESTION[tag]).replace('TAG', tag)
    open(f'images/panaf/{tag}/{tag}.html', 'w').write(page)
    print(f'wrote images/panaf/{tag}/{tag}.html,', len(rows), 'videos (test first)')


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:2])
