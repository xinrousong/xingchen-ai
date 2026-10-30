# -*- coding: utf-8 -*-
"""星辰 · 训练监控面板（零依赖：只用 Python 标准库 + 浏览器）

用法（训练运行期间或结束后都可以）：
  python scripts/monitor.py            # 默认读取 checkpoints/train_log.jsonl
  python scripts/monitor.py --port 8790

然后浏览器打开 http://127.0.0.1:8765 ，每 3 秒自动刷新。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------- 温度/功耗查询（2 秒缓存，避免高频拉起 nvidia-smi） ----------------
_temp_cache = {"t": 0.0, "data": {}}
_temp_lock = threading.Lock()


def query_temps() -> dict:
    with _temp_lock:
        now = time.time()
        if _temp_cache["data"] and now - _temp_cache["t"] < 2:
            return _temp_cache["data"]
        data = {"gpu_temp": None, "power_w": None, "gpu_util": None, "cpu": None}
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            g, p, u = [x.strip() for x in out.split(",")]
            data.update(gpu_temp=float(g), power_w=float(p), gpu_util=float(u))
        except Exception:
            pass
        try:
            import psutil
            data["cpu"] = psutil.cpu_percent()
        except Exception:
            pass
        _temp_cache.update(t=now, data=data)
        return data

HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>星辰 · 训练监控</title>
<style>
 body{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#0f1420;color:#dfe6f3;margin:0;padding:24px;max-width:980px;}
 h1{font-size:20px;margin:0 0 4px;} .sub{color:#8fa0bf;font-size:13px;margin-bottom:18px;}
 .stats{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px;}
 .card{background:#171f31;border:1px solid #26324c;border-radius:10px;padding:10px 16px;}
 .card .v{font-size:21px;font-weight:600;} .card .k{font-size:12px;color:#8fa0bf;margin-top:3px;}
 .panel{background:#171f31;border:1px solid #26324c;border-radius:10px;padding:14px;margin-bottom:18px;}
 .panel h2{font-size:14px;margin:0 0 8px;color:#a8b8d8;font-weight:500;} canvas{width:100%;height:260px;display:block;}
 .legend{font-size:12px;color:#8fa0bf;} .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 12px;}
 .done{color:#5ad19f;font-weight:600;font-size:16px;}
</style></head><body>
<h1>星辰 · 训练监控 <span id="done" class="done"></span></h1>
<div class="sub">每 3 秒自动刷新；训练结束或中断后仍可打开回看全过程。</div>
<div class="stats" id="stats"></div>
<div class="panel"><h2>损失曲线（越低 = 模型越会预测下一个字）</h2>
 <canvas id="c1"></canvas>
 <div class="legend"><span class="dot" style="background:#5b8dff"></span>训练 loss
 <span class="dot" style="background:#ff7a6b"></span>验证 loss（模型在没背过的数据上的成绩）</div></div>
<div class="panel"><h2>学习率（先爬坡 warmup，再沿余弦缓降）</h2><canvas id="c2"></canvas></div>
<script>
async function refresh(){
  try{
    const [logR, tempR] = await Promise.all([
      fetch('/log?ts='+Date.now(), {cache:'no-store'}),
      fetch('/temps', {cache:'no-store'})
    ]);
    const text = await logR.text();
    let temps = {}; try{ temps = await tempR.json(); }catch(e){}
    const rows = text.trim() ? text.trim().split('\\n')
      .map(s=>{try{return JSON.parse(s)}catch(e){return null}}).filter(Boolean) : [];
    render(rows, temps);
  }catch(e){}
}
function render(rows, temps){
  const meta = rows.find(r=>r.type==='meta');
  const tr = rows.filter(r=>r.type==='train').map(r=>({x:r.iter,y:r.loss,r}));
  const ev = rows.filter(r=>r.type==='eval').map(r=>({x:r.iter,y:r.val_loss}));
  const lr = rows.filter(r=>r.type==='train').map(r=>({x:r.iter,y:r.lr}));
  const lastTr = tr.length?tr[tr.length-1]:null;
  const lastEv = ev.length?ev[ev.length-1]:null;
  const done = rows.some(r=>r.type==='done');
  const target = meta ? meta.max_iters : null;
  const cards = [];
  if(temps && temps.gpu_temp != null){
    const t = temps.gpu_temp;
    const col = t>80 ? '#ff5a5a' : (t>70 ? '#ffb35a' : '#5ad19f');
    cards.push(['GPU 温度', '<span style="color:'+col+'">'+t+'°C</span>']);
    cards.push(['GPU 功耗', temps.power_w!=null ? temps.power_w+' W' : '—']);
    cards.push(['GPU 利用率', temps.gpu_util+'%']);
  }
  if(temps && temps.cpu != null) cards.push(['CPU 占用', temps.cpu+'%']);
  cards.push(['训练步数', lastTr ? lastTr.x + (target?' / '+target:'') : '0']);
  cards.push(['训练 loss', lastTr ? lastTr.y.toFixed(4) : '—']);
  if(lastEv) cards.push(['验证 loss', lastEv.y.toFixed(3)]);
  if(lastTr && lastTr.r.tok_s) cards.push(['速度', (lastTr.r.tok_s/1000).toFixed(1)+'k tok/s']);
  if(lastTr && lastTr.r.vram_gb) cards.push(['显存', lastTr.r.vram_gb.toFixed(2)+' GB']);
  if(meta) cards.push(['模型', meta.name+' ('+meta.params_m+'M)']);
  document.getElementById('stats').innerHTML =
    cards.map(c=>'<div class="card"><div class="v">'+c[1]+'</div><div class="k">'+c[0]+'</div></div>').join('');
  document.getElementById('done').textContent = done ? '✔ 训练已完成' : '';
  drawChart('c1',[{data:tr,color:'#5b8dff'},{data:ev,color:'#ff7a6b'}]);
  drawChart('c2',[{data:lr,color:'#ffd166'}]);
}
function drawChart(id, series){
  const cv = document.getElementById(id);
  const w = cv.clientWidth || 900, h = cv.clientHeight || 260;
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0,0,w,h);
  const pts = series.flatMap(s=>s.data);
  ctx.font='11px sans-serif';
  if(!pts.length){ ctx.fillStyle='#8fa0bf'; ctx.fillText('暂无数据，等待训练写入…', 16, 24); return; }
  const xs = pts.map(p=>p.x), ys = pts.map(p=>p.y);
  let x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  if(x1===x0) x1=x0+1;
  const padY=(y1-y0)*0.1 || 0.5; y0-=padY; y1+=padY;
  const L=52,R=12,Tp=10,B=26, cw=w-L-R, ch=h-Tp-B;
  const X=x=>L+(x-x0)/(x1-x0)*cw, Y=y=>Tp+ch-(y-y0)/(y1-y0)*ch;
  ctx.strokeStyle='#26324c'; ctx.fillStyle='#8fa0bf';
  for(let i=0;i<=4;i++){
    const yv=y0+(y1-y0)*i/4, yy=Y(yv);
    ctx.beginPath(); ctx.moveTo(L,yy); ctx.lineTo(w-R,yy); ctx.stroke();
    ctx.fillText(fmt(yv), 6, yy+4);
  }
  ctx.fillText(String(Math.round(x0)), L, h-8);
  ctx.fillText(String(Math.round((x0+x1)/2)), L+cw/2-12, h-8);
  ctx.fillText(String(Math.round(x1)), w-R-24, h-8);
  for(const s of series){
    if(!s.data.length) continue;
    ctx.strokeStyle=s.color; ctx.lineWidth=1.6; ctx.beginPath();
    s.data.forEach((p,i)=>{ const px=X(p.x), py=Y(p.y); i?ctx.lineTo(px,py):ctx.moveTo(px,py); });
    ctx.stroke();
  }
}
function fmt(v){ if(Math.abs(v)>=1000) return (v/1000).toFixed(1)+'k'; if(Math.abs(v)>=10) return v.toFixed(0); return v.toFixed(2); }
refresh(); setInterval(refresh, 3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    log_path: Path

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            ctype = "text/html; charset=utf-8"
        elif self.path.startswith("/log"):
            try:
                body = Path(self.log_path).read_text(encoding="utf-8").encode("utf-8")
            except FileNotFoundError:
                body = b""
            ctype = "text/plain; charset=utf-8"
        elif self.path.startswith("/temps"):
            body = json.dumps(query_temps()).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静默访问日志
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "checkpoints" / "train_log.jsonl"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.log_path = Path(args.log)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[monitor] 面板已就绪：http://{args.host}:{args.port}")
    print(f"[monitor] 数据源：{Handler.log_path}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[monitor] 已关闭")


if __name__ == "__main__":
    main()
