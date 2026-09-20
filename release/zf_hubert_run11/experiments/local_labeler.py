#!/usr/bin/env python
"""Render a batch of windows as a standalone HTML labeling page. No Jupyter required.

Each item shows a WIDE context spectrogram with the current 1 s window boxed in it, a zoomed
spectrogram of just that window underneath, and a player for the window's audio. Labels:

    1 call          a vocalization, essentially clean
    2 call+noise    a vocalization with substantial background over it
    3 noise         background sound, no vocalization (cage, wing, movement, other species)
    4 silence       nothing audible

Why call and call+noise are separate: for detection they collapse to one positive class, but
keeping them apart lets you ask afterwards whether the detector's misses are concentrated in
the noisy ones -- which is the single most likely explanation for recall being limited, and you
cannot recover that distinction once it is labeled away.

Why noise and silence are separate: the retracted 0.984 result happened because silence leaked
into the negative class and made the task trivially easy. Keeping them apart means you can
verify your negatives are mostly real sound rather than digital quiet, and report detection
against noise specifically, which is the honest number.

Usage
-----
  python local_labeler.py --batch labelset/batch_000.csv --out labelset/batch_000.html
  python local_labeler.py --batch-dir labelset/          # render every batch in the folder
"""
import argparse
import base64
import csv
import io
import json
import os
import sys

import numpy as np

try:
    import soundfile as sf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.mlab
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"needs soundfile and matplotlib: {e}")

SR = 16000
LABELS = ["call", "call+noise", "noise", "silence"]
CONTEXT_PAD = 3.0          # seconds either side of the window in the context view


def read(path, start, dur, sr=SR):
    info = sf.info(path)
    if start is None:
        x, in_sr = sf.read(path, dtype="float32", always_2d=True)
    else:
        a = max(0, int(start * info.samplerate))
        b = min(info.frames, a + int(dur * info.samplerate))
        x, in_sr = sf.read(path, start=a, stop=b, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if in_sr != sr and len(x):
        n = int(round(len(x) * sr / in_sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
    return x


def wav_b64(x, sr=SR, peak=None):
    """peak=None normalises each clip to itself. Pass an explicit peak when two clips cut from
    the same audio are offered side by side: normalising each to its own maximum would make the
    quieter one louder, and 'which of these two is louder' is a judgement the labeler makes."""
    x = np.asarray(x, dtype="float32")
    peak = (float(np.abs(x).max()) if len(x) else 0.0) if peak is None else float(peak)
    if peak > 0:
        x = 0.95 * x / peak          # listening aid only; never used for a loudness judgement
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _spec(ax, x, fmax):
    """Spectrogram with percentile-clipped contrast, so faint calls are visible instead of
    being flattened by one loud frame setting the colour scale."""
    if len(x) < 256:
        return
    P, f, t = matplotlib.mlab.specgram(x, NFFT=256, Fs=SR, noverlap=192)[:3]
    S = 10 * np.log10(P + 1e-12)
    lo, hi = np.percentile(S, 5), np.percentile(S, 99.5)
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=lo, vmax=hi,
              extent=[0, len(x) / SR, f[0], f[-1]])
    ax.set_ylim(0, fmax)


def _box(ax):
    """Axes rectangle as fractions of the image, for the playback cursor overlay.

    The spectrogram does not fill the JPEG -- matplotlib reserves margins for the ticks and
    title -- so a cursor positioned against the image edges drifts out of sync with the audio.
    Must be read AFTER tight_layout(), which is what finally moves the axes.
    """
    p = ax.get_position()
    return {"x": round(float(p.x0), 4), "y": round(float(1 - p.y1), 4),   # y flipped: CSS is top-down
            "w": round(float(p.width), 4), "h": round(float(p.height), 4)}


def figure_b64(ctx, win, t0, t1, ctx_start, fmax=8000):
    """Context spectrogram with the labeled window marked, plus a zoom of that window.

    Returns (jpeg_b64, box) where box locates the axes for the playhead overlay.

    Saved as JPEG, not PNG. A spectrogram is photographic, so PNG cannot compress it and each
    frame cost ~400 KB -- ten times the audio it accompanies, which made whole batches unopenable.
    JPEG artefacts are irrelevant for a visual position cue.
    """
    buf = io.BytesIO()
    if ctx is None:
        # No meaningful context available (concatenated bundle) -- show the window alone rather
        # than a neighbourhood that is not actually its neighbourhood.
        fig, a1 = plt.subplots(figsize=(7.6, 2.8), dpi=72)
        _spec(a1, win, fmax)
        a1.set_title(f"window  ({t1-t0:.2f}s)", fontsize=8)
        a1.set_xlabel("s", fontsize=7); a1.set_ylabel("Hz", fontsize=7)
        a1.set_yticks([0, 4000, 8000]); a1.set_yticklabels(["0", "4k", "8k"])
        a1.tick_params(labelsize=6)
        fig.tight_layout(pad=0.3)
        box = {"ctx": _box(a1)}
        fig.savefig(buf, format="jpg", pil_kwargs={"quality": 72, "optimize": True})
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode(), box

    fig, ax = plt.subplots(2, 1, figsize=(7.6, 3.6), dpi=72,
                           gridspec_kw={"height_ratios": [2, 1]})
    _spec(ax[0], ctx, fmax)
    if len(ctx) >= 256:
        # Markers only, no translucent fill -- a fill discolours precisely the region being
        # judged, which is the one part that must stay readable.
        for v in (t0 - ctx_start, t1 - ctx_start):
            ax[0].axvline(v, color="#39ff14", lw=2.0, zorder=5)
        ax[0].plot([t0 - ctx_start, t1 - ctx_start], [fmax * 0.97] * 2, color="#39ff14",
                   lw=4, solid_capstyle="butt", zorder=5)
    ax[0].set_title(f"context  {ctx_start:.1f}-{ctx_start+len(ctx)/SR:.1f}s   "
                    f"(labeling {t0:.1f}-{t1:.1f}s, between the green lines)", fontsize=8)
    ax[0].set_ylabel("Hz", fontsize=7)

    _spec(ax[1], win, fmax)
    ax[1].set_title("this window", fontsize=8)
    ax[1].set_xlabel("s", fontsize=7)
    for a in ax:
        a.set_yticks([0, 4000, 8000]); a.set_yticklabels(["0", "4k", "8k"])
        a.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    # The audio that plays is the CONTEXT, so the top cursor spans the whole clip. The bottom
    # cursor tracks the same playback but only while it is inside the window, which is what makes
    # "did that call fall inside the green lines or just outside" answerable by ear.
    box = {"ctx": _box(ax[0]), "win": _box(ax[1]),
           "wt0": round(float(t0 - ctx_start), 4), "wt1": round(float(t1 - ctx_start), 4)}
    fig.savefig(buf, format="jpg", pil_kwargs={"quality": 72, "optimize": True})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), box


def thumb_b64(win, w=2.0, h=1.15, dpi=55, q=60, fmax=8000):
    """Bare spectrogram, no axes/ticks/title -- a grid cell, not a labeling page.

    No context, no audio-quality tradeoffs to make: this exists purely so a human can
    pattern-match a clean harmonic stack (evenly spaced horizontal lines) at a glance across many
    windows at once. Anything that isn't obviously that pattern gets no shortcut and still goes
    through the one-by-one page with full context and audio.
    """
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    _spec(ax, win, fmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", pil_kwargs={"quality": q, "optimize": True})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_grid(rows, title):
    """Grid of thumbnails for fast triage of obvious calls. See GRID_TEMPLATE for the rule this
    enforces: clicking marks 'call' and saves immediately; nothing is ever inferred from a
    non-click, so an unmarked cell means 'not yet judged', not 'not a call'.
    """
    payload = []
    for r in rows:
        path = r["path"]
        has_start = str(r.get("start", "")).strip() != ""
        dur = float(r["dur"])
        t0 = float(r["start"]) if has_start else 0.0
        try:
            win = read(path, t0 if has_start else None, dur)
            img_b64 = thumb_b64(win)
        except Exception as e:                        # noqa: BLE001
            print(f"  SKIP {path} @{t0}: {e}")
            continue
        payload.append({"id": r["id"], "img": img_b64, "audio": wav_b64(win)})
    return (GRID_TEMPLATE.replace("__DATA__", json.dumps(payload))
                         .replace("__TITLE__", title))


def build(rows, title, review=False):
    payload = []
    for n, r in enumerate(rows):
        path = r["path"]
        has_start = str(r.get("start", "")).strip() != ""
        dur = float(r["dur"])
        t0 = float(r["start"]) if has_start else 0.0
        t1 = t0 + dur
        # Context handling. Three cases, and getting this wrong shows the labeler audio that
        # isn't actually adjacent to the window:
        #   ctx_start/ctx_dur present -> real context was exported alongside; use it
        #   orig_path present         -> concatenated bundle, neighbours are UNRELATED windows,
        #                                so show no context rather than something misleading
        #   otherwise                 -> reading from a real recording, take +/-CONTEXT_PAD
        try:
            win = read(path, t0 if has_start else None, dur)
            if r.get("ctx_start", "").strip() != "":
                cs = float(r["ctx_start"])
                ctx = read(path, cs, float(r["ctx_dur"]))
            elif r.get("orig_path") or not has_start:
                cs, ctx = None, None
            else:
                cs = max(0.0, t0 - CONTEXT_PAD)
                ctx = read(path, cs, (t1 + CONTEXT_PAD) - cs)
            img_b64, box = figure_b64(ctx, win, t0, t1, cs)
        except Exception as e:                        # noqa: BLE001
            print(f"  SKIP {path} @{t0}: {e}")
            continue
        clip = ctx if ctx is not None else win
        pk = float(np.abs(clip).max()) if len(clip) else 0.0
        payload.append({
            "id": r["id"],
            "n": n + 1,
            # When real context exists, PLAY the context, not just the marked window. A 0.5 s
            # window is too short to judge on its own, and hearing whether a call continues past
            # the boundary is exactly what decides the label.
            "audio": wav_b64(clip, peak=pk),
            # ...and ship the window on its own as a SECOND clip, rather than stopping the first
            # one early. A stop-timer or an animation-frame check overshoots the boundary by up
            # to a frame (~16 ms) -- enough to leak the onset of an adjacent call into exactly
            # the judgement this is for. A file that ends where the window ends cannot overshoot.
            # Same peak as the context clip, so switching between them never changes the level.
            "audio_win": (wav_b64(win, peak=pk) if ctx is not None else ""),
            "img": img_b64,
            "box": box,
            # Never show the raw filename when a `recording` column exists: export filenames can
            # encode how the window was selected (e.g. "..._flagged_neg_..."), which would tell
            # the labeler what the model already thought and make the labels circular.
            "loc": (r["recording"] if r.get("recording")
                    else (f"{os.path.basename(path)}  {t0:.1f}s" if has_start
                          else os.path.basename(path))),
            "dur": round(dur, 3),
        })
    key = "zf" + str(abs(hash(title)) % 10**8)
    return (TEMPLATE.replace("__DATA__", json.dumps(payload))
                    .replace("__LABELS__", json.dumps(LABELS))
                    .replace("__REVIEW__", "true" if review else "false")
                    .replace("__TITLE__", title)
                    .replace("__KEY__", key))


TEMPLATE = """<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<body style="font-family:-apple-system,system-ui,sans-serif;background:#fafafa;margin:0;padding:18px">
<div id="__KEY___root" style="max-width:900px;margin:auto;background:#fff;border:1px solid #ddd;
     border-radius:10px;padding:16px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <b>__TITLE__</b><span id="__KEY___prog" style="font-size:13px;color:#666"></span>
  </div>
  <div style="font-size:12px;margin:2px 0 4px">
    <a href="__TITLE___grid.html">&larr; back to grid view</a></div>
  <div id="__KEY___loc" style="font-size:12px;color:#888;margin:4px 0;min-height:16px"></div>
  <div id="__KEY___wrap" style="position:relative;line-height:0">
    <img id="__KEY___img" style="width:100%;border-radius:6px;background:#111"/>
    <div id="__KEY___ph" style="position:absolute;width:2px;background:#4da3ff;display:none;
         pointer-events:none;box-shadow:0 0 5px #4da3ff"></div>
    <div id="__KEY___ph2" style="position:absolute;width:2px;background:#4da3ff;display:none;
         pointer-events:none;box-shadow:0 0 5px #4da3ff"></div>
  </div>
  <audio id="__KEY___aud" controls style="width:100%;margin-top:8px"></audio>
  <audio id="__KEY___audw" style="display:none"></audio>
  <div id="__KEY___btns" style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap"></div>
  <div style="margin-top:10px;display:flex;gap:6px;align-items:center;flex-wrap:wrap">
    <button id="__KEY___prev">&larr; back</button>
    <button id="__KEY___skip">skip &rarr;</button>
    <button id="__KEY___play">replay (space)</button>
    <button id="__KEY___playwin">window only (p)</button>
    <label style="font-size:12px;color:#666"><input type="checkbox" id="__KEY___auto" checked>
      autoplay</label>
    <span style="flex:1"></span>
    <button id="__KEY___dl" style="font-weight:600">download CSV</button>
    <button id="__KEY___clear" style="color:#a00">reset</button>
  </div>
  <div style="font-size:12px;color:#777;margin-top:8px">
    1 call &middot; 2 call+noise &middot; 3 noise &middot; 4 silence &middot;
    space replays context &middot; p plays the window only &middot; &larr;/&rarr; move
  </div>
  <div id="__KEY___save" style="font-size:12px;margin-top:6px;font-weight:600"></div>
  <div id="__KEY___out" style="font-size:12px;color:#333;margin-top:6px"></div>
</div>
<script>
(function(){
 const D=__DATA__, L=__LABELS__, K="__KEY__", T="__TITLE__", REVIEW=__REVIEW__;
 const $=id=>document.getElementById(K+"_"+id);
 // Served over http -> labeler_server.py owns a real CSV and every label is POSTed as it is
 // pressed. Opened as file:// -> no server to talk to, so fall back to localStorage and the
 // manual "download CSV" button. Same page either way.
 const SRV=location.protocol.indexOf("http")===0;
 let store={}; try{store=JSON.parse(localStorage.getItem(K)||"{}")}catch(e){store={}}
 // A normal batch resumes at the first unlabeled window. A review page is built ENTIRELY from
 // already-labeled windows, so that same rule would skip straight to the end -- start at 0.
 // The grid page labels windows out of order (whichever look like obvious calls), so "next" has
 // to search for the next unlabeled id, not just step the index -- otherwise going forward walks
 // straight into windows the grid already resolved.
 function nextUnlabeled(from){ let j=from+1; while(j<D.length && store[D[j].id]) j++; return j; }
 function seek(){ if(REVIEW){ i=0; return; }
   i=0; while(i<D.length && store[D[i].id]) i++;
   if(i>=D.length) i=Math.max(0,D.length-1); }
 let i=0; seek();
 function status(t,c){ $("save").textContent=t; $("save").style.color=c; }
 if(SRV){
   status("connecting\\u2026","#888");
   // The CSV on disk is authoritative: it survives cleared caches and other browsers, so it
   // wins over whatever this browser happens to remember.
   fetch("/labels?batch="+encodeURIComponent(T)).then(r=>r.json()).then(o=>{
     store=Object.assign({},store,o); localStorage.setItem(K,JSON.stringify(store));
     seek(); render(); status("\\u25cf saving to labels.csv","#2a7");
   }).catch(()=>status("\\u26a0 server unreachable \\u2014 browser-only","#a70"));
 } else {
   status("\\u26a0 opened as a file \\u2014 browser-only, use download CSV","#a70");
 }
 const btns=$("btns");
 L.forEach((lab,j)=>{
   const b=document.createElement("button");
   b.textContent=(j+1)+"  "+lab;
   b.style.cssText="padding:8px 14px;border-radius:6px;border:1px solid #bbb;cursor:pointer;font-size:14px";
   b.onclick=()=>mark(lab); btns.appendChild(b);
 });
 function render(){
   const d=D[i];
   // Hide before swapping the image, or the old cursor position lingers over the new
   // spectrogram for a frame and reads as a real playhead.
   $("ph").style.display="none"; $("ph2").style.display="none";
   $("img").src="data:image/jpeg;base64,"+d.img;
   $("aud").src="data:audio/wav;base64,"+d.audio;
   $("audw").src=d.audio_win?("data:audio/wav;base64,"+d.audio_win):"";
   const done=Object.keys(store).length;
   $("prog").textContent=(i+1)+" / "+D.length+"  \\u00b7  "+done+" labelled";
   $("loc").textContent=d.loc+(store[d.id]?("   \\u2014 current: "+store[d.id]):"");
   Array.from(btns.children).forEach((b,j)=>{
     b.style.background=(store[d.id]===L[j])?"#cfe8ff":"#f7f7f7";});
   if($("auto").checked) $("aud").play().catch(()=>{});
 }
 // ---- playback ---------------------------------------------------------------------------
 // Two ways to listen, because they answer different questions. Context (space) tells you whether
 // a call runs across the window boundary. Window-only (p) plays strictly what is being labelled,
 // for the close calls where the surrounding audio is what makes the judgement ambiguous.
 //
 // These are two separate clips, not one clip stopped early. Stopping the context clip at the
 // boundary means polling currentTime, and the tightest poll available (an animation frame) still
 // runs up to ~16 ms late -- long enough to sound the onset of the next call, which is precisely
 // what p exists to exclude. A clip that ends where the window ends cannot overshoot at all.
 function playAll(){ $("audw").pause(); $("aud").currentTime=0; $("aud").play().catch(()=>{}); }
 function playWindow(){
   const w=$("audw");
   // No context exported for this row -> the clip already IS the window, so play the lot.
   if(!w.getAttribute("src")){ playAll(); return; }
   $("aud").pause(); w.currentTime=0; w.play().catch(()=>{});
 }
 // ---- playback cursor -------------------------------------------------------------------
 // Positioned against the exported axes rectangle, not the image edges, so it lines up with the
 // spectrogram itself. Driven by rAF only while audio is actually playing: a 7-hour session
 // should not burn a core animating a cursor that is standing still.
 let raf=0;
 function place(el,b,frac){
   el.style.display="block";
   el.style.left=((b.x+frac*b.w)*100).toFixed(3)+"%";
   el.style.top=(b.y*100).toFixed(3)+"%";
   el.style.height=(b.h*100).toFixed(3)+"%";
 }
 function upd(){
   const a=$("aud"), w=$("audw"), d=D[i], b=d&&d.box;
   if(!b||!a.duration||!isFinite(a.duration)){ $("ph").style.display="none";
                                               $("ph2").style.display="none"; return; }
   // One timeline for both players: the window clip starts at wt0 in context time, so shift it
   // there rather than letting 0.5 s of playback sweep the whole 1.5 s context panel.
   const onWin = !w.paused && b.wt0!=null;
   const t = onWin ? (b.wt0 + w.currentTime) : a.currentTime;
   place($("ph"), b.ctx, Math.min(1,Math.max(0,t/a.duration)));
   if(b.win){
     if(t>=b.wt0 && t<=b.wt1) place($("ph2"), b.win, (t-b.wt0)/(b.wt1-b.wt0));
     else $("ph2").style.display="none";
   }
 }
 function tick(){ upd(); raf=requestAnimationFrame(tick); }
 (function(){
   [$("aud"),$("audw")].forEach(el=>{
     el.addEventListener("play",()=>{ if(!raf) raf=requestAnimationFrame(tick); });
     // Only stop the cursor when BOTH are idle: switching players pauses one and starts the
     // other, and media events are queued, so the pause can land after the play.
     ["pause","ended"].forEach(ev=>el.addEventListener(ev,()=>{
       if($("aud").paused && $("audw").paused){
         if(raf) cancelAnimationFrame(raf); raf=0; }
       upd(); }));
     el.addEventListener("seeked",upd);
     el.addEventListener("loadedmetadata",upd);
   });
 })();

 let pending=0;
 function push(id,lab){
   pending++; status("saving\\u2026","#888");
   fetch("/save",{method:"POST",headers:{"Content-Type":"application/json"},
                  body:JSON.stringify({batch:T,id:id,label:lab})})
    .then(r=>{ if(!r.ok) throw new Error("http "+r.status); pending--;
               if(!pending) status("\\u25cf saved to labels.csv","#2a7"); })
    // Loud and red on failure: a silently dropped write means labeling into the void, and you
    // would not find out until the merge came up short.
    .catch(e=>{ pending--; status("\\u26a0 NOT SAVED ("+e.message+") \\u2014 use download CSV",
                                  "#c00"); });
 }
 function mark(lab){
   const id=D[i].id;
   store[id]=lab; localStorage.setItem(K,JSON.stringify(store));
   if(SRV) push(id,lab);
   const j=nextUnlabeled(i);
   if(j<D.length) i=j;          // else: nothing left unlabeled -- stay put, prog shows N/N
   render();
 }
 function csv(){ let s="id,human_label\\n";
   D.forEach(d=>{ if(store[d.id]) s+='"'+d.id+'",'+store[d.id]+"\\n"; }); return s; }
 $("prev").onclick=()=>{ if(i>0){i--;render();} };
 $("skip").onclick=()=>{ const j=nextUnlabeled(i); if(j<D.length){i=j;render();} };
 $("play").onclick=playAll;
 $("playwin").onclick=playWindow;
 $("dl").onclick=()=>{ const b=new Blob([csv()],{type:"text/csv"});
   const a=document.createElement("a"); a.href=URL.createObjectURL(b);
   a.download="__TITLE__".replace(/[^A-Za-z0-9_.-]/g,"_")+"_labels.csv"; a.click(); };
 $("clear").onclick=()=>{ if(confirm("Erase all labels for this batch?")){
   store={}; localStorage.removeItem(K); i=0; render(); } };
 document.addEventListener("keydown",e=>{
   if(e.key===" "){e.preventDefault();playAll();}
   else if(e.key==="p"||e.key==="P"){e.preventDefault();playWindow();}
   else if(e.key==="ArrowLeft"){$("prev").click();}
   else if(e.key==="ArrowRight"){$("skip").click();}
   else{const n=parseInt(e.key); if(n>=1&&n<=L.length) mark(L[n-1]);}
 });
 render();
})();
</script>"""


GRID_TEMPLATE = """<!doctype html><meta charset="utf-8"><title>__TITLE__ (grid)</title>
<body style="font-family:-apple-system,system-ui,sans-serif;background:#fafafa;margin:0;padding:18px">
<div style="max-width:1100px;margin:auto">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
    <b>__TITLE__ &mdash; grid triage</b>
    <span id="prog" style="font-size:13px;color:#666"></span>
  </div>
  <div style="font-size:12px;color:#777;margin:6px 0 10px;max-width:760px">
    Click a thumbnail only if it is an obvious, clean call &mdash; a clear harmonic stack (evenly
    spaced horizontal lines) with nothing else on it. Click &#9654; to listen first if unsure.
    Anything not obviously clean &mdash; quiet, buried in noise, ambiguous &mdash; leave it
    unclicked. Unclicked means "not yet judged", never "not a call" &mdash; it still gets a real
    look in the one-by-one page.
  </div>
  <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap">
    <button id="save" style="font-weight:600;padding:6px 14px;border-radius:6px;border:1px solid #bbb;
            cursor:pointer">save marked calls (<span id="n">0</span>)</button>
    <label style="font-size:12px;color:#666">
      <input type="checkbox" id="showdone"> show already-labeled</label>
    <span id="status" style="font-size:12px;font-weight:600"></span>
    <span style="flex:1"></span>
    <a href="__TITLE__.html" style="font-size:12px">one-by-one page for the rest &rarr;</a>
  </div>
  <div id="grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
       gap:8px"></div>
</div>
<audio id="aud" style="display:none"></audio>
<script>
(function(){
  const D=__DATA__, T="__TITLE__";
  const grid=document.getElementById("grid");
  let labeled={};              // id -> label already on the server (any source)
  // Marks survive a reload. They used to live only in this Set: because saving is a deliberate
  // second step, closing the tab between marking and pressing save silently took every mark with
  // it, and the page came back looking like no work had been done.
  const PK="grid_"+T;
  function load(){ try{ return JSON.parse(localStorage.getItem(PK)||"[]"); }catch(e){ return []; } }
  function keep(){ try{ localStorage.setItem(PK,JSON.stringify(Array.from(picked))); }catch(e){} }
  let picked=new Set(load());  // ids clicked but not yet saved to labels.csv
  const aud=document.getElementById("aud");

  function status(t,c){ const s=document.getElementById("status"); s.textContent=t; s.style.color=c; }

  function render(){
    const showDone=document.getElementById("showdone").checked;
    grid.innerHTML="";
    let shown=0;
    D.forEach(d=>{
      const isDone=labeled.hasOwnProperty(d.id);
      if(isDone && !showDone) return;
      shown++;
      const cell=document.createElement("div");
      cell.style.cssText="position:relative;border:3px solid "+
        (picked.has(d.id)?"#2a7":"transparent")+";border-radius:6px;overflow:hidden;"+
        "cursor:pointer;background:#111;opacity:"+(isDone?"0.35":"1");
      const img=document.createElement("img");
      img.src="data:image/jpeg;base64,"+d.img;
      img.style.cssText="display:block;width:100%";
      cell.appendChild(img);
      if(picked.has(d.id)){
        const tag=document.createElement("div");
        tag.textContent="\\u2713 call";
        tag.style.cssText="position:absolute;top:2px;left:4px;color:#fff;background:#2a7;"+
          "font-size:11px;font-weight:700;padding:1px 5px;border-radius:3px";
        cell.appendChild(tag);
      } else if(isDone){
        const tag=document.createElement("div");
        tag.textContent=labeled[d.id];
        tag.style.cssText="position:absolute;top:2px;left:4px;color:#fff;background:#666;"+
          "font-size:10px;padding:1px 5px;border-radius:3px";
        cell.appendChild(tag);
      }
      const play=document.createElement("button");
      play.textContent="\\u25b6";
      play.style.cssText="position:absolute;bottom:2px;right:2px;border:none;border-radius:3px;"+
        "background:rgba(0,0,0,.55);color:#fff;font-size:11px;padding:2px 6px;cursor:pointer";
      play.onclick=(e)=>{ e.stopPropagation();
        aud.src="data:audio/wav;base64,"+d.audio; aud.currentTime=0; aud.play(); };
      cell.appendChild(play);
      cell.onclick=()=>{ if(isDone) return;
        if(picked.has(d.id)){ picked.delete(d.id); }
        else{ picked.add(d.id);
          // Autoplay only on the marking click, as a quick sanity check -- not on unmarking,
          // and not a substitute for the explicit ▶ button, which still works for a
          // pre-listen before deciding.
          aud.src="data:audio/wav;base64,"+d.audio; aud.currentTime=0; aud.play(); }
        keep(); render(); };
      grid.appendChild(cell);
    });
    document.getElementById("n").textContent=picked.size;
    document.getElementById("prog").textContent=
      Object.keys(labeled).length+" already labeled \\u00b7 "+shown+" shown \\u00b7 "+
      D.length+" total";
  }

  fetch("/labels?batch="+encodeURIComponent(T)).then(r=>r.json()).then(o=>{
    labeled=o;
    // Anything the server already has is settled; a stale local mark for it is just noise.
    Object.keys(labeled).forEach(id=>picked.delete(id)); keep();
    if(picked.size) status("\u26a0 "+picked.size+" marked, NOT yet saved \u2014 press save","#a70");
    render();
  }).catch(()=>{ status("\\u26a0 server unreachable \\u2014 run via labeler_server.py","#c00");
                render(); });

  document.getElementById("showdone").onchange=render;

  // Last line of defence: marks are in localStorage now, but they are still not in labels.csv
  // until the button is pressed, and only labels.csv counts.
  window.addEventListener("beforeunload",e=>{
    if(picked.size){ e.preventDefault(); e.returnValue=""; } });

  document.getElementById("save").onclick=()=>{
    if(!picked.size) return;
    const ids=Array.from(picked);
    status("saving "+ids.length+"\\u2026","#888");
    let left=ids.length, failed=0;
    ids.forEach(id=>{
      fetch("/save",{method:"POST",headers:{"Content-Type":"application/json"},
                     body:JSON.stringify({batch:T,id:id,label:"call"})})
        .then(r=>{ if(!r.ok) throw new Error("http "+r.status);
                   labeled[id]="call"; picked.delete(id); keep(); left--;
                   if(!left){ status(failed?("\\u26a0 "+failed+" NOT SAVED \\u2014 retry"):
                                            "\\u25cf saved","#2a7"); render(); } })
        // Loud on failure, and left in `picked` (still marked, still counted, still clickable
        // to retry) rather than silently dropped -- same rule as the one-by-one page.
        .catch(()=>{ failed++; left--;
                   if(!left){ status("\\u26a0 "+failed+" NOT SAVED \\u2014 retry","#c00");
                              render(); } });
    });
  };
  render();
})();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch")
    ap.add_argument("--batch-dir")
    ap.add_argument("--out")
    ap.add_argument("--review", action="store_true",
                    help="page is a re-review of already-labeled windows: start at the first "
                         "item instead of skipping to the first unlabeled one")
    ap.add_argument("--grid", action="store_true",
                    help="render the fast grid-triage page instead of the one-by-one page")
    a = ap.parse_args()

    todo = []
    if a.batch_dir:
        import glob as g
        todo = [p for p in sorted(g.glob(os.path.join(a.batch_dir, "batch_*.csv")))]
        if not todo:
            sys.exit(f"no batch_*.csv under {a.batch_dir}")
    elif a.batch:
        todo = [a.batch]
    else:
        ap.error("give --batch or --batch-dir")

    for b in todo:
        rows = list(csv.DictReader(open(b)))
        title = os.path.basename(b)[:-4]
        print(f"{title}: {len(rows)} items")
        if a.grid:
            html = build_grid(rows, title)
            out = a.out if (a.out and len(todo) == 1) else b[:-4] + "_grid.html"
        else:
            html = build(rows, title, review=a.review)
            out = a.out if (a.out and len(todo) == 1) else b[:-4] + ".html"
        with open(out, "w") as fh:
            fh.write(html)
        print(f"  -> {out}  ({len(html)/1e6:.1f} MB)")
        if len(html) / 1e6 > 60:
            print("  WARNING: large page, the browser will be slow. Use smaller batches.")

    print("\nOpen the .html in a browser, label, then click 'download CSV'.")
    print("Put the downloaded CSVs back in the labelset folder and run:")
    print("  python build_local_labelset.py --merge <labelset dir>")


if __name__ == "__main__":
    main()
