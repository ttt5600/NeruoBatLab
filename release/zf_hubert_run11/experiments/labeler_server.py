#!/usr/bin/env python
"""Serve the labeling pages and write labels straight to a CSV as they are pressed.

Without this, the pages keep labels in browser localStorage and you have to click "download CSV"
per page and shuffle files out of ~/Downloads. That is 32 manual steps and one cleared browser
cache away from losing hours of work. With this running, every keypress is a POST that lands in
labels.csv before the next window renders.

  python labeler_server.py --dir ~/zf_labelset/neg_bundle
  # -> http://localhost:8765/   (index of batches, with progress)

The CSV is the single source of truth: reopening a page pulls its existing labels back from the
server, so progress survives reloads, browser restarts, and switching machines. localStorage stays
as a fallback so the pages still work opened directly as file:// -- they just do not auto-save.
"""
import argparse
import csv
import glob
import html
import json
import os
import socketserver
import threading
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

LOCK = threading.Lock()


class Server(ThreadingHTTPServer):
    def server_bind(self):
        # http.server's server_bind calls socket.getfqdn(), which on this Mac blocks ~35 s on a
        # reverse-DNS lookup that will never resolve. The value is only used for CGI env vars we
        # do not use, so set it directly and skip the lookup.
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


class Store:
    """id -> (label, batch). Rewritten in full on every save; 4900 rows is nothing."""

    def __init__(self, path):
        self.path = path
        self.d = {}
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                if r.get("human_label", "").strip():
                    self.d[r["id"]] = (r["human_label"].strip(), r.get("batch", ""))
            print(f"resuming: {len(self.d)} labels already in {os.path.basename(path)}")

    def save(self, wid, label, batch):
        with LOCK:
            self.d[wid] = (label, batch)
            tmp = self.path + ".tmp"
            with open(tmp, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["id", "human_label", "batch"])
                for k in sorted(self.d):
                    w.writerow([k, self.d[k][0], self.d[k][1]])
            os.replace(tmp, self.path)          # atomic: a crash mid-write cannot truncate it
            return len(self.d)

    def for_batch(self, batch):
        return {k: v[0] for k, v in self.d.items() if v[1] == batch}


def make_handler(root, store, total, ids_by_batch, batch_of):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

        def log_message(self, *a):                  # keep the console readable
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/labels"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                b = (q.get("batch") or [""])[0]
                # Membership comes from the batch CSVs, never from the stored batch column, so a
                # review page (whose name matches no batch) cannot reassign a window and make it
                # look unlabeled in the batch it really belongs to.
                if b in ids_by_batch:
                    return self._json({i: store.d[i][0] for i in ids_by_batch[b]
                                       if i in store.d})
                return self._json({k: v[0] for k, v in store.d.items()})
            if self.path in ("/", "/index.html"):
                return self._index()
            return super().do_GET()

        def do_POST(self):
            if self.path != "/save":
                return self._json({"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                wid, lab = body["id"], body["label"]
            except Exception as ex:                 # noqa: BLE001
                return self._json({"error": str(ex)}, 400)
            # Record the window's true home batch, not whichever page happened to submit it.
            done = store.save(wid, lab, batch_of.get(wid, body.get("batch", "")))
            print(f"\r{done}/{total} labelled  ({done*100.0/max(1,total):.1f}%)   ",
                  end="", flush=True)
            return self._json({"ok": True, "done": done, "total": total})

        def _index(self):
            rows = []
            for p in sorted(glob.glob(os.path.join(root, "batch_*.html"))):
                name = os.path.splitext(os.path.basename(p))[0]
                ids = [r["id"] for r in csv.DictReader(
                    open(os.path.join(root, name + ".csv")))]
                done = sum(1 for i in ids if i in store.d)
                pct = done * 100.0 / max(1, len(ids))
                colour = "#2a7" if done == len(ids) else ("#a70" if done else "#bbb")
                rows.append(
                    f'<tr><td><a href="{name}.html">{name}</a></td>'
                    f'<td style="text-align:right">{done}/{len(ids)}</td>'
                    f'<td style="width:220px"><div style="background:#eee;border-radius:4px">'
                    f'<div style="width:{pct:.0f}%;background:{colour};height:10px;'
                    f'border-radius:4px"></div></div></td></tr>')
            done_all = len(store.d)
            body = f"""<!doctype html><meta charset="utf-8"><title>ZF labeling</title>
<body style="font-family:-apple-system,system-ui,sans-serif;background:#fafafa;padding:24px">
<div style="max-width:640px;margin:auto;background:#fff;border:1px solid #ddd;border-radius:10px;
     padding:18px">
<h2 style="margin:0 0 4px">Vocalization vs. noise &mdash; unlabeled negatives</h2>
<div style="color:#666;font-size:13px;margin-bottom:14px">
  {done_all} / {total} labelled &middot; saving to
  <code>{html.escape(os.path.basename(store.path))}</code> as you go</div>
<table style="width:100%;border-collapse:collapse;font-size:14px">{''.join(rows)}</table>
</div>"""
            b = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--labels", default=None, help="default <dir>/labels.csv")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(a.dir))
    if not glob.glob(os.path.join(root, "batch_*.html")):
        raise SystemExit(f"no batch_*.html in {root} -- render them with local_labeler.py first")
    store = Store(a.labels or os.path.join(root, "labels.csv"))
    ids_by_batch, batch_of = {}, {}
    for p in sorted(glob.glob(os.path.join(root, "batch_*.csv"))):
        name = os.path.splitext(os.path.basename(p))[0]
        ids_by_batch[name] = [r["id"] for r in csv.DictReader(open(p))]
        for i in ids_by_batch[name]:
            batch_of[i] = name
    total = sum(len(v) for v in ids_by_batch.values())

    srv = Server(("127.0.0.1", a.port),
                 make_handler(root, store, total, ids_by_batch, batch_of))
    url = f"http://localhost:{a.port}/"
    print(f"serving {root}\n  {url}\nlabels -> {store.path}\nCtrl-C to stop "
          f"(labels are already on disk; nothing is buffered)\n")
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
