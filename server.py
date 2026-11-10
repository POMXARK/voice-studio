#!/usr/bin/env python3
"""VoiceStudio — локальный веб-интерфейс для генерации озвучки (stdlib, без зависимостей).
usage: python server.py [--config config.json]   затем открыть http://127.0.0.1:8777
Данные проекта — в <work>: texts.tsv (русский текст), texts_<lang>.tsv, <lang>_wav/, dub_wav/, dub_wem/; состояние студии — studio.json, дубли — studio/takes, задачи — studio/jobs."""
import argparse, json, os, re, shutil, subprocess, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
HERE = Path(__file__).parent
START_T = time.time(); SRV = None
ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=HERE / "config.json" if (HERE / "config.json").exists() else HERE / "config.example.json"); A = ap.parse_args()
CFG = json.loads(A.config.read_text(encoding="utf-8")); W = Path(CFG["work"]); ST = W / "studio"; ST.mkdir(exist_ok=True)
(ST / "jobs").mkdir(exist_ok=True); (ST / "takes").mkdir(exist_ok=True); (ST / "samples").mkdir(exist_ok=True)
LOCK = threading.RLock(); JOBS = {}   # id -> {proc, kind, names, done, pos}
_TSV = {}
def rd_tsv(p):   # с кэшем по времени изменения: у полного набора игры десятки тысяч строк
    if not p.exists(): return {}
    k = (p.stat().st_mtime_ns, p.stat().st_size)
    if _TSV.get(p, (None,))[0] != k: _TSV[p] = (k, dict(l.split("\t", 1) for l in p.read_text(encoding="utf-8").splitlines() if "\t" in l))
    return _TSV[p][1]
def load_models(): return json.loads((HERE / "models.json").read_text(encoding="utf-8"))["models"]
_ST = {"S": None, "k": None, "dirty": False, "t": 0.0}   # studio.json (~50 МБ на 60 тыс. реплик) держим в памяти: разбор и запись на каждый опрос интерфейса (2,5 с) доходили до 22 с
def _skey(p):
    try: st = p.stat(); return (st.st_mtime_ns, st.st_size)
    except OSError: return None
def load_state():
    p = W / "studio.json"
    with LOCK:
        if _ST["S"] is not None and not _ST["dirty"] and _skey(p) != _ST["k"]: _ST["S"] = None   # файл правили снаружи — перечитать
        if _ST["S"] is None:
            _ST["S"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"lines": {}, "samples": {}, "defaults": {"model": "qwen3-1.7b", "ref": {"src": "lang", "lang": "ja"}}}; _ST["k"] = _skey(p)
        return _ST["S"]
def save_state(s, now=False):   # в память сразу, на диск — не чаще раза в 15 с (поток flush_loop); results.jsonl воркеров всё равно хранит каждый дубль
    with LOCK:
        _ST["S"] = s; _ST["dirty"] = True
        if now: flush_state()
def flush_state():
    p = W / "studio.json"
    with LOCK:
        if not _ST["dirty"]: return
        for _ in range(5):
            try: txt = json.dumps(_ST["S"], ensure_ascii=False); break
            except RuntimeError: time.sleep(0.05)   # другой поток как раз меняет словарь
        else: return
        try: (W / "studio.json.tmp").write_text(txt, encoding="utf-8"); os.replace(W / "studio.json.tmp", p)
        except OSError: return   # Windows: файл кто-то держит открытым — запишем в следующий раз
        _ST["k"] = _skey(p); _ST["dirty"] = False; _ST["t"] = time.time()
def flush_loop():
    while True:
        time.sleep(15)
        try: flush_state()
        except Exception as e: print("flush_state:", e, flush=True)
_LG = [0, []]
def langs():   # кэш на 3 с: при запуске 60 тыс. реплик вызывается на каждую
    if time.time() - _LG[0] > 3: _LG[:] = [time.time(), sorted(p.name[:-4] for p in W.glob("*_wav") if p.name != "dub_wav" and p.is_dir())]
    return _LG[1]
def ref_for(name, spec, S):
    """-> dict(wav|wavs, text|None, lang) для воркера и описание для интерфейса"""
    src = spec.get("src", "lang")
    if src == "sample":
        f = ST / "samples" / spec["file"]; return {"wav": str(f), "text": S["samples"].get(spec["file"], {}).get("text") or None}, f"свой образец {spec['file']}"
    if src == "any":   # образец с любого языка: первый из списка (или из имеющихся папок), где есть запись этой реплики
        for lg in [x for x in spec.get("order", "").split() if x] + langs():
            if (W / f"{lg}_wav" / f"{name}.wav").exists(): spec = {"src": "lang", "lang": lg, "lines": spec.get("lines")}; break
        else: raise ValueError(f"нет образца ни на одном языке для {name}")
        src = "lang"
    lang = "ru" if src == "dub" else spec.get("lang", "ja"); folder = "dub_wav" if src == "dub" else f"{lang}_wav"
    tx = rd_tsv(W / ("texts_dub.tsv" if src == "dub" else f"texts_{lang}.tsv")); names = spec.get("lines") or [name]
    names = [n for n in names if (W / folder / f"{n}.wav").exists()]
    if not names: raise ValueError(f"нет образца {folder}/{name}.wav")
    if len(names) == 1: return {"wav": str(W / folder / f"{names[0]}.wav"), "text": tx.get(names[0], "").strip() or None, "lang": lang}, f"{lang}: {names[0][-24:]}"
    return {"wavs": [str(W / folder / f"{n}.wav") for n in names], "text": " ".join(tx.get(n, "").strip() for n in names), "lang": lang}, f"{lang}: склейка {len(names)} реплик"
def line_cfg(S, name):
    L = S["lines"].get(name, {}); d = S["defaults"]
    return {"model": L.get("model", d["model"]), "ref": L.get("ref", d["ref"]), "params": {**d.get("params", {}), **L.get("params", {})}}
_DEC = json.JSONDecoder()
def parse_rows(rows):   # строки results.jsonl: два воркера могли записать одну строку вперемешку — достаём целые объекты, битое пропускаем
    for ln in rows:
        i = 0; ln = ln.strip()
        while i < len(ln):
            try: o, i = _DEC.raw_decode(ln, i)
            except ValueError: break
            if isinstance(o, dict) and "name" in o and "kind" in o: yield o
            while i < len(ln) and ln[i] in " \t": i += 1
SUSP_LOG = W / "suspect.log"
def played_recent(minutes):
    """именованные реплики, которые игра запрашивала за последние N минут (VoiceLayer.lines.tsv: время<TAB>папка<TAB>имя); нет свежих (игра закрыта) — окно от последней записи"""
    p = Path(CFG["game_dir"]).parent.parent / "VoiceLayer.lines.tsv"
    if not p.exists(): return [], "нет VoiceLayer.lines.tsv"
    with open(p, "rb") as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 3_000_000)); rows = f.read().decode("utf-8", "ignore").splitlines()[1:]
    ev = []
    for ln in rows:
        c = ln.split("\t")
        if len(c) >= 3 and re.fullmatch(r"\d\d:\d\d:\d\d", c[0]) and not c[2].isdigit():
            ev.append((int(c[0][:2]) * 3600 + int(c[0][3:5]) * 60 + int(c[0][6:]), c[2]))
    if not ev: return [], "в логе нет именованных реплик"
    lt = time.localtime(); now = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    pick = [n for t, n in ev if (now - t) % 86400 <= minutes * 60]
    note = "по текущему времени"
    if not pick:
        end = ev[-1][0]; pick = [n for t, n in ev if (end - t) % 86400 <= minutes * 60]; note = "окно от последней записи лога (игра не запущена?)"
    return list(dict.fromkeys(pick)), note
def quick_dur(t):   # длительность к тексту: ≥3× от ожидаемой — речь без остановки (так выглядели «совсем искажённые» реплики)
    tx = t.get("text") or ""; d = t.get("dur")
    if tx and d:
        exp = 0.5 + 0.075 * len(tx)
        if d > 3 * exp: return [f"затянуто ×{d / exp:.1f}"]
        if d < 0.4 * exp: return [f"обрыв ×{d / exp:.2f}"]
    return []
def auto_suspect(L, name, why):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S"); L["suspect"] = stamp; L["suspect_why"] = ", ".join(why)
    try:
        with open(SUSP_LOG, "a", encoding="utf-8") as f: f.write(f"{stamp}\tauto\t{name}\t{L['suspect_why']}\n")
    except OSError: pass
def mark_suspect(S, minutes=5, name=None):
    """«слышу странную реплику»: все реплики за последние N минут (live — из лога игры; с name — по времени генерации) → на проверку + строки в suspect.log"""
    known = set(S["lines"]) | set(rd_tsv(W / "texts.tsv"))
    if name:
        L0 = S["lines"].get(name, {}); ts = max([take_ts(t) for t in L0.get("takes", [])] or [0]); src = "take"
        names = [n for n, L in S["lines"].items() if any(abs(take_ts(t) - ts) <= minutes * 60 for t in L.get("takes", []))] if ts else [name]; note = "по времени генерации"
    else:
        names, note = played_recent(minutes); src = "live"
    names = [n for n in names if n in known]; stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(SUSP_LOG, "a", encoding="utf-8") as f:
        f.write(f"# {stamp} {src} {minutes} мин: {len(names)} реплик ({note})\n")
        for n in names: S["lines"].setdefault(n, {})["suspect"] = stamp; f.write(f"{stamp}\t{src}\t{n}\n")
    return {"n": len(names), "note": note, "names": names[:50], "log": str(SUSP_LOG)}
def ingest():   # забрать результаты воркеров в studio.json
    with LOCK:
        S = load_state(); ch = False
        for jid, j in JOBS.items():
            rp = ST / "jobs" / jid / "results.jsonl"
            if not rp.exists(): continue
            bp = j.get("bpos", 0)
            if rp.stat().st_size < bp: bp = 0   # файл пересоздан
            with open(rp, "rb") as f: f.seek(bp); data = f.read()   # дочитываем только новое: целиком это 40+ МБ на каждый опрос
            end = data.rfind(b"\n") + 1   # недописанную воркером строку оставляем на следующий раз
            if not end: continue
            j["bpos"] = bp + end
            for r in parse_rows(data[:end].decode("utf-8", "replace").splitlines()):
                L = S["lines"].setdefault(r["name"], {}); ch = True
                if r["kind"] == "take":
                    if any(t["take"] == r["take"] for t in L.get("takes", [])): continue
                    L.setdefault("takes", []).append({k: r[k] for k in r if k not in ("kind", "name")}); L["chosen"] = L.get("chosen") or r["take"]
                    why = r.get("quick") or quick_dur(r)
                    if why: auto_suspect(L, r["name"], why)
                elif r["kind"] == "score":
                    for t in L.get("takes", []):
                        if t["take"] == r["take"]: t.update({k: r[k] for k in ("score", "wer", "tail", "dur_ratio", "cos", "heard", "hallucination")})
                    L.pop("suspect", None)   # проверена оценкой — снимаем отметку «на проверку»
        if ch: save_state(S)
def spawn(jid, kind, py, append=False, idx=0):
    jd = ST / "jobs" / jid
    if idx == 0:
        (jd / "ctl").unlink(missing_ok=True)
        if kind == "gen":   # новый запуск задачи: очередь реплик и счётчики потоков начинаем заново (готовое воркер пропустит по results.jsonl)
            shutil.rmtree(jd / "claims", ignore_errors=True); (jd / "claims").mkdir(); (jd / "workers").write_text("1")
            for f in jd.glob("progress_*.json"): f.unlink(missing_ok=True)
    log = open(jd / "job.log", "a" if (append or idx) else "w", encoding="utf-8"); env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    if kind == "gen":   # режим генерации (переключатель в шапке): «быстрые» живут в своём venv — у faster-qwen3-tts своя сборка qwen_tts
        eng = THR.get("engine", "hf"); env.update(STUDIO_ENGINE=eng, STUDIO_BATCH=str(THR.get("bsize", 8)))
        if eng in ("fast", "fastbatch") and Path(FAST_PY).exists(): py = FAST_PY
    return subprocess.Popen([py, str(HERE / "worker.py"), kind, str(jd / "job.json"), str(idx)], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
def start_job(kind, py, items, extra):
    jid = time.strftime("%H%M%S") + f"_{kind}"; jd = ST / "jobs" / jid; jd.mkdir()
    (jd / "job.json").write_text(json.dumps({"items": items, "py": py, **extra}, ensure_ascii=False), encoding="utf-8")
    JOBS[jid] = {"id": jid, "py": py, "proc": None, "kind": kind, "n": len(items), "pos": 0, "started": time.time(), "names": {i["name"] for i in items if "name" in i}}
    JOBS[jid]["proc"] = spawn(jid, kind, py); JOBS[jid]["procs"] = {0: JOBS[jid]["proc"]}; return jid
def load_old_jobs():   # задачи прошлых запусков сервера: незавершённые можно продолжить (воркер пропускает уже сделанное)
    for jd in sorted((ST / "jobs").iterdir()):
        try: job = json.loads((jd / "job.json").read_text(encoding="utf-8"))
        except Exception: continue
        JOBS[jd.name] = {"id": jd.name, "proc": None, "kind": jd.name.split("_", 1)[1], "n": len(job["items"]), "pos": 0, "started": 0, "py": job.get("py"), "names": {i["name"] for i in job["items"] if "name" in i}}
def _pid_alive(pid):
    try: import psutil; return bool(pid) and psutil.pid_exists(pid) and psutil.Process(pid).status() != "zombie"
    except Exception: return False
def _fresh(f):   # воркер жив: файл прогресса свежий или процесс с записанным pid существует (при нагрузке прогресс пишется редко)
    try:
        if time.time() - f.stat().st_mtime < 90: return True
        d = _rj(f); return d.get("state") not in ("done", "exited", "error") and _pid_alive(d.get("pid"))
    except OSError: return False
def _rj(f):
    try: return json.loads(f.read_text(encoding="utf-8"))
    except Exception: return {}
def live_idx(j):   # номера работающих потоков генерации (свои процессы + пережившие перезапуск сервера: пишут прогресс)
    s = {i for i, p in (j.get("procs") or {}).items() if p.poll() is None}
    for f in (ST / "jobs" / j["id"]).glob("progress_*.json"):
        if _fresh(f) and _rj(f).get("state") not in ("done", "exited", "error"): s.add(int(f.stem.split("_")[1]))
    return s
def alive(j):
    if j["proc"] is not None and j["proc"].poll() is None: return True
    if live_idx(j): return True
    f = ST / "jobs" / j["id"] / "progress.json"
    return _fresh(f) and _rj(f).get("state") not in ("done", "exited", "error")
def progress(jid):
    jd = ST / "jobs" / jid; fs = sorted(jd.glob("progress_*.json"))
    if not fs: return _rj(jd / "progress.json")
    ps = [(_fresh(f), _rj(f)) for f in fs]; ps = [(a, p) for a, p in ps if p]
    if not ps: return {}
    try: base = int((jd / "base.txt").read_text())
    except Exception: base = 0
    act = [p for fr, p in ps if fr and p["state"] not in ("done", "exited", "error")]
    done = base + sum(p.get("done", 0) for _, p in ps); total = ps[0][1].get("total", 0)
    if act: state = "running" if any(p["state"] == "running" for p in act) else "paused"
    elif any(p["state"] == "error" for _, p in ps): state = "error"
    elif any(p["state"] == "done" for _, p in ps): state = "done"
    else: state = "running"
    rate = sum(1 / p["per_item"] for p in act if p.get("per_item")); per = round(1 / rate, 1) if rate else None
    return {"state": state, "done": done, "total": total, "cur": max(ps, key=lambda x: x[1].get("t", 0))[1].get("cur", ""), "per_item": per,
            "eta_min": None if per is None else round(per * (total - done) / 60, 1), "workers": len(act), "model": (act[0].get("model") if act else None), "vram": sum(p.get("vram", 0) for p in act)}
def _kill(p):   # venv-лаунчер fq5\Scripts\python.exe запускает настоящий python дочерним — terminate() убил бы только лаунчер
    if p is None or p.poll() is not None: return
    if os.name == "nt": subprocess.run(["taskkill", "/f", "/t", "/pid", str(p.pid)], capture_output=True)
    else: p.terminate()
def _orphans(jid):   # потоки этой задачи, которых сервер уже не помнит (гонка перезапусков, перезапуск сервера)
    if os.name != "nt": return []
    q = f"Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ? {{ $_.CommandLine -like '*worker.py gen*\\{jid}\\job.json*' }} | % {{ $_.ProcessId }}"
    r = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split() if x.isdigit()]
def stop_job(j):
    for p in (j.get("procs") or {}).values(): _kill(p)
    _kill(j["proc"])
    for pid in _orphans(j["id"]): subprocess.run(["taskkill", "/f", "/t", "/pid", str(pid)], capture_output=True)
    for f in (ST / "jobs" / j["id"]).glob("progress_*.json"):   # потоки, пережившие перезапуск сервера
        pid = _rj(f).get("pid")
        if pid and _fresh(f): subprocess.run(["taskkill", "/f", "/pid", str(pid)], capture_output=True) if os.name == "nt" else os.kill(pid, 15)
    for f in (ST / "jobs" / j["id"]).glob("progress*.json"):   # процесс убит — иначе свежий файл прогресса ещё 90 с считается «живым» и блокирует продолжение
        d = _rj(f)
        if d: d["state"] = "exited"; f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
RESTART_LOCK = threading.Lock()   # два переключения режима подряд (HTTP-сервер многопоточный) запускали по потоку каждое — второй оставался сиротой и делил видеопамять
def restart_gen():
  with RESTART_LOCK:
    for j in list(JOBS.values()):
        if j["kind"] != "gen" or not alive(j) or not j.get("py"): continue
        if progress(j["id"]).get("state") == "paused": continue   # на паузе: новый режим применится при «Продолжить»
        stop_job(j); time.sleep(1); jd = ST / "jobs" / j["id"]
        shutil.rmtree(jd / "claims", ignore_errors=True); (jd / "claims").mkdir(); (jd / "workers").write_text("1")   # взятые, но не сделанные реплики — снова в очередь
        for f in jd.glob("progress_*.json"): f.unlink(missing_ok=True)
        j["proc"] = spawn(j["id"], "gen", j["py"], append=True); j["procs"] = {0: j["proc"]}
def gen(names, model_id=None, takes=1, variants=None):
    if variants: takes = len(variants)
    S = load_state(); models = {m["id"]: m for m in load_models()}; groups = {}
    queued = set().union(*[j.get("names", set()) for j in JOBS.values() if j["kind"] == "gen" and alive(j)]) if not variants else set()   # уже стоят в живой очереди (повторный клик)
    for n in names:
        if n in queued: continue
        c = line_cfg(S, n); mid = model_id or c["model"]; text = S["lines"].get(n, {}).get("text") or rd_tsv(W / "texts.tsv").get(n, "")
        try: ref, desc = ref_for(n, c["ref"], S)
        except ValueError as e: print("skip", n, e); continue
        for k in range(takes):
            tid = time.strftime("%H%M%S") + f"_{mid}_{k}"; p = {**models[mid].get("params", {}), **c["params"]}
            if takes > 1: p["seed"] = int(p.get("seed", 1)) + k
            if variants: p.update(variants[k])
            groups.setdefault(mid, []).append({"name": n, "text": text, "ref": dict(ref), "ref_desc": desc, "params": p, "take": tid, "out": str(ST / "takes" / n / f"{tid}.wav")})
    return [start_job("gen", models[mid]["python"], items, {"model": models[mid]}) for mid, items in groups.items()]
def score(names):
    S = load_state(); tx = rd_tsv(W / "texts.tsv"); items = []
    for n in names:
        L = S["lines"].get(n, {}); c = line_cfg(S, n)
        try: ref, _ = ref_for(n, c["ref"], S)
        except ValueError: ref = {}
        dub = W / "dub_wav" / f"{n}.wav"; dd = None
        if dub.exists():
            import wave
            try:
                with wave.open(str(dub)) as w: dd = w.getnframes() / w.getframerate()
            except Exception: pass
        for t in L.get("takes", []):
            if "score" not in t and Path(t["file"]).exists(): items.append({"name": n, "take": t["take"], "file": t["file"], "text": L.get("text") or tx.get(n, ""), "dub_dur": dd, "ref_wav": ref.get("wav") or (ref.get("wavs") or [None])[0]})
    return [start_job("score", ASR_PY, items, {"campplus": CFG["campplus"]})] if items else []
FIX_VARIANTS = [{"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1, "seed": 2}, {"seed": 3}, {"temperature": 0.7, "top_p": 0.9, "seed": 4}]
def stage_lists(lines, S):
    """этапы от лёгкого к сложному: 1 генерация без проверок → 2 проверка → 3 исправление плохих (другие seed/параметры) → 4 выбор лучшего"""
    g = {1: [], 2: [], 3: [], 4: []}
    for l in lines:
        if not l["text"]: continue
        T = l["takes"]; n = l["name"]
        if not T and l["chosen"] != "dub" and not l["installed"]: g[1].append(n); continue
        if any(t.get("score") is None for t in T): g[2].append(n); continue
        if not T: continue
        b = min(t["score"] for t in T)
        if b > 0.5 and not S["lines"].get(n, {}).get("fixed") and l["chosen"] != "dub": g[3].append(n)
        if l["chosen"] != "dub" and not S["lines"].get(n, {}).get("picked"): g[4].append(n)
    return g
def autopick(names):
    with LOCK:
        S = load_state(); n = 0
        for nm in names:
            L = S["lines"].get(nm, {}); T = [t for t in L.get("takes", []) if t.get("score") is not None]
            if not T or L.get("chosen") == "dub": continue
            b = min(T, key=lambda t: t["score"])
            if b["score"] <= 0.5: L["chosen"] = b["take"]; L.pop("flag", None)
            elif (W / "dub_wem" / f"{nm}.wem").exists(): L["chosen"] = "dub"; L.pop("flag", None)
            else: L["chosen"] = b["take"]; L["flag"] = "bad"
            L["picked"] = True; n += 1
        save_state(S); return n
def run_stage(st, names=None):
    S = load_state(); L0 = light_lines(S); g = stage_lists(L0, S); g[5] = [l["name"] for l in L0 if l["flag"] == "regen"]; nm = names or g[st]
    if not nm: return {"n": 0, "jobs": []}
    if st in (1, 5):
        jobs = gen(nm)
        with LOCK:
            S = load_state()
            for x in nm: S["lines"].get(x, {}).pop("flag", None) if S["lines"].get(x, {}).get("flag") == "regen" else None
            save_state(S)
        return {"n": len(nm), "jobs": jobs}
    if st == 2: return {"n": len(nm), "jobs": score(nm)}
    if st == 3:
        with LOCK:
            S = load_state()
            for x in nm: S["lines"].setdefault(x, {})["fixed"] = True
            save_state(S)
        return {"n": len(nm), "jobs": gen(nm, variants=FIX_VARIANTS)}
    return {"n": autopick(nm), "jobs": []}
STAGE = {"busy": False, "msg": ""}
TRIED = {1: set(), 2: set(), 3: set(), 4: set()}
THR = {"mode": "auto", "n": 2, "max": 4, "engine": "hf", "bsize": 8, "cur": 0, "msg": "", "gpu": {}, "vram_per": None}
THR_F = ST / "threads.json"
FAST_PY = CFG.get("fast_python") or r"C:\voice\fq5\Scripts\python.exe"   # install_fast.bat: faster-qwen3-tts из форка с пакетным путём
ASR_PY = CFG.get("asr_python") or (FAST_PY if Path(FAST_PY).exists() else CFG["score_python"])   # Whisper на transformers + torch 2.11: на Blackwell ×3,5 к CTranslate2
ENGINES = {"hf": "обычный", "batch": "пакет", "fast": "быстрый (CUDA graphs)", "fastbatch": "быстрый пакет"}
def thr_save(): THR_F.write_text(json.dumps({k: THR[k] for k in ("mode", "n", "max", "vram_per", "engine", "bsize")}), encoding="utf-8")
def sysstat():
    o = {}
    try:
        u, a, t = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split("\n")[0].split(",")
        o.update(util=int(u), vused=int(a), vtotal=int(t))
    except Exception: pass
    try: import psutil; o["ram_free"] = psutil.virtual_memory().available >> 20
    except Exception: pass
    return o
def scale_loop():   # потоки генерации: вручную (mode=manual) или по загрузке GPU/памяти (auto)
    hist = []; last_add = 0; pend = None
    try: THR.update(json.loads(THR_F.read_text(encoding="utf-8")))
    except Exception: pass
    while True:
        time.sleep(8)
        try:
            resume_after_asr()
            gj = [j for j in JOBS.values() if j["kind"] == "gen" and (ST / "jobs" / j["id"] / "claims").exists() and alive(j)]
            st = sysstat(); THR["gpu"] = st; THR["cur"] = sum(len(live_idx(j)) for j in gj)
            if not gj: THR["msg"] = ""; continue
            j = gj[0]; pr = progress(j["id"]); idxs = live_idx(j); n = len(idxs); rem = pr.get("total", 0) - pr.get("done", 0)
            hist = (hist + [st.get("util", 0)])[-4:]
            if pr.get("state") == "paused": continue
            if pend and time.time() - pend[0] > 100 and "vused" in st:   # прогрелся новый поток — запоминаем, сколько он съел видеопамяти
                THR["vram_per"] = max(800, st["vused"] - pend[1]); pend = None; thr_save()
            target = n; fv = st.get("vtotal", 0) - st.get("vused", 0); fr = st.get("ram_free", 99999); pv = THR.get("vram_per") or 5500
            if THR["mode"] == "manual":
                target = max(1, min(8, int(THR["n"]))); THR["msg"] = f"вручную: {target}"
            else:
                THR["msg"] = f"авто: загрузка GPU {sum(hist) // len(hist)}%, свободно {fv // 1024} ГБ видеопамяти, {fr // 1024} ГБ ОЗУ"
                if (fv < 600 or fr < 1500) and time.time() - last_add > 150: target = max(1, n - 1); THR["msg"] += " — мало памяти, убавляю"
                elif n < THR["max"] and rem > 40 and time.time() - last_add > 150 and len(hist) >= 3 and sum(hist) / len(hist) < 70 and fv > pv * 1.15 + 500 and fr > 9000: target = n + 1
            if target != n or (j["proc"] is not None):
                (ST / "jobs" / j["id"] / "workers").write_text(str(target))
            if target > n and rem > 5 and time.time() - last_add > 20:
                idx = next(i for i in range(target) if i not in idxs); j.setdefault("procs", {})[idx] = spawn(j["id"], "gen", j["py"], idx=idx)
                last_add = time.time(); pend = (time.time(), st.get("vused", 0))
        except Exception as e: THR["msg"] = f"ошибка: {e}"
RESUME_F = ST / "resume_after_asr.json"
def resume_after_asr():
    try: ids = json.loads(RESUME_F.read_text(encoding="utf-8"))
    except Exception: return
    if not ids or any(j["kind"] == "asr" and alive(j) for j in JOBS.values()): return
    time.sleep(3)
    for i in ids:
        j = JOBS.get(i)
        if j and not alive(j) and j.get("py"): j["proc"] = spawn(i, j["kind"], j["py"], append=True); j["procs"] = {0: j["proc"]}
    RESUME_F.write_text("[]", encoding="utf-8")
ASR_C = {"t": 0, "items": {}}
def asr_items(force=False):
    if not force and time.time() - ASR_C["t"] < 60: return ASR_C["items"]
    S = load_state(); items = {}
    for n in [l["name"] for l in light_lines(S)]:
        try: r, _ = ref_for(n, line_cfg(S, n)["ref"], S)
        except ValueError: continue
        if r.get("text") is None and r.get("wav") and not r["wav"].startswith(str(ST)):
            p = Path(r["wav"]); items[r["wav"]] = {"name": p.stem, "wav": r["wav"], "lang": r.get("lang") or "ja", "out": str(p.parent.parent / f"texts_{r.get('lang') or 'ja'}.tsv")}
    ASR_C.update(t=time.time(), items=items); return items
def asr_start(names=None):
    """образцы без расшифровки (японские и др.) → фоновая задача Whisper, результат в texts_<язык>.tsv; генерация потом не слушает образец сама"""
    items = asr_items(True)
    if not items: return {"n": 0}
    stopped = []   # две модели в видеопамять не помещаются: генерацию останавливаем и продолжим, когда расшифровка закончится
    for j in JOBS.values():
        if j["kind"] == "gen" and alive(j): stop_job(j); stopped.append(j["id"])
    RESUME_F.write_text(json.dumps(stopped), encoding="utf-8")
    return {"n": len(items), "id": start_job("asr", ASR_PY, list(items.values()), {})}
AUTO = {"on": False, "stage": 0, "msg": ""}
def auto_loop():   # фоновый конвейер: берёт следующий непустой этап, когда нет работающих задач (пауза держит задачу живой — цикл ждёт)
    while True:
        time.sleep(5)
        try:
            if not AUTO["on"] or any(alive(j) for j in JOBS.values()): continue
            S = load_state(); g = stage_lists(light_lines(S), S)
            for st in (1, 2, 3, 4):
                nm = [x for x in g[st] if x not in TRIED[st]]   # то, что этап уже пробовал и не смог (нет образца и т.п.), повторно не берём
                if nm:
                    AUTO["stage"] = st; TRIED[st].update(nm); r = run_stage(st, nm); AUTO["msg"] = f"этап {st}: {r['n']}"
                    if r["n"]: break
            else: AUTO["on"] = False; AUTO["stage"] = 0; AUTO["msg"] = "всё сделано"
        except Exception as e: AUTO["msg"] = f"ошибка: {e}"; AUTO["on"] = False
_INS = [0.0, set()]
def installed_set():   # содержимое папки игры: глоб тысяч файлов на каждый опрос — кэш на 10 с
    if time.time() - _INS[0] > 10:
        gd = Path(CFG["game_dir"]); _INS[:] = [time.time(), {p.stem for p in gd.glob("*.*")} if gd.exists() else set()]
    return _INS[1]
def light_lines(S=None):
    """все реплики в лёгком виде (для этапов, фильтров, счётчиков)"""
    S = S or load_state(); ru = rd_tsv(W / "texts.tsv"); inst = installed_set(); out = []
    for n in sorted(set(ru) | set(S["lines"])):
        L = S["lines"].get(n, {}); out.append({"name": n, "text": L.get("text") or ru.get(n, ""), "takes": L.get("takes", []), "chosen": L.get("chosen"), "flag": L.get("flag"), "suspect": L.get("suspect"), "installed": n in inst})
    return out
def status_of(l):
    c = l["chosen"]
    if not c: return "no"
    if c == "dub": return "dub"
    t = next((t for t in l["takes"] if t["take"] == c), None); sc = t and t.get("score")
    return "in" if sc is None else "bad" if sc > 0.5 else "ok"
def run_prep(adapter, params, work_flt=None):
    with LOCK:
        S = load_state(); S["prep"] = {"adapter": adapter, "params": params}; save_state(S)
    jid = time.strftime("%H%M%S") + "_prep"; jd = ST / "jobs" / jid; jd.mkdir()
    (jd / "job.json").write_text(json.dumps({"items": [{}], "py": CFG.get("prep_python") or CFG["score_python"], "adapter": adapter, "params": params, "work": str(W), "tools": CFG.get("tools", {})}, ensure_ascii=False), encoding="utf-8")
    JOBS[jid] = {"id": jid, "proc": spawn(jid, "prep", CFG.get("prep_python") or CFG["score_python"]), "kind": "prep", "n": 1, "pos": 0, "started": time.time()}; return jid
def take_ts(t):
    if t.get("ts"): return t["ts"]
    try: return os.path.getmtime(t["file"])   # старые дубли без ts — по времени файла
    except Exception: return 0
_SIX = {"k": None, "tr": {}, "sub": {}}
def search_index(LG, tl, dub):
    """строки для поиска в нижнем регистре: перевод другого автора и субтитры оригинала; пересобираются только при смене tsv"""
    k = tuple(id(tl[l]) for l in LG) + (id(dub),)
    if _SIX["k"] != k:
        _SIX["sub"] = {}
        for l in LG:
            for n, t in tl[l].items(): _SIX["sub"][n] = _SIX["sub"].get(n, "") + "\n" + t.lower()
        _SIX["tr"] = {n: t.lower() for n, t in dub.items()}; _SIX["k"] = k
    return _SIX
def state_json(q="", flt="all", limit=300, off=0, sort="", where="all"):
    ingest(); S = load_state(); ru = rd_tsv(W / "texts.tsv"); LG = langs(); tl = {l: rd_tsv(W / f"texts_{l}.tsv") for l in LG}; dub = rd_tsv(W / "texts_dub.tsv")
    allv = light_lines(S); q = q.lower().strip(); IX = search_index(LG, tl, dub) if q else None
    def hit(l):   # where: name — имя файла, text — перевод (наш текст и другого автора), sub — субтитры оригинала, all — везде
        n = l["name"]
        if where in ("all", "name") and q in n.lower(): return True
        if where in ("all", "text") and (q in l["text"].lower() or q in IX["tr"].get(n, "")): return True
        return where in ("all", "sub") and q in IX["sub"].get(n, "")
    def ok(l):
        if q and not hit(l): return False
        st = status_of(l)
        return flt == "all" or (flt == "none" and st == "no") or (flt == "dub" and st == "dub") or (flt == "bad" and st == "bad") or (flt == "unscored" and st == "in") or (flt == "inst" and l["installed"]) or (flt == "flagbad" and l["flag"] == "bad") or (flt == "regen" and l["flag"] == "regen") or (flt == "suspect" and l.get("suspect"))
    vis = [l for l in allv if ok(l)]; lines = []
    if sort == "time_desc":
        vis.sort(key=lambda l: max([take_ts(t) for t in l["takes"]] or [0]), reverse=True)
    if sort == "time":   # по времени последней генерации, новые в конце; off<0 — последняя страница (слежение за новыми)
        vis.sort(key=lambda l: max([take_ts(t) for t in l["takes"]] or [0]))
        if off < 0: off = max(0, len(vis) - limit)
    for l in vis[off:off + limit]:
        n = l["name"]; L = S["lines"].get(n, {})
        l = {**l, "takes": [t if t.get("ts") else {**t, "mt": take_ts(t)} for t in l["takes"]]}
        lines.append({**l, "orig_text": ru.get(n, ""), "edited": bool(L.get("text")), "src_text": {k: tl[k].get(n, "") for k in LG}, "dub_text": dub.get(n, ""), "has_dub": (W / "dub_wav" / f"{n}.wav").exists(),
            "has": [k for k in LG if (W / f"{k}_wav/{n}.wav").exists()], "cfg": line_cfg(S, n), "own": {k: L[k] for k in ("model", "ref", "params") if k in L}})
    jobs = [{"id": i, "kind": j["kind"], "n": j["n"], "running": alive(j), "code": None if j["proc"] is None else j["proc"].poll(), **progress(i)} for i, j in JOBS.items()]
    for x in jobs:
        if x["running"] and x.get("state") == "exited": x["state"] = "loading"   # процесс запущен, модель ещё грузится, прогресс прошлого запуска устарел
    return {"lines": lines, "off": off, "total": len(allv), "matched": len(vis), "models": load_models(), "langs": LG, "samples": S["samples"], "defaults": S["defaults"], "jobs": jobs, "auto": AUTO, "threads": THR, "asr_n": len(asr_items()) if not any(j["kind"] == "asr" and alive(j) for j in JOBS.values()) else 0, "stage_msg": STAGE["msg"], "stage_busy": STAGE["busy"], "autoinst": {**(S.get("autoinst") or {"on": False, "every": 60}), **{k: v for k, v in AINST.items() if k != "t"}},
            "stages": {k: len(v) for k, v in stage_lists(allv, S).items()}, "inst": install_stage(S), "adapters": __import__("prep").ADAPTERS, "prep_last": S.get("prep", {}), "game_dir": CFG["game_dir"]}
AINST = {"last": "", "n": 0, "msg": ""}
def _sig(src):
    try: st = src.stat(); return [str(src), st.st_size, int(st.st_mtime)]
    except OSError: return None
def _put(src, dst):
    """в папку игры кладём жёсткую ссылку (место не тратится), если диск другой — копию"""
    dst.unlink(missing_ok=True)
    try: os.link(src, dst)
    except OSError: shutil.copy(src, dst)
AUD_EXT = ("wem", "mp3", "wav", "flac")   # порядок поиска в VoiceLayer: первый найденный играет, остальные убираем
def _enc(src, dst):
    """дубль → mp3 для игры: WAV 24 кГц без сжатия занимает 48 КБ/с, mp3 64 кбит/с — 8 КБ/с (в 6 раз меньше), VoiceLayer читает mp3 через miniaudio"""
    tmp = dst.with_name(dst.stem + ".part.mp3")
    r = subprocess.run([CFG["tools"]["ffmpeg"], "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-c:a", "libmp3lame", "-b:a", f"{CFG.get('install_kbps', 64)}k", str(tmp)], capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode or not tmp.exists(): tmp.unlink(missing_ok=True); raise OSError(r.stderr.decode("utf-8", "replace")[-200:] or "ffmpeg failed")
    os.replace(tmp, dst)
_LED = {"k": None, "d": {}}
INST_LOCK = threading.Lock(); INST = {"run": False, "k": 0, "n": 0, "t": 0.0}
def led_load():   # installed.json (12 МБ) читаем заново только при изменении: его смотрит каждый опрос интерфейса
    f = W / "installed.json"
    try: k = (f.stat().st_mtime_ns, f.stat().st_size)
    except OSError: return {}
    if _LED["k"] != k:
        try: _LED["d"] = json.loads(f.read_text(encoding="utf-8")); _LED["k"] = k
        except Exception: pass
    return _LED["d"]
_GSZ = [0.0, {}]
def game_size():   # размер папки игры по расширениям: обход 60 тыс. файлов — раз в минуту
    if time.time() - _GSZ[0] > 60:
        out = {}
        try:
            with os.scandir(CFG["game_dir"]) as it:
                for e in it:
                    x = e.name.rsplit(".", 1)[-1].lower(); c = out.setdefault(x, [0, 0]); c[0] += 1; c[1] += e.stat().st_size
        except OSError: pass
        _GSZ[:] = [time.time(), out]
    return _GSZ[1]
_IST = [0.0, None]
def install_stage(S):
    """этап «в игру»: сколько выбранных реплик уже в игре в нужном формате и сколько ждёт; счёт по 60 тыс. реплик — не чаще раза в 5 с"""
    if time.time() - _IST[0] > 5 or _IST[1] is None: _IST[:] = [time.time(), _inst_count(S)]
    return {**_IST[1], "fmt": CFG.get("install_format", "mp3"), "kbps": CFG.get("install_kbps", 64), "size": game_size(), "run": INST["run"], "k": INST["k"], "n": INST["n"], "msg": AINST.get("msg", "")}
def _inst_count(S):
    led = led_load(); fmt = CFG.get("install_format", "mp3"); ok = pend = 0
    for n, L in S["lines"].items():
        c = L.get("chosen")
        if not c or (L.get("flag") == "bad" and c != "dub"): continue
        v = led.get(n); want = "wem" if c == "dub" else fmt
        if v and v[-1] == want: ok += 1
        else: pend += 1
    return {"ok": ok, "pending": pend}
def install_bg(only_new=True):
    if INST["run"]: return False
    threading.Thread(target=install, args=(None, only_new), daemon=True).start(); return True
def _enc_many(pairs):
    """пачка дублей одним ffmpeg (N входов → N выходов): запуск процесса на Windows дороже самого кодирования, по одному файлу выходило ~26 реплик/с"""
    a = [CFG["tools"]["ffmpeg"], "-y", "-loglevel", "error"]
    for src, _ in pairs: a += ["-i", str(src)]
    tmps = [d.with_name(d.stem + ".part.mp3") for _, d in pairs]
    for i, t in enumerate(tmps): a += ["-map", f"{i}:a", "-ac", "1", "-c:a", "libmp3lame", "-b:a", f"{CFG.get('install_kbps', 64)}k", str(t)]
    r = subprocess.run(a, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)); ok = []
    for (src, d), t in zip(pairs, tmps):
        if not r.returncode and t.exists() and t.stat().st_size: os.replace(t, d); ok.append(d)
        else: t.unlink(missing_ok=True)
    return ok
def install(names=None, only_new=False):
    if not INST_LOCK.acquire(timeout=1): return []   # одна установка за раз: ручная и авто не пишут в папку игры одновременно
    try:
        INST.update(run=True, k=0, n=0, t=time.time()); return _install(names, only_new)
    finally: INST["run"] = False; _GSZ[0] = 0; INST_LOCK.release()
def _install(names=None, only_new=False):
    """отдаёт выбранные дубли в папку игры; only_new — по таблице installed.json пропускает уже установленное без повторного копирования"""
    S = load_state(); gd = Path(CFG["game_dir"]); gd.mkdir(parents=True, exist_ok=True); done = []; todo = []
    led_f = W / "installed.json"
    led = dict(led_load())
    for n, L in S["lines"].items():
        if names and n not in names: continue
        c = L.get("chosen")
        if L.get("flag") == "bad" and c != "dub": continue   # плохие не отдаём в игру
        try:
            if c == "dub" and (W / "dub_wem" / f"{n}.wem").exists(): src, ext = W / "dub_wem" / f"{n}.wem", "wem"
            elif c:
                t = next((t for t in L.get("takes", []) if t["take"] == c), None)
                if not t: continue
                src, ext = Path(t["file"]), CFG.get("install_format", "mp3")   # mp3 (сжатие) или wav (жёсткая ссылка на дубль)
            else: continue
            sg = _sig(src)
            if sg is None: continue
            if only_new and led.get(n) == sg + [ext]: continue
            todo.append((n, src, ext, sg))
        except OSError as e: AINST["msg"] = f"{n}: {e}"
    def fin(n, ext):
        for o in AUD_EXT:
            if o != ext: (gd / f"{n}.{o}").unlink(missing_ok=True)
    def one(batch):   # пачка реплик: mp3 — одним ffmpeg, остальное — ссылкой/копией
        res = []; enc = []
        for n, src, ext, sg in batch:
            dst = gd / f"{n}.{ext}"
            try:
                if ext == "mp3" and src.suffix.lower() != ".mp3":
                    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size: fin(n, ext); res.append((n, sg + [ext])); continue   # уже перекодирован прошлым прогоном
                    enc.append((n, src, ext, sg)); continue
                _put(src, dst); fin(n, ext); res.append((n, sg + [ext]))
            except OSError as e: AINST["msg"] = f"{n}: {e}"
        if enc:
            ok = set(_enc_many([(src, gd / f"{n}.{ext}") for n, src, ext, sg in enc]))
            for n, src, ext, sg in enc:
                d = gd / f"{n}.{ext}"
                if d not in ok:
                    try: _enc(src, d)   # пачка упала (битый файл) — этот по одному
                    except OSError as e: AINST["msg"] = f"{n}: {e}"; continue
                fin(n, ext); res.append((n, sg + [ext]))
        return len(batch), res
    from concurrent.futures import ThreadPoolExecutor
    INST["n"] = len(todo); B = int(CFG.get("install_batch", 24)); k = 0; last = 0
    with ThreadPoolExecutor(int(CFG.get("install_threads", 8))) as ex:
        for m, res in ex.map(one, [todo[i:i + B] for i in range(0, len(todo), B)]):
            k += m; INST["k"] = k
            for n, v in res: led[n] = v; done.append(n)
            if k - last >= 2000: last = k; led_f.write_text(json.dumps(led, ensure_ascii=False), encoding="utf-8"); AINST["msg"] = f"в игру {k}/{len(todo)}"
    if done: led_f.write_text(json.dumps(led, ensure_ascii=False), encoding="utf-8")
    return done
def pending_count():
    S = load_state()
    led = led_load()
    return sum(1 for n, L in S["lines"].items() if L.get("chosen") and n not in led and not (L.get("flag") == "bad" and L.get("chosen") != "dub"))
def autoinst_loop():   # отдаёт в игру новые готовые реплики: раз в N секунд или когда накопилось K новых
    while True:
        time.sleep(5)
        try:
            a = load_state().get("autoinst") or {}
            if not a.get("on"): continue
            due = time.time() - AINST.get("t", 0) >= max(15, int(a.get("every", 60)))
            k = int(a.get("min_new", 0))
            if not due and not (k > 0 and pending_count() >= k): continue
            AINST["t"] = time.time(); d = install(None, True)
            AINST["last"] = time.strftime("%H:%M:%S"); AINST["n"] += len(d)
            if d: AINST["msg"] = f"+{len(d)} в игру"
        except Exception as e: AINST["msg"] = f"ошибка: {e}"
def browse(p):
    """Серверный «проводник» для формы подготовки: список папок и файлов (браузер не отдаёт полный путь)."""
    out = {"path": p, "parent": "", "dirs": [], "files": [], "err": ""}
    try:
        if not p:
            if os.name == "nt":
                out["dirs"] = [d + ":\\" for d in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(d + ":\\")]
                return out
            p = "/"
        P = Path(p)
        if P.is_file(): P = P.parent
        out["path"] = str(P)
        out["parent"] = "" if P.parent == P and os.name != "nt" else (str(P.parent) if P.parent != P else "")
        es = sorted(P.iterdir(), key=lambda e: e.name.lower())
        out["dirs"] = [e.name for e in es if e.is_dir()][:500]
        out["files"] = [e.name for e in es if e.is_file()][:500]
    except Exception as e:
        out["err"] = str(e)
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ct="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ct + ("; charset=utf-8" if "json" in ct or "html" in ct else "")); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query); path = urllib.parse.unquote(u.path)
        if path == "/": return self.send(200, (HERE / "index.html").read_bytes(), "text/html")
        if path == "/api/state": return self.send(200, state_json(q.get("q", [""])[0], q.get("flt", ["all"])[0], int(q.get("limit", ["300"])[0]), int(q.get("off", ["0"])[0]), q.get("sort", [""])[0], q.get("in", ["all"])[0]))
        if path == "/api/browse": return self.send(200, browse(q.get("path", [""])[0]))
        if path == "/api/ping": return self.send(200, {"ok": 1, "pid": os.getpid(), "started": START_T})
        if path == "/api/log":
            f = ST / "jobs" / q["id"][0] / "job.log"; return self.send(200, {"log": f.read_text(encoding="utf-8", errors="replace")[-6000:] if f.exists() else ""})
        if path.startswith("/audio/"):
            kind, _, rest = path[7:].partition("/"); S = load_state()
            f = {"orig": lambda: W / f"{rest.split('/')[0]}_wav" / (rest.split("/")[1] + ".wav"), "dub": lambda: W / "dub_wav" / (rest + ".wav"), "take": lambda: ST / "takes" / rest, "sample": lambda: ST / "samples" / rest}[kind]()
            if not f.exists(): return self.send(404, {"error": "нет файла"})
            return self.send(200, f.read_bytes(), "audio/wav")
        self.send(404, {"error": "?"})
    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query); raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if u.path == "/api/upload":   # свой образец: тело — wav
            name = re.sub(r"[^\w.\-]", "_", q["name"][0]); (ST / "samples" / name).write_bytes(raw)
            with LOCK: S = load_state(); S["samples"].setdefault(name, {"text": ""}); save_state(S)
            return self.send(200, {"ok": name})
        d = json.loads(raw or b"{}")
        with LOCK:
            S = load_state()
            if u.path == "/api/line":   # правка реплики: text/model/ref/params/chosen (null — сбросить к умолчанию)
                L = S["lines"].setdefault(d["name"], {})
                for k in ("text", "model", "ref", "params", "chosen", "flag"):
                    if k in d: (L.pop(k, None) if d[k] is None else L.__setitem__(k, d[k]))
                for n in d.get("also", []):   # применить то же к другим выбранным
                    L2 = S["lines"].setdefault(n, {})
                    for k in ("model", "ref", "params"):
                        if k in d: (L2.pop(k, None) if d[k] is None else L2.__setitem__(k, d[k]))
                save_state(S); return self.send(200, {"ok": 1})
            if u.path == "/api/suspect":   # горячая клавиша «странная реплика»: {minutes?, name?}
                if d.get("scan"):   # прогнать уже готовые дубли по длительности (без чтения звука)
                    n = 0
                    for nm, L in S["lines"].items():
                        for t in L.get("takes", []):
                            why = quick_dur(t)
                            if why and not L.get("suspect"): auto_suspect(L, nm, why); n += 1; break
                    save_state(S); return self.send(200, {"marked": n})
                if d.get("clear"):   # снять все отметки
                    n = sum(1 for L in S["lines"].values() if L.pop("suspect", None)); save_state(S); return self.send(200, {"cleared": n})
                r = mark_suspect(S, int(d.get("minutes", 5)), d.get("name")); save_state(S); return self.send(200, r)
            if u.path == "/api/bulk":   # массовая пометка: {names, flag: null|"bad"|"regen"} и/или {names, chosen: "dub"|"mine"}
                n = 0
                for nm in d["names"]:
                    L = S["lines"].setdefault(nm, {})
                    if "flag" in d: (L.pop("flag", None) if d["flag"] is None else L.__setitem__("flag", d["flag"])); n += 1
                    if d.get("chosen") == "dub" and (W / "dub_wem" / f"{nm}.wem").exists(): L["chosen"] = "dub"; L.pop("flag", None); n += 1
                    elif d.get("chosen") == "mine" and L.get("takes"):
                        ts = [t for t in L["takes"] if t.get("score") is not None] or L["takes"]; L["chosen"] = min(ts, key=lambda t: t.get("score", 9))["take"] if ts[0].get("score") is not None else ts[-1]["take"]; n += 1
                save_state(S); return self.send(200, {"ok": n})
            if u.path == "/api/defaults": S["defaults"].update(d); save_state(S); return self.send(200, {"ok": 1})
            if u.path == "/api/sample": S["samples"].setdefault(d["file"], {})["text"] = d["text"]; save_state(S); return self.send(200, {"ok": 1})
            if u.path == "/api/delete_take":
                L = S["lines"].get(d["name"], {}); L["takes"] = [t for t in L.get("takes", []) if t["take"] != d["take"]]
                if L.get("chosen") == d["take"]: L["chosen"] = None
                save_state(S); return self.send(200, {"ok": 1})
        if u.path == "/api/prep":
            if d.get("mode") == "dub":   # второй вариант: голос-образец = русская озвучка другого автора
                with LOCK: S = load_state(); S["defaults"]["ref"] = {"src": "dub"}; save_state(S)
            elif d.get("mode") == "any":
                with LOCK: S = load_state(); S["defaults"]["ref"] = {"src": "any", "order": d.get("order", "")}; save_state(S)
            return self.send(200, {"jobs": [run_prep(d["adapter"], d["params"])]})
        if u.path == "/api/stage":   # долго (десятки тысяч реплик) — запускаем в потоке и отвечаем сразу
            if STAGE["busy"]: return self.send(200, {"n": 0, "jobs": [], "busy": STAGE["msg"]})
            def _go(st=int(d["stage"]), nm=d.get("names")):
                STAGE["busy"] = True; STAGE["msg"] = f"этап {st}: готовлю задачу…"
                try: r = run_stage(st, nm); STAGE["msg"] = f"этап {st}: поставлено {r['n']}" if r["n"] else f"этап {st}: нечего делать"
                except Exception as e: STAGE["msg"] = f"этап {st}: ошибка {e}"
                finally: STAGE["busy"] = False
            threading.Thread(target=_go, daemon=True).start()
            return self.send(200, {"n": -1, "jobs": [], "starting": True})
        if u.path == "/api/autoinst":
            S = load_state(); S["autoinst"] = {"on": bool(d.get("on")), "every": max(15, int(d.get("every", 60))), "min_new": max(0, int(d.get("min_new", 0)))}; save_state(S); AINST["t"] = 0
            return self.send(200, S["autoinst"])
        if u.path == "/api/asr":
            def run():
                STAGE["busy"] = True; STAGE["msg"] = "считаю образцы без расшифровки…"
                try: r = asr_start(); STAGE["msg"] = f"расшифровка образцов: {r['n']}" if r["n"] else "все образцы уже расшифрованы"
                except Exception as e: STAGE["msg"] = f"ошибка: {e}"
                STAGE["busy"] = False
            threading.Thread(target=run, daemon=True).start(); return self.send(200, {"ok": 1})
        if u.path == "/api/threads":
            if "engine" in d or "bsize" in d:   # смена режима: потоки идущей генерации перезапускаются в новом режиме, сделанное сохраняется
                old = (THR["engine"], THR["bsize"])
                if d.get("engine") in ENGINES: THR["engine"] = d["engine"]
                if "bsize" in d: THR["bsize"] = max(2, min(32, int(d["bsize"])))
                thr_save()
                if (THR["engine"], THR["bsize"]) != old: restart_gen()
                return self.send(200, THR)
            THR["mode"] = "manual" if d.get("mode") == "manual" else "auto"; THR["n"] = max(1, min(8, int(d.get("n", THR["n"])))); thr_save(); return self.send(200, THR)
        if u.path == "/api/auto": AUTO["on"] = bool(d["on"]); AUTO["msg"] = "запущено" if AUTO["on"] else "выключено"; return self.send(200, AUTO)
        if u.path == "/api/generate":
            jobs = gen(d["names"], d.get("model"), int(d.get("takes", 1)))
            with LOCK:   # перегенерация поставлена в очередь — снять пометку
                S = load_state()
                for nm in d["names"]:
                    if S["lines"].get(nm, {}).get("flag") == "regen": S["lines"][nm].pop("flag")
                save_state(S)
            return self.send(200, {"jobs": jobs})
        if u.path == "/api/score": return self.send(200, {"jobs": score(d["names"])})
        if u.path == "/api/install":
            if d.get("bg"): return self.send(200, {"started": install_bg(bool(d.get("only_new", True)))})
            return self.send(200, {"installed": install(d.get("names"))})
        if u.path == "/api/uninstall":
            gd = Path(CFG["game_dir"]); n = 0
            for nm in d["names"]:
                for e in ("wav", "wem"): n += (gd / f"{nm}.{e}").exists(); (gd / f"{nm}.{e}").unlink(missing_ok=True)
            return self.send(200, {"removed": n})
        if u.path == "/api/restart":
            def _re():
                flush_state()
                RESUME_F.write_text("[]", encoding="utf-8")   # сначала: иначе scale_loop возобновит генерацию после остановки расшифровки
                for j in list(JOBS.values()):   # перезапуск = чистый лист: все проходы останавливаются, модели выгружаются из памяти (продолжить можно кнопкой «продолжить» — готовое пропустится)
                    try:
                        if alive(j): stop_job(j)
                    except Exception: pass
                time.sleep(0.4)
                try: SRV.server_close()
                except Exception: pass
                subprocess.Popen([sys.executable] + sys.argv, cwd=str(HERE))
                os._exit(0)
            threading.Thread(target=_re, daemon=True).start()
            return self.send(200, {"ok": 1})
        if u.path == "/api/stop":
            j = JOBS.get(d["id"]); j and alive(j) and stop_job(j); return self.send(200, {"ok": 1})
        if u.path in ("/api/pause", "/api/resume", "/api/pause_all", "/api/resume_all"):   # пауза: воркер доделывает текущую реплику и ждёт; модель остаётся в памяти
            ids = list(JOBS) if u.path.endswith("_all") else [d["id"]]
            for i in ids:
                j = JOBS.get(i); jd = ST / "jobs" / i
                if not j: continue
                if u.path.startswith("/api/pause"):
                    if alive(j): (jd / "ctl").write_text("pause")
                elif alive(j): (jd / "ctl").unlink(missing_ok=True)
                elif progress(i).get("state") != "done" and j.get("py") and (u.path == "/api/resume" or progress(i).get("state") == "paused"):
                    j["proc"] = spawn(i, j["kind"], j["py"], append=True); j["procs"] = {0: j["proc"]}   # процесс умер (перезапуск студии/компьютера) — продолжить с места остановки
            return self.send(200, {"ok": 1})
        self.send(404, {"error": "?"})
if __name__ == "__main__":
    load_old_jobs(); threading.Thread(target=auto_loop, daemon=True).start(); threading.Thread(target=autoinst_loop, daemon=True).start(); threading.Thread(target=scale_loop, daemon=True).start(); threading.Thread(target=flush_loop, daemon=True).start()
    print(f"VoiceStudio: http://{CFG['host']}:{CFG['port']}  (work={W})", flush=True); SRV = ThreadingHTTPServer((CFG["host"], CFG["port"]), H); SRV.serve_forever()
