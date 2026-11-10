#!/usr/bin/env python3
"""Воркер студии: `worker.py gen <job.json>` (генерация, python модели) и `worker.py score <job.json>` (оценка, python с whisper).
Результаты — строки JSON в <job dir>/results.jsonl, прогресс — в stdout (сервер пишет его в job.log)."""
import json, os, re, subprocess, sys, time
from pathlib import Path
job = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")); jd = Path(sys.argv[2]).parent
res = open(jd / "results.jsonl", "a", encoding="utf-8")
_rfd = os.open(str(jd / "results.jsonl"), os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))
def emit(**k): os.write(_rfd, (json.dumps(k, ensure_ascii=False) + "\n").encode("utf-8"))   # одна запись на строку: потоки не перемешивают строки
IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # номер потока генерации (несколько воркеров берут реплики из общей очереди)
def say(*a): print(time.strftime("[%H:%M:%S]"), *(["[п%d]" % IDX] if IDX else []), *a, flush=True)
done_keys = set()   # уже сделанное (продолжение после остановки): (имя, дубль)
if (jd / "results.jsonl").exists():
    for l in (jd / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try: r = json.loads(l); done_keys.add((r["name"], r.get("take")))
        except Exception: pass
spent = 0.0   # секунды активной работы (без пауз) — для скорости и оценки
proc_n = 0   # сколько реплик обработано в этом запуске
def _vused():
    try: return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception: return 0
VR = {"base": _vused(), "t": 0, "v": 0}   # занято до загрузки модели: разница = сколько видеопамяти занял этот воркер
MODEL = {"gen": "Qwen3-TTS (озвучка)", "asr": "Whisper large-v3 (расшифровка)", "score": "Whisper large-v3 + CAM++ (оценка)", "prep": "Whisper large-v3 (подготовка)"}.get(sys.argv[1], sys.argv[1])
def _vram():
    if time.time() - VR["t"] > 6: VR["t"] = time.time(); VR["v"] = max(0, _vused() - VR["base"])
    return VR["v"]
def prog(state, k, total, cur=""):
    per = spent / proc_n if proc_n else None
    pf = jd / (f"progress_{IDX}.json" if sys.argv[1] == "gen" else "progress.json"); tmp = pf.with_suffix(".tmp%d" % IDX)
    tmp.write_text(json.dumps({"state": state, "done": k, "total": total, "cur": cur, "per_item": None if per is None else round(per, 1), "eta_min": None if per is None else round(per * (total - k) / 60, 1), "t": time.time(), "pid": os.getpid(), "model": MODEL, "vram": _vram()}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, pf)
done_keys_start = set(done_keys)
_tc = {}
def tsv_text(ref):   # texts_<язык>.tsv лежит рядом с папкой <язык>_wav; имя образца = имя файла
    p = Path(ref["wav"]); f = p.parent.parent / f"texts_{ref.get('lang') or 'ja'}.tsv"
    try: m = f.stat().st_mtime
    except OSError: return None
    if _tc.get(f, (0,))[0] != m:
        d = {}
        for l in f.read_text(encoding="utf-8").splitlines():
            k, _, v = l.partition("\t")
            if v.strip(): d[k] = v.strip()
        _tc[f] = (m, d)
    return _tc[f][1].get(p.stem)
def workers_n():   # сколько потоков должно работать: поток с номером >= этого числа заканчивает после текущей реплики
    try: return max(1, int((jd / "workers").read_text().strip()))
    except Exception: return 1
def gate(k, total, cur):   # пауза: файл ctl = "pause" → ждём, пока его не уберут
    while (jd / "ctl").exists() and (jd / "ctl").read_text().strip() == "pause":
        prog("paused", k - 1, total, cur); time.sleep(1)

def quick_check(w, sr, text):
    """быстрая проверка дубля без нейросетей (миллисекунды, numpy): затянутая речь без EOS, обрыв, перегруз, шум, тишина → список причин"""
    try:
        y = np.asarray(w, dtype="float32").ravel(); why = []
        if y.size < sr // 10: return ["пусто"]
        dur = y.size / sr; exp = 0.5 + 0.075 * len(text or "")
        if text and dur > 3 * exp: why.append(f"затянуто ×{dur / exp:.1f}")
        if text and dur < 0.4 * exp: why.append(f"обрыв ×{dur / exp:.2f}")
        pk = float(np.abs(y).max())
        if pk < 0.02: why.append("тихо")
        if float((np.abs(y) > 0.99).mean()) > 0.002: why.append("перегруз")
        n, h = 2048, 1024
        if y.size > n * 2:
            S = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(y, n)[::h] * np.hanning(n), axis=1)) + 1e-9; e = S.sum(1); A = S[e > e.max() * 0.05]
            if len(A) and float(np.median(np.exp(np.log(A).mean(1)) / A.mean(1))) > 0.6: why.append("шум")   # спектр плоский — не речь
        return why
    except Exception: return []
# ---------- адаптеры: load(cfg) -> synth(text, ref_wav, ref_text, params) -> (samples, sr) ----------
ENGINE = os.environ.get("STUDIO_ENGINE", "hf")   # hf — по одной (штатный qwen_tts); batch — пакетом; fast — CUDA graphs (faster-qwen3-tts); fastbatch — CUDA graphs пакетом
BSIZE = max(1, int(os.environ.get("STUDIO_BATCH", "8") or 8))
def _qwen_text(text, p):
    t = text.strip()
    if p.get("tail", True) and not t.endswith(","): t += " ..."
    return t
def _qwen_kw(p): return {x: p[x] for x in ("temperature", "top_p", "top_k", "repetition_penalty") if x in p}
def adapter_qwen3(cfg):
    import torch
    fast = ENGINE in ("fast", "fastbatch")
    if fast:   # тот же чекпойнт, статический KV-кэш + CUDA graphs; выход не побитово равен штатному (другие ядра bf16)
        from faster_qwen3_tts import FasterQwen3TTS
        m = FasterQwen3TTS.from_pretrained(cfg["model"]); base = m.model
    else:
        from qwen_tts import Qwen3TTSModel
        m = base = Qwen3TTSModel.from_pretrained(cfg["model"], device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    cache = {}
    def prompt(ref, ref_text):
        k = (ref, ref_text)
        if k not in cache: cache[k] = base.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text, x_vector_only_mode=not ref_text.strip())
        return cache[k]
    def call(texts, prompts, p, cap, ref_text=""):
        kw = dict(text=texts if len(texts) > 1 else texts[0], language="Russian", voice_clone_prompt=prompts, max_new_tokens=cap, **_qwen_kw(p))
        if fast: kw.update(append_silence=False, ref_text=ref_text)   # как в штатном: без 0,5 с тишины после образца; ICL-образцу нужна его расшифровка
        kw["non_streaming_mode"] = bool(p.get("non_streaming", True))
        return m.generate_voice_clone(**kw)
    def synth(text, ref, ref_text, p):
        torch.manual_seed(int(p.get("seed", 1)))
        w, sr = call([_qwen_text(text, p)], prompt(ref, ref_text), p, 60 + 6 * len(text), ref_text)   # потолок: короткий образец иногда не даёт EOS
        return w[0], sr
    def many(rows):   # rows: [(text, ref, ref_text, params)] с одинаковыми params → [(w, sr)]; seed общий на пакет
        p = rows[0][3]; torch.manual_seed(int(p.get("seed", 1)))
        pr = [x for r in rows for x in prompt(r[1], r[2])]
        w, sr = call([_qwen_text(r[0], r[3]) for r in rows], pr, p, max(60 + 6 * len(r[0]) for r in rows))
        return [(x, sr) for x in w]
    if ENGINE == "fastbatch":   # пакет через CUDA graphs: форк POMXARK/faster-qwen3-tts@batched = 0.5.x + PR #150 (в апстрим не слит)
        if not hasattr(m, "_generate_voice_clone_batch"):
            say("ВНИМАНИЕ: установленный faster-qwen3-tts не умеет пакеты (нужен install_fast.bat) — работаю как «быстрый» по одной"); return synth
        def many_fast(rows):
            p = rows[0][3]; torch.manual_seed(int(p.get("seed", 1)))
            w, sr = m.generate_voice_clone(text=[_qwen_text(r[0], r[3]) for r in rows], language="Russian", ref_text=[r[2] for r in rows],
                                           voice_clone_prompt=[x for r in rows for x in prompt(r[1], r[2])], non_streaming_mode=bool(p.get("non_streaming", True)),
                                           max_new_tokens=max(60 + 6 * len(r[0]) for r in rows), **_qwen_kw(p))
            return [(x, sr) for x in w]
        synth.many = many_fast
    elif ENGINE == "batch": synth.many = many
    return synth
def adapter_cosyvoice3(cfg):
    repo = Path(cfg["repo"]); sys.path[:0] = [str(repo), str(repo / "third_party" / "Matcha-TTS")]
    import random, numpy, torch; from cosyvoice.cli.cosyvoice import AutoModel
    tts = AutoModel(model_dir=str(repo / cfg["model"]), fp16=False)
    def synth(text, ref, ref_text, p):
        s = int(p.get("seed", 1)); random.seed(s); numpy.random.seed(s); torch.manual_seed(s)
        it = tts.inference_zero_shot(text.strip(), "You are a helpful assistant.<|endofprompt|>" + ref_text, ref, stream=False)
        return torch.cat([j["tts_speech"] for j in it], dim=1)[0].cpu().numpy(), tts.sample_rate
    return synth
def adapter_script(cfg):   # "adapter": "script:C:\\path\\my_tts.py" — файл с функцией load(cfg) -> synth
    import importlib.util
    sp = importlib.util.spec_from_file_location("custom_tts", cfg["adapter"][7:]); mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod); return mod.load(cfg)
ADAPTERS = {"qwen3": adapter_qwen3, "cosyvoice3": adapter_cosyvoice3}

def audio16(path):
    """wav -> float32 моно 16 кГц массивом: faster-whisper сам декодирует через PyAV, а в старом av падает на 'metadata_errors'"""
    import numpy as np, soundfile as sf
    y, sr = sf.read(str(path), dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
    if sr != 16000: y = np.interp(np.linspace(0, len(y) - 1, max(1, int(len(y) * 16000 / sr))), np.arange(len(y)), y).astype("float32")
    return y
ASR_MODEL = os.environ.get("STUDIO_ASR_MODEL", "large-v3"); ASR_BATCH = max(1, int(os.environ.get("STUDIO_ASR_BATCH", "16") or 16))
def whisper_model():
    from faster_whisper import WhisperModel; return WhisperModel("large-v3", device="cuda", compute_type="float16")

if sys.argv[1] == "prep":
    sys.path.insert(0, str(Path(__file__).parent)); import prep
    (jd / "progress.json").write_text(json.dumps({"state": "running", "done": 0, "total": 1, "cur": "подготовка"}), encoding="utf-8")
    prep.run(job, say, lambda k, n, cur="": (jd / "progress.json").write_text(json.dumps({"state": "running", "done": k, "total": n, "cur": cur, "per_item": None, "eta_min": None}), encoding="utf-8"))
    (jd / "progress.json").write_text(json.dumps({"state": "done", "done": 1, "total": 1}), encoding="utf-8"); say("JOBDONE")

elif sys.argv[1] == "gen":
    import numpy as np, soundfile as sf
    cfg = job["model"]; a = cfg["adapter"]; say(f"загрузка модели {cfg['name']} …"); synth = (adapter_script if a.startswith("script:") else ADAPTERS[a])(cfg)
    wh = None; T0 = time.time(); streak = 0
    N = len(job["items"]); exited = False; rt_cache = {}; (jd / "claims").mkdir(exist_ok=True)
    if IDX == 0: (jd / "base.txt").write_text(str(sum((it["name"], it["take"]) in done_keys_start for it in job["items"])))
    prog("running", 0, N, "загрузка")
    many = getattr(synth, "many", None); B = BSIZE if many else 1
    say(f"режим: {ENGINE}" + (f", пакет {B}" if many else ""))
    def ready_ref(k, it):   # образец: склейка нескольких реплик и его расшифровка
        global wh
        ref = it["ref"]
        if ref.get("wavs"):   # несколько реплик склеиваем в один образец (до 20 с)
            parts, sr0, tot = [], None, 0.0
            for x in ref["wavs"]:
                y, sr = sf.read(x); y = y if y.ndim == 1 else y.mean(1)
                if tot + len(y) / sr > 20 and parts: break
                parts += [y, np.zeros(int(sr * 0.3), dtype=y.dtype)]; tot += len(y) / sr + 0.3; sr0 = sr
            ref["wav"] = str(jd / f"ref_{k}.wav"); sf.write(ref["wav"], np.concatenate(parts), sr0)
        if ref.get("text") is None:   # нет расшифровки образца — слушаем Whisper
            tt = None if ref.get("wavs") else tsv_text(ref)   # расшифровка могла быть сделана заранее (этап «Образцы»)
            if tt: ref["text"] = tt
        if ref.get("text") is None:
            ck = tuple(ref.get("wavs") or [ref["wav"]])   # один образец у многих реплик — слушаем один раз
            if ck not in rt_cache:
                wh = wh or whisper_model(); seg, _ = wh.transcribe(audio16(ref["wav"]), language=ref.get("lang") or None, beam_size=5)
                rt_cache[ck] = " ".join(s.text.strip() for s in seg).strip(); say("  расшифровка образца:", rt_cache[ck])
            ref["text"] = rt_cache[ck]
        return ref
    def save(k, it, w, sr, sec):
        global spent, proc_n
        n, ref = it["name"], it["ref"]
        out = Path(it["out"]); out.parent.mkdir(parents=True, exist_ok=True); sf.write(str(out), np.asarray(w), sr)
        emit(kind="take", ts=round(time.time(), 1), name=n, take=it["take"], file=str(out), dur=round(len(w) / sr, 2), sec=round(sec, 1), model=cfg["id"], params=it["params"], ref=it["ref_desc"], text=it["text"], ref_text=ref["text"], quick=quick_check(w, sr, it["text"]), engine=ENGINE)
        proc_n += 1; say(f"{k}/{N} {n[-34:]} {len(w)/sr:.1f}с за {sec:.1f}с")
    def fail(k, it, e):
        global proc_n, streak
        proc_n += 1; emit(kind="error", name=it["name"], error=str(e)[:200]); say(f"{k}/{N} {it['name']} ОШИБКА {e}"); streak += 1
        if streak >= 10:   # одна и та же поломка на каждой реплике — не жжём очередь
            say("ОСТАНОВЛЕНО: 10 ошибок подряд, см. текст выше"); import traceback; say(traceback.format_exc()[-1500:]); prog("error", proc_n, N, it["name"]); sys.exit(2)
    def run(batch):   # batch: [(k, it)] с готовыми образцами
        global spent, streak
        ts = time.time()
        groups = {}
        for k, it in batch: groups.setdefault(json.dumps(it["params"], sort_keys=True), []).append((k, it))
        for g in groups.values():
            t0 = time.time()
            try:
                if len(g) > 1: outs = many([(it["text"], it["ref"]["wav"], it["ref"]["text"], it["params"]) for _, it in g])
                else: outs = [synth(g[0][1]["text"], g[0][1]["ref"]["wav"], g[0][1]["ref"]["text"], g[0][1]["params"])]
                sec = (time.time() - t0) / len(g)
                for (k, it), (w, sr) in zip(g, outs): save(k, it, w, sr, sec)
                streak = 0
            except Exception as e:
                if len(g) > 1:   # пакет упал (память и т. п.) — повторяем по одной
                    say(f"пакет из {len(g)} упал ({e}), повторяю по одной")
                    for k, it in g:
                        t1 = time.time()
                        try: w, sr = synth(it["text"], it["ref"]["wav"], it["ref"]["text"], it["params"]); save(k, it, w, sr, time.time() - t1); streak = 0
                        except Exception as e2: fail(k, it, e2)
                else: fail(g[0][0], g[0][1], e)
        spent += time.time() - ts; prog("running", proc_n, N, batch[-1][1]["name"]); vram_guard()
    def vram_guard():   # Windows не даёт OOM, а докладывает ОЗУ вместо видеопамяти (sysmem fallback) — генерация падает в 5–10 раз; кэш аллокатора разрастается после длинных реплик
        tc = sys.modules.get("torch")
        if not tc or not tc.cuda.is_available(): return
        res, tot = tc.cuda.memory_reserved(), tc.cuda.mem_get_info()[1]
        if res > tot - (2 << 30):
            tc.cuda.empty_cache(); say(f"видеопамять: кэш {res >> 20} МБ из {tot >> 20} — сброшен до {tc.cuda.memory_reserved() >> 20} МБ")
    batch = []
    for k, it in enumerate(job["items"], 1):
        n = it["name"]
        if (n, it["take"]) in done_keys_start: continue
        if IDX >= workers_n(): exited = True; say("поток остановлен (уменьшено число потоков)"); break
        try: os.close(os.open(jd / "claims" / str(k), os.O_CREAT | os.O_EXCL | os.O_WRONLY))   # реплику уже взял другой поток
        except FileExistsError: continue
        gate(proc_n + 1, N, n); prog("running", proc_n, N, n)
        try: ready_ref(k, it); batch.append((k, it))
        except Exception as e: fail(k, it, e)
        if len(batch) >= B: run(batch); batch = []
    if batch: run(batch)
    if exited: prog("exited", proc_n, N)
    if not exited: prog("done", proc_n, N); say("JOBDONE")

elif sys.argv[1] == "asr":   # заранее расшифровать образцы (Whisper), чтобы генерация не грузила Whisper и не слушала образец каждый раз
    sys.path.insert(0, str(Path(__file__).parent)); from asr_batch import ASR
    M = ASR(ASR_MODEL); say(f"Whisper {ASR_MODEL}, движок {M.backend}, пакет {ASR_BATCH}")
    N = len(job["items"]); t0 = time.time(); out = Path(job["items"][0]["out"]) if N else None
    done = {l.split("\t")[0] for l in out.read_text(encoding="utf-8").splitlines()} if out and out.exists() else set()
    todo = [it for it in job["items"] if it["name"] not in done]; base = N - len(todo); k = 0
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for b in range(0, len(todo), ASR_BATCH * 8):   # кусок на 8 пакетов: сортировка по длине внутри куска, запись по мере готовности
            part = todo[b:b + ASR_BATCH * 8]; gate(base + k + 1, N, part[0]["name"])
            groups = {}
            for it in part: groups.setdefault(it.get("lang") or None, []).append(it)
            for lg, its in groups.items():
                try: T = M.transcribe([audio16(it["wav"]) for it in its], lang=lg, beam=1, batch=ASR_BATCH)   # жадный разбор — образцы короткие
                except Exception as e: say("пакет: ОШИБКА", e); T = [""] * len(its)
                for it, t in zip(its, T):
                    t = t.replace("\t", " ").replace("\n", " ")
                    if t: f.write(f"{it['name']}\t{t}\n")
            f.flush(); k += len(part); spent = time.time() - t0; proc_n = k; prog("running", base + k, N, part[-1]["name"]); say(f"{base + k}/{N} {part[-1]['name'][-30:]}")
    prog("done", N, N); say("JOBDONE")

elif sys.argv[1] == "score":
    import numpy as np, soundfile as sf
    sys.path.insert(0, str(Path(__file__).parent)); from asr_batch import ASR
    M = ASR(ASR_MODEL); S = None; say(f"Whisper {ASR_MODEL}, движок {M.backend}, пакет {ASR_BATCH}")
    try:
        import onnxruntime, torch, torchaudio, torchaudio.compliance.kaldi as kaldi
        S = onnxruntime.InferenceSession(job["campplus"], providers=["CPUExecutionProvider"])
    except Exception as e: say("тембр недоступен:", e)
    def emb(p):
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)   # не torchaudio.load: в torchaudio 2.9+ он требует torchcodec
        x = torchaudio.functional.resample(torch.from_numpy(y.mean(1)).unsqueeze(0), sr, 16000)
        f = kaldi.fbank(x, num_mel_bins=80, dither=0, sample_frequency=16000); f = f - f.mean(0, keepdim=True)
        e = S.run(None, {S.get_inputs()[0].name: f.unsqueeze(0).numpy()})[0].flatten(); return e / np.linalg.norm(e)
    HALL = re.compile(r"субтитр|dimatorzok|продолжение следует|ご視聴|ありがとうございました|字幕|sous-titr|suscr|thanks for watching", re.I)
    norm = lambda s: re.sub(r"[^\w ]", "", s.lower().replace("ё", "е")).split()
    def wer(a, b):
        d = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            p = d[:]; d[0] = i
            for j, y in enumerate(b, 1): d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (x != y))
        return d[-1] / max(1, len(a))
    N = len(job["items"]); todo = [(k, it) for k, it in enumerate(job["items"], 1) if (it["name"], it["take"]) not in done_keys_start]
    for bi in range(0, len(todo), ASR_BATCH):   # Whisper пакетом, тембр и метрики по одной
        part = todo[bi:bi + ASR_BATCH]; gate(part[0][0], N, part[0][1]["name"]); prog("running", part[0][0] - 1, N, part[0][1]["name"]); t0 = time.time()
        H = M.transcribe([audio16(it["file"]) for _, it in part], lang="ru", beam=5, batch=ASR_BATCH); share = (time.time() - t0) / len(part)
        for (k, it), h in zip(part, H):
            ts = time.time() - share
            a, b = norm(it["text"]), norm(h); e = wer(a, b); tail = 0
            for x in reversed(a):
                if x in b: break
                tail += 1
            d = sf.info(it["file"]).duration; r = d / it["dub_dur"] if it.get("dub_dur") else 1.0
            cs = float(emb(it["file"]) @ emb(it["ref_wav"])) if S and it.get("ref_wav") else None
            sc = e + 0.3 * tail + 0.8 * max(0, abs(np.log(r)) - 0.35) + (1.5 * max(0, 0.6 - cs) if cs is not None else 0)
            emit(kind="score", name=it["name"], take=it["take"], score=round(sc, 2), wer=round(e, 2), tail=tail, dur_ratio=round(r, 2), cos=None if cs is None else round(cs, 2), heard=h, hallucination=bool(HALL.search(h)))
            spent += time.time() - ts; proc_n += 1; prog("running", k, N, it["name"])
            say(f"{k}/{N} {it['name'][-34:]} score={sc:.2f} wer={e:.2f} tail={tail} dur={r:.2f} cos={cs}")
    prog("done", N, N); say("JOBDONE")
