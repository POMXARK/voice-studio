"""Подготовка проекта студии: достать голосовые записи и субтитры игры/проекта, положить в папку проекта.
Универсально: источник — адаптер из ADAPTERS (папка с аудио+текстами, Crimson Desert PAZ, свой скрипт `script:путь.py` с функцией collect(params, say)).
Результат в <work>: texts.tsv (имя<TAB>текст), <язык>_wav/<имя>.wav (оригинал), dub_wav/ + dub_wem/ + texts_dub.tsv (озвучка другого автора, если есть)."""
import os, re, shutil, struct, subprocess, sys
from pathlib import Path

# ---------- схемы параметров (по ним интерфейс рисует форму) ----------
ADAPTERS = {
    "folder": {"name": "Папка с записями (любая игра/проект)", "params": [
        {"key": "audio_dir", "label": "Папка с оригинальными записями (<имя>.wav/.wem/.ogg/.mp3/.flac, можно во вложенных)", "default": ""},
        {"key": "lang", "label": "Код языка оригинала", "default": "en"},
        {"key": "texts", "label": "Файл субтитров: строки «имя<TAB>текст» (имя = имя файла без расширения)", "default": ""},
        {"key": "dub_dir", "label": "Папка с русской озвучкой другого автора (необязательно)", "default": ""},
        {"key": "filter", "label": "Только имена, содержащие (необязательно)", "default": ""},
        {"key": "limit", "label": "Не больше реплик (0 = все)", "default": "0"}]},
    "crimson_paz": {"name": "Crimson Desert (PAZ-архивы игры)", "params": [
        {"key": "game", "label": "Папка игры", "default": r"C:\Games\Crimson Desert"},
        {"key": "lang", "label": "Языки оригинальной озвучки через пробел (ko en zh fr de ja pt es) — образцом можно брать любой", "default": "ja"},
        {"key": "paz_dir", "label": "Папка с pamt.py и paz_crypto.py", "default": r"C:\voice\paz"},
        {"key": "dub_dir", "label": "Папка с русской озвучкой другого автора (.wem, необязательно)", "default": ""},
        {"key": "filter", "label": "Только имена, содержащие (необязательно)", "default": ""},
        {"key": "limit", "label": "Не больше реплик (0 = все)", "default": "0"}]},
}
AUDIO = (".wav", ".wem", ".ogg", ".mp3", ".flac")
CRIMSON_LANGS = {"ko": ("0005", "korean"), "en": ("0006", "english(us)"), "zh": ("0035", "chinese(prc)"), "fr": ("0036", "french"), "de": ("0037", "german"), "ja": ("0038", "japanese"), "pt": ("0039", "portuguese(br)"), "es": ("0040", "spanish(es)")}

def read_tsv(p):
    return dict(l.split("\t", 1) for l in Path(p).read_text(encoding="utf-8-sig").splitlines() if "\t" in l)

def scan_audio(d):
    out = {}
    for root, _, fs in os.walk(d):
        for f in fs:
            b, e = os.path.splitext(f)
            if e.lower() in AUDIO and (b not in out or e.lower() == ".wav"): out[b] = os.path.join(root, f)
    return out

def collect_folder(P, work, say, prog):
    voices = scan_audio(P["audio_dir"]); texts = read_tsv(P["texts"]); dub = scan_audio(P["dub_dir"]) if P.get("dub_dir") else {}
    say(f"записей {len(voices)}, субтитров {len(texts)}, озвучки другого автора {len(dub)}")
    return [{"name": n, "text": texts[n], "voices": {P.get("lang", "orig"): voices[n]}, "dub": dub.get(n)} for n in sorted(voices) if n in texts]

def collect_crimson(P, work, say, prog):
    sys.path[:0] = [P["paz_dir"], str(Path(P["paz_dir"]).parent)]
    try:
        import pamt, lz4.block; from paz_crypto import decrypt
    except ImportError as e:
        raise SystemExit("Не хватает модуля Python (%s). Установите: %s -m pip install lz4 cryptography; и проверьте папку с pamt.py/paz_crypto.py" % (e.name, sys.executable))
    game = P["game"]; langs = P["lang"].replace(",", " ").split()
    def pairs(d):   # .paloc: [8][длина ключа][ключ][длина текста][текст]
        raw = lz4.block.decompress(d[512:], uncompressed_size=struct.unpack_from("<I", d, 12)[0]); o = 0
        while o + 12 <= len(raw):
            o += 8; kl = struct.unpack_from("<I", raw, o)[0]; o += 4; k = raw[o:o + kl].decode("utf-8", "replace"); o += kl
            tl = struct.unpack_from("<I", raw, o)[0]; o += 4; t = raw[o:o + tl].decode("utf-8", "replace"); o += tl; yield k, t
    subs = {}; G = os.path.join(game, "0022"); say("субтитры (русские) …")
    for e in pamt.parse(os.path.join(G, "0.pamt")):
        path, off, cs = e[0], e[1], e[2]; base = path.rsplit("/", 1)[-1]
        if "binary__/rus/" not in path or base not in ("questdialog", "narrationdialog", "aidialog", "sequencer"): continue
        with open(os.path.join(G, "%d.paz" % e[4]), "rb") as f: f.seek(off); raw = f.read(cs)
        for k, t in pairs(decrypt(raw, base + ".paloc")):
            t = re.sub(r"\{[^{}#]*#([^{}]*)\}", r"\1", t).replace("<br/>", " ").replace("\r", " ").replace("\n", " ").strip()
            if t and "\t" not in k: subs[k] = t
    say(f"субтитров {len(subs)}"); dub = scan_audio(P["dub_dir"]) if P.get("dub_dir") and os.path.isdir(P["dub_dir"]) else {}; flt = P.get("filter", ""); lim = int(P.get("limit") or 0)
    def match(name):   # ключ субтитра — хвост имени реплики без префикса голоса
        p = name.split("_")
        for i in range(1, len(p)):
            k = "_".join(p[i:])
            if k in subs: return subs[k]
    items = {}
    for lg in langs:   # любой набор языков: каждый — своя папка <язык>_wav, образцом можно брать любой
        grp, folder = CRIMSON_LANGS[lg]; G = os.path.join(game, grp); pre = f"sound/windows/{folder}/"; wd = work / f"{lg}_wem"; wd.mkdir(exist_ok=True)
        ents = [x for x in pamt.parse(os.path.join(G, "0.pamt")) if x[0].startswith(pre)]; say(f"{lg}: записей {len(ents)}, озвучки другого автора {len(dub)}"); k = 0
        for x in ents:
            name = x[0][len(pre):]
            if flt and flt not in name: continue
            if name not in items:
                t = match(name)
                if not t or (lim and len(items) >= lim): continue
                items[name] = {"name": name, "text": t, "voices": {}, "dub": dub.get(name)}
            _, off, cs, us, cid, fl, _x = x; f = wd / (name + ".wem")
            if not f.exists():
                with open(os.path.join(G, "%d.paz" % cid), "rb") as fh: fh.seek(off); f.write_bytes(fh.read(cs))
            items[name]["voices"][lg] = str(f); k += 1
            if k % 500 == 0: prog(k, len(ents), "извлечение " + lg); say(f"{lg}: извлечено {k}")
    return list(items.values())

def to_wav(src, dst, tools):
    if os.path.exists(dst): return True
    e = os.path.splitext(src)[1].lower()
    if e == ".wav": shutil.copy(src, dst); return True
    if e == ".wem" and tools.get("vgm") and os.path.exists(tools["vgm"]): subprocess.run([tools["vgm"], "-o", dst, src], capture_output=True)
    elif tools.get("ffmpeg"): subprocess.run([tools["ffmpeg"], "-y", "-loglevel", "error", "-i", src, dst], capture_output=True)
    return os.path.exists(dst)

def run(job, say, prog):
    work = Path(job["work"]); P = job["params"]; tools = job.get("tools", {}); ad = job["adapter"]
    if ad.startswith("script:"):
        import importlib.util; sp = importlib.util.spec_from_file_location("custom_prep", ad[7:]); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); items = m.collect(P, say)
    else: items = {"folder": collect_folder, "crimson_paz": collect_crimson}[ad](P, work, say, prog)
    lim = int(P.get("limit") or 0); flt = P.get("filter", "")
    items = [c for c in items if (not flt or flt in c["name"])]; items = items[:lim] if lim else items
    (work / "dub_wav").mkdir(exist_ok=True); (work / "dub_wem").mkdir(exist_ok=True)
    tx = work / "texts.tsv"; old = read_tsv(tx) if tx.exists() else {}; dubtx = work / "texts_dub.tsv"; olddub = read_tsv(dubtx) if dubtx.exists() else {}
    ok = okd = 0
    for k, c in enumerate(items, 1):
        n = c["name"]; old.setdefault(n, c["text"])
        for lg, v in c["voices"].items():
            (work / f"{lg}_wav").mkdir(exist_ok=True)
            if to_wav(v, str(work / f"{lg}_wav" / f"{n}.wav"), tools): ok += 1
        if c.get("dub"):
            d = work / "dub_wem" / (n + os.path.splitext(c["dub"])[1])
            if not d.exists(): shutil.copy(c["dub"], d)
            if to_wav(c["dub"], str(work / "dub_wav" / f"{n}.wav"), tools): okd += 1; olddub.setdefault(n, c["text"])
        if k % 200 == 0 or k == len(items): prog(k, len(items), n); say(f"{k}/{len(items)} оригинал {ok}, другого автора {okd}")
    tx.write_text("".join(f"{k}\t{v}\n" for k, v in sorted(old.items())), encoding="utf-8", newline="\n")
    dubtx.write_text("".join(f"{k}\t{v}\n" for k, v in sorted(olddub.items())), encoding="utf-8", newline="\n")
    say(f"готово: реплик {len(items)}, оригинал {ok}, другого автора {okd}")
