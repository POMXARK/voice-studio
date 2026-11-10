"""Пакетная расшифровка коротких записей Whisper.

Реплики по 1–10 с, их десятки тысяч. Энкодер Whisper всегда считает 30-секундное окно, поэтому выгодно
собрать в один пакет окна РАЗНЫХ записей: один проход энкодера на B реплик и один пакетный декодер.
Записи сортируются по длине — в пакете похожие по длине тексты, декодер меньше ждёт самый длинный.

Движки (STUDIO_ASR или аргумент backend):
  hf  — transformers + torch (SDPA, fp16). На RTX 5080 (Blackwell) энкодер 34 мс/реплику: torch 2.11 cu128 знает sm_120.
  ct2 — faster-whisper / CTranslate2 4.8: на Blackwell энкодер 148 мс/реплику (нет быстрых ядер), int8 падает в cuBLAS.
Записи длиннее 30 с идут по одной (длинная форма)."""
import os

MODELS = {"large-v3": "openai/whisper-large-v3", "large-v3-turbo": "openai/whisper-large-v3-turbo"}

class ASR:
    def __init__(self, name="large-v3", backend=None):
        self.backend = backend or os.environ.get("STUDIO_ASR", "hf")
        if self.backend == "hf":
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            mid = MODELS.get(name, name); self.torch = torch
            self.P = WhisperProcessor.from_pretrained(mid)
            self.M = WhisperForConditionalGeneration.from_pretrained(mid, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        else:
            from faster_whisper import WhisperModel
            self.M = WhisperModel(name, device="cuda", compute_type="float16")

    def transcribe(self, audios, lang=None, beam=1, batch=16):
        """audios: float32 16 кГц; lang: код языка или None (определить по каждой записи). -> список строк в том же порядке"""
        out = [""] * len(audios)
        order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
        short = [i for i in order if len(audios[i]) <= 30 * 16000]
        for i in order:
            if len(audios[i]) > 30 * 16000: out[i] = self._long(audios[i], lang, beam)
        for b in range(0, len(short), batch):
            ids = short[b:b + batch]
            for i, t in zip(ids, self._batch([audios[i] for i in ids], lang, beam)): out[i] = t.strip()
        return out

    def _batch(self, xs, lang, beam):
        if self.backend == "hf":
            torch = self.torch
            f = self.P.feature_extractor(xs, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
            with torch.inference_mode():
                ids = self.M.generate(f, language=lang, task="transcribe", num_beams=beam, do_sample=False)
            return self.P.batch_decode(ids, skip_special_tokens=True)
        import numpy as np
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer
        M = self.M; nf = M.feature_extractor.nb_max_frames
        enc = M.encode(np.stack([pad_or_trim(M.feature_extractor(x), nf) for x in xs]).astype("float32"))
        langs = [lang] * len(xs) if lang else [r[0][0][2:-2] for r in M.model.detect_language(enc)]
        toks = [Tokenizer(M.hf_tokenizer, M.model.is_multilingual, task="transcribe", language=lg) for lg in langs]
        res = M.model.generate(enc, [M.get_prompt(tk, [], without_timestamps=True) for tk in toks], beam_size=beam, max_length=224, suppress_blank=True, suppress_tokens=[-1])
        return [tk.decode([t for t in r.sequences_ids[0] if t < tk.eot]) for tk, r in zip(toks, res)]

    def _long(self, x, lang, beam):
        if self.backend == "hf":
            torch = self.torch
            inp = self.P.feature_extractor(x, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True)
            with torch.inference_mode():
                ids = self.M.generate(inp.input_features.to("cuda", torch.float16), attention_mask=inp.attention_mask.to("cuda"), language=lang, task="transcribe", num_beams=beam, condition_on_prev_tokens=False)
            return self.P.batch_decode(ids, skip_special_tokens=True)[0].strip()
        seg, _ = self.M.transcribe(x, language=lang, beam_size=beam, without_timestamps=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in seg).strip()
