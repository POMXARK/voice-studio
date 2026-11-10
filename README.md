# VoiceStudio

**Local AI dubbing studio for games: voice-clone every line of dialogue, check it automatically, fix what's broken, ship it into the game.**
**Under one second per line** on a single consumer GPU (RTX 5080): Qwen3-TTS with CUDA graphs and batched decoding, Whisper with batched windows.

[English](#english) · [Русский](#русский)

---

## English

VoiceStudio is a self-hosted web UI (pure Python stdlib server + one HTML page) for producing a full voice-over of a game or any project with tens of thousands of lines. It was built to dub an open-world RPG into Russian (62 000+ lines), but nothing in it is tied to one game: the source adapter takes any folder of recordings plus a `name<TAB>text` subtitle file.

### What it does
- **Voice cloning per line.** The original actor's line (any language) is the voice sample; the model speaks the translated text in that voice. Samples can also be your own wav, several original lines glued together, or another author's dub.
- **Staged pipeline with an auto mode.** ① generate → ② check (Whisper WER, missing tail words, duration vs. reference, timbre similarity via CAM++, hallucination markers) → ③ fix (new takes with other seeds/temperature) → ④ pick the best take or fall back to another author's dub. The auto pipeline walks the stages on its own.
- **Review UI.** Original / mine / other author side by side, hotkeys, flags (bad / regenerate / use dub), editable text and parameters per line, several takes per line.
- **Cheap distorted-take detector.** Duration-to-text ratio plus clipping/silence/noise checks in milliseconds — marks runaway generations (no EOS) and cut-offs before any Whisper pass.
- **"Something sounded wrong" hotkey.** Press it in-game, and lines played in the last 5 minutes go to review.
- **Install into the game** with auto-install mode; background job queue with pause/resume, multi-worker generation that scales with free VRAM.
- **Swappable models.** Qwen3-TTS 1.7B / 0.6B Base, CosyVoice 3, or your own adapter script (`script:path.py`).

### Speed (RTX 5080, Qwen3-TTS 1.7B, same 16 lines)
| generation mode | time | per line |
|---|---|---|
| normal (stock qwen-tts, one by one) | 107.7 s | 6.7 s |
| batch (stock, 8 lines per call) | 35.2 s | 2.2 s |
| fast (CUDA graphs, [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)) | 20.3 s | 1.3 s |
| **fast batch** (CUDA graphs + batched decode) | **10.4 s** | **0.65 s**, ≈0.3 s after warm-up |

Modes switch from the UI; running workers restart in the new mode, nothing generated is lost.

Whisper (160 short Japanese samples): faster-whisper / CTranslate2 293 ms per line → **84 ms** with `asr_batch.py` (transformers + torch SDPA fp16, windows of *different* lines batched together, sorted by length). On Blackwell GPUs CTranslate2 4.8 lacks fast kernels — the torch path is 4× faster in the encoder.

### Requirements
- Windows 10/11 (paths and the hotkey script are Windows-first; the server itself is cross-platform Python), NVIDIA GPU with 12–16 GB VRAM.
- Python 3.10+ with `qwen-tts` for the normal modes; `install_fast.bat` creates a separate venv for the fast modes and batched Whisper (torch 2.11 cu128, transformers 5.15.1, qwen-tts-hf, faster-qwen3-tts from the [`batched` branch of my fork](https://github.com/POMXARK/faster-qwen3-tts/tree/batched) — upstream 0.5.4 + unmerged PR #150).
- Optional: CosyVoice 3 (`campplus.onnx` for timbre score), ffmpeg, vgmstream for `.wem`.

### Quick start
```bat
git clone https://github.com/POMXARK/voice-studio
cd voice-studio
copy config.example.json config.json   & rem set work / game_dir / pythons
install_fast.bat                         & rem optional: fast modes + batched Whisper
python server.py                         & rem open http://127.0.0.1:8777
```
In the UI: **Prepare project** (pick a source adapter and folders) → **Samples** (pre-transcribe voice samples) → **Generate** → **Check** → **Fix** → **Pick** → **Install**. Or just turn on the auto pipeline.

Detailed notes (Russian): [docs/NOTES.ru.md](docs/NOTES.ru.md) — data layout, every feature, benchmarks, dead ends with library versions.

### Legal
The repository contains code only — no game audio, texts or models. Use it with games you own and respect their EULA and the voice actors' rights; generated voices are for personal/mod use.

---

## Русский

VoiceStudio — локальная веб-студия (сервер на чистой стандартной библиотеке Python + одна HTML-страница) для полной озвучки игры или любого проекта на десятки тысяч реплик. Сделана для русской озвучки RPG с открытым миром (62 000+ реплик), но к игре не привязана: источник — любая папка с записями и файл субтитров `имя<TAB>текст`.

### Что умеет
- **Клонирование голоса на каждую реплику.** Образец — реплика оригинального актёра (на любом языке), модель говорит переведённый текст его голосом. Образцом может быть и свой wav, склейка нескольких реплик, или озвучка другого автора.
- **Этапы и автоконвейер.** ① генерация → ② проверка (WER по Whisper, потерянные слова в конце, длительность к эталону, сходство тембра CAM++, метки галлюцинаций) → ③ исправление (новые дубли с другими seed/temperature) → ④ выбор лучшего или замена дублем другого автора. Автоконвейер идёт по этапам сам.
- **Прослушивание и правка.** Оригинал / моя / другого автора рядом, горячие клавиши, пометки («плохая», «на перегенерацию», «заменить дублем»), свой текст и параметры на реплику, несколько дублей.
- **Быстрый детектор искажённых дублей** за миллисекунды: длительность к тексту + клиппинг/тишина/шум — ловит «бесконечную» речь без EOS и обрывы ещё до Whisper.
- **Горячая клавиша «странно прозвучало»** прямо в игре: реплики за последние 5 минут уходят на проверку.
- **Установка в игру** (и автоустановка), очередь задач с паузой, несколько потоков генерации по свободной видеопамяти.
- **Модели сменные:** Qwen3-TTS 1.7B / 0.6B Base, CosyVoice 3 или свой адаптер (`script:путь.py`).

### Скорость (RTX 5080, Qwen3-TTS 1.7B, одни и те же 16 реплик)
| режим генерации | время | на реплику |
|---|---|---|
| обычный (штатный qwen-tts, по одной) | 107,7 с | 6,7 с |
| пакет (штатный, 8 реплик за вызов) | 35,2 с | 2,2 с |
| быстрый (CUDA graphs, [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)) | 20,3 с | 1,3 с |
| **быстрый пакет** (CUDA graphs + пакетный декодер) | **10,4 с** | **0,65 с**, после прогрева ≈0,3 с |

Режим переключается в интерфейсе; идущие потоки перезапускаются в новом режиме, готовое не теряется.

Whisper (160 коротких японских образцов): faster-whisper / CTranslate2 — 293 мс на реплику, `asr_batch.py` — **84 мс** (transformers + torch SDPA fp16, в один пакет собираются окна *разных* реплик, отсортированных по длине). На видеокартах Blackwell у CTranslate2 4.8 нет быстрых ядер — энкодер на torch в 4 раза быстрее.

### Что нужно
- Windows 10/11 (пути и скрипт горячей клавиши — под Windows; сам сервер — кроссплатформенный Python), NVIDIA с 12–16 ГБ видеопамяти.
- Python 3.10+ с `qwen-tts` для обычных режимов; `install_fast.bat` ставит отдельный venv для быстрых режимов и пакетного Whisper (torch 2.11 cu128, transformers 5.15.1, qwen-tts-hf, faster-qwen3-tts из [ветки `batched` моего форка](https://github.com/POMXARK/faster-qwen3-tts/tree/batched) — апстрим 0.5.4 + не слитый PR #150).
- По желанию: CosyVoice 3 (`campplus.onnx` для оценки тембра), ffmpeg, vgmstream для `.wem`.

### Запуск
```bat
git clone https://github.com/POMXARK/voice-studio
cd voice-studio
copy config.example.json config.json   & rem папки work / game_dir, питоны
install_fast.bat                         & rem по желанию: быстрые режимы и пакетный Whisper
python server.py                         & rem открыть http://127.0.0.1:8777
```
В интерфейсе: **Подготовка проекта** (адаптер источника и папки) → **Образцы** (заранее расшифровать) → **Генерация** → **Проверка** → **Исправление** → **Выбор** → **Установка**. Или включить автоконвейер.

Подробно: [docs/NOTES.ru.md](docs/NOTES.ru.md) — устройство папки проекта, все функции, замеры, тупики с версиями библиотек.

### Правовое
В репозитории только код — без звука, текстов и моделей игр. Используйте с играми, которые у вас есть, соблюдайте их лицензию и права актёров; сгенерированные голоса — для личного использования и модов.

---

### Credits
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0), [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) by Andres Marafioti (MIT), batched voice clone from PR #150 by yu45020 (MIT).
- [Whisper](https://github.com/openai/whisper) (MIT) via [transformers](https://github.com/huggingface/transformers) / [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) and CAM++ speaker embedding (Apache-2.0).

License: MIT.
