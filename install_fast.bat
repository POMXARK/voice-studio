@echo off
rem Быстрые режимы генерации: faster-qwen3-tts (CUDA graphs) в отдельном venv C:\voice\fq5.
rem Форк POMXARK/faster-qwen3-tts, ветка batched = апстрим 0.5.4 + PR #150 (пакетный voice clone, в апстрим не слит).
rem Версии: transformers 5.15.1 (как в pyproject 0.5.4) + qwen-tts-hf. 5.12 — речь без EOS, 5.22 — ошибка upsample; 0.5.x с transformers 4.57 — ошибка StaticLayer.
if not exist C:\voice\fq5\Scripts\python.exe C:\Python314\python.exe -m venv C:\voice\fq5
C:\voice\fq5\Scripts\python.exe -m pip install -U pip
C:\voice\fq5\Scripts\python.exe -m pip install torch==2.11.0 torchaudio --index-url https://download.pytorch.org/whl/cu128
C:\voice\fq5\Scripts\python.exe -m pip install "transformers==5.15.1" "qwen-tts-hf>=0.1.1.post1,<0.2" soundfile numpy psutil onnxruntime
C:\voice\fq5\Scripts\python.exe -m pip install --no-deps "git+https://github.com/POMXARK/faster-qwen3-tts@batched"
C:\voice\fq5\Scripts\python.exe -c "import torch,faster_qwen3_tts.batched;print('FQOK',torch.__version__,torch.cuda.is_available())"
