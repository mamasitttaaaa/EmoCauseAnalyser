import ssl
import certifi
import torch
import shutil
import os
import numpy as np
import soundfile as sf
import requests
import re
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from resemblyzer import VoiceEncoder, preprocess_wav, sampling_rate
from silero_vad import load_silero_vad, get_speech_timestamps
from sklearn.metrics.pairwise import cosine_similarity
from aniemore.recognizers.multimodal import VoiceTextRecognizer
from aniemore.models import HuggingFaceModel
from pyannote.audio import Pipeline
from huggingface_hub import login
import torchaudio
import whisper
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydub import AudioSegment, effects

from uuid import uuid4
from typing import Dict
import asyncio

task_store: Dict[str, dict] = {}  # Временное хранилище

def normalize_audio(input_path, output_path):
    """
    Нормализация аудиофайла с помощью метода peak normalization (пиковое выравнивание громкости).
    Используется для приведения уровня громкости к стандартному уровню без изменения соотношения амплитуд.
    Результат сохраняется в формате WAV.
    """
    audio = AudioSegment.from_file(input_path)
    normalized = effects.normalize(audio)
    normalized.export(output_path, format="wav")

def extract_emotions_causes(text: str) -> list:
    """
    Извлечение пар "эмоция — причина" из текста, с учётом форматирования (нумерация, тире и маркеры).
    Поддерживает гибкое распознавание списков, даже если структура слегка нарушена.
    Фильтрует заголовки и пустые строки. Используется регулярное выражение для поиска шаблона.
    """
    print(f"Before: {text}")
    lines = text.splitlines()
    results = []

    # Паттерн для новой строки с маркером начала (нумерация или "-")
    item_start = re.compile(r"^\s*([-–—•]|\d+\.)\s*")

    # Паттерн для пары "эмоция - причина"
    pattern = re.compile(r"^\s*(?:[-–—•]|\d+\.)?\s*([\wА-Яа-яёЁ\s]+?)\s*[-–—:]\s*(.+?)\s*$")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Пропуск строки-заголовка
        if re.match(r"^.{0,40}:\s*$", line) and "-" not in line and "—" not in line:
            continue
        # Проверка, начинается ли строка с маркера
        if item_start.match(line):
            match = pattern.match(line)
            if match:
                emotion = match.group(1).strip().capitalize()
                cause = match.group(2).strip().rstrip(".")
                results.append(f"{emotion} - {cause}")
        else:
            # Попытка извлечь даже из обычной строки
            match = pattern.match(line)
            if match:
                emotion = match.group(1).strip().capitalize()
                cause = match.group(2).strip().rstrip(".")
                results.append(f"{emotion} - {cause}")
    
    print(f"After: {results}")
    return results

def correct_text_yandex(text: str) -> str:
    """
    Орфографическая коррекция текста с помощью сервиса Яндекс.Спеллер.
    Возвращает исправленный текст на русском языке.
    При ошибке сервиса — возвращает оригинальный текст без изменений.
    """
    try:
        resp = requests.get(
            "https://speller.yandex.net/services/spellservice.json/checkText",
            params={"text": text, "lang": "ru"}, timeout=5
        )
        for err in resp.json():
            if err.get("s"):
                text = text.replace(err["word"], err["s"][0])
    except Exception:
        pass
    return text

def clear_upload_dir():
    """
    Очистка временной папки UPLOAD_DIR от всех файлов.
    Используется после завершения обработки или при ошибке.
    Безопасный обход ошибок удаления.
    """
    for fname in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, fname)
        try:
            os.remove(path)
        except Exception as e:
            print(f"Не удалось удалить {path}: {e}")

def analyze_text_with_gigachat(full_text: str, emotions: list) -> str:
    """
    Генерация анализа от лица психолога с помощью модели GigaChat.
    - Выполняется авторизация по OAuth и отправка запроса с диалогом пользователя.
    - Эмоции передаются как подсказка, но не определяют результат.
    - Модель формирует пары "эмоция — причина", которые далее парсятся.
    Возвращает готовый список таких пар или оригинальный текст, если пар не извлечено.
    """
    auth_key = "ПЕРСОНАЛЬНЫЙ_КЛЮЧ_ДОСТУПА"
    oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    oauth_headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': '67e16015-f7f5-4617-b9fd-8e19a82a8db6',
        'Authorization': f'Basic {auth_key}'
    }
    oauth_payload = {'scope': 'GIGACHAT_API_PERS'}

    try:
        oauth_response = requests.post(oauth_url, headers=oauth_headers, data=oauth_payload, verify=False)
        access_token = oauth_response.json().get("access_token")
        if not access_token:
            return "Ошибка получения токена"
    except Exception as e:
        return f"Ошибка при обращении к OAuth-серверу: {e}"

    gigachat_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    giga_headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    emotions_text = ", ".join(emotions)
    prompt = (
        "Ты — практикующий психолог с опытом более 10 лет.\n"
        "Твоя задача — проанализировать следующий диалог между Пользователем и Собеседниками.\n\n"
        f"Учитывай, что по результатам предварительного мультимодального анализа у Пользователя были выявлены следующие эмоции: {emotions_text}.\n"
        "Эти эмоции могут быть неполными или ошибочными, их можно использовать как подсказку, но не основываться на них без подтверждения в тексте.\n\n"
        "Твоя задача — проанализировать исключительно высказывания Пользователя и определить:\n"
        "1. Какие эмоции он проявляет;\n"
        "2. Какие причины этих эмоций указаны в его словах.\n\n"
        "Важно: указывай только те эмоции, для которых в тексте есть явное или очевидное подтверждение. Не делай предположений, если причина не названа прямо или недвусмысленно.\n"
        "Если причина не указана — не добавляй её в ответ.\n\n"
        "Выведи результат в следующем формате: пары эмоция — причина, разделённые запятыми, в одной строке.\n"
        f"Текст диалога:\n{full_text}"
    )

    giga_payload = {
        "model": "GigaChat-Max",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.9,
        "n": 1,
        "stream": False,
        "max_tokens": 512,
        "repetition_penalty": 1,
        "update_interval": 0
    }

    try:
        response = requests.post(gigachat_url, headers=giga_headers, json=giga_payload, verify=False)
        content = response.json()["choices"][0]["message"]["content"]
        parsed = extract_emotions_causes(content)
        return "\n".join(parsed) if parsed else content
    except Exception as e:
        return f"Ошибка при запросе к GigaChat: {e}"
    
def whisper_transcribe(audio_array: np.ndarray, sr: int = 16000) -> str:
    """
    Распознавание речи на русском языке с помощью модели Whisper (от OpenAI).
    Если частота дискретизации не равна 16000 Гц — выполняется ресемплирование.
    Возвращает расшифрованный текст.
    """
    if sr != 16000:
        audio_array = torchaudio.functional.resample(torch.tensor(audio_array), sr, 16000).numpy()
    result = asr_model.transcribe(audio_array.astype(np.float32), language="ru")
    return result['text']

EMOTION_TRANSLATIONS = {
    "neutral": "нейтральное",
    "anger": "гнев",
    "enthusiasm": "энтузиазм",
    "fear": "страх",
    "sadness": "грусть",
    "happiness": "счастье",
    "disgust": "отвращение"
}


ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
login(token=os.getenv("HUGGINGFACE_TOKEN"))

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diar_pipeline = pipeline.instantiate({
    "segmentation": {"threshold": 0.5},
    "clustering": {"method": "centroid", "threshold": 0.65}
})

device = "cuda:0" if torch.cuda.is_available() else "cpu"
emotion_model = VoiceTextRecognizer(model=HuggingFaceModel.MultiModal.WavLMBertFusion, device=device)
asr_model = whisper.load_model("medium")
encoder = VoiceEncoder()
vad_model = load_silero_vad()

@app.post("/submit_audio")
async def submit_audio(file: UploadFile = File(...), owner_file: UploadFile = File(...)):
    """
    Endpoint загрузки аудио:
    - Пользователь загружает два файла: свою речь и эталон владельца.
    - Генерируется task_id и создаётся задача в task_store.
    - Файлы сохраняются и обрабатываются асинхронно в фоне.
    Возвращается task_id для отслеживания статуса.
    """
    task_id = str(uuid4())
    task_store[task_id] = {"status": "processing", "result": None}

    user_path = os.path.join(UPLOAD_DIR, f"user_{task_id}.wav")
    owner_path = os.path.join(UPLOAD_DIR, f"owner_{task_id}.wav")

    with open(user_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    with open(owner_path, "wb") as f:
        shutil.copyfileobj(owner_file.file, f)

    # Запуск фоновой задачи
    asyncio.create_task(process_audio(task_id, user_path, owner_path))

    return {"task_id": task_id}


async def process_audio(task_id: str, user_path: str, owner_path: str):
    """
    Фоновая задача обработки аудиофайлов:
    - Запускает синхронную функцию обработки в threadpool.
    - Обновляет статус задачи и результат.
    - В конце очищает временные файлы, независимо от исхода.
    """
    try:
        result = await run_in_threadpool(lambda: sync_processing(user_path, owner_path))
        task_store[task_id]["status"] = "done"
        task_store[task_id]["result"] = result
    except Exception as e:
        task_store[task_id]["status"] = "error"
        task_store[task_id]["result"] = str(e)
    finally:
        try:
            clear_upload_dir()
        except Exception as e:
            print(f"Ошибка при удалении файлов: {e}")

@app.get("/check_status/{task_id}")
def get_status(task_id: str):
    """
    Проверка текущего статуса задачи по task_id.
    Возможные значения: "processing", "done", "error".
    """
    task = task_store.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return {"status": task["status"]}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    """
    Получение результата завершённой задачи:
    - Если задача не найдена или ещё не завершена — возвращает соответствующий статус.
    - Если завершена — возвращает результат и удаляет задачу из хранилища.
    """
    task = task_store.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    if task["status"] != "done":
        return JSONResponse(status_code=202, content={"message": "Still processing"})
    
    result = task["result"]
    del task_store[task_id]

    return {"result": result}

def sync_processing(user_path: str, owner_path: str) -> dict:
    """
    Основная синхронная обработка аудио:
    - Нормализует пользовательское аудио (peak normalization).
    - Выполняет диаризацию и идентификацию голосов с помощью pyannote и Resemblyzer.
    - Отбирает реплики владельца, выполняет VAD, ASR, орфокоррекцию, анализ эмоций.
    - Все эмоции переводятся на русский и отправляются в GigaChat для анализа причин.
    Возвращает JSON с анализом GPT.
    """

    normalized_user_path = user_path.replace(".wav", "_norm.wav")
    normalize_audio(user_path, normalized_user_path)
    user_path = normalized_user_path

    owner_emb = encoder.embed_utterance(preprocess_wav(owner_path))
    print("start diare")
    diarization = diar_pipeline(user_path)
    print("stop diare")
    wav, sr = sf.read(user_path)
    threshold = 0.65

    dialogue_turns = []
    all_emotions = set()
    segment_idx = 0
    owner_present = False

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start, end = segment.start, segment.end
        audio_segment = wav[int(start * sampling_rate):int(end * sampling_rate)]
        seg_emb = encoder.embed_utterance(audio_segment)
        sim = cosine_similarity([owner_emb], [seg_emb])[0][0]
        if sim >= threshold:
            owner_present = True
            break

    if not owner_present:
        try:
            clear_upload_dir()
        except Exception as e:
            print(f"Ошибка при удалении файлов: {e}")
        return JSONResponse(content={"gpt_analysis": ""})

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start, end = segment.start, segment.end
        audio_segment = wav[int(start * sampling_rate):int(end * sampling_rate)]

        try:
            seg_emb = encoder.embed_utterance(audio_segment)
            sim = cosine_similarity([owner_emb], [seg_emb])[0][0]
            text = whisper_transcribe(audio_segment)
            cleaned = correct_text_yandex(text)

            if sim >= threshold:
                speaker_name = "Пользователь"
                if cleaned.strip():
                    dialogue_turns.append(f"{speaker_name}: {cleaned.strip()}")

                vad_segments = get_speech_timestamps(audio_segment, vad_model, sampling_rate=sampling_rate)
                for j, vad in enumerate(vad_segments):
                    frag = audio_segment[vad['start']:vad['end']]
                    # неизбежен момент сохраниения файла фрагмента в системе в связи с требованием формата аргументов модели распознавания эмоций (необходим путь к файлу)
                    frag_path = os.path.join(UPLOAD_DIR, f"frag_{segment_idx}_{j}.wav")
                    sf.write(frag_path, frag, samplerate=sr)
                    try:
                        frag_text = correct_text_yandex(whisper_transcribe(frag))
                        emotion = emotion_model.recognize((frag_path, frag_text), return_single_label=True)
                        all_emotions.add(emotion)
                        os.remove(frag_path)
                    except Exception as e:
                        print(f"Ошибка эмоции: {e}")
            else:
                speaker_name = f"Собеседник {speaker[-2:]}"
                if cleaned.strip():
                    dialogue_turns.append(f"{speaker_name}: {cleaned.strip()}")

        except Exception as e:
            print(f"Ошибка обработки сегмента {segment_idx}: {e}")
        finally:
            segment_idx += 1

    translated_emotions = {
        EMOTION_TRANSLATIONS.get(emotion, emotion)
        for emotion in all_emotions
    }

    dialogue_text = "\n".join(dialogue_turns)
    print(f"Emotional set: {translated_emotions}")
    gpt_analysis = analyze_text_with_gigachat(dialogue_text, list(set(translated_emotions)))

    return JSONResponse(content={"gpt_analysis": gpt_analysis})

@app.get("/ping")
async def ping():
    """
    Простая проверка доступности сервера.
    """
    return JSONResponse(content={"status": "ok"}, status_code=200)