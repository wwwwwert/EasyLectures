import argparse
import gc
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import torch
import whisperx
from ffmpeg import FFmpeg
from tqdm import tqdm
import re


warnings.filterwarnings('ignore')

HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
FIREWORKS_TOKEN = os.environ.get('FIREWORKS_TOKEN')
PROMPT = """У меня была лекция по курсу "{class_title}". Я сделал транскрибацию по её видеозаписи. Пожалуйста сделай подробный конспект этой лекции в формате md для программы Obsidian. 
Ориентируйся также на следующий формат оформления: 
- Излагай материал последовательно. Делай логические разделы для того, чтобы читатель мог воспринимать информацию структурированно. 
- Оформляй подразделы, последовательности примеров или какие-то наборы кейсов маркированными списками с комментарием к каждому пункту.
- Для формул используй вставки через `$`. 
- Раскрой непонятные определения и приведи примеры кода там, где это уместно.

{transcribation}"""


def beautify_note(note: str) -> str:
    # Удалить ``` в начале и в конце файла, а также ```markdown
    note = re.sub(r'^```(markdown)?\s*', '', note)
    note = re.sub(r'\s*```$', '', note)
    
    # Заменить '\[ ' и ' \]' на '$$'
    note = re.sub(r'\\\[\s*|\s*\\\]', '$$', note)
    
    # Заменить '\( ' и ' \)' на '$'
    note = re.sub(r'\\\(\s*|\s*\\\)', '$', note)

    return note


def get_brief_note(transcribation: str, class_title: str) -> str:
    print('Summarizing...')
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
    "model": "accounts/fireworks/models/deepseek-v3",
    "max_tokens": 16384,
    "top_p": 1,
    "top_k": 40,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "temperature": 0.6,
    "messages": [
            {
            "role": "user",
            "content": PROMPT.format(transcribation=transcribation, class_title=class_title)
            }
        ]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_TOKEN}"
    }
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    note = response.json()['choices'][0]['message']['content']

    return beautify_note(note)


def transcribe_audio(audio_file: Path, output_dir: Path, language: str) -> str:
    device = "cuda"
    batch_size = 16 # reduce if low on GPU mem
    print('Loading Whisper...')
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root='my_cache', language=language)

    audio = whisperx.load_audio(audio_file)
    print()
    print('Transcribing...')
    result = model.transcribe(audio, batch_size=batch_size, print_progress=True, language=language, task="transcribe")
    # print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    gc.collect(); torch.cuda.empty_cache(); del model

    print('Loading aligner...')
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_dir='my_cache')
    print()
    print('Aligning...')
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False, print_progress=True)
    gc.collect(); torch.cuda.empty_cache(); del model_a

    print('Loading diarizer...')
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HUGGINGFACE_TOKEN, device=device)
    print()
    print('Diarizing...')
    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    transcribation_dir = output_dir / 'transcribations'
    transcribation_dir.mkdir(parents=True, exist_ok=True)
    with open(transcribation_dir / (audio_file.stem + '.json'), 'w') as fp:
        json.dump(result['segments'], fp, ensure_ascii=False)
        
    df_diarize = pd.DataFrame(result['segments'])
    lecturer = df_diarize['speaker'].value_counts().index[0]
    df_diarize['speaker'] = df_diarize['speaker'].apply(lambda x: 'Лектор' if x == lecturer else 'Студент')
    import numpy as np

    phrases = []
    for idx, row in df_diarize.iterrows():
        if not isinstance(row['speaker'], str) and np.isnan(row['speaker']):
            phrases.append(
                row['text'].strip()
            )
        else:
            phrases.append(
                f"{row['speaker'].strip()}: {row['text'].strip()}"
            )
    result = '\n'.join(phrases)
    with open(transcribation_dir / (audio_file.stem + '.txt'), 'w') as fp:
        fp.write(result)
    return result


def extract_mp3(mp4_path: Path) -> Path:
    print('Extracting mp3 from', mp4_path)
    output_dir = Path('tmp')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (mp4_path.stem + '.mp3')
    # ffmpeg.input(str(mp4_path)).output(str(output_path), vn=True).overwrite_output().run()
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(str(mp4_path), vn=None)
        .output(
            str(output_path), 
        )
    )

    ffmpeg.execute()
    return output_path


def get_lectures_files(root_path: Path) -> List[Path]:
    video_files = []
    for file in root_path.iterdir():
        if file.suffix == '.mp4' or file.suffix == '.mp3':
            video_files.append(file)
    return video_files


def main(lecture_path: Path, output_dir: Path, language: str, class_title: str, transcribation_only: bool):
    print('Started')
    if lecture_path.is_dir():
        lecture_files = get_lectures_files(lecture_path)
    else:
        lecture_files = [lecture_path]

    print('Processing following files: \n', '\n'.join(map(str, lecture_files)), sep='')
    mp3_files = [
        extract_mp3(lecture_file) if lecture_file.suffix == '.mp4' else lecture_file 
        for lecture_file in lecture_files
    ]

    # transcribations = [transcribe_audio(mp3_file, output_dir, language) for mp3_file in mp3_files]
    
    # if transcribation_only:
    #     return
    
    # notes = [get_brief_note(transcribation) for transcribation in transcribations]

    # for note, lecture_path in zip(notes, lecture_files):
    #     output_path = output_dir / lecture_path.stem + '.md'
    #     with open(output_path, 'w') as fp:
    #         fp.write(note)

    for mp3_file in mp3_files:
        transcribation = transcribe_audio(mp3_file, output_dir, language)
        if transcribation_only:
            continue
        note = get_brief_note(transcribation, class_title)
        output_path = output_dir / (mp3_file.stem + '.md')
        with open(output_path, 'w') as fp:
            fp.write(note)


if __name__ == '__main__':
    print(r"""
    ______                 __              __                      
   / ____/___ ________  __/ /   ___  _____/ /___  __________  _____
  / __/ / __ `/ ___/ / / / /   / _ \/ ___/ __/ / / / ___/ _ \/ ___/
 / /___/ /_/ (__  ) /_/ / /___/  __/ /__/ /_/ /_/ / /  /  __(__  ) 
/_____/\__,_/____/\__, /_____/\___/\___/\__/\__,_/_/   \___/____/  
                 /____/                                            
""")
    
    args = argparse.ArgumentParser(description="EasyLectures")
    args.add_argument(
        "lecture_path",
        type=str,
        help="Path to lecture mp3 or mp4 file or even a directory with them",
    )

    args.add_argument(
        'class_title', 
        type=str,
        help="Set the title of the class.",
    )

    args.add_argument(
        '-o',
        '--output', 
        type=str,
        default='lecture_notes',
        help="Directory to put brief notes and transcribations",
    )

    args.add_argument(
        '-l',
        '--language', 
        type=str,
        default='ru',
        help="Set the language: en, ru, fr, de, es, it, ...",
    )

    args.add_argument(
        '-t',
        '--transcribation-only', 
        action=argparse.BooleanOptionalAction,
        help="Only transcribe with WhisperX without making a brief note",
    )

    args = args.parse_args()

    main(Path(args.lecture_path), Path(args.output), args.language, args.class_title, args.transcribation_only)