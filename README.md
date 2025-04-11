
# EasyLectures

EasyLectures is a tool that automatically transcribes lecture recordings and generates structured markdown notes using WhisperX and LLM technologies.

```ascii
 ______                 __              __                      
   / ____/___ ________  __/ /   ___  _____/ /___  __________  _____
  / __/ / __ `/ ___/ / / / /   / _ \/ ___/ __/ / / / ___/ _ \/ ___/
 / /___/ /_/ (__  ) /_/ / /___/  __/ /__/ /_/ /_/ / /  /  __(__  ) 
/_____/\__,_/____/\__, /_____/\___/\___/\__/\__,_/_/   \___/____/  
                 /____/                                            
```

## Features

- Transcribes lecture recordings (MP3/MP4) using WhisperX
- Supports multiple languages
- Identifies speakers (lecturer vs students)
- Generates structured Markdown notes optimized for Obsidian
- Handles both single files and directories with multiple lectures
- Automatic MP4 to MP3 conversion
- Option to only transcribe without generating notes

## Prerequisites

- Python 3.x
- CUDA-capable GPU (for WhisperX)
- Fireworks API token (set as environment variable `FIREWORKS_TOKEN`)

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install whisperx pandas torch requests ffmpeg-python tqdm
```

## Usage

Basic usage:
```bash
python easy_lectures.py <path_to_lecture> "<class_title>"
```

Full usage with options:
```bash
python easy_lectures.py <path_to_lecture> "<class_title>" [-o OUTPUT_DIR] [-l LANGUAGE] [-t]
```

### Arguments

- `lecture_path`: Path to lecture MP3/MP4 file or directory containing multiple files
- `class_title`: Title of the course/class
- `-o, --output`: Output directory for notes and transcriptions (default: "lecture_notes")
- `-l, --language`: Language code (default: "ru", options: en, ru, fr, de, es, it, etc.)
- `-t, --transcription-only`: Only generate transcription without creating notes

### Example

```bash
python easy_lectures.py lectures/week1.mp4 "Introduction to Python" -l en -o notes
```

## Output Structure

The tool creates two types of output files in the specified directory:

```
lecture_notes/
├── transcribations/
│   ├── lecture1.json
│   └── lecture1.txt
└── lecture1.md
```

- `transcribations/*.json`: Detailed transcription with timing and speaker information
- `transcribations/*.txt`: Human-readable transcription with speaker labels
- `*.md`: Generated structured notes in Markdown format

## Note Format

The generated notes are formatted for Obsidian with:
- Logical sections and subsections
- Bullet points for examples and case studies
- LaTeX-style math formatting using `$` delimiters
- Code blocks where appropriate
- Speaker identification (Lecturer/Student)

## Environment Variables

- `FIREWORKS_TOKEN`: Your Fireworks API authentication token
