# Quizumi

Quizumi is an intelligent Japanese learning video player that combines video playback with interactive quizzes powered by Google's Gemini AI. It helps users learn Japanese through active engagement with anime, movies, or any Japanese video content.

## Features

- **Intelligent Video Player**
  - Support for both .srt and .ass subtitle formats
  - Toggle between Japanese and English subtitles
  - Frame capture and video segment analysis
  - Keyboard controls for easy navigation

- **AI-Powered Quizzes**
  - Dynamic question generation based on current video context
  - Multiple question types covering various JLPT levels
  - Adaptive difficulty based on user performance
  - Comprehensive feedback and explanations
  - Progress tracking and concept mastery analysis

## Installation

1. Clone the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export WANDB_API_KEY="your_wandb_api_key"
```

## Usage

1. Place your video and subtitle files in the example directory:
```
example/
  ├── your_video.mp4
  ├── your_video_jp.srt  # Japanese subtitles
  └── your_video_en.ass  # English subtitles
```

2. Update the video and subtitle paths in `quizumi.py`:
```python
player = CustomVideoPlayer(
    video_path="./example/your_video.mp4",
    jp_sub_path="./example/your_video_jp.srt",
    en_sub_path="./example/your_video_en.ass",
    app=app,
)
```

3. Run the application:
```bash
python quizumi.py
```

## Controls

- `p` - Play/pause video
- `j` - Toggle Japanese subtitles
- `e` - Toggle English subtitles
- `k` - Start quiz for current segment
- `←/→` - Seek backward/forward 5 seconds
- `q` - Quit player

## How It Works

1. **Video Processing**
   - The player captures frames and segments video in real-time
   - Subtitles are synchronized and extracted for the current segment
   - Media is processed for AI analysis

2. **Quiz Generation**
   - The Gemini AI analyzes video context, subtitles, and frames
   - Questions are generated based on JLPT levels and learning concepts
   - Adaptive difficulty adjusts to user performance

3. **Learning Analysis**
   - Performance tracking across multiple concepts
   - Difficulty adjustment based on success rate
   - Targeted feedback for improvement

## Requirements

- Python 3.8+
- VLC Media Player
- FFmpeg
- Google Gemini API key
- PyQt5
- Additional dependencies listed in requirements.txt

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.