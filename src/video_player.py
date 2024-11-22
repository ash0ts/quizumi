from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
import vlc
import sys
import asyncio
import os
import time
from rich.console import Console
from rich.logging import RichHandler
import logging
import subprocess
import pysrt
import ass
from .run_quiz_agent import run_quiz_agent  # Add this import


class CustomVideoPlayer(QMainWindow):
    def __init__(self, video_path, jp_sub_path=None, en_sub_path=None, app=None):
        super().__init__()
        self.app = app
        self.setWindowTitle("Intelligent Japanese Learning Player")

        # Set up rich logging with a more appropriate level
        logging.basicConfig(
            level=logging.INFO,  # Changed from DEBUG to INFO
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
        self.log = logging.getLogger("video_player")
        self.console = Console()

        # Create a central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a widget to hold the VLC player
        self.video_widget = QWidget(self)
        layout.addWidget(self.video_widget)

        # Create VLC instance with basic subtitle options
        subtitle_options = [
            "--freetype-rel-fontsize=20",  # Relative font size (16=normal, 20=larger)
            "--freetype-bold",  # Make subtitles bold
            "--sub-margin=25",  # Reduced margin to move subtitles closer to bottom
        ]

        self.instance = vlc.Instance(" ".join(subtitle_options))
        self.player = self.instance.media_player_new()
        self.media = self.instance.media_new(video_path)

        # Add subtitle files in correct order
        if en_sub_path:
            self.media.add_option(f":sub-file={en_sub_path}")
        if jp_sub_path:
            self.media.add_option(f":sub-file={jp_sub_path}")

        self.player.set_media(self.media)

        # Set the window ID where VLC will render the video
        if sys.platform.startswith("darwin"):  # macOS
            self.player.set_nsobject(int(self.video_widget.winId()))
        elif sys.platform.startswith("linux"):  # Linux
            self.player.set_xwindow(self.video_widget.winId())
        elif sys.platform.startswith("win"):  # Windows
            self.player.set_hwnd(self.video_widget.winId())

        # Set fullscreen mode
        self.showFullScreen()

        # Initialize subtitle states
        self.japanese_sub_enabled = False
        self.english_sub_enabled = False

        # Store subtitle track IDs (will be set after media starts playing)
        self.jp_track_id = None
        self.en_track_id = None

        # Update control print statements with rich formatting
        self.console.print(
            "\n[bold cyan]Welcome to the Intelligent Japanese Learning Player![/]\n"
        )
        self.console.print("[yellow]Controls:[/]")
        self.console.print("  [green]p[/] - Play/pause video")
        self.console.print("  [green]j[/] - Toggle Japanese subtitles")
        self.console.print("  [green]e[/] - Toggle English subtitles")
        self.console.print("  [green]k[/] - Start quiz for current segment")
        self.console.print("  [green]←/→[/] - Seek backward/forward 5 seconds")
        self.console.print("  [green]q[/] - Quit player\n")

        # Store the original video path
        self.video_path = video_path

        # Add new variables for scrubbing
        self.scrub_timer = QTimer()
        self.scrub_timer.setSingleShot(True)
        self.scrub_timer.timeout.connect(self.handle_scrub_end)
        self.SCRUB_BUFFER_MS = 2000  # Wait 2 second after last scrub
        self.was_playing_before_scrub = False
        self.log.info(
            f"Initialized scrubbing buffer timer with {self.SCRUB_BUFFER_MS}ms delay"
        )

        self.jp_sub_path = jp_sub_path
        self.en_sub_path = en_sub_path

        # Add output directory setup
        self.base_output_dir = self._create_output_directory(video_path)
        self.log.info(f"Saving outputs to: {self.base_output_dir}")

    def _create_output_directory(self, video_path):
        """Create an output directory for current segments"""
        # Create directory in current working directory
        output_dir = os.path.join(os.getcwd(), "current_segment")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    def play(self):
        self.player.play()
        # Wait for media to start playing before getting subtitle tracks
        time.sleep(0.5)  # Give VLC time to load the media
        self.initialize_subtitle_tracks()

    def pause(self):
        """Toggle play/pause state"""
        if self.player.is_playing():
            self.player.pause()
        else:
            self.player.play()

    def initialize_subtitle_tracks(self):
        """Initialize subtitle tracks after media starts playing"""
        time.sleep(1)

        tracks = self.player.video_get_spu_description()
        self.log.debug("Available subtitle tracks:")

        self.jp_track_id = None
        self.en_track_id = None

        # Store all tracks for debugging
        for track_id, track_name in tracks:
            if track_id == -1:  # Skip the "Disable" track
                continue

            track_str = track_name.decode("utf-8")
            self.log.debug(f"Track ID: {track_id}, Track Name: {track_str}")

            # Look for Japanese track by filename
            if "jp" in track_str.lower():
                self.jp_track_id = track_id
                self.log.debug(
                    f"Found Japanese track -> ID: {track_id}, Name: {track_str}"
                )
            # Assign Track 2 as English
            elif track_id == 3:
                self.en_track_id = track_id
                self.log.debug(
                    f"Assigned English track -> ID: {track_id}, Name: {track_str}"
                )

        self.log.info("Final track assignments:")
        self.log.info(f"Japanese track ID: {self.jp_track_id}")
        self.log.info(f"English track ID: {self.en_track_id}")

        # Ensure subtitles start disabled
        self.player.video_set_spu(-1)
        self.japanese_sub_enabled = False
        self.english_sub_enabled = False

    def toggle_japanese_subtitles(self):
        """Toggle Japanese subtitles on/off"""
        if self.jp_track_id is None:
            self.log.warning("Japanese subtitle track not found")
            return

        self.log.debug(
            "Current track before Japanese toggle: %d", self.player.video_get_spu()
        )

        if self.japanese_sub_enabled:
            self.log.info("Disabling Japanese subtitles")
            self.player.video_set_spu(-1)
            self.japanese_sub_enabled = False
        else:
            self.log.info("Enabling Japanese subtitles")
            self.player.video_set_spu(self.jp_track_id)
            self.japanese_sub_enabled = True
            self.english_sub_enabled = False

        self.log.debug(
            "Current track after Japanese toggle: %d", self.player.video_get_spu()
        )

    def toggle_english_subtitles(self):
        """Toggle English subtitles on/off"""
        if self.en_track_id is None:
            self.log.warning("English subtitle track not found")
            return

        self.log.debug(
            "Current track before English toggle: %d", self.player.video_get_spu()
        )

        if self.english_sub_enabled:
            self.log.info("Disabling English subtitles")
            self.player.video_set_spu(-1)
            self.english_sub_enabled = False
        else:
            self.log.info("Enabling English subtitles")
            self.player.video_set_spu(self.en_track_id)
            self.english_sub_enabled = True
            self.japanese_sub_enabled = False

        self.log.debug(
            "Current track after English toggle: %d", self.player.video_get_spu()
        )

    # TODO: Support equally spaced frames from begininng to current frame to use Gemini for multiple frames in one prompt
    def _capture_frame(self):
        """Capture current video frame"""
        try:
            # Save frame to current segment directory instead of working directory
            temp_frame_path = os.path.join(self.base_output_dir, "temp_frame.png")
            success = self.player.video_take_snapshot(0, temp_frame_path, 0, 0)
            self.log.debug(f"Frame capture success: {success}")
            return success
        except Exception as e:
            self.log.error(f"Frame capture error: {e}")
        return None

    def _get_current_subtitles(self, language: str) -> str:
        """Get current subtitles for specified language"""
        try:
            track_id = self.jp_track_id if language == "jp" else self.en_track_id
            if track_id is not None:
                subs = self.player.video_get_spu_text(track_id)
                if subs:
                    self.log.debug(f"{language.upper()} Subtitles: {subs}")
                return subs
        except Exception as e:
            self.log.error(f"Subtitle retrieval error: {e}")
        return ""

    def handle_scrub_end(self):
        """Called when scrubbing ends after buffer time"""
        self.log.debug("Scrub buffer time elapsed, handling end of scrub")
        current_time = self.player.get_time()
        self.log.debug(f"Current video position: {current_time}ms")

        self._capture_frame()
        self.save_video_segment()
        self.save_subtitle_segment()

        # Always keep video paused after scrubbing
        self.log.debug("Keeping video paused after scrub")
        self.player.pause()

    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_P:
            self.pause()
            self._capture_frame()
            self.save_video_segment()
            self.save_subtitle_segment()
        elif event.key() == Qt.Key_K:
            # Ensure video is paused
            if self.player.is_playing():
                self.player.pause()
            # Save current state
            self._capture_frame()
            self.save_video_segment()
            self.save_subtitle_segment()
            # Run quiz agent asynchronously
            asyncio.run(run_quiz_agent())
        elif event.key() in (Qt.Key_Left, Qt.Key_Right):
            # Store playing state and pause if playing
            if not self.scrub_timer.isActive():
                if self.player.is_playing():
                    self.log.debug("Pausing video for scrubbing")
                    self.player.pause()

            # Handle the seek
            current_time = self.player.get_time()
            direction = "backward" if event.key() == Qt.Key_Left else "forward"
            self.log.debug(f"Scrubbing {direction} from position {current_time}ms")

            if event.key() == Qt.Key_Left:
                new_time = max(0, current_time - 5000)
                self.player.set_time(new_time)
            else:  # Right key
                new_time = current_time + 5000
                self.player.set_time(new_time)

            self.log.debug(f"New position after scrub: {new_time}ms")

            # Reset the scrub timer
            if self.scrub_timer.isActive():
                self.log.debug("Resetting scrub buffer timer")
            else:
                self.log.debug("Starting scrub buffer timer")
            self.scrub_timer.start(self.SCRUB_BUFFER_MS)

        elif event.key() == Qt.Key_J:
            self.toggle_japanese_subtitles()
        elif event.key() == Qt.Key_E:
            self.toggle_english_subtitles()
        elif event.key() == Qt.Key_Q:
            self.close()
            self.app.quit()

    def run(self):
        """Start the video player"""
        self.show()
        self.play()

    def stop(self):
        """Stop the video player"""
        self.player.stop()
        self.close()

    def save_video_segment(self):
        """Save video segment from start to current position using ffmpeg"""
        try:
            current_pos = self.player.get_time() / 1000.0
            if current_pos <= 0:
                self.log.warning("Cannot save video segment: Invalid current position")
                return

            output_path = os.path.join(self.base_output_dir, "video_segment.mp4")
            self.log.info("Saving video segment...")

            command = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i",
                self.video_path,
                "-ss",
                "0",  # Start from beginning
                "-t",
                str(current_pos),
                "-c:v",
                "copy",  # Copy video stream
                "-c:a",
                "copy",  # Copy audio stream
                "-copyts",  # Copy timestamps
                "-avoid_negative_ts",
                "make_zero",  # Handle negative timestamps
                output_path,
            ]

            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode == 0:
                self.log.info("✓ Video segment saved successfully")
            else:
                self.log.error("Failed to save video segment: %s", result.stderr)

        except Exception as e:
            self.log.error("Error saving video segment: %s", str(e))

    def save_subtitle_segment(self):
        """Save subtitle segments from start to current position in SRT format"""
        try:
            current_pos_ms = self.player.get_time()  # Already in milliseconds
            if current_pos_ms <= 0:
                self.log.warning(
                    "Cannot save subtitle segment: Invalid current position"
                )
                return

            self.log.debug(f"Saving subtitles up to position: {current_pos_ms}ms")

            # Process Japanese subtitles
            if self.jp_sub_path:
                self._process_subtitle_file(
                    self.jp_sub_path, current_pos_ms, "jp_segment.srt"
                )

            # Process English subtitles
            if self.en_sub_path:
                self._process_subtitle_file(
                    self.en_sub_path, current_pos_ms, "en_segment.srt"
                )

        except Exception as e:
            self.log.error(f"Error saving subtitle segment: {e}")

    def _process_subtitle_file(self, subtitle_path, end_time_ms, output_filename):
        """Process a subtitle file and save segments up to end_time_ms (in milliseconds)"""
        output_path = os.path.join(self.base_output_dir, output_filename)
        file_ext = subtitle_path.lower().split(".")[-1]

        try:
            if file_ext == "srt":
                try:
                    subs = pysrt.open(subtitle_path, encoding="utf-8-sig")
                except:
                    subs = pysrt.open(subtitle_path, encoding="utf-8")

                filtered_subs = [
                    sub for sub in subs if self._time_to_ms(sub.end) <= end_time_ms
                ]

            elif file_ext == "ass":
                with open(subtitle_path, encoding="utf-8-sig") as f:
                    ass_file = ass.parse(f)

                filtered_subs = []
                for event in ass_file.events:
                    # Skip non-dialogue events, comments, or empty lines
                    if not isinstance(event, ass.Dialogue) or not event.text.strip():
                        continue

                    self.log.debug(
                        f"Processing event with style: {event.style}, text: {event.text[:30]}..."
                    )

                    # For English subtitles
                    if output_filename == "en_segment.srt":
                        if event.style != "Default":  # Main dialogue style
                            continue
                    # For Japanese subtitles
                    elif output_filename == "jp_segment.srt":
                        if "Default" not in event.style:
                            continue

                    # Convert timedelta to milliseconds
                    start_ms = int(event.start.total_seconds() * 1000)
                    end_ms = int(event.end.total_seconds() * 1000)

                    if end_ms <= end_time_ms:
                        # Clean up ASS formatting tags
                        text = self._clean_ass_text(event.text)

                        sub = pysrt.SubRipItem(
                            index=len(filtered_subs) + 1,
                            start=self._ms_to_time(start_ms),
                            end=self._ms_to_time(end_ms),
                            text=text,
                        )
                        filtered_subs.append(sub)
                        self.log.debug(f"Added subtitle: {text[:30]}...")

                self.log.debug(
                    f"Found {len(filtered_subs)} valid subtitles in ASS file for {output_filename}"
                )
            else:
                raise ValueError(f"Unsupported subtitle format: {file_ext}")

            # Save filtered subtitles
            if filtered_subs:
                for i, sub in enumerate(filtered_subs, 1):
                    sub.index = i
                subs_file = pysrt.SubRipFile(filtered_subs)
                subs_file.save(output_path, encoding="utf-8-sig")
                self.log.info(f"Saved {len(filtered_subs)} subtitles to {output_path}")
            else:
                self.log.warning(
                    f"No subtitles found in the specified time range for {output_path}"
                )
                with open(output_path, "w", encoding="utf-8-sig") as f:
                    f.write("")

        except Exception as e:
            self.log.error(f"Error processing subtitle file: {str(e)}")
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write("")

    def _time_to_ms(self, time):
        """Convert SubRipTime to milliseconds"""
        return (
            time.hours * 3600 + time.minutes * 60 + time.seconds
        ) * 1000 + time.milliseconds

    def _ms_to_time(self, milliseconds):
        """Convert milliseconds to pysrt.SubRipTime"""
        hours = int(milliseconds // (3600 * 1000))
        minutes = int((milliseconds % (3600 * 1000)) // (60 * 1000))
        seconds = int((milliseconds % (60 * 1000)) // 1000)
        ms = int(milliseconds % 1000)
        return pysrt.SubRipTime(
            hours=hours, minutes=minutes, seconds=seconds, milliseconds=ms
        )

    def _clean_ass_text(self, text):
        """Remove ASS tags and formatting from text"""
        # Remove {...} style tags
        import re

        text = re.sub(r"{[^}]*}", "", text)
        # Remove \N and \n line breaks, replace with standard line break
        text = text.replace("\\N", "\n").replace("\\n", "\n")
        return text.strip()
