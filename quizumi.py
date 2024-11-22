from src.video_player import CustomVideoPlayer
import sys
from PyQt5.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    player = CustomVideoPlayer(
        video_path="./example/nichijou_1.mp4",
        jp_sub_path="./example/nichijou_1_jp.srt",  # or .srt
        en_sub_path="./example/nichijou_1_en.ass",  # or .ass
        app=app,
    )
    player.run()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
