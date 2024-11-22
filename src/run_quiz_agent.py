from src.quiz_agent import QuizAgent, QuizContext
from set_env import set_env
import os
import weave
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QRadioButton,
    QPushButton,
    QButtonGroup,
    QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

set_env("GOOGLE_API_KEY")
set_env("WANDB_API_KEY")


class QuizDialog(QDialog):
    def __init__(self, question_data, question_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quiz Question")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        type_label = QLabel(f"Question Type: {question_type}")
        type_label.setFont(QFont("Arial", 10, QFont.Bold))
        type_label.setStyleSheet("color: #666;")

        question_label = QLabel(question_data["question"])
        question_label.setFont(QFont("Arial", 12))
        question_label.setWordWrap(True)

        layout.addWidget(type_label)
        layout.addWidget(question_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #ccc;")
        layout.addWidget(line)

        self.button_group = QButtonGroup()
        self.answers = []

        for i, choice in enumerate(question_data["choices"]):
            radio = QRadioButton(f"{chr(65+i)}. {choice}")
            radio.setFont(QFont("Arial", 11))
            radio.setStyleSheet("""
                QRadioButton {
                    padding: 5px;
                }
                QRadioButton::indicator {
                    width: 13px;
                    height: 13px;
                }
                QRadioButton::indicator:unchecked {
                    border: 2px solid #999;
                    border-radius: 7px;
                    background-color: transparent;
                }
                QRadioButton::indicator:checked {
                    border: 2px solid #2196F3;
                    border-radius: 7px;
                    background-color: #2196F3;
                }
            """)
            self.answers.append(chr(65 + i))
            self.button_group.addButton(radio, i)
            layout.addWidget(radio)

        submit_btn = QPushButton("Submit")
        submit_btn.setFont(QFont("Arial", 11, QFont.Bold))
        submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        submit_btn.clicked.connect(self.accept)
        layout.addWidget(submit_btn)

        self.setLayout(layout)

    def get_answer(self):
        selected_id = self.button_group.checkedId()
        return self.answers[selected_id] if selected_id >= 0 else None


def create_styled_dialog(title, is_feedback=False):
    dialog = QDialog()
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(400)
    dialog.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

    layout = QVBoxLayout()
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(15)
    return dialog, layout


@weave.op
async def run_quiz_agent():
    weave.init("quiz-agent-test")
    api_key = os.getenv("GOOGLE_API_KEY")
    quiz_agent = QuizAgent(api_key)

    # Prepare available media
    frame, video = quiz_agent.prepare_media(
        image_path="./current_segment/temp_frame.png",
        video_path="./current_segment/video_segment.mp4",
    )

    context = QuizContext.from_files(
        jp_path="./current_segment/jp_segment.srt",
        en_path="./current_segment/en_segment.srt",
        frame=frame,
        video=video,
    )

    quiz_session = quiz_agent.run_quiz_session(context)
    ctr = 0
    try:
        while True:
            action = await quiz_session.__anext__()
            action_type = action[0]

            if action_type == "wait_for_answer":
                _, question_data, question_type = action

                # Create and show dialog
                dialog = QuizDialog(question_data, question_type)
                if dialog.exec_() == QDialog.Accepted:
                    user_answer = dialog.get_answer()
                    if user_answer:
                        await quiz_session.asend(user_answer)
                    else:
                        print("No answer selected!")

            elif action_type == "show_feedback":
                _, is_correct, feedback = action
                feedback_dialog, layout = create_styled_dialog("Quiz Feedback", True)

                result_label = QLabel("✓ Correct!" if is_correct else "✗ Incorrect!")
                result_label.setFont(QFont("Arial", 14, QFont.Bold))
                result_label.setStyleSheet(
                    f"color: {'#4CAF50' if is_correct else '#F44336'};"
                )

                feedback_label = QLabel(f"Feedback: {feedback}")
                feedback_label.setFont(QFont("Arial", 11))
                feedback_label.setWordWrap(True)

                ok_btn = QPushButton("OK")
                ok_btn.setFont(QFont("Arial", 11))
                ok_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2196F3;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                    }
                    QPushButton:hover {
                        background-color: #1976D2;
                    }
                """)
                ok_btn.clicked.connect(feedback_dialog.accept)

                layout.addWidget(result_label)
                layout.addWidget(feedback_label)
                layout.addWidget(ok_btn)
                feedback_dialog.setLayout(layout)
                feedback_dialog.exec_()

            elif action_type == "conclude_session":
                break

            ctr += 1
            if ctr > 5:
                break

    except StopAsyncIteration:
        pass

    score_dialog, layout = create_styled_dialog("Quiz Complete")
    score_label = QLabel(
        f"Final Score: {quiz_agent.current_score}/{quiz_agent.questions_asked}"
    )
    score_label.setFont(QFont("Arial", 14, QFont.Bold))

    ok_btn = QPushButton("OK")
    ok_btn.setFont(QFont("Arial", 11))
    ok_btn.setStyleSheet("""
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
    """)
    ok_btn.clicked.connect(score_dialog.accept)

    layout.addWidget(score_label)
    layout.addWidget(ok_btn)
    score_dialog.setLayout(layout)
    score_dialog.exec_()


if __name__ == "__main__":
    import asyncio
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    asyncio.run(run_quiz_agent())
