import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
from typing_extensions import TypedDict
import enum
from datetime import datetime
import re
import time
import weave


@dataclass
class QuizContext:
    """Represents the current quiz state and available media"""

    en_text: str
    jp_text: str
    frame: Optional[genai.types.File] = None
    video: Optional[genai.types.File] = None

    @classmethod
    def from_files(
        cls,
        jp_path: str,
        en_path: str,
        frame: Optional[genai.types.File] = None,
        video: Optional[genai.types.File] = None,
    ) -> "QuizContext":
        """Create QuizContext from file paths, supporting both .txt and .srt formats"""
        jp_text = cls._load_text_file(jp_path)
        en_text = cls._load_text_file(en_path)
        return cls(en_text=en_text, jp_text=jp_text, frame=frame, video=video)

    @staticmethod
    def _load_text_file(file_path: str) -> str:
        """Load text from either .txt or .srt files"""
        if file_path.lower().endswith(".srt"):
            return QuizContext._parse_srt_file(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def _parse_srt_file(srt_path: str) -> str:
        """Parse SRT file and return text content with timestamps preserved"""
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into subtitle blocks
        blocks = re.split(r"\n\n+", content.strip())

        # Extract text content from each block with timestamps
        texts = []
        for block in blocks:
            lines = block.split("\n")
            if len(lines) >= 3:  # Valid subtitle block should have at least 3 lines
                timestamp = lines[1]  # Format: 00:00:03,021 --> 00:00:04,523
                text = " ".join(lines[2:]).strip()
                texts.append(f"[{timestamp}] {text}")

        return " ".join(texts)


# Define enums for structured choices
class QuestionType(enum.Enum):
    # Reading (文字・語彙・文法)
    KANJI_READING = "KANJI_READING"  # Kanji reading questions
    VOCABULARY = "VOCABULARY"  # Vocabulary usage and meaning
    GRAMMAR = "GRAMMAR"  # Grammar structure and usage

    # Listening (聴解)
    LISTENING_COMPREHENSION = (
        "LISTENING_COMPREHENSION"  # General listening comprehension
    )
    LISTENING_SUMMARY = "LISTENING_SUMMARY"  # Summary of longer passages
    LISTENING_CONTEXT = "LISTENING_CONTEXT"  # Understanding situational context

    # Reading Comprehension (読解)
    READING_SHORT = "READING_SHORT"  # Short passage comprehension
    READING_MEDIUM = "READING_MEDIUM"  # Medium passage comprehension
    READING_LONG = "READING_LONG"  # Long passage comprehension

    # Language Knowledge (言語知識)
    WORD_USAGE = "WORD_USAGE"  # Proper word usage in context
    GRAMMAR_PATTERN = "GRAMMAR_PATTERN"  # Specific grammar pattern usage
    TEXT_COMPLETION = "TEXT_COMPLETION"  # Fill in the blank questions

    # Integrated Skills (統合問題)
    INTEGRATED_COMPREHENSION = "INTEGRATED_COMPREHENSION"  # Combined reading/listening
    INTEGRATED_EXPRESSION = (
        "INTEGRATED_EXPRESSION"  # Information integration/expression
    )


class Difficulty(enum.Enum):
    # N5 (Beginner)
    N5_BASIC = "N5-1"  # Complete beginner
    N5_INTERMEDIATE = "N5-2"  # Early beginner
    N5_ADVANCED = "N5-3"  # Confident beginner

    # N4 (Upper Beginner)
    N4_BASIC = "N4-1"  # Early upper beginner
    N4_INTERMEDIATE = "N4-2"  # Developing upper beginner
    N4_ADVANCED = "N4-3"  # Confident upper beginner

    # N3 (Intermediate)
    N3_BASIC = "N3-1"  # Early intermediate
    N3_INTERMEDIATE = "N3-2"  # Developing intermediate
    N3_ADVANCED = "N3-3"  # Confident intermediate

    # N2 (Upper Intermediate)
    N2_BASIC = "N2-1"  # Early upper intermediate
    N2_INTERMEDIATE = "N2-2"  # Developing upper intermediate
    N2_ADVANCED = "N2-3"  # Confident upper intermediate

    # N1 (Advanced)
    N1_BASIC = "N1-1"  # Early advanced
    N1_INTERMEDIATE = "N1-2"  # Developing advanced
    N1_ADVANCED = "N1-3"  # Confident advanced/Native-like


class Concept(enum.Enum):
    # Core Language Elements
    KANA = "Hiragana and Katakana Usage"
    BASIC_KANJI = "Basic Kanji (JLPT N5-N4)"
    ADVANCED_KANJI = "Advanced Kanji (JLPT N3-N1)"

    # Grammar
    PARTICLES = "Particle Usage (Basic and Advanced)"
    VERB_CONJUGATION = "Verb Conjugation Patterns"
    ADJECTIVES = "Adjective Usage and Conjugation"
    TE_FORM = "て Form and Related Patterns"
    CONDITIONALS = "Conditional Forms (と、ば、たら、なら)"

    # Communication
    POLITE_SPEECH = "Polite Language (です・ます)"
    CASUAL_SPEECH = "Casual Language Patterns"
    HONORIFICS = "Honorific and Humble Language"
    GIVING_RECEIVING = "Giving and Receiving Expressions"

    # Vocabulary
    ESSENTIAL_VOCAB = "Core Vocabulary Usage"
    NUMBERS_TIME = "Numbers and Time Expressions"
    IDIOMATIC = "Idiomatic and Cultural Expressions"

    # Comprehension
    READING = "Reading Comprehension"
    LISTENING = "Listening Comprehension"
    CONTEXT = "Contextual Understanding"

    # Study Skills
    KANJI_STUDY = "Kanji Learning Strategies"
    ERROR_ANALYSIS = "Error Recognition and Correction"


# Define TypedDict classes for structured responses
class QuestionTypeResponse(TypedDict):
    question_type: QuestionType
    reasoning: str


class QuestionChoice(enum.Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class QuestionData(TypedDict):
    question: str


class QuestionOptions(TypedDict):
    choices: list[str]
    correct_answer: QuestionChoice
    explanation: str


class EvaluationResponse(TypedDict):
    is_correct: bool
    reasoning: str


class ContinueResponse(TypedDict):
    continue_quiz: bool
    reasoning: str


@dataclass
class QuizMemory:
    """Represents a single question-answer interaction"""

    timestamp: datetime
    question_type: QuestionType
    question_data: QuestionData
    question_options: QuestionOptions
    user_answer: str
    is_correct: bool
    feedback: str
    concepts_covered: list[Concept]
    concepts_to_review: list[Concept]


class AgentAction(enum.Enum):
    """Available actions for the quiz agent"""

    ASK_QUESTION = "ask_question"
    ADJUST_DIFFICULTY = "adjust_difficulty"
    ADJUST_GOAL = "adjust_goal"
    CONCLUDE_SESSION = "conclude_session"


class ActionResponse(TypedDict):
    """Structured response for action decisions"""

    action: AgentAction
    reasoning: str


class ConceptSelectionResponse(TypedDict):
    """Response type for concept selection"""

    concepts: List[Concept]
    reasoning: str


class ConceptReviewResponse(TypedDict):
    """Response type for concept review determination"""

    concepts_to_review: List[Concept]
    reasoning: str


class GoalAdjustmentResponse(TypedDict):
    """Response type for goal adjustment"""

    new_goal: str
    reasoning: str


class DifficultyAdjustmentResponse(TypedDict):
    """Response type for difficulty adjustment"""

    new_difficulty: Difficulty
    reasoning: str


class QuizAgent(weave.Model):
    model: genai.GenerativeModel = None
    current_score: int = 0
    questions_asked: int = 0
    memory: list[QuizMemory] = field(default_factory=list)
    current_goal: str = "Assess initial student level"
    current_difficulty: Difficulty = field(default=Difficulty.N5_BASIC)
    available_actions: Dict[AgentAction, str] = field(
        default_factory=lambda: {
            AgentAction.ASK_QUESTION: "generate_question",
            AgentAction.ADJUST_DIFFICULTY: "modify_difficulty",
            AgentAction.ADJUST_GOAL: "adjust_goal",
            AgentAction.CONCLUDE_SESSION: "should_continue",
        }
    )

    def __init__(
        self, api_key: str, initial_difficulty: Difficulty = Difficulty.N5_BASIC
    ):
        super().__init__()

        genai.configure(api_key=api_key)
        object.__setattr__(self, "model", genai.GenerativeModel("gemini-1.5-pro"))
        object.__setattr__(self, "current_score", 0)
        object.__setattr__(self, "questions_asked", 0)
        object.__setattr__(self, "memory", [])
        object.__setattr__(self, "current_goal", "Assess initial student level")
        object.__setattr__(self, "current_difficulty", initial_difficulty)
        object.__setattr__(
            self,
            "available_actions",
            {
                AgentAction.ASK_QUESTION: "generate_question",
                AgentAction.ADJUST_DIFFICULTY: "modify_difficulty",
                AgentAction.ADJUST_GOAL: "adjust_goal",
                AgentAction.CONCLUDE_SESSION: "should_continue",
            },
        )

    @weave.op
    def prepare_media(
        self, image_path: Optional[str] = None, video_path: Optional[str] = None
    ) -> Tuple[Optional[genai.types.File], Optional[genai.types.File]]:
        """Prepare media files for Gemini API"""
        frame = None
        video = None

        if image_path:
            frame = genai.upload_file(image_path)
            # Wait for frame processing to complete
            while frame.state.name == "PROCESSING":
                time.sleep(1)  # Check every 1 second
                frame = genai.get_file(frame.name)

            if frame.state.name == "FAILED":
                raise ValueError(f"Frame processing failed: {image_path}")

        if video_path:
            video = genai.upload_file(video_path)
            # Wait for video processing to complete
            while video.state.name == "PROCESSING":
                time.sleep(1)  # Check every 1 second
                video = genai.get_file(video.name)

            if video.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {video_path}")

        return frame, video

    @weave.op
    async def determine_question_type(self, context: QuizContext) -> QuestionType:
        """Determines best question type based on available media and learning history"""
        prompt = """Given the following Japanese language learning context and history, determine the most effective question type and concepts to focus on.

Context:
- English Text: {en_text}
- Japanese Text: {jp_text}
- Available Media: {media_types}
- Current Difficulty: {difficulty}
- Current Learning Goal: {goal}

Available Question Types:
{question_types}

Learning History:
{learning_history}

Analyze the following aspects:
1. Previous question types and performance
2. Concepts that need reinforcement
3. Linguistic complexity of the Japanese text
4. Available visual/audio cues
5. Alignment with current difficulty level and learning goal

Select the most appropriate question type and relevant concepts to test.

Return a JSON response:
{{
    "question_type": "QUESTION_TYPE_ENUM",
    "reasoning": "Detailed explanation of why this question type was chosen, considering the current difficulty level and learning goal."
}}"""

        # Format question types as a list
        question_types_text = "\n".join(
            [f"- {qt.name}: {qt.value}" for qt in QuestionType]
        )

        # Format learning history
        history_text = self._format_learning_history()

        media_types = []
        if context.frame:
            media_types.append("Static Frame")
        if context.video:
            media_types.append("Video Segment")

        content = [
            prompt.format(
                en_text=context.en_text,
                jp_text=context.jp_text,
                media_types=", ".join(media_types) if media_types else "Text Only",
                difficulty=self.current_difficulty.value,
                goal=self.current_goal,
                learning_history=history_text,
                question_types=question_types_text,
            )
        ]

        if context.frame:
            content.append(context.frame)
        if context.video:
            content.append(context.video)

        response = await self.model.generate_content_async(
            content,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=QuestionTypeResponse,
            ),
        )
        response_text = response.candidates[0].content.parts[0].text
        response_data: QuestionTypeResponse = json.loads(response_text)
        return response_data["question_type"]

    def _format_learning_history(self) -> str:
        """Formats the learning history for prompt inclusion"""
        if not self.memory:
            return "No previous questions asked."

        history = []
        # Get last 5 items without using slice
        start_idx = max(0, len(self.memory) - 5)
        memory_len = len(self.memory)

        for i in range(start_idx, memory_len):
            mem = self.memory[i]
            history.append(f"""Question {i - start_idx + 1}:
- Type: {mem.question_type}
- Question: {mem.question_data["question"]}
- Question Options: {mem.question_options["choices"]}
- Correct Answer: {mem.question_options["correct_answer"]}
- Student Answer: {mem.user_answer}
- Student Correct: {mem.is_correct}
- Concepts Covered: {', '.join(mem.concepts_covered)}
- Concepts Needing Review: {', '.join(mem.concepts_to_review)}
- Feedback: {mem.feedback[:100]}...""")  # Truncate feedback for brevity

        return "\n\n".join(history)

    @weave.op
    async def determine_concepts(
        self, context: QuizContext, question_type: str
    ) -> List[Concept]:
        """Determines appropriate concepts to test based on question type and learning history"""
        prompt = """Given the following Japanese language learning context and question type, determine the most appropriate concepts to focus on.

Context:
- English Text: {en_text}
- Japanese Text: {jp_text}
- Question Type: {question_type}
- Available Media: {media_types}
- Current Difficulty: {difficulty}
- Current Learning Goal: {goal}

Learning History:
{learning_history}

Available Concepts:
{concepts}

Consider:
1. Previous concept performance
2. Concepts that align with the question type
3. Natural concept progression at current difficulty level
4. Available context and media
5. Alignment with current learning goal

Return a JSON response with 2-3 related concepts:
{{
    "concepts": ["CONCEPT_ENUM1", "CONCEPT_ENUM2", "CONCEPT_ENUM3"],
    "reasoning": "Explanation of why these concepts were chosen, considering difficulty level and learning goal."
}}"""

        # Format concepts as a list
        concepts_text = "\n".join([f"- {c.name}: {c.value}" for c in Concept])

        media_types = []
        if context.frame:
            media_types.append("Static Frame")
        if context.video:
            media_types.append("Video Segment")

        content = [
            prompt.format(
                en_text=context.en_text,
                jp_text=context.jp_text,
                question_type=question_type,
                media_types=", ".join(media_types) if media_types else "Text Only",
                difficulty=self.current_difficulty.value,
                goal=self.current_goal,
                learning_history=self._format_learning_history(),
                concepts=concepts_text,
            )
        ]

        if context.frame:
            content.append(context.frame)
        if context.video:
            content.append(context.video)

        response = await self.model.generate_content_async(
            content,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ConceptSelectionResponse,
            ),
        )
        response_text = response.candidates[0].content.parts[0].text
        response_data: ConceptSelectionResponse = json.loads(response_text)

        # Default to basic concepts if none provided or invalid
        default_concepts = [Concept.KANA, Concept.ESSENTIAL_VOCAB, Concept.PARTICLES]
        concepts = response_data.get("concepts", default_concepts)

        return concepts

    @weave.op
    async def generate_question(
        self, context: QuizContext, question_type: QuestionType, concepts: list[str]
    ) -> QuestionData:
        """Generates question using all available media and learning history"""
        prompt = """Generate a short focused Japanese language question based on the following context and history.

Learning Context:
- English: {en_text}
- Japanese: {jp_text}
- Question Focus: {q_type}
- Target Concepts: {concepts}
- Current Difficulty: {difficulty}
- Available Media: {media_types}

Previous Learning History:
{learning_history}

Requirements:
1. Question should directly test the specified concepts: {concepts}
2. Question type should be: {q_type}
3. Question difficulty should match: {difficulty}
4. Choices must be plausible but clearly distinguishable

Return a JSON response:
{{
    "question": "Clear, short, focused question text that matches the specified difficulty level",
}}"""
        # Format learning history
        history_text = self._format_learning_history()

        media_types = []
        if context.frame:
            media_types.append("Static Frame")
        if context.video:
            media_types.append("Video Segment")

        content = [
            prompt.format(
                en_text=context.en_text,
                jp_text=context.jp_text,
                q_type=question_type,
                concepts=concepts,
                difficulty=self.current_difficulty.value,
                media_types=", ".join(media_types) if media_types else "Text Only",
                learning_history=history_text,
            )
        ]

        if context.frame:
            content.append(context.frame)
        if context.video:
            content.append(context.video)

        response = await self.model.generate_content_async(
            content,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=QuestionData
            ),
        )
        response_text = response.candidates[0].content.parts[0].text
        response_data: QuestionData = json.loads(response_text)
        return response_data

    @weave.op
    async def evaluate_answer(
        self,
        user_answer: str,
        question_data: QuestionData,
        question_options: QuestionOptions,
        question_type: QuestionType,
        concepts: list[str],
        context: QuizContext,
    ) -> Tuple[bool, str]:
        """Evaluate user's answer and provide feedback"""
        prompt = """Provide detailed feedback on the student's Japanese language quiz response.

Question Context:
Question: {question}
Expected Answer: {correct}
Student Response: {student}
Current JLPT Level: {difficulty}
Current Learning Goal: {goal}
Target Concepts: {concepts}

Provide:
1. Correctness assessment
2. Detailed explanation why the answer is correct/incorrect
3. Feedback specifically addressing:
   - Target concepts: {concepts}
   - Current difficulty level expectations
   - Progress toward learning goal
4. Common misconceptions if relevant
5. Study tips for improvement focused on these concepts

Return a JSON response:
{{
    "is_correct": boolean,
    "reasoning": "Detailed, constructive feedback with examples"
}}"""

        content = [
            prompt.format(
                question=question_data["question"],
                correct=question_options["correct_answer"],
                student=user_answer,
                difficulty=self.current_difficulty.value,
                goal=self.current_goal,
                concepts=concepts,
            )
        ]

        if context.frame:
            content.append(context.frame)
        if context.video:
            content.append(context.video)

        response = await self.model.generate_content_async(
            content,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=EvaluationResponse,
            ),
        )
        response_text = response.candidates[0].content.parts[0].text
        result: EvaluationResponse = json.loads(response_text)

        # Update score
        if result["is_correct"]:
            self.current_score += 1

        self.questions_asked += 1

        # Determine concepts to review if answer is incorrect
        concepts_to_review = []
        if not result["is_correct"]:
            concepts_to_review = await self.determine_concepts_to_review(
                question_data,
                question_options,
                user_answer,
                result["is_correct"],
                result["reasoning"],
            )

        # Store the interaction in memory
        memory_entry = QuizMemory(
            timestamp=datetime.now(),
            question_type=question_type,
            question_data=question_data,
            question_options=question_options,
            user_answer=user_answer,
            is_correct=result["is_correct"],
            feedback=result["reasoning"],
            concepts_covered=concepts,
            concepts_to_review=concepts_to_review,
        )
        self.memory.append(memory_entry)

        return result["is_correct"], result["reasoning"]

    @weave.op
    async def should_continue(self) -> bool:
        prompt = """Evaluate the student's Japanese language learning progress and recommend next steps.

Current Status:
- Learning Goal: {goal}
- Difficulty Level: {difficulty}

Complete Learning History:
{detailed_history}

Summary Statistics:
- Total Questions: {num_questions}
- Correct Answers: {score}
- Recent Performance: {recent_performance}
- Most Challenging Concepts: {challenging_concepts}

Analysis Criteria:
1. Progress toward current learning goal
2. Mastery at current difficulty level
3. Error Patterns
4. Engagement Level
5. Readiness for difficulty adjustment

Return a JSON response:
{{
    "continue_quiz": boolean,
    "reasoning": "Detailed pedagogical justification for continuing or ending, considering current goal and difficulty level"
}}
"""
        # Calculate recent performance and challenging concepts
        recent_correct = sum(1 for m in self.memory[-5:] if m.is_correct)
        recent_performance = (
            f"{recent_correct}/min(5, len(self.memory)) recent questions correct"
        )

        # Identify challenging concepts
        concept_counts = {}
        for mem in self.memory:
            if not mem.is_correct:
                for concept in mem.concepts_to_review:
                    concept_counts[concept] = concept_counts.get(concept, 0) + 1
        challenging_concepts = sorted(
            concept_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        response = await self.model.generate_content_async(
            prompt.format(
                goal=self.current_goal,
                difficulty=self.current_difficulty.value,
                detailed_history=self._format_learning_history(),
                num_questions=self.questions_asked,
                score=self.current_score,
                recent_performance=recent_performance,
                challenging_concepts=challenging_concepts,
            ),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=ContinueResponse
            ),
        )
        response_text = response.candidates[0].content.parts[0].text
        result: ContinueResponse = json.loads(response_text)
        return result["continue_quiz"]

    @weave.op
    async def run_quiz_session(self, context: QuizContext):
        """Main agent loop - observe, think, act"""
        while True:
            # THINK - Determine next action
            next_action = await self._decide_next_action(context)

            # ACT - Execute chosen action
            if next_action == AgentAction.ASK_QUESTION:
                # First determine question type
                question_type = await self.determine_question_type(context)
                # Then determine appropriate concepts
                concepts = await self.determine_concepts(context, question_type)
                # Generate base question
                question_data = await self.generate_question(
                    context, question_type, concepts
                )
                # Generate options and correct answer
                question_options = await self.generate_question_options(
                    context, question_data["question"], question_type, concepts
                )

                # Combine question data with options
                full_question = {**question_data, **question_options}

                # Yield control to caller for user input
                user_answer = yield ("wait_for_answer", full_question, question_type)

                # Process user's answer
                is_correct, feedback = await self.evaluate_answer(
                    user_answer,
                    question_data,
                    question_options,
                    question_type,
                    concepts,
                    context,
                )
                yield ("show_feedback", is_correct, feedback)

            elif next_action == AgentAction.ADJUST_DIFFICULTY:
                new_difficulty = await self._modify_difficulty()
                yield ("adjust_difficulty", new_difficulty)

            elif next_action == AgentAction.ADJUST_GOAL:
                new_goal = await self._adjust_goal()
                yield ("adjust_goal", new_goal)

            elif next_action == AgentAction.CONCLUDE_SESSION:
                should_continue = await self.should_continue()
                if not should_continue:
                    yield ("conclude_session", None)
                    return

    @weave.op
    async def _decide_next_action(self, context: QuizContext) -> AgentAction:
        """Agent decides what action to take next using structured output"""
        prompt = """As a Japanese language tutor, decide the next best action.

Current Status:
- Learning Goal: {goal}
- Difficulty Level: {difficulty}
- Questions Asked: {questions}
- Current Score: {score}
- Success Rate: {success_rate}%

Student History:
{history}

Available Actions:
- ASK_QUESTION: Generate and ask a new question
- ADJUST_DIFFICULTY: Modify the current difficulty level
- ADJUST_GOAL: Update the learning goal based on performance
- CONCLUDE_SESSION: End the current learning session

Think through:
1. Current progress toward learning goal
2. Performance at current difficulty level
3. Need for goal or difficulty adjustment
4. Overall engagement and progress
5. Best action to optimize learning

Return a JSON response:
{{
    "action": "ASK_QUESTION|ADJUST_DIFFICULTY|ADJUST_GOAL|CONCLUDE_SESSION",
    "reasoning": "Detailed explanation of why this action was chosen"
}}"""

        # Calculate success rate
        if self.questions_asked > 0:
            success_rate = (self.current_score / self.questions_asked) * 100
        else:
            success_rate = 0

        response = await self.model.generate_content_async(
            prompt.format(
                goal=self.current_goal,
                difficulty=self.current_difficulty.value,
                questions=self.questions_asked,
                score=self.current_score,
                success_rate=round(success_rate, 1),
                history=self._format_learning_history(),
            ),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=ActionResponse
            ),
        )
        response_text = response.candidates[0].content.parts[0].text
        result: ActionResponse = json.loads(response_text)
        return AgentAction(result["action"])

    @weave.op
    async def _modify_difficulty(self) -> Difficulty:
        """Adjust difficulty based on performance and learning progression using LLM analysis"""
        prompt = """Analyze the student's Japanese language learning performance and determine appropriate difficulty adjustment.

Current Status:
- Current Difficulty: {difficulty}
- Questions Asked: {total_questions}
- Overall Success Rate: {overall_rate}%
- Recent Performance: {recent_performance}
- Learning Goal: {goal}

Recent Learning History:
{history}

Available Difficulty Levels:
{difficulty_levels}

Consider:
1. Recent performance trend
2. Consistency at current level
3. Error patterns and types
4. Readiness for advancement
5. Appropriate challenge level
6. Minimum questions needed for assessment

You must choose a difficulty level different from the current one.

Return a JSON response:
{{
    "new_difficulty": "DIFFICULTY_ENUM",
    "reasoning": "Detailed explanation of why this difficulty level is appropriate"
}}"""

        # Calculate performance metrics
        recent_correct = 0
        total_recent = min(5, len(self.memory))
        memory_len = len(self.memory)
        start_idx = max(0, memory_len - 5)

        for i in range(start_idx, memory_len):
            if self.memory[i].is_correct:
                recent_correct += 1

        recent_rate = (recent_correct / total_recent) if total_recent > 0 else 0
        overall_rate = (
            (self.current_score / self.questions_asked)
            if self.questions_asked > 0
            else 0
        )

        # Format difficulty levels as a list with descriptions
        difficulty_levels = "\n".join([f"- {d.name}: {d.value}" for d in Difficulty])

        response = await self.model.generate_content_async(
            prompt.format(
                difficulty=self.current_difficulty.value,
                total_questions=self.questions_asked,
                overall_rate=round(overall_rate * 100, 1),
                recent_performance=f"{recent_correct}/{total_recent} recent questions correct ({round(recent_rate * 100, 1)}%)",
                goal=self.current_goal,
                history=self._format_learning_history(),
                difficulty_levels=difficulty_levels,
            ),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=DifficultyAdjustmentResponse,
            ),
        )

        response_text = response.candidates[0].content.parts[0].text
        result: DifficultyAdjustmentResponse = json.loads(response_text)

        # Update the agent's current difficulty
        try:
            new_difficulty = Difficulty[
                result["new_difficulty"]
            ]  # Convert string to enum
            object.__setattr__(self, "current_difficulty", new_difficulty)
            print(f"New difficulty set to: {new_difficulty}")
            return new_difficulty
        except KeyError:
            # Fallback to current difficulty if invalid enum value received
            return self.current_difficulty

    @weave.op
    async def _adjust_goal(self) -> str:
        """Determine and set a new learning goal based on performance analysis"""
        prompt = """Analyze the student's learning progress and determine the most appropriate learning goal.

Current Status:
- Current Goal: {current_goal}
- Difficulty Level: {difficulty}
- Questions Asked: {total_questions}
- Current Score: {score}
- Success Rate: {success_rate}%

Recent Performance History:
{history}

Consider:
1. Overall performance trend
2. Consistency at current difficulty level
3. Areas of strength and weakness
4. Learning progression
5. Appropriate challenge level

Return a JSON response:
{{
    "new_goal": "Clear learning goal from the available types",
    "reasoning": "Detailed explanation of why this goal is appropriate given the current context"
}}"""

        # Calculate success rate
        if self.questions_asked > 0:
            success_rate = (self.current_score / self.questions_asked) * 100
        else:
            success_rate = 0

        response = await self.model.generate_content_async(
            prompt.format(
                current_goal=self.current_goal,
                difficulty=self.current_difficulty.value,
                total_questions=self.questions_asked,
                score=self.current_score,
                success_rate=round(success_rate, 1),
                history=self._format_learning_history(),
            ),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=GoalAdjustmentResponse,
            ),
        )

        response_text = response.candidates[0].content.parts[0].text
        result: GoalAdjustmentResponse = json.loads(response_text)

        # Update the agent's goal
        object.__setattr__(self, "current_goal", result["new_goal"])
        return result["new_goal"]

    @weave.op
    async def generate_question_options(
        self,
        context: QuizContext,
        question: str,
        question_type: QuestionType,
        concepts: list[str],
    ) -> QuestionOptions:
        """Generates multiple choice options and correct answer for a given question"""
        prompt = """Generate multiple choice options for the following Japanese language question.

Question: {question}
Question Type: {q_type}
Target Concepts: {concepts}
Context:
- English: {en_text}
- Japanese: {jp_text}
- Available Media: {media_types}

Requirements:
1. Generate 4 distinct choices (A, B, C, D)
2. One choice must be clearly correct
3. Other choices should be plausible but clearly incorrect
4. Include brief explanation of why the correct answer is right

Return a JSON response:
{{
    "choices": ["Choice A text", "Choice B text", "Choice C text", "Choice D text"],
    "correct_answer": "A|B|C|D",
    "explanation": "Clear explanation of why the correct answer is right and common misconceptions"
}}"""

        media_types = []
        if context.frame:
            media_types.append("Static Frame")
        if context.video:
            media_types.append("Video Segment")

        content = [
            prompt.format(
                question=question,
                q_type=question_type,
                concepts=concepts,
                en_text=context.en_text,
                jp_text=context.jp_text,
                media_types=", ".join(media_types) if media_types else "Text Only",
            )
        ]

        if context.frame:
            content.append(context.frame)
        if context.video:
            content.append(context.video)

        response = await self.model.generate_content_async(
            content,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=QuestionOptions
            ),
        )
        response_text = response.candidates[0].content.parts[0].text
        question_options: QuestionOptions = json.loads(response_text)
        return question_options

    @weave.op
    async def determine_concepts_to_review(
        self,
        question_data: QuestionData,
        question_options: QuestionOptions,
        user_answer: str,
        is_correct: bool,
        feedback: str,
    ) -> List[Concept]:
        """Analyzes student response to determine which concepts need additional review"""
        prompt = """Analyze the student's response to determine which concepts need additional review.

Question Context:
Question: {question}
Answer Choices:
{choices}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}
Is Correct: {is_correct}
Evaluation Feedback: {feedback}
Current JLPT Level: {difficulty}
Current Learning Goal: {goal}

Available Concepts:
{all_concepts}

Consider:
1. Specific errors in the student's response
2. Concepts directly related to those errors
3. Prerequisite concepts that might need strengthening
4. Current difficulty level expectations
5. Overall learning progression

Return a JSON response:
{{
    "concepts_to_review": ["CONCEPT_ENUM1", "CONCEPT_ENUM2"],
    "reasoning": "Detailed explanation of why these concepts need review based on the student's response"
}}"""

        # Format choices as a numbered list
        choices_text = "\n".join(
            [
                f"{chr(65 + i)}. {choice}"
                for i, choice in enumerate(question_options["choices"])
            ]
        )

        # Format all available concepts as a list
        concepts_text = "\n".join([f"- {c.name}: {c.value}" for c in Concept])

        response = await self.model.generate_content_async(
            prompt.format(
                question=question_data["question"],
                choices=choices_text,
                correct_answer=f"{question_options['correct_answer']}: {question_options['choices'][ord(question_options['correct_answer']) - 65]}",
                student_answer=user_answer,
                is_correct=is_correct,
                feedback=feedback,
                difficulty=self.current_difficulty.value,
                goal=self.current_goal,
                all_concepts=concepts_text,
            ),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ConceptReviewResponse,
            ),
        )
        response_text = response.candidates[0].content.parts[0].text
        result: ConceptReviewResponse = json.loads(response_text)

        return result["concepts_to_review"]
