import os
import asyncio
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Poll
from telegram.ext import (
    filters,
    ContextTypes,
    CommandHandler,
    ApplicationBuilder,
    MessageHandler,
    ConversationHandler,
    CallbackQueryHandler,
    Application
)
import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import time
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable logging with reduced verbosity
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Constants and setup
WRITING, OCR_CHOICE, EDITED_TEXT, CONFIRM_QA, AWAITING_QUESTIONS, AWAITING_ANSWERS, AWAITING_NOTE = range(7)

users = []
task_data = {}  # To manage tasks with unique IDs
bot_token = "7797307575:AAFJClPS7X0_sMp25v7VUiSMXbjfaZ7f6JA"
if not bot_token:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set!")

MAX_POLL_QUESTION_LENGTH = 300  # Telegram limit for poll question
MAX_POLL_OPTION_LENGTH = 100   # Telegram limit for poll option
MAX_EXPLANATION_LENGTH = 200   # Telegram limit for poll explanation
MAX_MESSAGE_LENGTH = 4096      # Telegram message length limit

# Helper Functions

def get_user_data(user_id):
    """Retrieve or initialize user data."""
    for user_data in users:
        if user_data['id'] == user_id:
            return user_data
    new_user_data = {
        'id': user_id,
        'quizzes': [],
        'ocr_text': None,
        'state': WRITING,
        'current_question_index': None,
        'temp_file_path': None,
        'file_ext': None
    }
    users.append(new_user_data)
    return new_user_data

def update_user_data(user_id, key, value):
    """Update user data with a key-value pair."""
    user_data = get_user_data(user_id)
    user_data[key] = value

def split_text(text, max_length=MAX_MESSAGE_LENGTH):
    """Split text into chunks smaller than max_length, preserving structure."""
    if not text or len(text.strip()) == 0:
        return ["No text extracted."]
    parts = []
    while text:
        if len(text) <= max_length:
            parts.append(text.strip())
            break
        split_at = text.rfind('\n', 0, max_length) or text.rfind(' ', 0, max_length) or max_length
        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()
    return parts

def resize_image(image, max_width):
    """Resize an image proportionally if it exceeds max_width."""
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        image = image.resize((max_width, new_height), Image.LANCZOS)
    return image

def is_two_column(image):
    """Determine if an image has two columns based on horizontal projection."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    projection = np.sum(thresh, axis=0)
    max_proj = np.max(projection)
    projection = projection / max_proj if max_proj > 0 else np.zeros_like(projection)
    width = len(projection)
    third = width // 3
    left = projection[:third]
    middle = projection[third:2*third]
    right = projection[2*third:]
    middle_min = np.min(middle) if len(middle) > 0 else 0
    left_max = np.max(left) if len(left) > 0 else 0
    right_max = np.max(right) if len(right) > 0 else 0
    return middle_min < 0.3 * min(left_max, right_max) and middle_min < 0.1 * max(projection)

def process_page(page_num, img):
    """Process an image page with OCR, handling single or two-column layouts."""
    start_time = time.time()
    try:
        img_ocr = resize_image(img, max_width=800)
        logger.info(f"Resized page {page_num} to 800px in {time.time() - start_time:.2f}s")
        
        start_detect = time.time()
        img_detect = resize_image(img_ocr, max_width=300)
        if is_two_column(img_detect):
            split_idx = img_ocr.width // 2
            left_img = img_ocr.crop((0, 0, split_idx, img_ocr.height))
            right_img = img_ocr.crop((split_idx, 0, img_ocr.width, img_ocr.height))
            logger.info(f"Detected two columns for page {page_num} in {time.time() - start_detect:.2f}s")
            start_ocr = time.time()
            left_text = pytesseract.image_to_string(left_img, lang='eng', config='--psm 4 --oem 1')
            right_text = pytesseract.image_to_string(right_img, lang='eng', config='--psm 4 --oem 1')
            logger.info(f"OCR for page {page_num} columns in {time.time() - start_ocr:.2f}s")
            result = f"{left_text}\n\n{right_text}"
        else:
            logger.info(f"Detected single column for page {page_num} in {time.time() - start_detect:.2f}s")
            start_ocr = time.time()
            page_text = pytesseract.image_to_string(img_ocr, lang='eng', config='--psm 4 --oem 1')
            logger.info(f"OCR for page {page_num} in {time.time() - start_ocr:.2f}s")
            result = page_text
    except Exception as e:
        result = f"Page {page_num}: Error - {str(e)}"
    finally:
        logger.info(f"Total time for page {page_num}: {time.time() - start_time:.2f}s")
    return result

def extract_text_from_pdf_page(page_num, page, file_path):
    """Attempt direct text extraction from a PDF page, fall back to OCR if necessary."""
    try:
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        device = TextConverter(rsrcmgr, retstr, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        interpreter.process_page(page)
        text = retstr.getvalue()
        device.close()
        retstr.close()
        if text.strip():
            logger.info(f"Extracted text directly from page {page_num}")
            return text
        else:
            logger.info(f"No text extracted from page {page_num}, performing OCR")
            images = convert_from_path(file_path, first_page=page_num, last_page=page_num)
            if images:
                img = images[0]
                page_result = process_page(page_num, img)
                return page_result
            else:
                return f"Error - Could not convert page {page_num} to image"
    except Exception as e:
        return f"Error - {str(e)}"

async def parse_questions(user_input, update, context):
    """Parse user input into a list of question/answer/note dictionaries."""
    questions = []
    question_blocks = re.split(r'\n\s*\n+', user_input.strip())
    for block in question_blocks:
        block = block.strip()
        if not block:
            continue
        question, choices, explanation = await parse_question_data(block)
        if question and choices:
            questions.append({
                "question": question,
                "choices": choices,
                "explanation": explanation
            })
        elif question or choices:
            questions.append({
                "question": question or "PARSING FAILED - QUESTION TEXT MISSING",
                "choices": choices or ["PARSING FAILED - CHOICES MISSING"],
                "explanation": explanation or "PARSING PARTIALLY FAILED, PLEASE CHECK FORMAT"
            })
        else:
            questions.append({
                "question": "PARSING FAILED - EMPTY BLOCK",
                "choices": ["PARSING FAILED - EMPTY BLOCK"],
                "explanation": "PARSING FAILED - EMPTY BLOCK"
            })
    return questions

async def parse_question_data(text):
    """Parse a single question block with flexible choice handling."""
    lines = text.split("\n")
    question_lines = []
    choices = []
    explanation = None
    choice_pattern = re.compile(r"^[a-z][). -]", re.IGNORECASE)
    has_labeled_choices = any(choice_pattern.match(line.strip()) for line in lines[1:])

    if has_labeled_choices:
        # Labeled choices: collect question lines until first choice
        for line in lines:
            stripped_line = line.strip()
            if choice_pattern.match(stripped_line):
                choice_text = re.split(r"^[a-z][). -]", stripped_line, 1, re.IGNORECASE)[1].strip()
                choices.append(choice_text)
            elif "#NOTE" in stripped_line:
                explanation = stripped_line.replace("#NOTE", "").strip()
            else:
                if not choices:  # Still part of the question
                    question_lines.append(stripped_line)
    else:
        # Unlabeled choices: first line is question, subsequent lines are choices
        question_lines = [lines[0].strip()] if lines else []
        for line in lines[1:]:
            stripped_line = line.strip()
            if "#NOTE" in stripped_line:
                explanation = stripped_line.replace("#NOTE", "").strip()
                break
            if stripped_line:  # Only add non-empty lines as choices
                choices.append(stripped_line)

    question = " ".join(question_lines).strip()
    return question, choices, explanation

async def reformat_ocr_text(text):
    """Reformat OCR-extracted text into a clean question-choice format."""
    questions = []
    question_blocks = re.split(r'\n\s*\n+', text.strip())
    for block in question_blocks:
        block = block.strip()
        if not block:
            continue
        question, choices, _ = await parse_question_data(block)
        if question and choices:
            formatted_block = f"{question}\n" + "\n".join([f"{chr(97 + i)}. {choice}" for i, choice in enumerate(choices)])
            questions.append(formatted_block)
    return "\n\n".join(questions)

async def parse_answers(update, context, user_input, questions):
    """Parse answers, extracting the first letter regardless of additional text."""
    answers = []
    lines = user_input.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Find the first letter 'a' to 'j' in the line
        match = re.search(r'[a-j]', line, re.IGNORECASE)
        if match:
            answer_char = match.group(0).lower()
            answer_index = ord(answer_char) - ord('a')
            if len(answers) < len(questions):
                current_question = questions[len(answers)]
                if 'choices' not in current_question or answer_index >= len(current_question['choices']):
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=f"Invalid answer '{line}'. Choice '{answer_char}' does not exist for question {len(answers) + 1}."
                    )
                    return None
                answers.append(answer_index)
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Too many answers provided."
                )
                return None
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Invalid answer format: '{line}'. Please provide answers with a letter (a, b, c, etc.)."
            )
            return None
    return answers

# Bot Functions

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the conversation and introduce the bot."""
    user = update.effective_user
    user_data = get_user_data(user.id)
    user_data['state'] = WRITING
    user_data['quizzes'] = []
    reply_markup = InlineKeyboardMarkup(get_main_keyboard())
    await update.message.reply_text(
        f"Hello, {user.first_name}! I'm ready to help you create quizzes. Send images/PDFs or type questions and answers. Use the buttons below.",
        reply_markup=reply_markup
    )
    return WRITING

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle file uploads (images and PDFs)."""
    user_id = update.effective_user.id
    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        file_ext = "image"
    elif update.message.document and update.message.document.mime_type in ("application/pdf", "image/jpeg", "image/png"):
        file_id = update.message.document.file_id
        file_ext = "pdf" if update.message.document.mime_type == "application/pdf" else "image"
    else:
        await update.message.reply_text("Please send a PDF or image.")
        return WRITING

    file = await context.bot.get_file(file_id)
    temp_file_path = f"{uuid.uuid4()}.{file_ext}"
    await file.download_to_drive(temp_file_path)

    task_id = str(uuid.uuid4())
    task_data[task_id] = (file_ext, temp_file_path)

    keyboard = [
        [InlineKeyboardButton("Extract Text", callback_data=f"extract_text_{task_id}")] if file_ext == "pdf" else [],
        [InlineKeyboardButton("Use Tesseract OCR", callback_data=f"use_tesseract_{task_id}")],
        [InlineKeyboardButton("Edit manually", callback_data=f"edit_ocr_{task_id}")]
    ]
    keyboard = [row for row in keyboard if row]  # Remove empty rows
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Processing file... Please choose an option:",
        reply_markup=reply_markup
    )
    return OCR_CHOICE

async def process_pdf_extract_text(update, context, file_path, task_id, reformat=False):
    """Extract text from PDF with progress updates."""
    user_id = update.effective_user.id
    progress_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Processing PDF... 0%")
    try:
        with open(file_path, 'rb') as fp:
            parser = PDFParser(fp)
            document = PDFDocument(parser)
            pages = list(PDFPage.create_pages(document))
            total_pages = len(pages)
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [loop.run_in_executor(executor, extract_text_from_pdf_page, i+1, page, file_path) for i, page in enumerate(pages)]
                progress_task = asyncio.create_task(update_progress(futures, progress_message, total_pages))
                results = await asyncio.gather(*futures)
                await progress_task
                text = "\n\n".join(results)
                text = re.sub(r'\n{3,}', '\n\n', text)  # Clean up extra newlines
                if reformat:
                    text = await reformat_ocr_text(text)
                update_user_data(user_id, 'ocr_text', text)
                for part in split_text(text):
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=part)
                keyboard = [
                    [InlineKeyboardButton("Use this text", callback_data=f"use_edited_{task_id}"),
                     InlineKeyboardButton("Edit manually", callback_data=f"edit_ocr_{task_id}")],
                    [InlineKeyboardButton("Reformat OCR Text", callback_data=f"reformat_ocr_{task_id}")]
                ]
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do?", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        await progress_message.edit_text(f"Error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if task_id in task_data:
            del task_data[task_id]

async def process_file_with_tesseract(update, context, file_path, file_ext, task_id, reformat=False):
    """Perform OCR on the file with progress updates for PDFs."""
    user_id = update.effective_user.id
    if file_ext == "pdf":
        progress_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Processing PDF... 0%")
        try:
            with open(file_path, 'rb') as fp:
                parser = PDFParser(fp)
                document = PDFDocument(parser)
                pages = list(PDFPage.create_pages(document))
                total_pages = len(pages)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    futures = [loop.run_in_executor(executor, process_page, i+1, convert_from_path(file_path, first_page=i+1, last_page=i+1)[0]) for i in range(total_pages)]
                    progress_task = asyncio.create_task(update_progress(futures, progress_message, total_pages))
                    results = await asyncio.gather(*futures)
                    await progress_task
                    text = "\n\n".join(results)
                    text = re.sub(r'\n{3,}', '\n\n', text)  # Clean up extra newlines
                    if reformat:
                        text = await reformat_ocr_text(text)
                    update_user_data(user_id, 'ocr_text', text)
                    for part in split_text(text):
                        await context.bot.send_message(chat_id=update.effective_chat.id, text=part)
                    keyboard = [
                        [InlineKeyboardButton("Use this text", callback_data=f"use_edited_{task_id}"),
                         InlineKeyboardButton("Edit manually", callback_data=f"edit_ocr_{task_id}")],
                        [InlineKeyboardButton("Reformat OCR Text", callback_data=f"reformat_ocr_{task_id}")]
                    ]
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do?", reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception as e:
            await progress_message.edit_text(f"Error: {str(e)}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            if task_id in task_data:
                del task_data[task_id]
    elif file_ext == "image":
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Processing image...")
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            text = await loop.run_in_executor(executor, process_page, 1, Image.open(file_path))
            text = re.sub(r'\n{3,}', '\n\n', text)  # Clean up extra newlines
            if reformat:
                text = await reformat_ocr_text(text)
            update_user_data(user_id, 'ocr_text', text)
            for part in split_text(text):
                await context.bot.send_message(chat_id=update.effective_chat.id, text=part)
            keyboard = [
                [InlineKeyboardButton("Use this text", callback_data=f"use_edited_{task_id}"),
                 InlineKeyboardButton("Edit manually", callback_data=f"edit_ocr_{task_id}")],
                [InlineKeyboardButton("Reformat OCR Text", callback_data=f"reformat_ocr_{task_id}")]
            ]
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do?", reply_markup=InlineKeyboardMarkup(keyboard))
        if os.path.exists(file_path):
            os.remove(file_path)
        if task_id in task_data:
            del task_data[task_id]

async def update_progress(futures, progress_message, total_pages):
    """Update the progress message during processing."""
    current_text = "Processing PDF... 0%"
    while not all(f.done() for f in futures):
        completed = sum(1 for f in futures if f.done())
        percentage = int((completed / total_pages) * 100)
        new_text = f"Processing PDF... {percentage}%"
        if new_text != current_text:
            await progress_message.edit_text(new_text)
            current_text = new_text
        await asyncio.sleep(5)
    await progress_message.edit_text("Processing complete.")

async def receive_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive text messages and file uploads."""
    message = update.message
    user_id = update.effective_user.id
    user_data = get_user_data(user_id)

    if message.photo or (message.document and message.document.mime_type in ("application/pdf", "image/jpeg", "image/png")):
        return await handle_file(update, context)
    else:
        if user_data['state'] == AWAITING_QUESTIONS:
            await process_text(update, context, message.text, "questions")
            return AWAITING_QUESTIONS
        elif user_data['state'] == AWAITING_ANSWERS:
            await process_text(update, context, message.text, "answers")
            return AWAITING_ANSWERS
        elif user_data['state'] == AWAITING_NOTE:
            await process_text(update, context, message.text, "note")
            return AWAITING_NOTE
        elif user_data['state'] == EDITED_TEXT:
            return await receive_edited_text(update, context)
        await process_text(update, context, message.text)
        return WRITING

async def receive_edited_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle edited text from the user."""
    user_id = update.effective_user.id
    edited_text = update.message.text
    if edited_text:
        update_user_data(user_id, 'ocr_text', edited_text)
        keyboard = [
            [InlineKeyboardButton("Use this text", callback_data="use_edited"),
             InlineKeyboardButton("Edit again", callback_data="reedit_ocr")]
        ]
        await update.message.reply_text(
            "Edited text received. What would you like to do?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return OCR_CHOICE
    else:
        await update.message.reply_text("No text received. Please send the edited text.")
        return EDITED_TEXT

async def generate_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate and send the quiz as polls, handling long questions and choices."""
    user_id = update.effective_user.id
    user_data = get_user_data(user_id)
    if not user_data['quizzes']:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please add questions and answers before generating a quiz."
        )
        return
    for quiz_index, quiz in enumerate(user_data['quizzes']):
        if not all(key in quiz for key in ('question', 'choices', 'answer')):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Skipping quiz {quiz_index + 1} due to missing data."
            )
            continue
        question_text = quiz['question']
        choices = [choice[:97] + "..." if len(choice) > MAX_POLL_OPTION_LENGTH else choice for choice in quiz['choices']]
        explanation = quiz.get('explanation')
        if explanation and len(explanation) > MAX_EXPLANATION_LENGTH:
            explanation = explanation[:197] + "..."
        if len(question_text) > MAX_POLL_QUESTION_LENGTH:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Question {quiz_index + 1}: {question_text}"
            )
            poll_question = "Which of the following is the correct answer?"
        else:
            poll_question = question_text
        try:
            await context.bot.send_poll(
                chat_id=update.effective_chat.id,
                question=poll_question,
                options=choices,
                type=Poll.QUIZ,
                correct_option_id=quiz['answer'],
                explanation=explanation,
                is_anonymous=True,
            )
        except Exception as e:
            logger.error(f"Error creating poll for quiz {quiz_index + 1}: {e}")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Error creating poll for quiz {quiz_index + 1}: {e}"
            )

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle button presses from inline keyboards."""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    user_data = get_user_data(user_id)
    data = query.data

    if data.startswith("extract_text_"):
        task_id = data.split("_", 2)[2]
        file_ext, file_path = task_data.get(task_id, (None, None))
        if file_ext == "pdf":
            await process_pdf_extract_text(update, context, file_path, task_id)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Extract Text is only available for PDFs.")
        return OCR_CHOICE
    elif data.startswith("use_tesseract_"):
        task_id = data.split("_", 2)[2]
        file_ext, file_path = task_data.get(task_id, (None, None))
        await process_file_with_tesseract(update, context, file_path, file_ext, task_id)
        return OCR_CHOICE
    elif data.startswith("edit_ocr_"):
        task_id = data.split("_", 2)[2]
        await edit_ocr_text(update, context)
        return EDITED_TEXT
    elif data.startswith("reformat_ocr_"):
        task_id = data.split("_", 2)[2]
        file_ext, file_path = task_data.get(task_id, (None, None))
        if file_ext == "pdf":
            await process_pdf_extract_text(update, context, file_path, task_id, reformat=True)
        elif file_ext == "image":
            await process_file_with_tesseract(update, context, file_path, file_ext, task_id, reformat=True)
        return OCR_CHOICE
    elif data.startswith("use_edited_"):
        task_id = data.split("_", 2)[2]
        await confirm_qa_type(update, context, user_data['ocr_text'])
        return CONFIRM_QA
    elif data == "add_questions":
        user_data['state'] = AWAITING_QUESTIONS
        await query.edit_message_text("Please send your questions. Separate each question with a blank line.")
        return AWAITING_QUESTIONS
    elif data == "add_answers":
        user_data['state'] = AWAITING_ANSWERS
        await query.edit_message_text("Please send your answers (one per line, e.g., 'a', 'b', 'c', etc.).")
        return AWAITING_ANSWERS
    elif data == "add_note":
        if not user_data['quizzes']:
            await query.edit_message_text("Please add questions before adding notes.")
            return WRITING
        question_buttons = [[InlineKeyboardButton(f"Question {i+1}", callback_data=f"select_question_{i}")] for i in range(len(user_data['quizzes']))]
        await query.edit_message_text("Select which question to add a note to:", reply_markup=InlineKeyboardMarkup(question_buttons))
        return AWAITING_NOTE
    elif data.startswith("select_question_"):
        question_index = int(data.split("_")[-1])
        if question_index < len(user_data['quizzes']):
            user_data['current_question_index'] = question_index
            user_data['state'] = AWAITING_NOTE
            await query.edit_message_text(f"Okay, send your note for Question {question_index + 1}:")
            return AWAITING_NOTE
        else:
            await query.edit_message_text("Invalid question selected.")
            return WRITING
    elif data == "generate_quiz":
        await generate_quiz(update, context)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Quiz generated! What next?", reply_markup=InlineKeyboardMarkup(get_main_keyboard()))
        return WRITING
    elif data == "reset_data":
        update_user_data(user_id, 'quizzes', [])
        await context.bot.send_message(chat_id=update.effective_chat.id, text="All quiz data has been reset.", reply_markup=InlineKeyboardMarkup(get_main_keyboard()))
        return WRITING
    elif data == "use_edited":
        await confirm_qa_type(update, context, user_data['ocr_text'])
        return CONFIRM_QA
    elif data == "reedit_ocr":
        await edit_ocr_text(update, context)
        return EDITED_TEXT
    elif data == "confirm_questions":
        await process_confirmed_ocr(update, context, "questions")
        return WRITING
    elif data == "confirm_answers":
        await process_confirmed_ocr(update, context, "answers")
        return WRITING
    elif data == "get_added":
        await get_added(update, context)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Added information retrieved.", reply_markup=InlineKeyboardMarkup(get_main_keyboard()))
        return WRITING
    elif data == "cancel":
        await cancel(update, context)
        return WRITING
    return WRITING

async def edit_ocr_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate OCR text editing."""
    user_id = update.effective_user.id
    text = get_user_data(user_id).get('ocr_text')
    if text:
        for part in split_text(text):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Please edit the following text and send it back:\n\n{part}"
            )
        user_data = get_user_data(user_id)
        user_data['state'] = EDITED_TEXT
        return EDITED_TEXT
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="No OCR text available to edit."
        )
        return WRITING

async def confirm_qa_type(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Ask user to confirm if text is questions or answers."""
    keyboard = [
        [InlineKeyboardButton("Questions", callback_data="confirm_questions"),
         InlineKeyboardButton("Answers", callback_data="confirm_answers")]
    ]
    await context.bot.edit_message_text(
        chat_id=update.effective_chat.id,
        message_id=update.callback_query.message.message_id,
        text="Should this text be interpreted as questions or answers?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def process_confirmed_ocr(update: Update, context: ContextTypes.DEFAULT_TYPE, qa_type: str):
    """Process OCR text after user confirmation."""
    user_id = update.effective_user.id
    user_data = get_user_data(user_id)
    ocr_text = user_data.get('ocr_text')
    if ocr_text:
        await process_text(update, context, ocr_text, qa_type)
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="No OCR text available. Please send an image or PDF first."
        )

async def process_text(update: Update, context: ContextTypes.DEFAULT_TYPE, user_input: str, qa_type=None):
    """Process text input as questions, answers, or notes with enhanced flexibility."""
    user_id = update.effective_user.id
    user_data = get_user_data(user_id)

    if "#ANSWERS" in user_input.upper():
        parts = re.split(r'#ANSWERS', user_input, flags=re.IGNORECASE)
        questions_text = parts[0].strip()
        answers_text = parts[1].strip() if len(parts) > 1 else None
        if questions_text:
            questions = await parse_questions(questions_text, update, context)
            update_user_data(user_id, "quizzes", questions)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"{len(questions)} Questions added."
            )
        if answers_text:
            answers = await parse_answers(update, context, answers_text, user_data['quizzes'])
            if answers:
                for i, quiz in enumerate(user_data['quizzes']):
                    quiz['answer'] = answers[i]
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"{len(answers)} Answers added."
                )
        user_data['state'] = WRITING
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="What next?",
            reply_markup=InlineKeyboardMarkup(get_main_keyboard())
        )
        return

    if qa_type is None:
        if "#QUESTIONS" in user_input:
            qa_type, user_input = "questions", user_input.replace("#QUESTIONS", "").strip()
        elif "#ADD_QUESTIONS" in user_input:
            qa_type, user_input = "add_questions", user_input.replace("#ADD_QUESTIONS", "").strip()
        elif "#ANSWERS" in user_input:
            qa_type, user_input = "answers", user_input.replace("#ANSWERS", "").strip()
        elif "#ADD_ANSWERS" in user_input:
            qa_type, user_input = "add_answers", user_input.replace("#ADD_ANSWERS", "").strip()
        elif "#NOTE" in user_input:
            qa_type = "note"
        else:
            if user_data['state'] == AWAITING_QUESTIONS:
                qa_type = "questions"
            elif user_data['state'] == AWAITING_ANSWERS:
                qa_type = "answers"
            elif user_data['state'] == AWAITING_NOTE:
                qa_type = "note"
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Please specify whether you are adding questions, answers, or a note using #QUESTIONS, #ANSWERS, #ADD_QUESTIONS, #ADD_ANSWERS, or #NOTE.",
                    reply_markup=InlineKeyboardMarkup(get_main_keyboard())
                )
                return

    if qa_type in ("questions", "add_questions"):
        questions = await parse_questions(user_input, update, context)
        if questions:
            if qa_type == "add_questions" and user_data['quizzes']:
                user_data['quizzes'].extend(questions)
            else:
                update_user_data(user_id, "quizzes", questions)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"{len(questions)} Questions added. Use the buttons below or send more questions/answers.",
                reply_markup=InlineKeyboardMarkup(get_main_keyboard())
            )
        user_data['state'] = WRITING

    elif qa_type in ("answers", "add_answers"):
        answers = await parse_answers(update, context, user_input, user_data['quizzes'])
        if not answers:
            return
        if len(user_data['quizzes']) != len(answers):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"The number of answers ({len(answers)}) does not match the number of questions ({len(user_data['quizzes'])}). Please check your input.",
                reply_markup=InlineKeyboardMarkup(get_main_keyboard())
            )
            return
        for i, quiz in enumerate(user_data['quizzes']):
            quiz['answer'] = answers[i]
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"{len(answers)} Answers added. Use the buttons below or send more information.",
            reply_markup=InlineKeyboardMarkup(get_main_keyboard())
        )
        user_data['state'] = WRITING

    elif qa_type == "note":
        if "#NOTE" in user_input:
            question_index = None
            for i, line in enumerate(user_input.split('\n')):
                if "#NOTE" in line:
                    question_index = i
                    break
            if question_index is not None and user_data['quizzes']:
                current_question_real_index, current_question_index = -1, -1
                for i, question in enumerate(user_data['quizzes']):
                    current_question_real_index += 1
                    if current_question_real_index == question_index:
                        current_question_index = i
                        break
                if current_question_index != -1:
                    user_data['quizzes'][current_question_index]['explanation'] = user_input.split("#NOTE", 1)[1].strip()
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=f"Note added to question {current_question_index + 1}.",
                        reply_markup=InlineKeyboardMarkup(get_main_keyboard())
                    )
                else:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="Question index mismatch.",
                        reply_markup=InlineKeyboardMarkup(get_main_keyboard())
                    )
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="No question found corresponding to the #NOTE, or no questions have been added yet.",
                    reply_markup=InlineKeyboardMarkup(get_main_keyboard())
                )
        elif user_data['current_question_index'] is not None:
            if user_data['current_question_index'] < len(user_data['quizzes']):
                user_data['quizzes'][user_data['current_question_index']]['explanation'] = user_input
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"Note added to question {user_data['current_question_index'] + 1}.",
                    reply_markup=InlineKeyboardMarkup(get_main_keyboard())
                )
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Invalid question index.",
                    reply_markup=InlineKeyboardMarkup(get_main_keyboard())
                )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="To add a note, either include '#NOTE' in your text or use the 'Add Note' button and select a question.",
                reply_markup=InlineKeyboardMarkup(get_main_keyboard())
            )
        user_data['state'] = WRITING

async def get_added(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display currently added questions, answers, and notes."""
    user_id = update.effective_user.id
    user_data = get_user_data(user_id)
    if user_data['quizzes']:
        message_text = "Added Questions and Answers:\n\n"
        for i, quiz in enumerate(user_data['quizzes']):
            message_text += f"**Question {i + 1}:** {quiz['question']}\n"
            for j, choice in enumerate(quiz['choices']):
                message_text += f"   {chr(65 + j)}) {choice}\n"
            if 'answer' in quiz:
                message_text += f"   **Answer:** {chr(65 + quiz['answer'])}\n"
            if 'explanation' in quiz and quiz['explanation']:
                message_text += f"   **Note:** {quiz['explanation']}\n"
            message_text += "\n"
    else:
        message_text = "No questions and answers have been added yet."
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=message_text,
        parse_mode="Markdown"
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the current operation."""
    user_id = update.effective_user.id
    get_user_data(user_id)['state'] = WRITING
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Current operation cancelled.",
        reply_markup=InlineKeyboardMarkup(get_main_keyboard())
    )
    return WRITING

def get_main_keyboard():
    """Return the main inline keyboard."""
    return [
        [InlineKeyboardButton("Add Questions", callback_data="add_questions"),
         InlineKeyboardButton("Add Answers", callback_data="add_answers")],
        [InlineKeyboardButton("Add Note", callback_data="add_note")],
        [InlineKeyboardButton("Generate Quiz", callback_data="generate_quiz")],
        [InlineKeyboardButton("Reset Data", callback_data="reset_data"),
         InlineKeyboardButton("Get Added", callback_data="get_added")],
        [InlineKeyboardButton("Cancel", callback_data="cancel")]
    ]

def main() -> None:
    """Set up and run the bot."""
    application = ApplicationBuilder().token(bot_token).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WRITING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_data),
                MessageHandler(filters.PHOTO | filters.Document.PDF | filters.Document.IMAGE, receive_data),
                CallbackQueryHandler(handle_button),
            ],
            AWAITING_QUESTIONS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_data),
                CallbackQueryHandler(handle_button)
            ],
            AWAITING_ANSWERS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_data),
                CallbackQueryHandler(handle_button)
            ],
            AWAITING_NOTE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_data),
                CallbackQueryHandler(handle_button)
            ],
            OCR_CHOICE: [
                CallbackQueryHandler(handle_button),
            ],
            EDITED_TEXT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_edited_text)
            ],
            CONFIRM_QA: [
                CallbackQueryHandler(handle_button),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("generate", generate_quiz))
    application.add_handler(CommandHandler("reset", lambda update, context: reset_command(update, context)))
    application.add_handler(CommandHandler("getadded", get_added))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(CommandHandler("edit", edit_ocr_text))
    application.add_handler(CommandHandler("process", process_text))
    application.run_polling(allowed_updates=Update.ALL_TYPES)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /reset command."""
    user_id = update.effective_user.id
    update_user_data(user_id, 'quizzes', [])
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Quiz data reset.",
        reply_markup=InlineKeyboardMarkup(get_main_keyboard())
    )

if __name__ == "__main__":
    main()