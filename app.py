from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from concurrent.futures import ThreadPoolExecutor
import os
import re
import sqlite3
import time
import uuid
from local_engine import (
    process_voice_command,
    extract_text_from_pdf,
    extract_text_from_image,
    generate_summary,
    generate_audiobook,
    create_resume_chunks,
    set_resume_for_student,
    get_missing_dependencies,
    _resolve_piper_voice_paths,
    _any_piper_model_available,
)

app = Flask(__name__, instance_relative_config=True)
os.makedirs(app.instance_path, exist_ok=True)
app.secret_key = "offline-library-secret-key"
db_file = os.path.join(app.instance_path, 'OfflineLibrary.db')
root_db_file = os.path.join(app.root_path, 'OfflineLibrary.db')
if os.path.exists(root_db_file) and os.path.getsize(root_db_file) == 0:
    try:
        os.remove(root_db_file)
    except OSError:
        pass
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_file.replace(os.sep, '/')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
AUDIO_FOLDER = os.path.join(app.root_path, 'static', 'audiobooks')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

AUDIO_JOB_POOL = ThreadPoolExecutor(max_workers=2)
AUDIO_JOBS = {}


def _update_audio_job(job_id, status, percent=0, message='Queued', **extra):
    if not job_id:
        return
    current = AUDIO_JOBS.setdefault(job_id, {})
    current.update({'status': status, 'percent': int(percent), 'message': message})
    for key, value in extra.items():
        if value is not None:
            current[key] = value


def _job_payload(job_id):
    job = AUDIO_JOBS.get(job_id, {})
    percent = int(job.get('percent', 0) or 0)
    started_at = float(job.get('started_at') or 0)
    remaining_seconds = None
    if started_at and percent > 0 and percent < 100:
        elapsed = max(1.0, time.time() - started_at)
        estimated_total = elapsed * 100.0 / percent
        remaining_seconds = max(0, int(round(estimated_total - elapsed)))
    return {
        'job_id': job_id,
        'status': job.get('status', 'queued'),
        'percent': percent,
        'message': job.get('message', 'Waiting...'),
        'book_id': job.get('book_id'),
        'estimated_remaining_seconds': remaining_seconds,
    }


def _enqueue_book_audio_job(job_id, book_id, language, tasks):
    AUDIO_JOBS[job_id] = {
        'job_id': job_id,
        'book_id': book_id,
        'status': 'queued',
        'percent': 0,
        'message': 'Queued for audio generation',
        'started_at': time.time(),
    }

    def _run():
        total_tasks = max(1, len(tasks))
        for idx, task in enumerate(tasks, start=1):
            task_path = task['path']
            task_output = task['output']
            try:
                _update_audio_job(job_id, 'running', int((idx - 1) / total_tasks * 100), f"Generating {task['label']}...", book_id=book_id)

                def _progress_callback(payload):
                    local_percent = int(payload.get('percent', 0) or 0)
                    overall = int(((idx - 1) / total_tasks) * 100 + (local_percent / 100.0) * (100 / total_tasks))
                    _update_audio_job(
                        job_id,
                        payload.get('status', 'running'),
                        max(0, min(100, overall)),
                        payload.get('message', f"Generating {task['label']}..."),
                        book_id=book_id,
                    )

                generate_audiobook(task_path, task_output, preferred_language=language, progress_callback=_progress_callback)
                _update_audio_job(job_id, 'running', int((idx / total_tasks) * 100), f"Completed {task['label']}.", book_id=book_id)
            except Exception as exc:
                _update_audio_job(job_id, 'error', 100, f"Audio generation failed: {exc}", book_id=book_id)
                return
        _update_audio_job(job_id, 'complete', 100, 'Audio generation complete.', book_id=book_id)

    AUDIO_JOB_POOL.submit(_run)
    return job_id


class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    filepath = db.Column(db.String(250), nullable=False)
    syllabus_tag = db.Column(db.String(100), nullable=True)
    language = db.Column(db.String(20), nullable=True, default='en')


class BookChapter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False)
    chapter_number = db.Column(db.Integer, nullable=False)
    filepath = db.Column(db.String(250), nullable=False)
    audio_path = db.Column(db.String(250), nullable=True)
    book = db.relationship('Book', backref=db.backref('chapters', cascade='all, delete-orphan', lazy=True))


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(150), nullable=False)
    semester = db.Column(db.String(50), nullable=False)
    preferred_language = db.Column(db.String(20), nullable=True)
    interests = db.Column(db.String(250), nullable=True)
    last_book_id = db.Column(db.Integer, nullable=True)
    last_chapter = db.Column(db.Integer, nullable=True)
    last_position_type = db.Column(db.String(50), nullable=True)
    last_position_seconds = db.Column(db.Integer, nullable=True)


def _remove_all_book_audio(book_id):
    """Remove generated audiobook files for a book in either language."""
    for language in ('en', 'hi'):
        index_path = os.path.join(
            app.config['AUDIO_FOLDER'],
            f'book_{book_id}_{language}_index.mp3',
        )
        if os.path.exists(index_path):
            os.remove(index_path)

        for chapter in BookChapter.query.filter_by(book_id=book_id).all():
            chapter_path = os.path.join(
                app.config['AUDIO_FOLDER'],
                f'book_{book_id}_{language}_chapter_{chapter.chapter_number}.mp3',
            )
            if os.path.exists(chapter_path):
                os.remove(chapter_path)


with app.app_context():
    db.create_all()
    # Ensure Student table has newer columns added when upgrading older DBs.
    try:
        sqlite_path = db_file
        if os.path.exists(sqlite_path):
            with sqlite3.connect(sqlite_path) as conn:
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(student)")
                cols = [row[1] for row in cur.fetchall()]
                needed = [
                    ('preferred_language', 'TEXT'),
                    ('interests', 'TEXT'),
                    ('last_book_id', 'INTEGER'),
                    ('last_chapter', 'INTEGER'),
                    ('last_position_type', 'TEXT'),
                    ('last_position_seconds', 'INTEGER'),
                ]
                for name, _type in needed:
                    if name not in cols:
                        cur.execute(f"ALTER TABLE student ADD COLUMN {name} {_type}")
                if cols:
                    print('Existing student columns:', cols)
                else:
                    print('Student table created or empty schema.')
                book_cols = [row[1] for row in conn.execute("PRAGMA table_info(book)").fetchall()]
                if 'language' not in book_cols:
                    conn.execute("ALTER TABLE book ADD COLUMN language TEXT")
                conn.commit()
        else:
            print('SQLite file does not exist yet, skipping migration until created.')
    except Exception as e:
        print('Database migration error:', e)
    missing = get_missing_dependencies()
    if missing:
        print('WARNING: Missing external dependencies:', ', '.join(missing))
        print('Install Tesseract for OCR and FFmpeg for MP3 export. See README for setup instructions.')
    print(f"Database initialized. Audiobook generation will run on upload or on demand. Using DB: {db_file}")


@app.route("/health", methods=["GET"])
def health():
    missing = get_missing_dependencies()
    return jsonify({
        "ok": len(missing) == 0,
        "ffmpeg_installed": "FFmpeg" not in missing,
        "tesseract_installed": "Tesseract" not in missing,
        "piper_model_available": bool(_any_piper_model_available()),
        "english_model_available": bool(_resolve_piper_voice_paths('en')[0] and _resolve_piper_voice_paths('en')[1]),
        "hindi_model_available": bool(_resolve_piper_voice_paths('hi')[0] and _resolve_piper_voice_paths('hi')[1]),
        "missing_dependencies": missing,
        "database": db_file,
    })


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/summarize_book/<int:book_id>', methods=['GET'])
def summarize_book(book_id):
    book = Book.query.get_or_404(book_id)
    text_parts = []

    if os.path.exists(book.filepath):
        text_parts.append(extract_text_from_pdf(book.filepath))

    chapter_files = BookChapter.query.filter_by(book_id=book.id).order_by(BookChapter.chapter_number).all()
    for chapter in chapter_files:
        if os.path.exists(chapter.filepath):
            text_parts.append(extract_text_from_pdf(chapter.filepath))

    extracted_text = "\n\n".join([part for part in text_parts if part]).strip()
    if not extracted_text:
        return jsonify({"error": "No text found for this book."}), 404

    summary = generate_summary(extracted_text)
    return jsonify({
        "book_id": book.id,
        "title": book.title,
        "summary": summary,
    })


def _extract_student_profile(command):
    cmd = command.lower().strip()
    name_match = re.search(r"(?:my name is|i am|name is|call me)\s+([a-zA-Z][a-zA-Z .'-]{1,40})", cmd)
    dept_match = re.search(r"(?:department|dept)(?:\s+is|\s*:)?\s*([a-zA-Z][a-zA-Z0-9 .'-]{1,40})", cmd)
    sem_match = re.search(r"(?:semester|sem)(?:\s+is|\s*:)?\s*(\d{1,2}(?:st|nd|rd|th)?)", cmd)
    return {
        "name": name_match.group(1).strip().title() if name_match else None,
        "department": dept_match.group(1).strip().title() if dept_match else None,
        "semester": sem_match.group(1).strip() if sem_match else None,
    }


def _is_profile_update_command(command):
    if not command:
        return False
    cmd = command.lower()
    return bool(re.search(r"\b(my name is|name is|call me|i am|department|dept|semester|sem)\b", cmd))


def _get_profile_state():
    return session.get("student_profile_state") or {}


def _set_profile_state(state):
    session["student_profile_state"] = state


def _clear_profile_state():
    session.pop("student_profile_state", None)


@app.route("/voice_command", methods=["POST"])
def voice_command():
    data = request.json or {}
    command = (data.get("command") or "").strip()
    current_student = session.get('current_student')
    state = _get_profile_state()

    if not command:
        if not current_student:
            if not state:
                state = {"step": "name", "name": None, "department": None, "semester": None}
                _set_profile_state(state)
            if state.get("step") == "name":
                return jsonify({"chunks": [{"type": "normal", "text": "Hi , I am Vrinda , your assistant for digilibrary, Please tell me your name."}]})
            if state.get("step") == "department":
                return jsonify({"chunks": [{"type": "normal", "text": "Please tell me your department."}]})
            if state.get("step") == "semester":
                return jsonify({"chunks": [{"type": "normal", "text": "Please tell me your semester."}]})
        return jsonify({"chunks": [{"type": "normal", "text": "I didn't hear anything."}]})

    if current_student is None and (_is_profile_update_command(command) or state):
        if not state:
            state = {"step": "name", "name": None, "department": None, "semester": None}
        profile = _extract_student_profile(command)
        if profile.get("name"):
            state["name"] = profile["name"]
        if profile.get("department"):
            state["department"] = profile["department"]
        if profile.get("semester"):
            state["semester"] = profile["semester"]
        _set_profile_state(state)

        if state.get("step") == "name":
            if state.get("name"):
                existing = Student.query.filter_by(name=state["name"]).first()
                if existing:
                    session["current_student"] = existing.name
                    set_resume_for_student(existing)
                    books = Book.query.all()
                    resume_chunks = create_resume_chunks(existing, books)
                    if resume_chunks:
                        return jsonify({"chunks": resume_chunks})
                state["step"] = "department"
                _set_profile_state(state)
                return jsonify({"chunks": [{"type": "normal", "text": "Now tell me your department."}]})
            return jsonify({"chunks": [{"type": "normal", "text": "Hi , I am Vrinda , your assistant for digilibrary, Please tell me your name."}]})

        if state.get("step") == "department":
            if state.get("department"):
                state["step"] = "semester"
                _set_profile_state(state)
                return jsonify({"chunks": [{"type": "normal", "text": "Please tell me your semester."}]})
            return jsonify({"chunks": [{"type": "normal", "text": "Please tell me your department."}]})

        if state.get("step") == "semester":
            if state.get("semester"):
                student = Student.query.filter_by(name=state["name"]).first()
                if student:
                    student.department = state["department"]
                    student.semester = state["semester"]
                    student.preferred_language = session.get('preferred_language') or student.preferred_language
                    session['current_student'] = student.name
                else:
                    db.session.add(Student(name=state["name"], department=state["department"], semester=state["semester"]))
                    session['current_student'] = state["name"]
                db.session.commit()
                _clear_profile_state()
                return jsonify({"chunks": [{"type": "normal", "text": "Thank you. Your profile has been saved. You can now use regular commands."}]})
            return jsonify({"chunks": [{"type": "normal", "text": "Hi , I am Vrinda , your assistant for digilibrary, Please tell me your name."}]})

        # If the command looked like profile input but we still couldn't continue, use normal flow instead.
        if not _is_profile_update_command(command):
            _clear_profile_state()

    if state and current_student is None and not _is_profile_update_command(command):
        _clear_profile_state()

    books = Book.query.all()
    book_chapters = BookChapter.query.all()
    # attach current student profile if available
    student_obj = None
    current_name = session.get('current_student')
    if current_name:
        student_obj = Student.query.filter_by(name=current_name).first()
    student_profile = None
    if student_obj:
        student_profile = {
            'name': student_obj.name,
            'department': student_obj.department,
            'semester': student_obj.semester,
            'preferred_language': student_obj.preferred_language,
            'interests': student_obj.interests,
            'last_book_id': student_obj.last_book_id,
            'last_chapter': student_obj.last_chapter,
            'last_position_type': student_obj.last_position_type,
        }
    response_chunks = process_voice_command(command, books, book_chapters, student_profile)
    return jsonify({"chunks": response_chunks})


@app.route('/audio_job_status/<job_id>', methods=['GET'])
def audio_job_status(job_id):
    return jsonify(_job_payload(job_id))


@app.route('/books', methods=['GET', 'POST'])
def manage_books():
    if request.method == 'POST':
        if request.form.get('student_profile') == '1':
            student_id = request.form.get('student_id')
            student_name = request.form.get('student_name', '').strip()
            department = request.form.get('department', '').strip()
            semester = request.form.get('semester', '').strip()
            preferred_language = request.form.get('preferred_language') or None
            interests = request.form.get('interests') or None
            if student_name and department and semester:
                existing_student = None
                if student_id:
                    existing_student = Student.query.get(int(student_id))
                if not existing_student:
                    existing_student = Student.query.filter_by(name=student_name).first()
                if existing_student:
                    existing_student.name = student_name
                    existing_student.department = department
                    existing_student.semester = semester
                    existing_student.preferred_language = preferred_language or existing_student.preferred_language
                    existing_student.interests = interests or existing_student.interests
                else:
                    db.session.add(Student(name=student_name, department=department, semester=semester, preferred_language=preferred_language, interests=interests))
                db.session.commit()
            return redirect(url_for('manage_books'))

        book_id = request.form.get('book_id')
        title = request.form.get('title', '').strip()
        author = request.form.get('author', '').strip()
        syllabus = request.form.get('syllabus_tag', '').strip()
        language = request.form.get('language', 'en') or 'en'
        chapter_count = int(request.form.get('chapter_count', '0') or 0)
        remove_chapter_ids = [int(x) for x in (request.form.get('remove_chapter_ids') or '').split(',') if x.strip().isdigit()]
        index_file = request.files.get('index_file')

        def _get_safe_name_from_book(book):
            if not book or not book.filepath:
                return f'book_{uuid.uuid4().hex[:8]}'
            base = os.path.splitext(os.path.basename(book.filepath))[0]
            if base.endswith('_index'):
                base = base[:-6]
            return re.sub(r'[^A-Za-z0-9_-]+', '_', base).strip('_') or f'book_{book.id}'

        def _book_audio_language(book_or_lang):
            if hasattr(book_or_lang, 'language'):
                return str(getattr(book_or_lang, 'language') or 'en').strip().lower()
            return str(book_or_lang or 'en').strip().lower()

        def _audio_file_for_book(book_id, item_type, language='en', chapter_number=None):
            language = 'hi' if str(language or 'en').strip().lower().startswith('hi') else 'en'
            if item_type == 'index':
                return os.path.join(app.config['AUDIO_FOLDER'], f'book_{book_id}_{language}_index.mp3')
            if item_type == 'chapter':
                return os.path.join(app.config['AUDIO_FOLDER'], f'book_{book_id}_{language}_chapter_{chapter_number}.mp3')
            return None

        def _remove_language_audio_for_book(book_id, language='en'):
            language = 'hi' if str(language or 'en').strip().lower().startswith('hi') else 'en'
            files = [
                os.path.join(app.config['AUDIO_FOLDER'], f'book_{book_id}_{language}_index.mp3'),
            ]
            for chapter in BookChapter.query.filter_by(book_id=book_id).all():
                files.append(os.path.join(app.config['AUDIO_FOLDER'], f'book_{book_id}_{language}_chapter_{chapter.chapter_number}.mp3'))
            for file_path in files:
                if os.path.exists(file_path):
                    os.remove(file_path)

        def _remove_old_language_audio_for_book(book_id, previous_language, new_language):
            previous_lang = 'hi' if str(previous_language or 'en').strip().lower().startswith('hi') else 'en'
            new_lang = 'hi' if str(new_language or 'en').strip().lower().startswith('hi') else 'en'
            if previous_lang == new_lang:
                return
            for suffix in [previous_lang, new_lang]:
                index_path = os.path.join(app.config['AUDIO_FOLDER'], f'book_{book_id}_{suffix}_index.mp3')
                if os.path.exists(index_path):
                    os.remove(index_path)
                for chapter in BookChapter.query.filter_by(book_id=book_id).all():
                    chapter_path = os.path.join(app.config['AUDIO_FOLDER'], f'book_{book_id}_{suffix}_chapter_{chapter.chapter_number}.mp3')
                    if os.path.exists(chapter_path):
                        os.remove(chapter_path)

        if book_id:
            existing_book = Book.query.get(int(book_id))
            if existing_book:
                previous_language = existing_book.language or 'en'
                language_changed = existing_book.language != language
                existing_book.title = title or existing_book.title
                existing_book.author = author or existing_book.author
                existing_book.syllabus_tag = syllabus or existing_book.syllabus_tag
                existing_book.language = language or existing_book.language
                if language_changed:
                    _remove_old_language_audio_for_book(existing_book.id, previous_language, language)
                index_updated = False
                if index_file and index_file.filename.lower().endswith('.pdf'):
                    index_file.save(existing_book.filepath)
                    index_updated = True
                db.session.commit()
                if not (_resolve_piper_voice_paths(language)[0] and _resolve_piper_voice_paths(language)[1]):
                    session['books_status_message'] = f"{language.upper()} Piper model is missing. The book was saved, but audio generation was skipped. Install the {language} model in model/ to use this audio language."
                    return redirect(url_for('manage_books'))

                for chapter_id in remove_chapter_ids:
                    chapter = BookChapter.query.filter_by(id=chapter_id, book_id=existing_book.id).first()
                    if chapter:
                        if os.path.exists(chapter.filepath):
                            os.remove(chapter.filepath)
                        if chapter.audio_path and os.path.exists(chapter.audio_path):
                            os.remove(chapter.audio_path)
                        db.session.delete(chapter)
                db.session.commit()

                existing_numbers = [chapter.chapter_number for chapter in existing_book.chapters]
                next_chapter_number = max(existing_numbers, default=0) + 1
                safe_name = _get_safe_name_from_book(existing_book)
                new_chapter_entries = []
                for i in range(1, chapter_count + 1):
                    chapter_file = request.files.get(f'chapter_file_{i}')
                    if not chapter_file or not chapter_file.filename.lower().endswith('.pdf'):
                        continue
                    chapter_filename = f"{safe_name}_chapter_{next_chapter_number}.pdf"
                    chapter_path = os.path.join(app.config['UPLOAD_FOLDER'], chapter_filename)
                    chapter_file.save(chapter_path)
                    chapter_entry = BookChapter(
                        book_id=existing_book.id,
                        chapter_number=next_chapter_number,
                        filepath=chapter_path,
                        audio_path=_audio_file_for_book(existing_book.id, 'chapter', existing_book.language, next_chapter_number),
                    )
                    db.session.add(chapter_entry)
                    new_chapter_entries.append(chapter_entry)
                    next_chapter_number += 1
                db.session.commit()

                if os.path.exists(existing_book.filepath) and (language_changed or index_updated):
                    index_audio_path = _audio_file_for_book(existing_book.id, 'index', existing_book.language)
                    _remove_language_audio_for_book(existing_book.id, existing_book.language)
                    generate_audiobook(existing_book.filepath, index_audio_path, preferred_language=existing_book.language)

                if language_changed:
                    for chapter_entry in existing_book.chapters:
                        if os.path.exists(chapter_entry.filepath):
                            generate_audiobook(chapter_entry.filepath, chapter_entry.audio_path, preferred_language=existing_book.language)
                else:
                    for chapter_entry in new_chapter_entries:
                        if os.path.exists(chapter_entry.filepath):
                            generate_audiobook(chapter_entry.filepath, chapter_entry.audio_path, preferred_language=existing_book.language)

            return redirect(url_for('manage_books'))

        if title and author and index_file and index_file.filename.lower().endswith('.pdf') and chapter_count > 0:
            if not (_resolve_piper_voice_paths(language)[0] and _resolve_piper_voice_paths(language)[1]):
                session['books_status_message'] = f"{language.upper()} Piper model is missing. The book was saved, but audio generation was skipped. Install the {language} model in model/ to use this audio language."
                safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', title).strip('_') or f'book_{uuid.uuid4().hex[:8]}'
                index_filename = f"{safe_name}_index.pdf"
                index_path = os.path.join(app.config['UPLOAD_FOLDER'], index_filename)
                index_file.save(index_path)

                new_book = Book(title=title, author=author, filepath=index_path, syllabus_tag=syllabus, language=language)
                db.session.add(new_book)
                db.session.commit()

                for chapter_number in range(1, chapter_count + 1):
                    chapter_file = request.files.get(f'chapter_file_{chapter_number}')
                    if not chapter_file or not chapter_file.filename.lower().endswith('.pdf'):
                        continue

                    chapter_filename = f"{safe_name}_chapter_{chapter_number}.pdf"
                    chapter_path = os.path.join(app.config['UPLOAD_FOLDER'], chapter_filename)
                    chapter_file.save(chapter_path)

                    chapter_entry = BookChapter(
                        book_id=new_book.id,
                        chapter_number=chapter_number,
                        filepath=chapter_path,
                        audio_path=_audio_file_for_book(new_book.id, 'chapter', language, chapter_number),
                    )
                    db.session.add(chapter_entry)

                db.session.commit()
                return redirect(url_for('manage_books'))

            safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', title).strip('_') or f'book_{uuid.uuid4().hex[:8]}'
            index_filename = f"{safe_name}_index.pdf"
            index_path = os.path.join(app.config['UPLOAD_FOLDER'], index_filename)
            index_file.save(index_path)

            new_book = Book(title=title, author=author, filepath=index_path, syllabus_tag=syllabus, language=language)
            db.session.add(new_book)
            db.session.commit()

            index_audio_path = _audio_file_for_book(new_book.id, 'index', language)
            _remove_language_audio_for_book(new_book.id, language)

            tasks = [{'label': 'index', 'path': index_path, 'output': index_audio_path}]
            for chapter_number in range(1, chapter_count + 1):
                chapter_file = request.files.get(f'chapter_file_{chapter_number}')
                if not chapter_file or not chapter_file.filename.lower().endswith('.pdf'):
                    continue

                chapter_filename = f"{safe_name}_chapter_{chapter_number}.pdf"
                chapter_path = os.path.join(app.config['UPLOAD_FOLDER'], chapter_filename)
                chapter_file.save(chapter_path)

                chapter_entry = BookChapter(
                    book_id=new_book.id,
                    chapter_number=chapter_number,
                    filepath=chapter_path,
                    audio_path=_audio_file_for_book(new_book.id, 'chapter', language, chapter_number),
                )
                db.session.add(chapter_entry)
                tasks.append({'label': f'chapter {chapter_number}', 'path': chapter_path, 'output': chapter_entry.audio_path})

            db.session.commit()
            session['audio_job_id'] = _enqueue_book_audio_job(str(uuid.uuid4()), new_book.id, language, tasks)

        return redirect(url_for('manage_books'))

    status_message = session.pop('books_status_message', None)
    all_books = Book.query.all()
    all_students = Student.query.order_by(Student.name).all()
    active_job_id = session.get('audio_job_id')
    audio_job = _job_payload(active_job_id) if active_job_id else None
    if audio_job and audio_job['status'] in ('complete', 'error'):
        session.pop('audio_job_id', None)
        active_job_id = None
    return render_template(
        'books.html',
        books=all_books,
        students=all_students,
        status_message=status_message,
        english_model_available=bool(_resolve_piper_voice_paths('en')[0] and _resolve_piper_voice_paths('en')[1]),
        hindi_model_available=bool(_resolve_piper_voice_paths('hi')[0] and _resolve_piper_voice_paths('hi')[1]),
        audio_job_id=active_job_id,
        audio_job=audio_job,
        audio_available={
            book.id: os.path.exists(
                os.path.join(
                    app.config['AUDIO_FOLDER'],
                    f"book_{book.id}_{'hi' if str(book.language or 'en').lower().startswith('hi') else 'en'}_index.mp3",
                )
            )
            for book in all_books
        },
    )


@app.route('/delete_student/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    student = Student.query.get_or_404(student_id)
    db.session.delete(student)
    db.session.commit()
    return redirect(url_for('manage_books'))


@app.route('/delete_book/<int:book_id>', methods=['POST'])
def delete_book(book_id):
    book_to_delete = Book.query.get_or_404(book_id)

    if os.path.exists(book_to_delete.filepath):
        os.remove(book_to_delete.filepath)

    for chapter in book_to_delete.chapters:
        if os.path.exists(chapter.filepath):
            os.remove(chapter.filepath)
        if chapter.audio_path and os.path.exists(chapter.audio_path):
            os.remove(chapter.audio_path)

    _remove_all_book_audio(book_to_delete.id)

    db.session.delete(book_to_delete)
    db.session.commit()

    return redirect(url_for('manage_books'))


@app.route('/save_progress', methods=['POST'])
def save_progress():
    data = request.json or {}
    current_name = session.get('current_student')
    if not current_name:
        return jsonify({'error': 'no current student in session'}), 400
    student = Student.query.filter_by(name=current_name).first()
    if not student:
        return jsonify({'error': 'student not found'}), 404

    try:
        book_id = data.get('book_id')
        chapter = data.get('chapter')
        position_type = data.get('position_type')
        position_seconds = data.get('position_seconds')

        if book_id is not None:
            student.last_book_id = int(book_id)
        if chapter is not None and chapter != '':
            try:
                student.last_chapter = int(chapter)
            except Exception:
                student.last_chapter = None
        else:
            student.last_chapter = None
        student.last_position_type = position_type or student.last_position_type
        if position_seconds is not None and position_seconds != '':
            try:
                student.last_position_seconds = int(position_seconds)
            except Exception:
                student.last_position_seconds = None
        db.session.commit()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)