"""
StudyBuddy: trợ lý ôn tập AI mã nguồn mở (Gradio + mô hình Llama qua Groq hoặc Ollama)

Tính năng:
- Quiz trắc nghiệm tạo từ chủ đề, từ văn bản dán vào hoặc từ file .txt/.md/.pdf.
  Với tài liệu, mỗi câu kèm trích dẫn nguyên văn và được kiểm tra có thật trong tài liệu hay không.
- Chấm điểm, giải thích, ôn lại câu sai, tải báo cáo kết quả (.md).
- Theo dõi tiến độ theo chủ đề nhỏ và chỉ ra chủ đề còn yếu.
- Lập kế hoạch học tập dựa trên mục tiêu và kết quả quiz.
- Hỏi đáp với trợ lý.
- Chạy được cả khi không có AI: bộ câu hỏi mẫu và kế hoạch mẫu vẫn dùng bình thường.
"""
import json
import os
import random
import re
import tempfile
import threading
import time
import unicodedata
from collections import deque
from datetime import datetime

import gradio as gr
from openai import OpenAI
from pypdf import PdfReader

# ======================================================================
# CẤU HÌNH
# ======================================================================
SUBJECT = os.getenv("SUBJECT", "Khoa học tự nhiên")
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "60"))  # bảo vệ khóa khi chia sẻ link

MAX_HISTORY = 6
MAX_INPUT_CHARS = 2000
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_DOC_CHARS = 300_000
MAX_SOURCE_CHARS = 12_000  # lượng chữ gửi cho AI mỗi lần tạo quiz
MAX_PDF_PAGES = 40
LETTERS = "ABCD"
WEAK_THRESHOLD = 0.7

DIFFICULTY_HINTS = {
    "Dễ": "ghi nhớ kiến thức cơ bản, câu hỏi ngắn gọn",
    "Vừa": "hiểu và vận dụng đơn giản, có một chút suy luận",
    "Khó": "vận dụng, phân tích, tình huống thực tế, phương án gây nhiễu hợp lý",
}


def load_provider():
    """Chọn nhà cung cấp AI: tùy chỉnh > Groq > Ollama (chạy trên máy)."""
    if os.getenv("LLM_BASE_URL"):
        return (
            os.environ["LLM_BASE_URL"],
            os.getenv("LLM_API_KEY", "none"),
            os.getenv("LLM_MODEL", "llama3.2"),
            "Tùy chỉnh",
        )
    if os.getenv("GROQ_API_KEY"):
        return (
            "https://api.groq.com/openai/v1",
            os.environ["GROQ_API_KEY"],
            os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            "Groq (đám mây)",
        )
    return (
        "http://localhost:11434/v1",
        "ollama",  # Ollama không cần khóa thật
        os.getenv("LLM_MODEL", "llama3.2"),
        "Ollama (chạy trên máy)",
    )


BASE_URL, API_KEY, MODEL, PROVIDER = load_provider()
llm = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=90, max_retries=1)

CHAT_SYSTEM = (
    f"Bạn là trợ lý ôn tập môn {SUBJECT} cho học sinh. Trả lời bằng tiếng Việt, "
    "giải thích dễ hiểu, có ví dụ đời sống khi phù hợp. Nếu không chắc chắn, hãy nói thẳng "
    "thay vì đoán."
)
PLAN_SYSTEM = (
    "Bạn là cố vấn học tập. Hãy lập kế hoạch bằng tiếng Việt, định dạng Markdown, có cấu trúc: "
    "mục tiêu, lịch theo tuần (và theo ngày nếu phù hợp), phương pháp ôn tập, cách tự kiểm tra "
    "tiến độ, lưu ý. Nếu thiếu thông tin, hãy đưa ra giả định hợp lý và nêu rõ."
)

# ======================================================================
# BỘ CÂU HỎI MẪU (dùng khi không có AI)
# ======================================================================
SAMPLE_BANK = [
    {"tag": "Vật lí", "question": "Đơn vị đo lực trong hệ SI là gì?",
     "options": ["Kilogram", "Joule", "Newton", "Watt"], "answer": 2,
     "explanation": "Lực đo bằng newton (N). Kilogram là đơn vị khối lượng, joule là năng lượng, watt là công suất."},
    {"tag": "Vật lí", "question": "Ánh sáng truyền trong chân không với tốc độ xấp xỉ bao nhiêu?",
     "options": ["340 m/s", "300 000 km/s", "30 000 km/s", "3 000 km/s"], "answer": 1,
     "explanation": "Tốc độ ánh sáng trong chân không khoảng 300 000 km/s. 340 m/s là tốc độ âm thanh trong không khí."},
    {"tag": "Vật lí", "question": "Dụng cụ nào dùng để đo cường độ dòng điện?",
     "options": ["Vôn kế", "Ampe kế", "Nhiệt kế", "Lực kế"], "answer": 1,
     "explanation": "Ampe kế đo cường độ dòng điện, còn vôn kế đo hiệu điện thế."},
    {"tag": "Vật lí", "question": "Vật liệu nào sau đây dẫn điện tốt?",
     "options": ["Nhựa", "Cao su", "Đồng", "Gỗ khô"], "answer": 2,
     "explanation": "Kim loại như đồng dẫn điện tốt; nhựa, cao su và gỗ khô là chất cách điện."},
    {"tag": "Hóa học", "question": "Công thức hóa học của nước là gì?",
     "options": ["CO2", "NaCl", "H2O", "O2"], "answer": 2,
     "explanation": "Mỗi phân tử nước gồm 2 nguyên tử hydrogen và 1 nguyên tử oxygen."},
    {"tag": "Hóa học", "question": "Khí nào chiếm tỉ lệ thể tích lớn nhất trong không khí?",
     "options": ["Oxygen", "Carbon dioxide", "Argon", "Nitrogen"], "answer": 3,
     "explanation": "Nitrogen chiếm khoảng 78% thể tích không khí, oxygen khoảng 21%."},
    {"tag": "Hóa học", "question": "Chất nào sau đây là đơn chất?",
     "options": ["Nước (H2O)", "Khí oxygen (O2)", "Muối ăn (NaCl)", "Khí carbon dioxide (CO2)"], "answer": 1,
     "explanation": "Đơn chất do một nguyên tố hóa học tạo nên. O2 chỉ có oxygen, còn các chất còn lại là hợp chất."},
    {"tag": "Hóa học", "question": "Dung dịch có pH nhỏ hơn 7 có tính chất gì?",
     "options": ["Tính acid", "Tính base", "Trung tính", "Tính muối"], "answer": 0,
     "explanation": "pH nhỏ hơn 7 là acid, bằng 7 là trung tính, lớn hơn 7 là base."},
    {"tag": "Sinh học", "question": "Quá trình cây xanh dùng ánh sáng để tạo chất hữu cơ từ nước và khí carbon dioxide gọi là gì?",
     "options": ["Hô hấp", "Quang hợp", "Thoát hơi nước", "Tiêu hóa"], "answer": 1,
     "explanation": "Quang hợp tạo ra chất hữu cơ và giải phóng khí oxygen."},
    {"tag": "Sinh học", "question": "Bào quan nào là nơi diễn ra quang hợp ở tế bào thực vật?",
     "options": ["Ti thể", "Lục lạp", "Nhân", "Không bào"], "answer": 1,
     "explanation": "Lục lạp chứa diệp lục, hấp thụ ánh sáng để quang hợp."},
    {"tag": "Sinh học", "question": "Cơ quan nào bơm máu đi khắp cơ thể người?",
     "options": ["Phổi", "Gan", "Tim", "Thận"], "answer": 2,
     "explanation": "Tim co bóp để đẩy máu lưu thông trong hệ tuần hoàn."},
    {"tag": "Sinh học", "question": "Đơn vị cấu tạo và chức năng cơ bản của cơ thể sống là gì?",
     "options": ["Mô", "Cơ quan", "Tế bào", "Hệ cơ quan"], "answer": 2,
     "explanation": "Mọi cơ thể sống đều được cấu tạo từ tế bào, tế bào thực hiện các hoạt động sống cơ bản."},
    {"tag": "Trái Đất", "question": "Trái Đất tự quay quanh trục một vòng hết khoảng bao lâu?",
     "options": ["12 giờ", "24 giờ", "7 ngày", "365 ngày"], "answer": 1,
     "explanation": "Trái Đất tự quay một vòng quanh trục hết khoảng 24 giờ, tức một ngày đêm."},
    {"tag": "Trái Đất", "question": "Hiện tượng ngày và đêm luân phiên là do đâu?",
     "options": ["Trái Đất quay quanh Mặt Trời", "Trái Đất tự quay quanh trục",
                 "Mặt Trăng che Mặt Trời", "Trục Trái Đất nghiêng"], "answer": 1,
     "explanation": "Do Trái Đất tự quay quanh trục nên mỗi thời điểm chỉ một nửa được Mặt Trời chiếu sáng."},
]

# ======================================================================
# GỌI MÔ HÌNH AI
# ======================================================================
_calls = deque()
_lock = threading.Lock()


class UserFacingError(Exception):
    """Lỗi có thông báo thân thiện cho người dùng."""


def check_rate_limit():
    now = time.time()
    with _lock:
        while _calls and now - _calls[0] > 3600:
            _calls.popleft()
        if len(_calls) >= RATE_LIMIT:
            raise UserFacingError(
                "Hệ thống đang nhận quá nhiều yêu cầu, bạn thử lại sau ít phút nhé. "
                "Trong lúc chờ, vẫn dùng được bộ câu hỏi mẫu và kế hoạch mẫu."
            )
        _calls.append(now)


def ask_llm(messages, json_mode=False, max_tokens=2000, temperature=0.7) -> str:
    check_rate_limit()
    kwargs = dict(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
    try:
        if json_mode:
            response = llm.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        else:
            response = llm.chat.completions.create(**kwargs)
    except Exception:
        if not json_mode:
            raise
        response = llm.chat.completions.create(**kwargs)  # một số mô hình không hỗ trợ JSON mode
    return response.choices[0].message.content or ""


def friendly_error(e: Exception) -> str:
    if isinstance(e, UserFacingError):
        return f"⚠️ {e}"
    print(f"[Lỗi] {type(e).__name__}: {e}")
    name = type(e).__name__
    if "Connection" in name or "Timeout" in name:
        if PROVIDER.startswith("Ollama"):
            return "⚠️ Không kết nối được Ollama. Hãy mở ứng dụng Ollama và chạy `ollama pull llama3.2`."
        return "⚠️ Không kết nối được dịch vụ AI. Hãy kiểm tra mạng rồi thử lại."
    if "Authentication" in name or "PermissionDenied" in name:
        return "⚠️ Khóa API không hợp lệ hoặc không có quyền. Hãy kiểm tra lại GROQ_API_KEY."
    if "RateLimit" in name:
        return "⚠️ Dịch vụ AI đang giới hạn tần suất, bạn thử lại sau ít phút nhé."
    if "NotFound" in name:
        return f"⚠️ Không tìm thấy mô hình `{MODEL}`. Hãy đặt biến LLM_MODEL thành một mô hình đang có."
    return "⚠️ Đã xảy ra lỗi, bạn thử lại nhé!"


def test_ai() -> str:
    start = time.time()
    try:
        ask_llm([{"role": "user", "content": "Trả lời đúng một từ: OK"}], max_tokens=10, temperature=0)
    except Exception as e:
        return friendly_error(e)
    ms = int((time.time() - start) * 1000)
    return f"✅ AI hoạt động ({PROVIDER}, `{MODEL}`, {ms} ms)"


def as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(as_text(c) for c in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


def write_temp(text: str, prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ======================================================================
# HỎI ĐÁP
# ======================================================================
def chat_fn(message, history):
    recent = [
        {"role": m["role"], "content": as_text(m["content"])}
        for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ][-MAX_HISTORY:]
    while recent and recent[0]["role"] != "user":
        recent.pop(0)
    try:
        return ask_llm(
            [
                {"role": "system", "content": CHAT_SYSTEM},
                *recent,
                {"role": "user", "content": as_text(message)[:MAX_INPUT_CHARS]},
            ]
        )
    except Exception as e:
        return friendly_error(e)


# ======================================================================
# TÀI LIỆU VÀ XÁC MINH TRÍCH DẪN
# ======================================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip().casefold()


def verify_evidence(evidence: str, source: str) -> bool:
    e = _norm(evidence)
    return len(e) >= 10 and e in _norm(source)


def read_file(path: str) -> str:
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise UserFacingError("File quá lớn (tối đa 5 MB).")
    lower = path.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(path)
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:MAX_PDF_PAGES])
    elif lower.endswith((".txt", ".md")):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise UserFacingError("Chỉ hỗ trợ file .txt, .md hoặc .pdf.")
    text = text.strip()
    if not text:
        raise UserFacingError("Không đọc được nội dung file (có thể là PDF dạng ảnh scan).")
    return text[:MAX_DOC_CHARS]


def sample_source(text: str, limit: int = MAX_SOURCE_CHARS) -> str:
    """Tài liệu dài: lấy 4 đoạn rải đều để câu hỏi bao quát cả tài liệu."""
    if len(text) <= limit:
        return text
    parts = 4
    size = limit // parts
    step = (len(text) - size) // (parts - 1)
    return "\n[...]\n".join(text[i * step : i * step + size] for i in range(parts))


# ======================================================================
# TẠO VÀ KIỂM TRA CÂU HỎI
# ======================================================================
def normalize_question(q: dict):
    """Kiểm tra một câu hỏi và xáo trộn đáp án; trả None nếu không hợp lệ."""
    try:
        options = [str(o).strip() for o in q["options"]]
        answer = int(q["answer"])
        question = str(q["question"]).strip()
        if (
            len(options) != 4
            or not all(options)
            or len({o.casefold() for o in options}) != 4
            or not 0 <= answer < 4
            or not question
        ):
            return None
        correct_text = options[answer]
        random.shuffle(options)
        return {
            "question": question,
            "options": options,
            "answer": options.index(correct_text),
            "explanation": str(q.get("explanation", "")).strip(),
            "tag": str(q.get("tag", "")).strip()[:40],
            "evidence": str(q.get("evidence", "")).strip()[:300],
            "verified": q.get("verified"),
        }
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def parse_questions(raw: str) -> list:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Không tìm thấy JSON")
    data = json.loads(raw[start : end + 1])
    result = []
    for q in data.get("questions", []):
        if isinstance(q, dict):
            q = dict(q)
            q.pop("verified", None)  # không tin giá trị do AI tự khai
            item = normalize_question(q)
            if item:
                result.append(item)
    return result


def generate_questions(topic: str, count: int, difficulty: str, source_text) -> list:
    if source_text:
        context = (
            "Chỉ dựa trên tài liệu sau, không dùng kiến thức bên ngoài:\n"
            f'"""\n{sample_source(source_text)}\n"""'
        )
        evidence_rule = '"evidence": đoạn trích NGUYÊN VĂN (10 đến 200 ký tự) từ tài liệu chứng minh đáp án đúng.'
    else:
        context = f'Môn học: {SUBJECT}. Chủ đề: "{topic or SUBJECT}"'
        evidence_rule = '"evidence": để chuỗi rỗng.'

    prompt = f"""{context}

Hãy tạo {count} câu hỏi trắc nghiệm bằng tiếng Việt. Mức độ: {difficulty} ({DIFFICULTY_HINTS.get(difficulty, "")}).
Yêu cầu:
- Mỗi câu có đúng 4 lựa chọn khác nhau và đúng 1 đáp án đúng; các câu không trùng ý nhau.
- "explanation": giải thích ngắn gọn, KHÔNG nhắc tới chữ cái A/B/C/D.
- "tag": tên chủ đề nhỏ của câu hỏi (1 đến 4 từ).
- {evidence_rule}
Chỉ trả về JSON đúng định dạng sau, không thêm chữ nào khác:
{{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."], "answer": 0, "explanation": "...", "tag": "...", "evidence": "..."}}]}}
"answer" là chỉ số (0 đến 3) của đáp án đúng trong "options"."""

    last_error = None
    for _ in range(2):  # thử tối đa 2 lần
        try:
            raw = ask_llm(
                [
                    {"role": "system", "content": "Bạn chỉ trả về JSON hợp lệ."},
                    {"role": "user", "content": prompt},
                ],
                json_mode=True,
                max_tokens=3500,
                temperature=0.5,
            )
            questions = parse_questions(raw)
            if questions:
                if source_text:
                    norm_source = _norm(source_text)
                    for q in questions:
                        e = _norm(q["evidence"])
                        q["verified"] = len(e) >= 10 and e in norm_source
                return questions[:count]
        except UserFacingError:
            raise
        except Exception as e:
            last_error = e
            print(f"[Lỗi tạo quiz] {type(e).__name__}: {e}")
    if last_error is not None and any(k in type(last_error).__name__ for k in ("Connection", "Timeout", "Authentication", "NotFound")):
        raise last_error
    raise UserFacingError("Không tạo được câu hỏi hợp lệ, bạn thử lại hoặc dùng bộ câu hỏi mẫu nhé.")


# ======================================================================
# TIẾN ĐỘ
# ======================================================================
def new_progress() -> dict:
    return {"answers": [], "quizzes": []}


def ensure_progress(p) -> dict:
    return p if isinstance(p, dict) and "answers" in p and "quizzes" in p else new_progress()


def topic_stats(progress: dict) -> dict:
    stats = {}
    for a in progress["answers"]:
        s = stats.setdefault(a["tag"], [0, 0])
        s[1] += 1
        s[0] += 1 if a["ok"] else 0
    return stats


def weak_topics(progress: dict, k: int = 5) -> list:
    ranked = sorted(
        (c / t, -t, tag) for tag, (c, t) in topic_stats(progress).items() if c / t < WEAK_THRESHOLD
    )
    return [tag for _, _, tag in ranked[:k]]


def progress_md(progress) -> str:
    p = ensure_progress(progress)
    if not p["answers"]:
        return "Bạn chưa làm câu nào. Hãy làm một quiz ở tab **Quiz ôn tập** để xem thống kê tại đây."
    total = len(p["answers"])
    correct = sum(1 for a in p["answers"] if a["ok"])
    lines = [
        "### Tổng quan",
        f"- Số bài quiz đã hoàn thành: **{len(p['quizzes'])}**",
        f"- Số câu đã trả lời: **{total}**, đúng **{correct}** ({correct / total:.0%})",
        "",
        "### Theo chủ đề",
        "| Chủ đề | Đúng/Tổng | Tỉ lệ |",
        "|---|---|---|",
    ]
    for tag, (c, t) in sorted(topic_stats(p).items(), key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1])):
        lines.append(f"| {tag} | {c}/{t} | {c / t:.0%} |")
    weak = weak_topics(p)
    lines.append("")
    if weak:
        lines.append("**Nên ôn thêm:** " + ", ".join(weak))
    else:
        lines.append("**Tuyệt vời!** Chưa có chủ đề nào dưới 70%.")
    return "\n".join(lines)


def show_progress(progress):
    return progress_md(progress)


def reset_progress():
    return None, progress_md(None)


# ======================================================================
# QUIZ: LUỒNG LÀM BÀI
# ======================================================================
def hide():
    return gr.update(visible=False)


def pack(state, question="", choices=None, feedback="", check=None, nxt=None, retry=None, file=None):
    """Gói kết quả theo đúng thứ tự các thành phần đầu ra của quiz."""
    return (
        state,
        question,
        choices if choices is not None else hide(),
        feedback,
        check if check is not None else hide(),
        nxt if nxt is not None else hide(),
        retry if retry is not None else hide(),
        file if file is not None else hide(),
    )


def new_quiz_state(questions: list, topic: str) -> dict:
    return {"questions": questions, "index": 0, "score": 0, "topic": topic, "log": [], "answered": False}


def labels_of(q: dict) -> list:
    return [f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"])]


def show_question(state: dict):
    q = state["questions"][state["index"]]
    text = f"### Câu {state['index'] + 1}/{len(state['questions'])}\n{q['question']}"
    radio = gr.update(choices=labels_of(q), value=None, visible=True, interactive=True)
    return text, radio


def begin(state: dict):
    question_md, radio = show_question(state)
    return pack(state, question_md, radio, "", gr.update(visible=True))


def start_quiz(topic, count, difficulty, pasted, file_path):
    topic = (topic or "").strip()
    try:
        source, label = None, topic
        if file_path:
            source = read_file(file_path)
            label = os.path.basename(file_path)
        elif (pasted or "").strip():
            source = pasted.strip()[:MAX_DOC_CHARS]
            label = "Văn bản dán vào"
        questions = generate_questions(topic, int(count), difficulty, source)
    except Exception as e:
        return pack(None, friendly_error(e))
    return begin(new_quiz_state(questions, label or SUBJECT))


def start_sample(count):
    bank = [normalize_question(q) for q in SAMPLE_BANK]
    questions = random.sample(bank, min(int(count), len(bank)))
    return begin(new_quiz_state(questions, "Câu hỏi mẫu"))


def check_answer(state, progress, choice):
    keep = (gr.update(), gr.update(), gr.update(), gr.update())
    progress = ensure_progress(progress)
    if not state:
        return (state, progress, "Hãy tạo quiz trước.") + keep[1:]
    if state.get("answered"):
        return (state, progress) + keep
    q = state["questions"][state["index"]]
    labels = labels_of(q)
    if choice not in labels:
        return (state, progress, "👆 Hãy chọn một đáp án trước.") + keep[1:]

    chosen = labels.index(choice)
    ok = chosen == q["answer"]
    state["answered"] = True
    if ok:
        state["score"] += 1
        verdict = "✅ **Đúng rồi!**"
    else:
        verdict = f"❌ **Chưa đúng.** Đáp án đúng: {LETTERS[q['answer']]}. {q['options'][q['answer']]}"
    state["log"].append({"q": q, "chosen": chosen, "ok": ok})

    tag = (q.get("tag") or state["topic"]).replace("|", "/")
    progress["answers"].append({"tag": tag, "ok": ok, "ts": time.time()})

    feedback = f"{verdict}\n\n{q['explanation']}"
    if q.get("verified") is True and q.get("evidence"):
        feedback += f"\n\n📖 Trích từ tài liệu: “{q['evidence']}”"
    elif q.get("verified") is False:
        feedback += "\n\n⚠️ Chưa xác minh được câu này với tài liệu, hãy tự đối chiếu."
    if q.get("tag"):
        feedback += f"\n\n🏷️ Chủ đề: {q['tag']}"

    is_last = state["index"] == len(state["questions"]) - 1
    return (
        state,
        progress,
        feedback,
        gr.update(interactive=False),
        hide(),
        gr.update(visible=True, value="Xem kết quả 🏁" if is_last else "Câu tiếp ▶"),
    )


def build_report(state: dict) -> str:
    total = len(state["questions"])
    lines = [
        f"# Kết quả ôn tập: {state['topic']}",
        f"Ngày: {datetime.now():%d/%m/%Y %H:%M}",
        f"Điểm: {state['score']}/{total}",
        "",
    ]
    for i, e in enumerate(state["log"], 1):
        q = e["q"]
        lines.append(f"## Câu {i} {'✅' if e['ok'] else '❌'}")
        lines.append(q["question"])
        for j, o in enumerate(q["options"]):
            note = ""
            if j == q["answer"]:
                note = "  ← đáp án đúng"
            elif j == e["chosen"]:
                note = "  ← bạn đã chọn"
            lines.append(f"- {LETTERS[j]}. {o}{note}")
        if q.get("explanation"):
            lines.append(f"\n*Giải thích:* {q['explanation']}")
        lines.append("")
    return "\n".join(lines)


def next_question(state, progress):
    progress = ensure_progress(progress)
    if not state or not state.get("answered"):
        return (state, progress) + tuple(gr.update() for _ in range(7))

    state["index"] += 1
    state["answered"] = False

    if state["index"] < len(state["questions"]):
        question_md, radio = show_question(state)
        return (state, progress, question_md, radio, "", gr.update(visible=True), hide(), hide(), hide())

    total = len(state["questions"])
    ratio = state["score"] / total
    comment = "Xuất sắc! 🎉" if ratio == 1 else "Khá tốt, cố thêm chút nữa nhé!" if ratio >= 0.7 else "Hãy ôn lại các phần còn yếu rồi thử lại nhé!"
    progress["quizzes"].append({"topic": state["topic"], "score": state["score"], "total": total, "ts": time.time()})

    wrong = [e for e in state["log"] if not e["ok"]]
    result = f"## 🏁 Kết quả: {state['score']}/{total}\n{comment}"
    if wrong:
        result += "\n\n**Các câu cần ôn lại:**\n"
        for e in wrong[:10]:
            q = e["q"]
            result += f"- {q['question']} → **{q['options'][q['answer']]}**\n"
    path = write_temp(build_report(state), "ketqua")
    return (
        state,
        progress,
        result,
        hide(),
        "",
        hide(),
        hide(),
        gr.update(visible=bool(wrong)),
        gr.update(value=path, visible=True),
    )


def retry_wrong(state):
    wrong = [e["q"] for e in (state or {}).get("log", []) if not e["ok"]]
    questions = [q for q in (normalize_question(q) for q in wrong) if q]
    if not questions:
        return pack(state, "Không có câu sai để ôn lại.")
    base = str(state["topic"]).replace(" (ôn câu sai)", "")
    return begin(new_quiz_state(questions, f"{base} (ôn câu sai)"))


# ======================================================================
# KẾ HOẠCH HỌC TẬP
# ======================================================================
def offline_plan(goal: str, weeks: int, hours: float, level: str, weak: list) -> str:
    focus = weak[:] or ["Kiến thức nền tảng", "Bài tập vận dụng", "Đề tổng hợp"]
    lines = [
        "## Kế hoạch ôn tập (bản mẫu, tạo không cần AI)",
        f"**Mục tiêu:** {goal}",
        f"**Thời gian:** {weeks} tuần, khoảng {hours:g} giờ mỗi ngày · **Trình độ:** {level}",
        "",
    ]
    for w in range(1, weeks + 1):
        if w == weeks:
            phase = "Ôn tổng hợp và tự kiểm tra"
        elif w / weeks <= 0.4:
            phase = "Nắm lại kiến thức nền"
        else:
            phase = "Luyện tập vận dụng"
        lines += [
            f"### Tuần {w}: {phase}. Trọng tâm: {focus[(w - 1) % len(focus)]}",
            f"- Thứ 2, 4, 6: học lý thuyết và làm ví dụ (khoảng {hours:g} giờ mỗi ngày)",
            "- Thứ 3, 5: làm bài tập và ghi lại các câu sai",
            "- Thứ 7: làm quiz tự kiểm tra 10 câu bằng ứng dụng này, rồi ôn lại câu sai",
            "- Chủ nhật: nghỉ nhẹ hoặc xem lại ghi chú, sơ đồ tư duy",
            "",
        ]
    return "\n".join(lines)


def plan_fn(goal, weeks, hours, level, use_progress, progress):
    goal = (goal or "").strip()
    if not goal:
        return "Hãy nhập mục tiêu, ví dụ: *Ôn thi cuối kỳ môn Khoa học tự nhiên*", hide()
    weeks, hours = int(weeks), float(hours)
    weak = weak_topics(ensure_progress(progress)) if use_progress else []
    weak_line = (
        "Các chủ đề học sinh đang yếu, cần ưu tiên: " + ", ".join(weak)
        if weak
        else "Chưa có dữ liệu về chủ đề yếu."
    )
    prompt = (
        f"Mục tiêu: {goal[:MAX_INPUT_CHARS]}\n"
        f"Thời gian: {weeks} tuần, khoảng {hours:g} giờ mỗi ngày\n"
        f"Trình độ hiện tại: {level}\n"
        f"{weak_line}\n"
        "Hãy lập kế hoạch ôn tập chi tiết theo từng tuần, có bảng tóm tắt ở đầu."
    )
    note = ""
    try:
        text = ask_llm(
            [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=2500,
        )
    except Exception as e:
        note = f"{friendly_error(e)}\n\nDưới đây là kế hoạch mẫu tạo không cần AI:\n\n"
        text = offline_plan(goal, weeks, hours, level, weak)
    path = write_temp(text, "kehoach")
    return note + text, gr.update(value=path, visible=True)


# ======================================================================
# GIAO DIỆN
# ======================================================================
CHAT_EXAMPLES = [
    "Giải thích quang hợp bằng ví dụ dễ hiểu",
    "Phân biệt khối lượng và trọng lượng",
    "Vì sao nước đá nổi trên mặt nước?",
]


def build_ui():
    with gr.Blocks() as demo:
        progress = gr.State(None)

        gr.Markdown(
            f"# 📚 StudyBuddy: trợ lý ôn tập {SUBJECT}\n"
            f"Mô hình mở: `{MODEL}` · Chạy qua: **{PROVIDER}**"
        )
        with gr.Row():
            status = gr.Markdown("Bấm nút bên cạnh để kiểm tra kết nối AI.")
            test_btn = gr.Button("🔌 Kiểm tra kết nối AI")
        test_btn.click(test_ai, None, status)

        # ---------------- Quiz
        with gr.Tab("📝 Quiz ôn tập"):
            with gr.Row():
                topic = gr.Textbox(label="Chủ đề", placeholder="Ví dụ: Lực và chuyển động", scale=3)
                count = gr.Slider(1, 10, value=5, step=1, label="Số câu", scale=1)
                difficulty = gr.Dropdown(["Dễ", "Vừa", "Khó"], value="Vừa", label="Mức độ", scale=1)
            pasted = gr.Textbox(
                label="Hoặc dán nội dung bài học vào đây (nếu có, câu hỏi sẽ dựa trên đoạn này)",
                lines=4,
            )
            file_in = gr.File(label="Hoặc tải tài liệu (.txt, .md, .pdf)", type="filepath")
            with gr.Row():
                start_btn = gr.Button("Tạo quiz bằng AI", variant="primary")
                sample_btn = gr.Button("Bộ câu hỏi mẫu (không cần AI)")

            quiz_state = gr.State()
            question_md = gr.Markdown()
            choices = gr.Radio(label="Chọn đáp án", visible=False)
            feedback = gr.Markdown()
            check_btn = gr.Button("Kiểm tra", variant="primary", visible=False)
            next_btn = gr.Button("Câu tiếp ▶", visible=False)
            retry_btn = gr.Button("🔁 Ôn lại các câu sai", visible=False)
            result_file = gr.File(label="Tải báo cáo kết quả (.md)", visible=False)

            quiz_outputs = [quiz_state, question_md, choices, feedback, check_btn, next_btn, retry_btn, result_file]
            start_btn.click(start_quiz, [topic, count, difficulty, pasted, file_in], quiz_outputs)
            sample_btn.click(start_sample, [count], quiz_outputs)
            check_btn.click(
                check_answer,
                [quiz_state, progress, choices],
                [quiz_state, progress, feedback, choices, check_btn, next_btn],
            )
            next_btn.click(
                next_question,
                [quiz_state, progress],
                [quiz_state, progress, question_md, choices, feedback, check_btn, next_btn, retry_btn, result_file],
            )
            retry_btn.click(retry_wrong, [quiz_state], quiz_outputs)

        # ---------------- Tiến độ
        with gr.Tab("📊 Tiến độ") as tab_progress:
            progress_out = gr.Markdown(progress_md(None))
            with gr.Row():
                refresh_btn = gr.Button("Cập nhật")
                reset_btn = gr.Button("Xóa dữ liệu tiến độ")
            refresh_btn.click(show_progress, [progress], progress_out)
            reset_btn.click(reset_progress, None, [progress, progress_out])
            try:
                tab_progress.select(show_progress, [progress], progress_out)
            except Exception:
                pass  # phiên bản Gradio không hỗ trợ: vẫn còn nút Cập nhật

        # ---------------- Kế hoạch
        with gr.Tab("🗓️ Kế hoạch học tập"):
            goal = gr.Textbox(
                label="Mục tiêu của bạn",
                lines=2,
                placeholder="Ôn thi cuối kỳ môn Khoa học tự nhiên",
            )
            with gr.Row():
                weeks = gr.Slider(1, 12, value=3, step=1, label="Số tuần")
                hours = gr.Slider(0.5, 6, value=1.5, step=0.5, label="Giờ học mỗi ngày")
                level = gr.Dropdown(["Mới bắt đầu", "Trung bình", "Khá"], value="Trung bình", label="Trình độ")
            use_progress = gr.Checkbox(value=True, label="Ưu tiên các chủ đề tôi còn yếu (dựa trên kết quả quiz)")
            plan_btn = gr.Button("Lập kế hoạch", variant="primary")
            plan_out = gr.Markdown()
            plan_file = gr.File(label="Tải kế hoạch (.md)", visible=False)
            plan_btn.click(plan_fn, [goal, weeks, hours, level, use_progress, progress], [plan_out, plan_file])

        # ---------------- Hỏi đáp
        with gr.Tab("💬 Hỏi đáp"):
            try:
                gr.ChatInterface(chat_fn, type="messages", examples=CHAT_EXAMPLES)
            except TypeError:  # phiên bản Gradio mới không còn tham số type
                gr.ChatInterface(chat_fn, examples=CHAT_EXAMPLES)

        gr.Markdown("<small>Lưu ý: AI có thể sai. Hãy đối chiếu đáp án với sách giáo khoa hoặc giáo viên.</small>")
    return demo


if __name__ == "__main__":
    port = os.getenv("PORT")  # các dịch vụ lưu trữ (Render...) cấp cổng qua biến PORT
    build_ui().queue(max_size=20).launch(
        server_name="0.0.0.0" if port else None,
        server_port=int(port) if port else None,
        share=os.getenv("SHARE") == "1",
    )
