"""
StudyBuddy: trợ lý ôn tập AI mã nguồn mở, học mà chơi (Gradio + mô hình Llama qua Groq hoặc Ollama)

Học tập:
- Quiz trắc nghiệm tạo từ chủ đề, nội dung dán vào hoặc file .txt/.md/.pdf.
  Với tài liệu, mỗi câu kèm trích dẫn nguyên văn và được kiểm tra có thật trong tài liệu hay không.
- Giải thích, ôn lại câu sai, báo cáo kết quả, theo dõi tiến độ theo chủ đề, kế hoạch học tập, hỏi đáp.

Trò chơi:
- Đấu quái vật: 3 mạng, trả lời đúng để làm quái vật yếu đi, chuỗi đúng gây đòn chí mạng.
- Chiến dịch nhiều màn: sau mỗi ván bấm "Ván tiếp theo" để nhận câu hỏi chưa gặp,
  quái vật mạnh dần và độ khó tăng dần. Hết câu mới thì xáo lại từ đầu.
- Điểm kinh nghiệm (XP), cấp độ, linh thú tiến hóa, huy hiệu, gợi ý 50/50.
- Thẻ ghi nhớ (lật thẻ, tự đánh giá, thẻ chưa nhớ quay lại sau).
- Nhắc nghỉ ngơi sau vài ván. Không lưu thông tin cá nhân, không bảng xếp hạng công khai.

Dành cho học sinh hôm nay:
- Chọn môn (Toán, Văn, Anh, Sử, Địa, Tin, KHTN, GDCD) và lớp 6 đến 12; AI ra câu hỏi đúng trình độ.
- Chuỗi ngày học, mục tiêu mỗi ngày, tiến độ tự lưu trên trình duyệt (nếu Gradio hỗ trợ BrowserState).
- Sổ tay lỗi sai: câu làm sai được lưu lại, ôn bằng thẻ ghi nhớ cho tới khi nhớ.
- Nút "Giải thích dễ hiểu hơn", hỏi đáp có nút hỏi nhanh, giao diện gọn cho điện thoại.
- Khung "khoe thành tích" để sao chép chia sẻ với bạn bè, mẹo học tập sau mỗi ván.

Chạy được cả khi không có AI: bộ câu hỏi mẫu, thẻ mẫu và kế hoạch mẫu vẫn dùng bình thường.
"""
import json
import math
import os
import random
import re
import tempfile
import threading
import time
import unicodedata
from collections import deque
from datetime import datetime, timedelta, timezone

import gradio as gr
from openai import OpenAI
from pypdf import PdfReader

# ======================================================================
# CẤU HÌNH
# ======================================================================
SUBJECT = os.getenv("SUBJECT", "Khoa học tự nhiên")
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "60"))  # bảo vệ khóa khi chia sẻ link

# --- Môn học và lớp (học sinh chọn ở đầu trang) kèm chủ đề gợi ý
SUBJECTS = {
    "Khoa học tự nhiên": ["Lực và chuyển động", "Tế bào", "Phản ứng hóa học", "Hệ Mặt Trời",
                          "Quang hợp", "Điện và mạch điện", "Acid và base"],
    "Toán": ["Phân số", "Phương trình bậc nhất", "Tam giác", "Tỉ lệ thức", "Hàm số", "Xác suất cơ bản"],
    "Ngữ văn": ["Biện pháp tu từ", "Từ loại", "Truyện ngắn", "Thơ trung đại", "Nghị luận xã hội"],
    "Tiếng Anh": ["Thì hiện tại đơn", "Thì quá khứ đơn", "Từ vựng về trường học", "Câu điều kiện",
                  "Giới từ chỉ nơi chốn"],
    "Lịch sử": ["Kháng chiến chống Pháp", "Các triều đại phong kiến", "Thời Trần", "Cách mạng tháng Tám 1945"],
    "Địa lí": ["Khí hậu Việt Nam", "Địa hình Việt Nam", "Dân cư", "Châu Á"],
    "Tin học": ["Phần cứng máy tính", "Internet và an toàn mạng", "Thuật toán cơ bản", "Bảng tính"],
    "Giáo dục công dân": ["Tôn trọng sự thật", "Bảo vệ môi trường", "Quyền và nghĩa vụ của công dân"],
}
if SUBJECT not in SUBJECTS:
    SUBJECTS[SUBJECT] = []
DEFAULT_SUBJECT = SUBJECT
GRADES = [f"Lớp {n}" for n in range(6, 13)]
DEFAULT_GRADE = os.getenv("GRADE", "Lớp 8")
if DEFAULT_GRADE not in GRADES:
    DEFAULT_GRADE = "Lớp 8"
SAMPLE_DEFAULT_SUBJECT = "Khoa học tự nhiên"  # môn của các câu mẫu cũ chưa gắn nhãn môn

MAX_HISTORY = 6
MAX_INPUT_CHARS = 2000
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_DOC_CHARS = 300_000
MAX_SOURCE_CHARS = 12_000  # lượng chữ gửi cho AI mỗi lần tạo quiz
MAX_PDF_PAGES = 40
LETTERS = "ABCD"
WEAK_THRESHOLD = 0.7

# --- Trò chơi
MODE_BOSS = "⚔️ Đấu quái vật (3 mạng)"
MODE_PRACTICE = "📖 Luyện tập (không mất mạng)"
MAX_LIVES = 3
HINTS_PER_QUIZ = 2
BOSS_HP_RATIO = 0.7  # màn 1: cần đúng khoảng 70% để hạ quái vật
BOSS_HP_STEP = 0.05  # mỗi màn sau quái vật khỏe thêm 5%
BOSS_HP_MAX_RATIO = 0.9
BADGE_BONUS_XP = 10
REST_EVERY = 3  # nhắc nghỉ sau mỗi 3 ván
DIFFICULTY_ORDER = ["Dễ", "Vừa", "Khó"]
SEEN_MEMORY = 60  # số câu AI đã tạo được nhớ để tránh lặp lại
MAX_ANSWERS = 600  # số câu trả lời gần nhất giữ trong hồ sơ
MAX_MISTAKES = 40  # sức chứa của sổ tay lỗi sai
DAILY_GOAL = 10  # mục tiêu số câu mỗi ngày
TZ = timezone(timedelta(hours=float(os.getenv("TZ_OFFSET_HOURS", "7"))))  # mặc định giờ Việt Nam

STUDY_TIPS = [
    "🍅 **Mẹo Pomodoro:** học 25 phút, nghỉ 5 phút. Sau 4 chu kỳ thì nghỉ dài hơn một chút.",
    "🧠 **Tự kiểm tra** (như ván vừa rồi) nhớ lâu hơn nhiều so với đọc đi đọc lại. Cứ đóng sách rồi thử nhớ lại!",
    "📅 **Ôn lại đúng lúc:** xem lại bài sau 1 ngày, 3 ngày, rồi 7 ngày để kiến thức bám chắc hơn.",
    "😴 **Ngủ đủ giấc** giúp não củng cố trí nhớ. Học dồn đến khuya thường không hiệu quả đâu.",
    "✍️ **Thử giảng lại bài** cho bạn bè (hoặc cho thú cưng). Giảng được là hiểu thật sự.",
    "📵 **Bật chế độ Không làm phiền** khi học. Mỗi lần xem thông báo, bạn mất vài phút để tập trung lại.",
    "💧 **Uống nước, vươn vai** sau mỗi 30 đến 40 phút ngồi học để đầu óc tỉnh táo.",
    "🧩 **Câu sai là mỏ vàng:** xem lại sổ tay lỗi sai trong tab Thẻ ghi nhớ, mỗi ngày vài câu là đủ.",
]

BOSSES = [
    {"name": "Quái vật Quên Bài", "emoji": "👾"},
    {"name": "Bóng ma Đãng Trí", "emoji": "👻"},
    {"name": "Robot Nhầm Lẫn", "emoji": "🤖"},
    {"name": "Slime Lười Học", "emoji": "🟢"},
]

# (điểm XP tối thiểu, danh hiệu, linh thú, tên linh thú)
LEVELS = [
    (0, "Tập sự phòng thí nghiệm", "🥚", "Trứng bí ẩn"),
    (100, "Nhà thám hiểm nhí", "🐣", "Chim non"),
    (250, "Nhà khoa học trẻ", "🐱", "Mèo con tò mò"),
    (450, "Nhà phát minh", "🦉", "Cú thông thái"),
    (700, "Giáo sư tí hon", "🦊", "Cáo tinh khôn"),
    (1000, "Bậc thầy khoa học", "🐲", "Rồng trí tuệ"),
]

BADGES = {
    "first_quiz": ("🚀", "Khởi đầu", "Hoàn thành quiz đầu tiên"),
    "streak3": ("🔥", "Chuỗi lửa", "Trả lời đúng 3 câu liên tiếp"),
    "streak5": ("⚡", "Bão sét", "Trả lời đúng 5 câu liên tiếp"),
    "perfect": ("💯", "Điểm tuyệt đối", "Đúng hết một quiz (từ 3 câu trở lên)"),
    "boss_win": ("🏆", "Thợ săn quái vật", "Đánh bại một quái vật"),
    "campaign3": ("🗺️", "Chiến binh nhiều màn", "Đánh bại quái vật ở màn 3"),
    "comeback": ("🔁", "Sửa sai thành công", "Đúng hết khi ôn lại các câu sai"),
    "explorer": ("🧭", "Nhà thám hiểm", "Trả lời câu hỏi thuộc 5 chủ đề khác nhau"),
    "marathon": ("📚", "Chăm chỉ", "Hoàn thành 3 quiz"),
    "cards": ("🃏", "Nhà ghi nhớ", "Nhớ 10 thẻ ghi nhớ"),
    "daily_goal": ("🎯", "Đủ chỉ tiêu", f"Làm đủ {DAILY_GOAL} câu trong một ngày"),
    "streak_days3": ("📅", "Kiên trì", "Học 3 ngày liên tiếp"),
}

CORRECT_MSGS = ["Chính xác!", "Tuyệt vời!", "Quá đỉnh!", "Làm tốt lắm!", "Đúng rồi!"]
WRONG_MSGS = [
    "Chưa đúng, nhưng không sao!",
    "Gần rồi, xem giải thích nhé!",
    "Sai cũng là cơ hội để học thêm!",
    "Không sao, lần sau sẽ nhớ!",
]

DIFFICULTY_HINTS = {
    "Dễ": "ghi nhớ kiến thức cơ bản, câu hỏi ngắn gọn",
    "Vừa": "hiểu và vận dụng đơn giản, có một chút suy luận",
    "Khó": "vận dụng, phân tích, tình huống thực tế, phương án gây nhiễu hợp lý",
}

SAFE_RULE = (
    "Nội dung phải phù hợp học sinh phổ thông: ngôn ngữ thân thiện, tích cực, "
    "không bạo lực, không nội dung nhạy cảm hay người lớn."
)


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

def chat_system(subject=None, grade=None) -> str:
    who = f"học sinh {grade}" if grade else "học sinh"
    return (
        f"Bạn là trợ lý ôn tập môn {subject or SUBJECT} cho {who}, nói chuyện thân thiện như một người anh chị đi trước. "
        "Trả lời bằng tiếng Việt, câu ngắn, dễ hiểu, dùng ví dụ gần gũi (trò chơi, thể thao, nấu ăn, mạng xã hội...). "
        "Nếu không chắc chắn, hãy nói thẳng thay vì đoán. "
        "Chỉ hướng dẫn cách học và cách hiểu bài, không làm hộ bài kiểm tra; với bài tập, hãy gợi ý từng bước để bạn tự làm. "
        "Nếu câu hỏi không phù hợp với học sinh, hãy từ chối nhẹ nhàng và đưa câu chuyện về việc học."
    )


PLAN_SYSTEM = (
    "Bạn là cố vấn học tập thân thiện cho học sinh. Hãy lập kế hoạch bằng tiếng Việt, định dạng Markdown, "
    "có cấu trúc: mục tiêu, lịch theo tuần (và theo ngày nếu phù hợp), phương pháp ôn tập, cách tự kiểm tra "
    "tiến độ, thời gian nghỉ ngơi hợp lý, lưu ý. Nếu thiếu thông tin, hãy đưa ra giả định hợp lý và nêu rõ."
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
    # ---- Toán
    {"subject": "Toán", "tag": "Lũy thừa", "question": "Kết quả của phép tính 3² + 4² là bao nhiêu?",
     "options": ["7", "12", "25", "49"], "answer": 2,
     "explanation": "3² = 9 và 4² = 16 nên tổng là 9 + 16 = 25."},
    {"subject": "Toán", "tag": "Hình học", "question": "Tổng ba góc trong một tam giác bằng bao nhiêu độ?",
     "options": ["90°", "180°", "270°", "360°"], "answer": 1,
     "explanation": "Tổng ba góc của mọi tam giác luôn bằng 180°."},
    {"subject": "Toán", "tag": "Hình học", "question": "Diện tích hình chữ nhật dài 8 cm, rộng 5 cm là bao nhiêu?",
     "options": ["13 cm²", "26 cm²", "40 cm²", "80 cm²"], "answer": 2,
     "explanation": "Diện tích = dài × rộng = 8 × 5 = 40 cm². Còn 26 cm là chu vi, không phải diện tích."},
    {"subject": "Toán", "tag": "Phương trình", "question": "Giải phương trình 2x + 6 = 14, x bằng bao nhiêu?",
     "options": ["3", "4", "7", "10"], "answer": 1,
     "explanation": "Chuyển vế: 2x = 14 − 6 = 8, nên x = 4."},
    # ---- Tiếng Anh
    {"subject": "Tiếng Anh", "tag": "Ngữ pháp", "question": "Choose the correct word: She ___ to school every day.",
     "options": ["go", "goes", "going", "gone"], "answer": 1,
     "explanation": "Chủ ngữ She (ngôi thứ ba số ít) ở thì hiện tại đơn thì động từ thêm -es: goes."},
    {"subject": "Tiếng Anh", "tag": "Từ vựng", "question": "What is the past tense of the verb “buy”?",
     "options": ["buyed", "bought", "boughted", "buys"], "answer": 1,
     "explanation": "“Buy” là động từ bất quy tắc: buy – bought – bought."},
    {"subject": "Tiếng Anh", "tag": "Từ vựng", "question": "Từ nào có nghĩa là “thư viện”?",
     "options": ["Library", "Bookstore", "Laboratory", "Cafeteria"], "answer": 0,
     "explanation": "Library là thư viện. Bookstore là hiệu sách, laboratory là phòng thí nghiệm, cafeteria là căng-tin."},
    {"subject": "Tiếng Anh", "tag": "Ngữ pháp", "question": "I ___ a book when my mom called me.",
     "options": ["read", "was reading", "am reading", "have read"], "answer": 1,
     "explanation": "Hành động đang diễn ra trong quá khứ thì bị một hành động khác chen vào: dùng quá khứ tiếp diễn (was reading)."},
    # ---- Lịch sử
    {"subject": "Lịch sử", "tag": "Lịch sử Việt Nam", "question": "Chiến thắng Điện Biên Phủ diễn ra vào năm nào?",
     "options": ["1945", "1954", "1968", "1975"], "answer": 1,
     "explanation": "Chiến thắng Điện Biên Phủ năm 1954 kết thúc cuộc kháng chiến chống thực dân Pháp."},
    {"subject": "Lịch sử", "tag": "Lịch sử Việt Nam", "question": "Ai là người lãnh đạo chiến thắng Bạch Đằng năm 938?",
     "options": ["Ngô Quyền", "Lê Lợi", "Trần Hưng Đạo", "Quang Trung"], "answer": 0,
     "explanation": "Ngô Quyền đánh bại quân Nam Hán trên sông Bạch Đằng năm 938, mở ra thời kỳ độc lập lâu dài."},
    {"subject": "Lịch sử", "tag": "Lịch sử Việt Nam", "question": "Ngày Quốc khánh của nước Việt Nam là ngày nào?",
     "options": ["2/9", "30/4", "19/5", "1/5"], "answer": 0,
     "explanation": "Ngày 2/9/1945, Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập, khai sinh nước Việt Nam Dân chủ Cộng hòa."},
    # ---- Địa lí
    {"subject": "Địa lí", "tag": "Địa lí Việt Nam", "question": "Việt Nam nằm ở khu vực nào của châu Á?",
     "options": ["Đông Nam Á", "Tây Á", "Bắc Á", "Nam Á"], "answer": 0,
     "explanation": "Việt Nam nằm ở phía đông của bán đảo Đông Dương, thuộc khu vực Đông Nam Á."},
    {"subject": "Địa lí", "tag": "Địa lí Việt Nam", "question": "Đỉnh núi cao nhất Việt Nam là đỉnh nào?",
     "options": ["Phan-xi-păng", "Ngọc Linh", "Bạch Mã", "Tây Côn Lĩnh"], "answer": 0,
     "explanation": "Phan-xi-păng (Fansipan) cao khoảng 3 143 m, được gọi là “nóc nhà Đông Dương”."},
    {"subject": "Địa lí", "tag": "Địa lí thế giới", "question": "Đường xích đạo chia Trái Đất thành hai bán cầu nào?",
     "options": ["Bắc và Nam", "Đông và Tây", "Nóng và lạnh", "Trên và dưới"], "answer": 0,
     "explanation": "Xích đạo là đường vĩ tuyến 0°, chia Trái Đất thành bán cầu Bắc và bán cầu Nam."},
    # ---- Ngữ văn
    {"subject": "Ngữ văn", "tag": "Văn học", "question": "Tác giả của tác phẩm “Truyện Kiều” là ai?",
     "options": ["Nguyễn Du", "Nguyễn Trãi", "Hồ Xuân Hương", "Nguyễn Đình Chiểu"], "answer": 0,
     "explanation": "Nguyễn Du là tác giả của Truyện Kiều (Đoạn trường tân thanh)."},
    {"subject": "Ngữ văn", "tag": "Tiếng Việt", "question": "Từ nào sau đây là từ láy?",
     "options": ["Xinh xắn", "Học sinh", "Bàn ghế", "Sách vở"], "answer": 0,
     "explanation": "“Xinh xắn” là từ láy (láy phụ âm đầu). Các từ còn lại là từ ghép."},
    {"subject": "Ngữ văn", "tag": "Tiếng Việt", "question": "Biện pháp tu từ trong câu “Mặt trời xuống biển như hòn lửa” là gì?",
     "options": ["So sánh", "Ẩn dụ", "Hoán dụ", "Điệp ngữ"], "answer": 0,
     "explanation": "Câu dùng từ “như” để đối chiếu mặt trời với hòn lửa nên là biện pháp so sánh."},
    # ---- Tin học
    {"subject": "Tin học", "tag": "Phần cứng", "question": "Bộ phận nào được coi là “bộ não” của máy tính?",
     "options": ["CPU", "Chuột", "Màn hình", "Loa"], "answer": 0,
     "explanation": "CPU (bộ xử lí trung tâm) thực hiện các phép tính và điều khiển hoạt động của máy."},
    {"subject": "Tin học", "tag": "Biểu diễn thông tin", "question": "1 byte bằng bao nhiêu bit?",
     "options": ["4", "8", "10", "16"], "answer": 1,
     "explanation": "1 byte gồm 8 bit."},
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
                "Trong lúc chờ, vẫn dùng được bộ câu hỏi mẫu, thẻ mẫu và kế hoạch mẫu."
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
def chat_fn(message, history, subject=None, grade=None):
    recent = [
        {"role": m["role"], "content": as_text(m["content"])}
        for m in (history or [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ][-MAX_HISTORY:]
    while recent and recent[0]["role"] != "user":
        recent.pop(0)
    try:
        return ask_llm(
            [
                {"role": "system", "content": chat_system(subject, grade)},
                *recent,
                {"role": "user", "content": as_text(message)[:MAX_INPUT_CHARS]},
            ]
        )
    except Exception as e:
        return friendly_error(e)


def chat_respond(message, history, subject=None, grade=None):
    """Hộp chat tự dựng: trả về (ô nhập rỗng, lịch sử mới)."""
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        return "", history
    reply = chat_fn(message, history, subject, grade)
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return "", history


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


def generate_questions(
    topic: str, count: int, difficulty: str, source_text, avoid=None, subject=None, grade=None
) -> list:
    """Tạo câu hỏi bằng AI. `avoid` là danh sách câu hỏi đã gặp để không lặp lại."""
    subject = subject or SUBJECT
    if source_text:
        context = (
            "Chỉ dựa trên tài liệu sau, không dùng kiến thức bên ngoài:\n"
            f'"""\n{sample_source(source_text)}\n"""'
        )
        evidence_rule = '"evidence": đoạn trích NGUYÊN VĂN (10 đến 200 ký tự) từ tài liệu chứng minh đáp án đúng.'
    else:
        level = f" Trình độ: học sinh {grade}, nội dung bám sát chương trình phổ thông Việt Nam." if grade else ""
        context = f'Môn học: {subject}.{level} Chủ đề: "{topic or subject}"'
        evidence_rule = '"evidence": để chuỗi rỗng.'

    lang_rule = "bằng tiếng Việt"
    if subject == "Tiếng Anh" and not source_text:
        lang_rule = "kiểm tra tiếng Anh (câu hỏi và 4 lựa chọn viết bằng tiếng Anh, riêng phần explanation viết bằng tiếng Việt)"
    avoid = list(avoid or [])
    avoid_set = {_norm(a) for a in avoid}
    avoid_line = ""
    if avoid:
        avoid_line = (
            "\nCác câu hỏi sau ĐÃ được dùng, tuyệt đối không lặp lại hoặc hỏi lại cùng một ý:\n"
            + "\n".join(f"- {a[:100]}" for a in avoid[-20:])
            + "\n"
        )

    prompt = f"""{context}

Hãy tạo {count} câu hỏi trắc nghiệm {lang_rule}. Mức độ: {difficulty} ({DIFFICULTY_HINTS.get(difficulty, "")}).
{SAFE_RULE}{avoid_line}
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
                temperature=0.5 if not avoid else 0.8,
            )
            questions = parse_questions(raw)
            if avoid_set:
                questions = [q for q in questions if _norm(q["question"]) not in avoid_set]
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
    if last_error is not None and any(
        k in type(last_error).__name__ for k in ("Connection", "Timeout", "Authentication", "NotFound")
    ):
        raise last_error
    raise UserFacingError("Không tạo được câu hỏi hợp lệ, bạn thử lại hoặc dùng bộ câu hỏi mẫu nhé.")


def sample_pool(subject=None) -> list:
    """Chỉ số các câu mẫu thuộc môn đã chọn (môn chưa có câu mẫu thì dùng toàn bộ)."""
    subject = subject or SAMPLE_DEFAULT_SUBJECT
    pool = [i for i, q in enumerate(SAMPLE_BANK) if q.get("subject", SAMPLE_DEFAULT_SUBJECT) == subject]
    return pool or list(range(len(SAMPLE_BANK)))


def pick_sample(seen: list, count: int, subject=None):
    """Chọn câu mẫu chưa gặp. Hết câu mới thì xáo lại từ đầu.

    Trả về (danh sách câu hỏi, danh sách chỉ số đã gặp mới, có xáo lại không).
    """
    pool = sample_pool(subject)
    count = max(1, min(int(count), len(pool)))
    seen_set = set(seen)
    unseen = [i for i in pool if i not in seen_set]
    random.shuffle(unseen)
    picked = unseen[:count]
    reshuffled = False
    if len(picked) < count:
        # Hết câu mới: bắt đầu vòng mới, bổ sung từ các câu còn lại (không trùng trong ván này)
        reshuffled = True
        rest = [i for i in pool if i not in picked]
        random.shuffle(rest)
        picked += rest[: count - len(picked)]
        new_seen = list(picked)
    else:
        new_seen = list(seen) + picked
    questions = [normalize_question(SAMPLE_BANK[i]) for i in picked]
    return [q for q in questions if q], new_seen, reshuffled


# ======================================================================
# HỒ SƠ NGƯỜI CHƠI: XP, CẤP ĐỘ, HUY HIỆU, TIẾN ĐỘ
# ======================================================================
def new_progress() -> dict:
    return {
        "answers": [], "quizzes": [], "xp": 0, "badges": [],
        "best_streak": 0, "cards_known": 0, "best_round": 0,
        "days": [], "mistakes": [],
    }


def ensure_progress(p) -> dict:
    if not isinstance(p, dict):
        return new_progress()
    for key, default in new_progress().items():
        if key not in p:
            p[key] = default.copy() if isinstance(default, list) else default
    return p


def level_of(xp: int):
    """Trả về (chỉ số cấp, danh hiệu, linh thú, tên linh thú, mốc hiện tại, mốc kế tiếp hoặc None)."""
    idx = max(i for i, lv in enumerate(LEVELS) if xp >= lv[0])
    nxt = LEVELS[idx + 1][0] if idx + 1 < len(LEVELS) else None
    return idx, LEVELS[idx][1], LEVELS[idx][2], LEVELS[idx][3], LEVELS[idx][0], nxt


def xp_bar(xp: int) -> str:
    _, _, _, _, floor, nxt = level_of(xp)
    if nxt is None:
        return "▰" * 10
    filled = int((xp - floor) / (nxt - floor) * 10)
    return "▰" * filled + "▱" * (10 - filled)


def add_xp(progress: dict, amount: int) -> str:
    """Cộng XP; trả về thông báo nếu lên cấp."""
    before = level_of(progress["xp"])[0]
    progress["xp"] += amount
    idx, title, pet, pet_name, _, _ = level_of(progress["xp"])
    if idx > before:
        return f"\n\n🎉 **Lên cấp {idx + 1}!** Bạn trở thành *{title}*, linh thú tiến hóa thành {pet} {pet_name}!"
    return ""


def award(progress: dict, key: str) -> str:
    """Trao huy hiệu (nếu chưa có) và thưởng XP."""
    if key in progress["badges"]:
        return ""
    progress["badges"].append(key)
    emoji, name, _ = BADGES[key]
    return f"\n\n🏅 **Huy hiệu mới:** {emoji} {name} (+{BADGE_BONUS_XP} XP)" + add_xp(progress, BADGE_BONUS_XP)


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


def today_str(ts=None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else time.time(), TZ).strftime("%Y-%m-%d")


def mark_today(progress: dict):
    """Ghi nhận hôm nay có học (để tính chuỗi ngày học)."""
    t = today_str()
    if t not in progress["days"]:
        progress["days"].append(t)
        progress["days"] = progress["days"][-120:]


def today_count(progress: dict) -> int:
    t = today_str()
    return sum(1 for a in progress["answers"] if a.get("ts") and today_str(a["ts"]) == t)


def day_streak(progress: dict) -> int:
    """Số ngày học liên tiếp (tính cả khi hôm nay chưa học nhưng hôm qua có học)."""
    days = set(progress["days"])
    d = datetime.now(TZ).date()
    if d.strftime("%Y-%m-%d") not in days:
        d -= timedelta(days=1)
    n = 0
    while d.strftime("%Y-%m-%d") in days:
        n += 1
        d -= timedelta(days=1)
    return n


def goal_bar(progress: dict) -> str:
    n = min(today_count(progress), DAILY_GOAL)
    filled = round(n / DAILY_GOAL * 10)
    return "▰" * filled + "▱" * (10 - filled)


def remember_mistake(progress: dict, q: dict):
    """Lưu câu làm sai vào sổ tay (không trùng, giữ các câu mới nhất)."""
    forget_mistake(progress, q)
    progress["mistakes"].append({k: q.get(k, "") for k in ("question", "options", "answer", "explanation", "tag")})
    progress["mistakes"] = progress["mistakes"][-MAX_MISTAKES:]


def forget_mistake(progress: dict, q: dict):
    key = _norm(q["question"])
    progress["mistakes"] = [m for m in progress["mistakes"] if _norm(m.get("question", "")) != key]


def welcome_md(progress) -> str:
    p = ensure_progress(progress)
    if not p["answers"] and not p["quizzes"]:
        return "👋 **Chào bạn!** Chọn môn, lớp và chủ đề rồi bắt đầu ván đầu tiên nhé. Mỗi ngày chỉ cần 10 câu là đã tiến bộ!"
    idx, title, pet, pet_name, _, _ = level_of(p["xp"])
    return (
        f"{pet} **Chào mừng quay lại!** Cấp {idx + 1} · ⭐ {p['xp']} XP · "
        f"📅 {day_streak(p)} ngày liên tiếp · 🎯 {min(today_count(p), DAILY_GOAL)}/{DAILY_GOAL} câu hôm nay"
    )


def share_text(state: dict, progress: dict, won: bool) -> str:
    total = len(state["questions"])
    lv = level_of(progress["xp"])[0] + 1
    win = f", hạ {state['boss']['name']} ở màn {state['round']}" if won else ""
    msg = (
        f"🎮 Mình vừa ôn {state['topic']} trên StudyBuddy: {state['score']}/{total} điểm{win}! "
        f"Cấp {lv}, {progress['xp']} XP. Thử sức không? 😎"
    )
    return "\n\n📤 **Khoe thành tích với bạn bè** (bấm nút sao chép ở góc khung):\n```\n" + msg + "\n```"


def profile_md(p: dict) -> str:
    idx, title, pet, pet_name, _, nxt = level_of(p["xp"])
    remain = f"(còn {nxt - p['xp']} XP để lên cấp)" if nxt else "(cấp cao nhất!)"
    lines = [
        f"## {pet} {pet_name} · Cấp {idx + 1}: {title}",
        f"⭐ **{p['xp']} XP** {xp_bar(p['xp'])} {remain}",
        f"🔥 Chuỗi đúng dài nhất: **{p['best_streak']}**",
        f"🗺️ Màn cao nhất đã vượt: **{p['best_round']}**",
        f"📅 Chuỗi ngày học: **{day_streak(p)}** ngày",
        f"🎯 Hôm nay: **{min(today_count(p), DAILY_GOAL)}/{DAILY_GOAL} câu** {goal_bar(p)}",
        f"📝 Sổ tay lỗi sai: **{len(p['mistakes'])}** câu (ôn ở tab Thẻ ghi nhớ)",
        "",
        "### Huy hiệu",
    ]
    for key, (emoji, name, desc) in BADGES.items():
        if key in p["badges"]:
            lines.append(f"- {emoji} **{name}**: {desc}")
        else:
            lines.append(f"- 🔒 {name}: {desc}")
    return "\n".join(lines)


def progress_md(progress) -> str:
    p = ensure_progress(progress)
    parts = [profile_md(p), ""]
    if not p["answers"]:
        parts.append("Bạn chưa làm câu nào. Hãy chơi một ván ở tab **Quiz ôn tập** để xem thống kê tại đây.")
        return "\n".join(parts)
    total = len(p["answers"])
    correct = sum(1 for a in p["answers"] if a["ok"])
    parts += [
        "### Tổng quan",
        f"- Số ván đã hoàn thành: **{len(p['quizzes'])}**",
        f"- Số câu đã trả lời: **{total}**, đúng **{correct}** ({correct / total:.0%})",
        "",
        "### Theo chủ đề",
        "| Chủ đề | Đúng/Tổng | Tỉ lệ |",
        "|---|---|---|",
    ]
    for tag, (c, t) in sorted(topic_stats(p).items(), key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1])):
        parts.append(f"| {tag} | {c}/{t} | {c / t:.0%} |")
    weak = weak_topics(p)
    parts.append("")
    if weak:
        parts.append("**Nên ôn thêm:** " + ", ".join(weak))
    else:
        parts.append("**Tuyệt vời!** Chưa có chủ đề nào dưới 70%.")
    return "\n".join(parts)


def show_progress(progress):
    return progress_md(progress)


def reset_progress():
    return None, progress_md(None)


# ======================================================================
# QUIZ: LUỒNG CHƠI
# ======================================================================
def hide():
    return gr.update(visible=False)


def pack(state, hud="", question="", choices=None, feedback="", hint=None, nxt=None, retry=None, round_btn=None, file=None):
    """Gói kết quả theo đúng thứ tự các thành phần đầu ra của quiz (10 thành phần)."""
    return (
        state,
        hud,
        question,
        choices if choices is not None else hide(),
        feedback,
        hint if hint is not None else hide(),
        nxt if nxt is not None else hide(),
        retry if retry is not None else hide(),
        round_btn if round_btn is not None else hide(),
        file if file is not None else hide(),
    )


def boss_ratio(round_no: int) -> float:
    """Tỉ lệ số câu cần đúng để hạ quái vật: tăng dần theo màn."""
    return min(BOSS_HP_MAX_RATIO, BOSS_HP_RATIO + BOSS_HP_STEP * (round_no - 1))


def round_difficulty(base: str, round_no: int) -> str:
    """Độ khó tăng một bậc sau mỗi 2 màn (tối đa 'Khó')."""
    idx = DIFFICULTY_ORDER.index(base) if base in DIFFICULTY_ORDER else 1
    return DIFFICULTY_ORDER[min(len(DIFFICULTY_ORDER) - 1, idx + (round_no - 1) // 2)]


def new_quiz_state(
    questions: list, topic: str, mode: str = MODE_BOSS, retry: bool = False,
    ctx: dict | None = None, round_no: int = 1, cleared: bool = False,
) -> dict:
    ratio = boss_ratio(round_no)
    boss_hp = max(1, math.ceil(ratio * len(questions)))
    if ctx is None:
        ctx = {
            "kind": "sample", "raw_topic": "", "label": topic, "difficulty": "Vừa",
            "source": None, "count": len(questions), "mode": mode, "seen": [],
        }
    return {
        "questions": questions,
        "index": 0,
        "score": 0,
        "topic": topic,
        "log": [],
        "answered": False,
        "finished": False,
        "over": False,
        "retry": retry,
        "mode": "boss" if mode == MODE_BOSS else "practice",
        "lives": MAX_LIVES,
        "boss": BOSSES[(round_no - 1) % len(BOSSES)],
        "boss_hp": boss_hp,
        "boss_hp_max": boss_hp,
        "ratio": ratio,
        "streak": 0,
        "best_streak": 0,
        "hints_left": HINTS_PER_QUIZ,
        "hinted": False,
        "xp": 0,
        "round": round_no,
        "cleared": cleared,  # đã vượt màn này (để biết "Ván tiếp theo" có lên màn mới không)
        "ctx": ctx,
    }


def hud_md(state) -> str:
    """HUD hiển thị bằng HTML: các viên thuốc thông tin và thanh tiến độ."""
    if not state:
        return ""
    total = len(state["questions"])
    boss_mode = state["mode"] == "boss"
    pills = [f"🗺️ Màn {state['round']}"]
    if boss_mode:
        lives = max(0, state["lives"])
        pills.append("❤️" * lives + "🖤" * (MAX_LIVES - lives))
        pills.append(f"{state['boss']['emoji']} {state['boss']['name']}")
    pills += [f"🔥 Chuỗi {state['streak']}", f"⭐ +{state['xp']} XP", f"💡 ×{state['hints_left']}"]
    html = '<div class="hud">' + "".join(f'<span class="pill">{p}</span>' for p in pills) + "</div>"
    if boss_mode:
        pct = int(state["boss_hp"] / max(1, state["boss_hp_max"]) * 100)
        html += (
            f'<div class="barlabel">Máu quái vật: {state["boss_hp"]}/{state["boss_hp_max"]}</div>'
            f'<div class="bar hp"><span style="width:{pct}%"></span></div>'
        )
    done = len(state["log"])
    html += (
        f'<div class="barlabel">Tiến độ ván: {done}/{total} câu</div>'
        f'<div class="bar"><span style="width:{int(done / max(1, total) * 100)}%"></span></div>'
    )
    return html


def intro_text(state) -> str:
    if state["mode"] == "boss":
        return (
            f"⚔️ **Màn {state['round']}:** {state['boss']['name']} {state['boss']['emoji']} xuất hiện! "
            f"Mỗi câu đúng làm quái vật yếu đi, chuỗi 3 câu đúng gây đòn chí mạng. "
            f"Bạn có {MAX_LIVES} mạng và {HINTS_PER_QUIZ} lượt gợi ý 50/50. "
            "**Bấm vào một đáp án là được chấm ngay.**"
        )
    return (
        f"📖 Chế độ luyện tập (màn {state['round']}): không mất mạng, cứ thoải mái học nhé. "
        f"Bạn có {HINTS_PER_QUIZ} lượt gợi ý 50/50. **Bấm vào một đáp án là được chấm ngay.**"
    )


def labels_of(q: dict) -> list:
    return [f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"])]


def show_question(state: dict):
    q = state["questions"][state["index"]]
    text = f"### Câu {state['index'] + 1}/{len(state['questions'])}\n{q['question']}"
    radio = gr.update(choices=labels_of(q), value=None, visible=True, interactive=True)
    return text, radio


def hint_update(state):
    return gr.update(visible=state["hints_left"] > 0, value=f"💡 Gợi ý 50/50 (còn {state['hints_left']})")


def begin(state: dict, note: str = ""):
    question_md, radio = show_question(state)
    intro = intro_text(state) + (f"\n\n{note}" if note else "")
    return pack(state, hud_md(state), question_md, radio, intro, hint=hint_update(state))


def start_quiz(topic, count, difficulty, mode, pasted, file_path, subject=None, grade=None):
    topic = (topic or "").strip()
    subject = subject or SUBJECT
    try:
        source, label = None, topic
        if file_path:
            source = read_file(file_path)
            label = os.path.basename(file_path)
        elif (pasted or "").strip():
            source = pasted.strip()[:MAX_DOC_CHARS]
            label = "Văn bản dán vào"
        questions = generate_questions(topic, int(count), difficulty, source, subject=subject, grade=grade)
    except Exception as e:
        return pack(None, "", friendly_error(e))
    ctx = {
        "kind": "ai", "raw_topic": topic, "label": label or subject, "difficulty": difficulty,
        "source": source, "count": int(count), "mode": mode, "subject": subject, "grade": grade,
        "seen": [q["question"] for q in questions][-SEEN_MEMORY:],
    }
    return begin(new_quiz_state(questions, ctx["label"], mode, ctx=ctx))


def start_sample(count, mode, subject=None):
    subject = subject or SAMPLE_DEFAULT_SUBJECT
    questions, seen, _ = pick_sample([], count, subject)
    ctx = {
        "kind": "sample", "raw_topic": "", "label": f"Câu hỏi mẫu · {subject}", "difficulty": "Vừa",
        "source": None, "count": int(count), "mode": mode, "subject": subject, "seen": seen,
    }
    return begin(new_quiz_state(questions, ctx["label"], mode, ctx=ctx))


def collapse_if_started(state):
    """Thu gọn phần cài đặt khi ván chơi đã bắt đầu."""
    return gr.update(open=not state)


def next_round(state):
    """Ván tiếp theo: câu hỏi chưa gặp, quái vật mạnh hơn nếu đã vượt màn trước."""
    keep = (gr.update(),) * 9
    if not state or not state.get("finished"):
        return (state,) + keep
    ctx = state["ctx"]
    rnd = state["round"] + (1 if state.get("cleared") else 0)
    difficulty = round_difficulty(ctx["difficulty"], rnd)
    note = ""
    try:
        if ctx["kind"] == "sample":
            questions, ctx["seen"], reshuffled = pick_sample(ctx["seen"], ctx["count"], ctx.get("subject"))
            if reshuffled:
                note = "🔄 Bạn đã gặp hết bộ câu hỏi mẫu, mình xáo lại từ đầu nhé!"
        else:
            questions = generate_questions(
                ctx["raw_topic"], ctx["count"], difficulty, ctx["source"], avoid=ctx["seen"],
                subject=ctx.get("subject"), grade=ctx.get("grade"),
            )
            ctx["seen"] = (ctx["seen"] + [q["question"] for q in questions])[-SEEN_MEMORY:]
    except Exception as e:
        gr.Warning(friendly_error(e).replace("⚠️ ", ""))
        return (state,) + keep
    if ctx["kind"] == "ai":
        note = f"🎚️ Độ khó màn này: **{difficulty}**."
    return begin(new_quiz_state(questions, ctx["label"], ctx["mode"], ctx=ctx, round_no=rnd), note)


def use_hint(state):
    """Gợi ý 50/50: loại bỏ 2 đáp án sai."""
    keep4 = tuple(gr.update() for _ in range(3))
    if not state or state.get("answered") or state.get("finished"):
        return (state,) + keep4 + (gr.update(),)
    if state["hinted"]:
        return state, hud_md(state), "💡 Câu này đã dùng gợi ý rồi.", gr.update(), gr.update()
    if state["hints_left"] <= 0:
        return state, hud_md(state), "💡 Bạn đã hết lượt gợi ý.", gr.update(), hint_update(state)
    q = state["questions"][state["index"]]
    wrong = [i for i in range(4) if i != q["answer"]]
    removed = set(random.sample(wrong, 2))
    labels = labels_of(q)
    remaining = [labels[i] for i in range(4) if i not in removed]
    state["hints_left"] -= 1
    state["hinted"] = True
    return (
        state,
        hud_md(state),
        "💡 Đã loại bỏ 2 đáp án sai! (câu này nhận ít XP hơn)",
        gr.update(choices=remaining, value=None),
        hint_update(state),
    )


def check_answer(state, progress, choice):
    """Chấm ngay khi chọn đáp án. Đầu ra: state, progress, hud, feedback, choices, hint_btn, next_btn."""
    progress = ensure_progress(progress)
    if not state:
        return (state, progress, gr.update(), "Hãy tạo quiz trước.", gr.update(), gr.update(), gr.update())
    if state.get("answered") or state.get("finished"):
        return (state, progress) + (gr.update(),) * 5
    q = state["questions"][state["index"]]
    labels = labels_of(q)
    if choice not in labels:
        return (state, progress, gr.update(), "👆 Hãy chọn một đáp án.", gr.update(), gr.update(), gr.update())

    chosen = labels.index(choice)
    ok = chosen == q["answer"]
    state["answered"] = True
    extra = ""
    if ok:
        state["score"] += 1
        state["streak"] += 1
        state["best_streak"] = max(state["best_streak"], state["streak"])
        progress["best_streak"] = max(progress["best_streak"], state["streak"])
        gained = 5 if state["hinted"] else 10
        combo = 10 if state["streak"] >= 5 else 5 if state["streak"] >= 3 else 0
        gained += combo
        state["xp"] += gained
        verdict = f"✅ **{random.choice(CORRECT_MSGS)}** +{gained} XP"
        if combo:
            verdict += f" (thưởng chuỗi {state['streak']} câu 🔥)"
        if state["mode"] == "boss":
            damage = 2 if state["streak"] >= 3 else 1
            state["boss_hp"] = max(0, state["boss_hp"] - damage)
            verdict += "\n\n💥 Đòn chí mạng!" if damage == 2 else "\n\n💥 Quái vật bị thương!"
        extra += add_xp(progress, gained)
        if state["streak"] >= 3:
            extra += award(progress, "streak3")
        if state["streak"] >= 5:
            extra += award(progress, "streak5")
    else:
        state["streak"] = 0
        verdict = f"❌ **{random.choice(WRONG_MSGS)}** Đáp án đúng: {LETTERS[q['answer']]}. {q['options'][q['answer']]}"
        if state["mode"] == "boss":
            state["lives"] -= 1
            verdict += f"\n\n💔 Mất 1 mạng, còn {max(state['lives'], 0)} mạng."
            if state["lives"] <= 0:
                state["over"] = True
    state["log"].append({"q": q, "chosen": chosen, "ok": ok})

    tag = (q.get("tag") or state["topic"]).replace("|", "/")
    progress["answers"].append({"tag": tag, "ok": ok, "ts": time.time()})
    progress["answers"] = progress["answers"][-MAX_ANSWERS:]
    mark_today(progress)
    if ok:
        forget_mistake(progress, q)  # làm đúng thì bỏ khỏi sổ tay lỗi sai
    else:
        remember_mistake(progress, q)
    if today_count(progress) >= DAILY_GOAL:
        extra += award(progress, "daily_goal")

    body = f"{verdict}\n\n{q['explanation']}"
    if q.get("verified") is True and q.get("evidence"):
        body += f"\n\n📖 Trích từ tài liệu: “{q['evidence']}”"
    elif q.get("verified") is False:
        body += "\n\n⚠️ Chưa xác minh được câu này với tài liệu, hãy tự đối chiếu."
    if q.get("tag"):
        body += f"\n\n🏷️ Chủ đề: {q['tag']}"
    box = "fb-ok" if ok else "fb-bad"
    feedback = f'<div class="fb {box}">\n\n{body}\n\n</div>' + extra

    is_last = state["index"] == len(state["questions"]) - 1 or state["over"]
    return (
        state,
        progress,
        hud_md(state),
        feedback,
        gr.update(interactive=False),
        hide(),
        gr.update(visible=True, value="Xem kết quả 🏁" if is_last else "Câu tiếp ▶"),
    )


def show_explain(state):
    return gr.update(visible=bool(state and state.get("answered")))


def clear_explain():
    return "", gr.update(visible=False)


def explain_more(state, subject=None, grade=None):
    """Nhờ AI giải thích lại câu vừa làm, dễ hiểu hơn."""
    if not state or not state.get("log"):
        return "Hãy trả lời một câu hỏi trước nhé."
    e = state["log"][-1]
    q = e["q"]
    prompt = (
        f"Câu hỏi: {q['question']}\n"
        f"Đáp án đúng: {q['options'][q['answer']]}\n"
        f"Học sinh đã chọn: {q['options'][e['chosen']]}\n"
        "Hãy giải thích lại thật dễ hiểu bằng tiếng Việt: (1) vì sao đáp án đúng là đúng, "
        "(2) nếu học sinh chọn sai thì vì sao phương án đó chưa đúng, "
        "(3) một ví dụ đời thường hoặc mẹo ghi nhớ. Tối đa 8 câu."
    )
    try:
        text = ask_llm(
            [{"role": "system", "content": chat_system(subject, grade)}, {"role": "user", "content": prompt}],
            max_tokens=600,
        )
    except Exception as ex:
        return friendly_error(ex)
    return "🧠 **Giải thích thêm**\n\n" + text


def build_report(state: dict) -> str:
    total = len(state["questions"])
    lines = [
        f"# Kết quả ôn tập: {state['topic']} (màn {state['round']})",
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
    pending = state["questions"][len(state["log"]):]
    if pending:
        lines.append("## Các câu chưa làm (để tự ôn)")
        for q in pending:
            lines.append(f"- {q['question']} → **{q['options'][q['answer']]}**")
    return "\n".join(lines)


def finish_quiz(state: dict, progress: dict) -> str:
    """Tổng kết ván chơi: cộng thưởng, trao huy hiệu, tạo thông báo."""
    total = len(state["questions"])
    score = state["score"]
    completed = len(state["log"]) == total
    perfect = completed and score == total and total >= 3
    won = state["mode"] == "boss" and not state["over"] and state["boss_hp"] <= 0
    if won or state["mode"] == "practice":
        state["cleared"] = True

    bonus = 5
    if perfect:
        bonus += 20
    if won:
        bonus += 30
    state["xp"] += bonus
    extra = add_xp(progress, bonus)
    progress["quizzes"].append({"topic": state["topic"], "score": score, "total": total, "ts": time.time()})
    extra += award(progress, "first_quiz")
    if len(progress["quizzes"]) >= 3:
        extra += award(progress, "marathon")
    if perfect:
        extra += award(progress, "perfect")
    if won:
        progress["best_round"] = max(progress["best_round"], state["round"])
        extra += award(progress, "boss_win")
        if state["round"] >= 3:
            extra += award(progress, "campaign3")
    if state["retry"] and completed and score == total:
        extra += award(progress, "comeback")
    if len({a["tag"] for a in progress["answers"]}) >= 5:
        extra += award(progress, "explorer")
    mark_today(progress)
    if day_streak(progress) >= 3:
        extra += award(progress, "streak_days3")

    boss = state["boss"]
    if state["mode"] == "boss":
        if won:
            head = f"## 🏆 Chiến thắng! Bạn đã đánh bại {boss['name']} {boss['emoji']} ở màn {state['round']}"
        elif state["over"]:
            head = (
                f"## 💫 Hết mạng rồi! {boss['name']} thắng ván này, nhưng bạn đã học được rất nhiều. "
                "Thử lại nhé!"
            )
        else:
            head = (
                f"## 😅 {boss['name']} vẫn còn sức! Cần đúng khoảng {int(state['ratio'] * 100)}% số câu "
                "để chiến thắng. Thử lại nhé!"
            )
    else:
        ratio = score / total
        comment = "Xuất sắc! 🎉" if ratio == 1 else "Khá tốt, cố thêm chút nữa nhé!" if ratio >= 0.7 else "Hãy ôn lại các phần còn yếu rồi thử lại nhé!"
        head = f"## 🏁 Kết quả: {score}/{total}. {comment}"

    result = f"{head}\n\n**Điểm:** {score}/{total} · **XP ván này:** +{state['xp']} · **Chuỗi tốt nhất:** {state['best_streak']}"
    wrong = [e for e in state["log"] if not e["ok"]]
    if wrong:
        result += "\n\n**Các câu cần ôn lại:**\n"
        for e in wrong[:10]:
            q = e["q"]
            result += f"- {q['question']} → **{q['options'][q['answer']]}**\n"
    pending = total - len(state["log"])
    if pending:
        result += f"\n\nCòn {pending} câu chưa làm, đáp án có trong báo cáo tải về để bạn tự ôn."
    if state["mode"] == "boss":
        if state["cleared"]:
            nxt = state["round"] + 1
            result += (
                f"\n\n🗺️ Màn {state['round']} hoàn thành! Ở màn {nxt}, quái vật sẽ mạnh hơn "
                f"(cần đúng khoảng {int(boss_ratio(nxt) * 100)}% số câu). Bấm **Ván tiếp theo** để tiếp tục."
            )
        else:
            result += "\n\n🗺️ Chưa vượt được màn này. Bấm nút bên dưới để thử lại với câu hỏi mới."
    else:
        result += "\n\n🗺️ Bấm **Ván tiếp theo** để nhận câu hỏi mới chưa gặp."
    cnt = min(today_count(progress), DAILY_GOAL)
    result += (
        f"\n\n🎯 Mục tiêu hôm nay: **{cnt}/{DAILY_GOAL} câu** {goal_bar(progress)}"
        f"  ·  📅 Chuỗi ngày học: **{day_streak(progress)}** ngày"
    )
    result += extra
    result += share_text(state, progress, won)
    if len(progress["quizzes"]) % REST_EVERY == 0:
        result += "\n\n💧 Bạn đã chơi khá nhiều ván rồi! Hãy nghỉ 5 phút, uống nước và vận động nhẹ cho đỡ mỏi mắt nhé."
    result += '\n\n<div class="tip">\n\n' + random.choice(STUDY_TIPS) + "\n\n</div>"
    return result


def next_question(state, progress):
    """Đầu ra: state, progress, hud, question_md, choices, feedback, hint, next, retry, round_btn, file."""
    progress = ensure_progress(progress)
    if not state or not state.get("answered") or state.get("finished"):
        return (state, progress) + (gr.update(),) * 9

    if not state["over"] and state["index"] < len(state["questions"]) - 1:
        state["index"] += 1
        state["answered"] = False
        state["hinted"] = False
        question_md, radio = show_question(state)
        return (
            state, progress, hud_md(state), question_md, radio, "",
            hint_update(state), hide(), hide(), hide(), hide(),
        )

    state["finished"] = True
    result = finish_quiz(state, progress)
    has_wrong = any(not e["ok"] for e in state["log"])
    path = write_temp(build_report(state), "ketqua")
    if state["cleared"]:
        round_label = f"Ván tiếp theo ▶ (màn {state['round'] + 1})"
    else:
        round_label = f"Thử lại màn {state['round']} với câu mới ▶"
    return (
        state,
        progress,
        hud_md(state),
        result,
        hide(),
        "",
        hide(),
        hide(),
        gr.update(visible=has_wrong),
        gr.update(visible=True, value=round_label),
        gr.update(value=path, visible=True),
    )


def retry_wrong(state):
    wrong = [e["q"] for e in (state or {}).get("log", []) if not e["ok"]]
    questions = [q for q in (normalize_question(q) for q in wrong) if q]
    if not questions:
        return pack(state, hud_md(state), "Không có câu sai để ôn lại.")
    base = str(state["topic"]).replace(" (ôn câu sai)", "")
    mode = MODE_BOSS if state["mode"] == "boss" else MODE_PRACTICE
    return begin(
        new_quiz_state(
            questions, f"{base} (ôn câu sai)", mode, retry=True,
            ctx=state["ctx"], round_no=state["round"], cleared=state.get("cleared", False),
        )
    )


# ======================================================================
# THẺ GHI NHỚ
# ======================================================================
def cpack(state, card="", flip=None, know=None, again=None):
    return (
        state,
        card,
        flip if flip is not None else hide(),
        know if know is not None else hide(),
        again if again is not None else hide(),
    )


def questions_to_cards(questions: list) -> list:
    return [
        {
            "front": q["question"],
            "back": f"**{q['options'][q['answer']]}**\n\n{q['explanation']}".strip(),
            "tag": q.get("tag", ""),
        }
        for q in questions
    ]


def card_md(cs: dict, flipped: bool) -> str:
    card = cs["cards"][cs["queue"][0]]
    head = f"Thẻ còn lại: **{len(cs['queue'])}** · Đã nhớ: **{cs['known']}/{cs['total']}**\n\n"
    if flipped:
        return head + f"### 🃏 Đáp án\n{card['back']}"
    return head + f"### 🃏 Câu hỏi\n{card['front']}"


def cards_begin(cards: list):
    if not cards:
        return cpack(None, "Không có thẻ nào để học.")
    cs = {"cards": cards, "queue": list(range(len(cards))), "known": 0, "total": len(cards), "flipped": False}
    return cpack(cs, card_md(cs, False), flip=gr.update(visible=True))


def start_cards(topic, count, subject=None, grade=None):
    try:
        questions = generate_questions((topic or "").strip(), int(count), "Vừa", None, subject=subject, grade=grade)
    except Exception as e:
        return cpack(None, friendly_error(e))
    return cards_begin(questions_to_cards(questions))


def start_cards_sample(count, subject=None):
    pool = sample_pool(subject or SAMPLE_DEFAULT_SUBJECT)
    bank = [normalize_question(SAMPLE_BANK[i]) for i in pool]
    bank = [q for q in bank if q]
    return cards_begin(questions_to_cards(random.sample(bank, min(int(count), len(bank)))))


def cards_from_mistakes(progress):
    """Biến sổ tay lỗi sai thành thẻ ghi nhớ."""
    p = ensure_progress(progress)
    questions = [q for q in (normalize_question(m) for m in p["mistakes"]) if q]
    if not questions:
        return cpack(None, "📝 Sổ tay lỗi sai đang trống. Làm quiz, câu nào sai sẽ tự được lưu vào đây để bạn ôn lại.")
    return cards_begin(questions_to_cards(questions))


def cards_from_quiz(quiz_state):
    if not quiz_state or not quiz_state.get("questions"):
        return cpack(None, "Hãy chơi một quiz trước, rồi quay lại đây để biến các câu hỏi thành thẻ ghi nhớ.")
    return cards_begin(questions_to_cards(quiz_state["questions"]))


def card_flip(cs):
    if not cs or not cs["queue"] or cs["flipped"]:
        return (cs,) + tuple(gr.update() for _ in range(4))
    cs["flipped"] = True
    return cpack(cs, card_md(cs, True), know=gr.update(visible=True), again=gr.update(visible=True))


def card_again(cs):
    if not cs or not cs["queue"] or not cs["flipped"]:
        return (cs,) + tuple(gr.update() for _ in range(4))
    cs["queue"].append(cs["queue"].pop(0))  # thẻ chưa nhớ quay lại cuối hàng
    cs["flipped"] = False
    return cpack(cs, card_md(cs, False), flip=gr.update(visible=True))


def card_know(cs, progress):
    progress = ensure_progress(progress)
    if not cs or not cs["queue"] or not cs["flipped"]:
        return (cs, progress) + tuple(gr.update() for _ in range(4))
    cs["queue"].pop(0)
    cs["known"] += 1
    progress["cards_known"] += 1
    mark_today(progress)
    extra = add_xp(progress, 2)
    if progress["cards_known"] >= 10:
        extra += award(progress, "cards")
    if cs["queue"]:
        cs["flipped"] = False
        out = cpack(cs, card_md(cs, False) + extra, flip=gr.update(visible=True))
    else:
        extra += add_xp(progress, 10)
        out = cpack(cs, f"## 🎉 Bạn đã nhớ hết {cs['total']} thẻ! (+10 XP thưởng hoàn thành){extra}")
    return (out[0], progress) + out[1:]


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
            "- Thứ 7: chơi một ván đấu quái vật 10 câu để tự kiểm tra, rồi ôn lại câu sai",
            "- Chủ nhật: nghỉ ngơi, xem lại ghi chú hoặc sơ đồ tư duy",
            "",
        ]
    return "\n".join(lines)


def plan_fn(goal, weeks, hours, level, use_progress, progress, subject=None, grade=None):
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
        f"Môn học: {subject or SUBJECT} {grade or ''}\n"
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
    "Giải thích quang hợp như thể mình 10 tuổi",
    "Phân biệt khối lượng và trọng lượng",
    "Cho mình mẹo nhớ các thì trong tiếng Anh",
    "Vì sao nước đá nổi trên mặt nước?",
]
QUICK_PROMPTS = [
    ("🧒 Giải thích dễ hiểu", "Giải thích thật dễ hiểu như mình 10 tuổi: "),
    ("🧩 Cho ví dụ", "Cho mình một ví dụ đời thường về: "),
    ("🧠 Mẹo ghi nhớ", "Cho mình mẹo ghi nhớ nhanh về: "),
    ("📌 Tóm tắt 5 ý", "Tóm tắt bài này trong 5 ý chính: "),
]

HERO_HTML = (
    '<div id="hero"><h1>📚 StudyBuddy</h1>'
    "<p>Ôn bài kiểu chơi game: đấu quái vật, lên cấp, giữ chuỗi ngày học 🔥</p></div>"
)

CSS = """
.gradio-container {max-width: 860px !important; margin: auto;}
#hero {background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #ec4899 100%);
       border-radius: 18px; padding: 18px 22px; margin-bottom: 6px;}
#hero h1 {color: #fff; margin: 0 0 4px; font-size: 1.7em;}
#hero p {color: #fff; margin: 0; opacity: .95;}
.hud {display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px;}
.pill {background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
       border-radius: 999px; padding: 3px 12px; font-weight: 600; font-size: .92em;}
.barlabel {font-size: .82em; opacity: .75; margin: 4px 0 2px;}
.bar {height: 10px; border-radius: 99px; overflow: hidden; background: var(--background-fill-secondary);
      border: 1px solid var(--border-color-primary);}
.bar > span {display: block; height: 100%; border-radius: 99px; transition: width .4s ease;
             background: linear-gradient(90deg, #22c55e, #84cc16);}
.bar.hp > span {background: linear-gradient(90deg, #ef4444, #f97316);}
.fb {border-radius: 12px; padding: 4px 16px; border-left: 6px solid; margin: 6px 0;}
.fb-ok {background: rgba(34, 197, 94, .12); border-color: #22c55e;}
.fb-bad {background: rgba(239, 68, 68, .10); border-color: #ef4444;}
.tip {background: rgba(99, 102, 241, .10); border-radius: 12px; padding: 2px 14px; font-size: .95em;}
#choices .wrap {flex-direction: column !important; gap: 8px !important;}
#choices label {width: 100%; border: 2px solid var(--border-color-primary) !important; border-radius: 14px !important;
                padding: 12px 14px !important; font-size: 1.05em; cursor: pointer; transition: all .15s;}
#choices label:hover {border-color: #8b5cf6 !important; transform: translateY(-1px);}
#start-btn {font-size: 1.1em;}
@media (max-width: 640px) {
  .gradio-container {padding: 6px !important;}
  #hero {padding: 14px 16px;} #hero h1 {font-size: 1.3em;}
  button {min-height: 46px;}
}
"""


def make_blocks():
    try:
        return gr.Blocks(theme=gr.themes.Soft(), css=CSS)
    except Exception:
        return gr.Blocks()


def make_progress_state():
    """Tiến độ lưu ngay trong trình duyệt của học sinh (nếu Gradio hỗ trợ), nếu không thì chỉ trong phiên."""
    try:
        return gr.BrowserState(None)
    except Exception:
        return gr.State(None)


def make_chatbot():
    try:
        return gr.Chatbot(type="messages", height=380, label="Trợ lý ôn tập")
    except TypeError:
        return gr.Chatbot(height=380, label="Trợ lý ôn tập")


def suggest_topic(subject):
    return random.choice(SUBJECTS.get(subject) or [subject or SUBJECT])


def topic_placeholder(subject):
    return gr.update(placeholder=f"Ví dụ: {suggest_topic(subject)}")


def make_filler(text):
    return lambda: text


def build_ui():
    with make_blocks() as demo:
        progress = make_progress_state()

        gr.HTML(HERO_HTML)
        welcome = gr.Markdown(welcome_md(None))
        with gr.Accordion("❓ Cách chơi (4 bước)", open=False):
            gr.Markdown(
                "① Chọn **môn** và **lớp**  ② Nhập chủ đề rồi bấm **Bắt đầu**  "
                "③ Bấm vào đáp án để được chấm ngay  ④ Hạ quái vật, nhận XP, lên cấp và giữ **chuỗi ngày học** 🔥\n\n"
                "Câu nào sai sẽ tự vào **sổ tay lỗi sai** (tab Thẻ ghi nhớ). Không có mạng hoặc AI? Bấm **Chơi thử ngay**."
            )
        with gr.Row():
            subject = gr.Dropdown(list(SUBJECTS), value=DEFAULT_SUBJECT, label="📘 Môn học", scale=2)
            grade = gr.Dropdown(GRADES, value=DEFAULT_GRADE, label="🎓 Lớp", scale=1)

        # ---------------- Quiz
        with gr.Tab("🎮 Quiz ôn tập"):
            quiz_state = gr.State()
            with gr.Accordion("⚙️ Chọn bài học và bắt đầu (bấm để mở/đóng)", open=True) as setup:
                with gr.Row():
                    topic = gr.Textbox(
                        label="Chủ đề bạn muốn ôn",
                        placeholder=f"Ví dụ: {(SUBJECTS[DEFAULT_SUBJECT] or [DEFAULT_SUBJECT])[0]}",
                        scale=4,
                    )
                    suggest_btn = gr.Button("🎲 Gợi ý chủ đề", scale=1)
                count = gr.Slider(1, 10, value=5, step=1, label="Số câu mỗi ván")
                with gr.Row():
                    start_btn = gr.Button("▶ Bắt đầu (AI tạo câu hỏi)", variant="primary", elem_id="start-btn")
                    sample_btn = gr.Button("🎲 Chơi thử ngay (không cần AI)")
                with gr.Accordion("Tùy chọn nâng cao: độ khó, chế độ, dán bài, tải file", open=False):
                    difficulty = gr.Dropdown(["Dễ", "Vừa", "Khó"], value="Vừa", label="Mức độ bắt đầu")
                    mode = gr.Dropdown([MODE_BOSS, MODE_PRACTICE], value=MODE_BOSS, label="Chế độ chơi")
                    pasted = gr.Textbox(
                        label="Hoặc dán nội dung bài học vào đây (nếu có, câu hỏi sẽ dựa trên đoạn này)",
                        lines=4,
                    )
                    file_in = gr.File(label="Hoặc tải tài liệu (.txt, .md, .pdf)", type="filepath")

            hud = gr.HTML()
            question_md = gr.Markdown()
            choices = gr.Radio(label="Chọn đáp án (bấm vào là chấm ngay)", visible=False, elem_id="choices")
            feedback = gr.Markdown()
            with gr.Row():
                hint_btn = gr.Button("💡 Gợi ý 50/50", visible=False)
                explain_btn = gr.Button("🧠 Giải thích dễ hiểu hơn", visible=False)
                next_btn = gr.Button("Câu tiếp ▶", visible=False, variant="primary")
            explain_md = gr.Markdown()
            with gr.Row():
                round_btn = gr.Button("Ván tiếp theo ▶", visible=False, variant="primary")
                retry_btn = gr.Button("🔁 Ôn lại các câu sai", visible=False)
            result_file = gr.File(label="Tải báo cáo kết quả (.md)", visible=False)

            quiz_outputs = [
                quiz_state, hud, question_md, choices, feedback,
                hint_btn, next_btn, retry_btn, round_btn, result_file,
            ]
            clear_targets = [explain_md, explain_btn]

            suggest_btn.click(suggest_topic, [subject], topic)
            subject.change(topic_placeholder, [subject], topic)
            start_btn.click(
                start_quiz, [topic, count, difficulty, mode, pasted, file_in, subject, grade], quiz_outputs
            ).then(collapse_if_started, [quiz_state], setup).then(clear_explain, None, clear_targets)
            sample_btn.click(start_sample, [count, mode, subject], quiz_outputs).then(
                collapse_if_started, [quiz_state], setup
            ).then(clear_explain, None, clear_targets)
            choices.input(
                check_answer,
                [quiz_state, progress, choices],
                [quiz_state, progress, hud, feedback, choices, hint_btn, next_btn],
            ).then(show_explain, [quiz_state], explain_btn).then(welcome_md, [progress], welcome)
            hint_btn.click(use_hint, [quiz_state], [quiz_state, hud, feedback, choices, hint_btn])
            explain_btn.click(explain_more, [quiz_state, subject, grade], explain_md)
            next_btn.click(
                next_question,
                [quiz_state, progress],
                [quiz_state, progress, hud, question_md, choices, feedback,
                 hint_btn, next_btn, retry_btn, round_btn, result_file],
            ).then(clear_explain, None, clear_targets).then(welcome_md, [progress], welcome)
            round_btn.click(next_round, [quiz_state], quiz_outputs).then(clear_explain, None, clear_targets)
            retry_btn.click(retry_wrong, [quiz_state], quiz_outputs).then(clear_explain, None, clear_targets)

        # ---------------- Thẻ ghi nhớ
        with gr.Tab("🃏 Thẻ ghi nhớ"):
            gr.Markdown(
                "Lật thẻ, tự đánh giá **Nhớ rồi** hoặc **Chưa nhớ**. Thẻ chưa nhớ sẽ quay lại sau để bạn ôn tiếp. "
                "Nút **Sổ tay lỗi sai** gom các câu bạn từng làm sai."
            )
            with gr.Row():
                cards_topic = gr.Textbox(label="Chủ đề", placeholder="Ví dụ: Tế bào", scale=3)
                cards_count = gr.Slider(3, 10, value=5, step=1, label="Số thẻ", scale=1)
            with gr.Row():
                cards_ai_btn = gr.Button("Tạo thẻ bằng AI", variant="primary")
                cards_mistake_btn = gr.Button("📝 Sổ tay lỗi sai")
            with gr.Row():
                cards_sample_btn = gr.Button("Thẻ mẫu (không cần AI)")
                cards_quiz_btn = gr.Button("Từ quiz vừa chơi")
            cards_state = gr.State()
            card_out = gr.Markdown()
            with gr.Row():
                flip_btn = gr.Button("🔄 Lật thẻ", visible=False)
                know_btn = gr.Button("✅ Nhớ rồi", visible=False)
                again_btn = gr.Button("🔁 Chưa nhớ", visible=False)

            card_outputs = [cards_state, card_out, flip_btn, know_btn, again_btn]
            cards_ai_btn.click(start_cards, [cards_topic, cards_count, subject, grade], card_outputs)
            cards_sample_btn.click(start_cards_sample, [cards_count, subject], card_outputs)
            cards_quiz_btn.click(cards_from_quiz, [quiz_state], card_outputs)
            cards_mistake_btn.click(cards_from_mistakes, [progress], card_outputs)
            flip_btn.click(card_flip, [cards_state], card_outputs)
            know_btn.click(
                card_know, [cards_state, progress], [cards_state, progress, card_out, flip_btn, know_btn, again_btn]
            ).then(welcome_md, [progress], welcome)
            again_btn.click(card_again, [cards_state], card_outputs)

        # ---------------- Hồ sơ và tiến độ
        with gr.Tab("🏆 Hồ sơ & tiến độ") as tab_progress:
            progress_out = gr.Markdown(progress_md(None))
            with gr.Row():
                refresh_btn = gr.Button("Cập nhật")
                reset_btn = gr.Button("Xóa dữ liệu và chơi lại từ đầu")
            refresh_btn.click(show_progress, [progress], progress_out)
            reset_btn.click(reset_progress, None, [progress, progress_out]).then(welcome_md, [progress], welcome)
            try:
                tab_progress.select(show_progress, [progress], progress_out)
            except Exception:
                pass  # phiên bản Gradio không hỗ trợ: vẫn còn nút Cập nhật

        # ---------------- Kế hoạch
        with gr.Tab("🗓️ Kế hoạch học tập"):
            goal = gr.Textbox(
                label="Mục tiêu của bạn",
                lines=2,
                placeholder="Ví dụ: Ôn thi cuối kỳ, đạt 8 điểm trở lên",
            )
            with gr.Row():
                weeks = gr.Slider(1, 12, value=3, step=1, label="Số tuần")
                hours = gr.Slider(0.5, 6, value=1.5, step=0.5, label="Giờ học mỗi ngày")
                level = gr.Dropdown(["Mới bắt đầu", "Trung bình", "Khá"], value="Trung bình", label="Trình độ")
            use_progress = gr.Checkbox(value=True, label="Ưu tiên các chủ đề tôi còn yếu (dựa trên kết quả quiz)")
            plan_btn = gr.Button("Lập kế hoạch", variant="primary")
            plan_out = gr.Markdown()
            plan_file = gr.File(label="Tải kế hoạch (.md)", visible=False)
            plan_btn.click(
                plan_fn, [goal, weeks, hours, level, use_progress, progress, subject, grade], [plan_out, plan_file]
            )

        # ---------------- Hỏi đáp
        with gr.Tab("💬 Hỏi đáp"):
            gr.Markdown("Hỏi bất cứ điều gì về bài học. Bấm nút hỏi nhanh để điền sẵn câu mở đầu rồi gõ tiếp chủ đề.")
            chatbot = make_chatbot()
            with gr.Row():
                chat_in = gr.Textbox(
                    placeholder="Nhập câu hỏi rồi bấm Enter...", show_label=False, lines=1, scale=5
                )
                send_btn = gr.Button("Gửi ➤", variant="primary", scale=1)
            with gr.Row():
                for label, template in QUICK_PROMPTS:
                    gr.Button(label, size="sm").click(make_filler(template), None, chat_in)
            gr.Examples(CHAT_EXAMPLES, inputs=[chat_in], label="Ví dụ câu hỏi")
            clear_chat_btn = gr.Button("🧹 Xóa hội thoại", size="sm")
            chat_in.submit(chat_respond, [chat_in, chatbot, subject, grade], [chat_in, chatbot])
            send_btn.click(chat_respond, [chat_in, chatbot, subject, grade], [chat_in, chatbot])
            clear_chat_btn.click(lambda: [], None, chatbot)

        # ---------------- Kiểm tra kết nối (cất gọn, ít khi cần)
        with gr.Accordion("🔧 Kiểm tra kết nối AI", open=False):
            gr.Markdown(f"Mô hình: `{MODEL}` · Chạy qua: **{PROVIDER}**")
            with gr.Row():
                status = gr.Markdown("Bấm nút bên cạnh để kiểm tra.")
                test_btn = gr.Button("🔌 Kiểm tra")
            test_btn.click(test_ai, None, status)

        gr.Markdown(
            "<small>Lưu ý: AI có thể sai. Hãy đối chiếu đáp án với sách giáo khoa hoặc giáo viên. "
            "Ứng dụng không thu thập thông tin cá nhân; điểm, huy hiệu và sổ tay lỗi sai được lưu "
            "trong trình duyệt của bạn trên thiết bị này (hoặc chỉ trong phiên chơi nếu trình duyệt không hỗ trợ).</small>"
        )

        demo.load(show_progress, [progress], progress_out)
        demo.load(welcome_md, [progress], welcome)
    return demo


if __name__ == "__main__":
    port = os.getenv("PORT")  # các dịch vụ lưu trữ (Render...) cấp cổng qua biến PORT
    build_ui().queue(max_size=20).launch(
        server_name="0.0.0.0" if port else None,
        server_port=int(port) if port else None,
        share=os.getenv("SHARE") == "1",
    )
