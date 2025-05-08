import smtplib
from email.mime.text import MIMEText
from email_validator import validate_email, EmailNotValidError
from fastapi import HTTPException

# 인증 코드 생성 함수
def generate_verification_code():
    import random
    return str(random.randint(100000, 999999))

# 이메일 전송 함수
def send_email(smtp_user: str, smtp_password: str, email: str, code: str):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    subject = "AI Keeper 인증 메일입니다."
    body = f"인증코드: {code}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = email

    # 이메일 유효성 검사
    try:
        valid = validate_email(email)
        email = valid.email  # 정제된 이메일 주소 반환
    except EmailNotValidError as e:
        raise HTTPException(status_code=400, detail="Invalid email address.")

    try:
        # SMTP 서버 설정 및 이메일 전송
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, email, msg.as_string())
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to send email.")
