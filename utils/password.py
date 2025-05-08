import random
import string
import bcrypt

def generate_temp_password(length=10):
    """ 랜덤한 임시 비밀번호 생성 (영문 + 숫자 조합) """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def hash_password(password: str) -> str:
    """ 주어진 비밀번호를 bcrypt 해시로 변환 """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')

def verify_password(input_password: str, hashed_password: str) -> bool:
    """ 입력된 비밀번호가 해시된 비밀번호와 일치하는지 확인 """
    return bcrypt.checkpw(input_password.encode('utf-8'), hashed_password.encode('utf-8'))