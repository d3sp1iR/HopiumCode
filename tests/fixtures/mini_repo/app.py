from storage import save_user
from validation import validate_email


def register(name, email):
    """Регистрирует пользователя: проверяет почту и сохраняет запись."""
    validate_email(email)
    return save_user({"name": name, "email": email})
