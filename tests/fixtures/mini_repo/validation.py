class InvalidEmail(ValueError):
    """Почта без символа @."""


def validate_email(email):
    if "@" not in email:
        raise InvalidEmail(email)
    return email.lower()
