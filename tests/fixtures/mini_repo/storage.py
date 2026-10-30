_DB = {}


def next_id():
    return len(_DB) + 1


def save_user(user):
    key = next_id()
    _DB[key] = user
    return key
