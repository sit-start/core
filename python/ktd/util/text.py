import re


def strip_ansi_codes(s: str) -> str:
    return re.sub(r"\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?", "", s)


def truncate(s: str, max_len: int = 50) -> str:
    suffix = "..."
    return s if len(s) <= max_len else s[: max_len - len(suffix)] + suffix


# https://gist.github.com/egmontkob/eb114294efbcd5adb1944c9f3cb5feda
def terminal_hyperlink(url: str, text: str) -> str:
    # Works as expected on iTerm2, but "file://" URLs don't work in VS Code's
    # terminal
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"
