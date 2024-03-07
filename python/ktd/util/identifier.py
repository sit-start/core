import string
from dataclasses import dataclass
from typing import Callable

from ktd.util.string import is_from_alphabet, int_to_str, str_to_int, rand_str
from ktd.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StringIdType:
    prefix: str
    suffix_len: int = 5
    alphabet: str = string.digits
    sequential: bool = True
    start: int = 0
    fixed_len: bool = True

    def is_valid(self, s: str) -> bool:
        suffix_len = len(self._suffix(s))
        return (
            s.startswith(self.prefix)
            and (
                suffix_len == self.suffix_len
                or (not self.fixed_len and suffix_len > self.suffix_len)
            )
            and is_from_alphabet(s[len(self.prefix) :], self.alphabet)
        )

    def _suffix(self, s: str) -> str:
        return s[len(self.prefix) :]

    def _id(self, suffix: str) -> str:
        return f"{self.prefix}{suffix}"

    def _next_sequential(self, existing: list[str]) -> str:
        assert self.sequential
        existing_vals = [
            str_to_int(self._suffix(s), self.alphabet) for s in existing
        ] + [self.start - 1]
        next_val = max(existing_vals) + 1
        suffix = int_to_str(next_val, self.alphabet, length=self.suffix_len)
        if len(suffix) > self.suffix_len:
            raise ValueError(
                f"Next ID {self._id(suffix)} is too large for fixed suffix "
                f"length {self.suffix_len}. Use a larger `suffix_len` or "
                f"set `fixed_len` to False."
            )
        return self._id(suffix)

    def _next_random(self, exists: Callable[[str], bool], max_attempts: int) -> str:
        assert not self.sequential
        suffix = rand_str(
            self.suffix_len,
            self.alphabet,
            lambda s: not exists(self._id(s)),
            max_attempts=max_attempts,
        )
        return self._id(suffix)

    def next(
        self,
        last: str | None = None,
        exists: Callable[[str], bool] | None = None,
        existing: list[str] | None = None,
        max_attempts: int = 100,
    ) -> str:
        if last is not None and not self.is_valid(last):
            raise ValueError(f"Invalid last id: {last}")
        existing = (existing or []) + ([last] if last else [])

        if self.sequential:
            if exists is not None:
                logger.warning("Ignoring `exists` argument for sequential StringIdType")
            return self._next_sequential(existing)

        exists = exists or (lambda s: False)
        return self._next_random(lambda s: s in existing or exists(s), max_attempts)


RUN_ID = StringIdType(
    prefix="R",
    suffix_len=5,
    alphabet=string.digits,
    sequential=True,
    fixed_len=True,
    start=16180,
)
