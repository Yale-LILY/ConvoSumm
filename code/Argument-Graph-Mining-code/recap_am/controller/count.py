from typing import Generator


def generate(current_id: int = 1) -> Generator[int, None, None]:
    """Generate unique id"""

    while True:
        yield current_id
        current_id += 1
