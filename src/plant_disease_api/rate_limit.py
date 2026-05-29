import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = self._buckets[client]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()

        if len(bucket) >= self.requests_per_minute:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Rate limit exceeded. Try again later.",
            )

        bucket.append(now)
