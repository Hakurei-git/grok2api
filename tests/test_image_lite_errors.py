from types import SimpleNamespace

import pytest

from app.control.account.enums import FeedbackKind
from app.control.model.enums import ModeId
from app.platform.errors import UpstreamError
from app.products.openai import images


@pytest.mark.asyncio
async def test_lite_batch_unwraps_task_group_app_error(monkeypatch):
    async def _fail_request(**_kwargs):
        raise UpstreamError("Imagine is busy", status=429)

    monkeypatch.setattr(images, "_run_lite_request", _fail_request)

    with pytest.raises(UpstreamError) as exc_info:
        await images._run_lite_batch(
            spec=object(),
            prompt="test",
            n=1,
            timeout_s=1,
            response_format="url",
        )

    assert exc_info.value.status == 429
    assert exc_info.value.message == "Imagine is busy"


@pytest.mark.asyncio
async def test_lite_request_switches_account_after_retryable_failure(monkeypatch):
    class FakeDirectory:
        def __init__(self):
            self.accounts = [
                SimpleNamespace(token="bad-token"),
                SimpleNamespace(token="good-token"),
            ]
            self.reserve_exclusions = []
            self.released = []
            self.feedbacks = []

        async def reserve(self, *, exclude_tokens=None, **_kwargs):
            self.reserve_exclusions.append(list(exclude_tokens or []))
            return self.accounts.pop(0)

        async def release(self, account):
            self.released.append(account.token)

        async def feedback(self, token, kind, mode_id):
            self.feedbacks.append((token, kind, mode_id))

    class FakeSpec:
        mode_id = ModeId.FAST

        @staticmethod
        def pool_candidates():
            return (0,)

    class FakeAdapter:
        def feed(self, _data):
            return [SimpleNamespace(kind="image", content="https://example.test/image.jpg")]

    async def fake_stream(token, *_args, **_kwargs):
        if token == "bad-token":
            raise UpstreamError("rate limited", status=429)
        yield "data: {}"

    async def fake_resolve(**_kwargs):
        return images._ImageOutput(
            api_value="https://example.test/image.jpg",
            markdown_value="![image](https://example.test/image.jpg)",
        )

    async def no_op(*_args, **_kwargs):
        return None

    directory = FakeDirectory()
    import app.dataplane.account as account_module

    monkeypatch.setattr(account_module, "_directory", directory)
    monkeypatch.setattr(images, "selection_max_retries", lambda: 2)
    monkeypatch.setattr(images, "_configured_retry_codes", lambda _cfg: frozenset({429}))
    monkeypatch.setattr(images, "StreamAdapter", FakeAdapter)
    monkeypatch.setattr(images, "_stream_lite_generate", fake_stream)
    monkeypatch.setattr(images, "_resolve_image_output", fake_resolve)
    monkeypatch.setattr(images, "_quota_sync", no_op)
    monkeypatch.setattr(images, "_fail_sync", no_op)

    result = await images._run_lite_request(
        spec=FakeSpec(),
        prompt="test",
        timeout_s=1,
        response_format="url",
    )

    assert result.api_value == "https://example.test/image.jpg"
    assert directory.reserve_exclusions == [[], ["bad-token"]]
    assert directory.released == ["bad-token", "good-token"]
    assert directory.feedbacks == [
        ("bad-token", FeedbackKind.RATE_LIMITED, int(ModeId.FAST)),
        ("good-token", FeedbackKind.SUCCESS, int(ModeId.FAST)),
    ]
