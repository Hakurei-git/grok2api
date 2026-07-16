import pytest

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
