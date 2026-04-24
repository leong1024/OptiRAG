import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("RUN_OPTIRAG_INTEGRATION"),
    reason="Set RUN_OPTIRAG_INTEGRATION=1 to run (requires API keys and network).",
)
def test_placeholder_integration_not_run_by_default() -> None:
    assert True
