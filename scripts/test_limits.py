"""Integration and unit tests for DocChat-AI limit tracking.

Tests that in DEMO_MODE:
1. Distinct session registration enforces MAX_SESSIONS_PER_DAY=2, rejecting the 3rd new session.
2. Existing sessions already registered continue to pass.
3. Daily LLM call counter blocks after N calls.
4. Owner key bypass allows sessions and LLM calls even when limits are reached.
"""

import hmac
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Enable DEMO_MODE for test execution
os.environ["DEMO_MODE"] = "true"

from src.limits import (
    check_and_increment_daily_llm_calls,
    check_and_register_session,
    get_daily_llm_calls_count,
    get_daily_sessions_count,
    is_session_registered,
    reset_daily_limits,
)


def run_tests() -> None:
    print("=" * 60)
    print("Running DocChat-AI Limit Tracking Tests (DEMO_MODE=true)")
    print("=" * 60)

    # ---------------------------------------------------------
    # Test 1 & 2: Daily distinct sessions limit (MAX_SESSIONS_PER_DAY=2)
    # ---------------------------------------------------------
    print("\n[Test 1] Testing distinct session registration with MAX_SESSIONS_PER_DAY=2...")
    reset_daily_limits()

    # Session 1: first meaningful action -> should pass
    sess_1_ok = check_and_register_session("session_001", demo_mode=True, max_sessions=2)
    assert sess_1_ok is True, "Session 1 should be allowed"
    print("  [PASS] Session 1 ('session_001') registered successfully.")

    # Session 2: first meaningful action -> should pass
    sess_2_ok = check_and_register_session("session_002", demo_mode=True, max_sessions=2)
    assert sess_2_ok is True, "Session 2 should be allowed"
    print("  [PASS] Session 2 ('session_002') registered successfully.")

    assert get_daily_sessions_count() == 2, "Session count should be exactly 2"

    # Session 3: 3rd new session -> should be REJECTED
    sess_3_ok = check_and_register_session("session_003", demo_mode=True, max_sessions=2)
    assert sess_3_ok is False, "Session 3 should be rejected as MAX_SESSIONS_PER_DAY=2 is reached"
    print("  [PASS] 3rd new session ('session_003') rejected when MAX_SESSIONS_PER_DAY=2.")

    # [Test 2] Existing sessions still pass
    print("\n[Test 2] Testing that existing sessions continue to pass...")
    sess_1_repeat = check_and_register_session("session_001", demo_mode=True, max_sessions=2)
    assert sess_1_repeat is True, "Existing Session 1 should still pass"

    sess_2_repeat = check_and_register_session("session_002", demo_mode=True, max_sessions=2)
    assert sess_2_repeat is True, "Existing Session 2 should still pass"
    print("  [PASS] Existing sessions ('session_001', 'session_002') still pass.")

    # ---------------------------------------------------------
    # Test 3: Daily LLM call cap blocks after N calls
    # ---------------------------------------------------------
    n_calls = 3
    print(f"\n[Test 3] Testing that daily LLM call cap blocks after N={n_calls} calls...")

    # First N calls should succeed
    for i in range(1, n_calls + 1):
        allowed = check_and_increment_daily_llm_calls(bypass=False, demo_mode=True, max_calls=n_calls)
        assert allowed is True, f"LLM call #{i} should be allowed"
        print(f"  [PASS] LLM call #{i} permitted.")

    assert get_daily_llm_calls_count() == n_calls

    # Call N + 1 should be blocked
    blocked_call = check_and_increment_daily_llm_calls(bypass=False, demo_mode=True, max_calls=n_calls)
    assert blocked_call is False, f"LLM call #{n_calls + 1} should be blocked"
    print(f"  [PASS] LLM call #{n_calls + 1} blocked after reaching cap of {n_calls}.")

    # ---------------------------------------------------------
    # Test 4: Owner key bypass works
    # ---------------------------------------------------------
    print("\n[Test 4] Testing owner key bypass...")

    # Constant-time digest comparison verification
    test_owner_key = "secure_owner_passphrase_882"
    assert hmac.compare_digest(test_owner_key, "secure_owner_passphrase_882") is True
    assert hmac.compare_digest("incorrect_key", test_owner_key) is False
    print("  [PASS] hmac.compare_digest validates owner key correctly.")

    # Owner bypass on session limit: Session 3 (which was rejected) with is_owner=True
    owner_sess_ok = check_and_register_session("session_003", is_owner=True, demo_mode=True, max_sessions=2)
    assert owner_sess_ok is True, "Owner session should bypass session limits"
    print("  [PASS] Owner session bypasses MAX_SESSIONS_PER_DAY successfully.")

    # Owner bypass on LLM calls: call with bypass=True when limit is exhausted
    owner_llm_ok = check_and_increment_daily_llm_calls(bypass=True, demo_mode=True, max_calls=n_calls)
    assert owner_llm_ok is True, "Owner call with bypass=True should succeed despite quota exhaustion"
    print("  [PASS] Owner LLM call bypasses daily LLM call cap successfully.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (4/4 test suites passed)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
