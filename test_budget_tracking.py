#!/usr/bin/env python3
"""Test script to demonstrate Global Daily Risk Limit."""

from engine import BudgetTracker

# Create a tracker with $100/day limit
budget = BudgetTracker(daily_limit_usd=100.0, bankroll_usd=1000.0)

print("[TEST] Global Daily Risk Limit System")
print("=" * 60)
print(f"Daily Limit: ${budget.daily_limit_usd}")
print(f"Bankroll: ${budget.bankroll_usd}")
print()

# Test 1: Normal trade (50% Kelly)
print("Test 1: 50% Kelly bet")
bet, execute = budget.check_and_cap_bet(0.05)  # 5% Kelly = $50
print(f"  Result: ${bet:.2f}, Execute: {execute}")
if execute:
    budget.record_trade(bet)
print()

# Test 2: Another trade (60% Kelly)
print("Test 2: 60% Kelly bet")
bet, execute = budget.check_and_cap_bet(0.06)  # 6% Kelly = $60
print(f"  Result: ${bet:.2f}, Execute: {execute}")
if execute:
    budget.record_trade(bet)
print()

# Test 3: Trade that would exceed budget
print("Test 3: 80% Kelly bet (would exceed budget)")
bet, execute = budget.check_and_cap_bet(0.08)  # 8% Kelly = $80
print(f"  Result: ${bet:.2f}, Execute: {execute}")
if execute:
    budget.record_trade(bet)
else:
    print("  -> Trade blocked!")
print()

# Test 4: Check remaining budget
print("Test 4: Check remaining budget")
remaining = budget.get_remaining_budget()
print(f"  Remaining: ${remaining:.2f}")
print(f"  Spent today: ${budget.total_spent_today:.2f}")
print()

# Test 5: Final trade attempt
print("Test 5: Final trade attempt")
bet, execute = budget.check_and_cap_bet(0.02)  # 2% Kelly = $20
print(f"  Result: ${bet:.2f}, Execute: {execute}")
if execute:
    budget.record_trade(bet)
else:
    print("  -> Daily limit reached!")
