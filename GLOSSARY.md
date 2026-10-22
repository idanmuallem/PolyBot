# PolyBot

An automated trading bot that prices Polymarket prediction markets and places bets where its own probability disagrees with the market's.

## Language

### Trading mode

**Dry run**:
The trading mode in which the bot places simulated bets against a paper balance and no real orders reach Polymarket.
_Avoid_: Paper mode, test mode, simulation

**Live run**:
The trading mode in which the bot places real orders on Polymarket with real money.
_Avoid_: Prod mode, real mode

### Environments

**Production**:
The deployed bot whose trades count; it runs in whichever trading mode its operator sets, so Production is not the same thing as Live run.
_Avoid_: Prod (when you mean Live run), live

**Staging**:
A second deployed copy of the bot that every change reaches before Production; it is always in Dry run, holds no wallet key, and its trade history is thrown away on every deploy.
_Avoid_: Test environment, sandbox
