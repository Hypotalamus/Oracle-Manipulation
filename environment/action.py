from enum import Enum

# class syntax
class Action(Enum):
    SWAP_USDC_TO_MNGO = 0
    PUT_COLLATERAL = 1
    BORROW = 2
    REPAY = 3
    NOP = 4
    BUY_MNGO = 5

ACTIONS_NUM = len(Action)
AMOUNT_ORDERS_HIGH = 8
AMOUNT_REDUCED = 3