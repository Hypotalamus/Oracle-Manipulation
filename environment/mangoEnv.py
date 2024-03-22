import gymnasium
from gymnasium import spaces
import numpy as np

from environment.amm import AMM
from environment.arbitrageur import Arbitrageur
from environment.protocol import Protocol
from environment.action import Action
from environment.action import ACTIONS_NUM, AMOUNT_ORDERS_HIGH, AMOUNT_REDUCED

PENALTY = -0.5
MAX_AMOUNT = 1e9
THRESHOLD_HEALTH = -1e-5
REDUCED = False
    
class MangoEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, 
                 render_mode=None, 
                 ext_mngo_price_mean=0.1, 
                 init_mngo_pool_balance=1e4 / MAX_AMOUNT,
                 init_usdc_pool_balance=1e3 / MAX_AMOUNT,
                 init_treasury_size_usdc=120e6 / MAX_AMOUNT,
                 mngo_collateral_factor=1.5,
                 arb_efficiency_factor=0.6,
                 usdc_user_balance_init=1e7 / MAX_AMOUNT,
                 random_magnitude_pct = 0.1,
                 seed=None
                 ):
        # Actions are dictionaries with the action type and some amount
        # 0 - Swap USDC to MNGO + amount of USDC to swap
        # 1 - Put collateral into protocol in MNGO + amount of collateral
        # 2 - Borrow funds from protocol + amount of funds to borrow
        # 3 - Repay to protocol + don't care
        # 4 - NOPE + don't care
        # 5 - Buy MNGO from external market
        if REDUCED:
            self.action_space = spaces.Discrete( (ACTIONS_NUM - 2) * AMOUNT_REDUCED + 2)
        else:
            self.action_space = spaces.Discrete((ACTIONS_NUM - 2) * AMOUNT_ORDERS_HIGH + 2)

        # State consists of:
        # - 1. External MNGO price
        # - 2. User balance of MNGO
        # - 3. User balance of USDC
        # - 4. Balance of USDC in pool
        # - 5. Balance of MNGO in pool
        # - 6. Size of protocol's treasure in USDC
        # - 7. Size of agent's collateral
        # - 8. Health factor of agent's account (if > 0, then how much USDC he 
        #   could borrow)
        self.observation_space = spaces.Box(
            low=np.array(10 * [-10.0]),
            high=np.array(10 * [10.0]),
            dtype= np.float64
        )

        self.amm = AMM(init_mngo_pool_balance, init_usdc_pool_balance)
        self.arb = Arbitrageur(self.amm, ext_mngo_price_mean, arb_efficiency_factor)
        self.mango = Protocol(init_treasury_size_usdc, mngo_collateral_factor, self.amm)
        self.init_treasury_nominal = init_treasury_size_usdc
        self.ext_mngo_price_mean = ext_mngo_price_mean
        self.usdc_user_balance_init = usdc_user_balance_init
        self.random_magnitude_pct = random_magnitude_pct
        self.reset(seed=seed)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def _get_obs(self):
        return np.array([
            self.ext_mngo_price,
            self.mngo_user_balance,
            self.usdc_user_balance,
            self.amm.usdc_balance,
            self.amm.mngo_balance,
            self.amm.get_price(),
            self.mango.user_debt_usdc,
            self.mango.treasury_usdc,
            self.mango.user_collateral_mngo,
            self.mango.get_user_health_factor()],
            dtype=np.float64
        )

    def _get_info(self):
        return {}
    
    def _ext_mango_price_randomize(self):
        rand = (self.np_random.random() - 0.5) * self.random_magnitude_pct
        self.ext_mngo_price = self.ext_mngo_price_mean * (1 + rand)

    def _usdc_user_balance_randomize(self):
        rand = (self.np_random.random() - 0.5) * self.random_magnitude_pct
        self.usdc_user_balance = self.usdc_user_balance_init * (1 + rand)

    def _mango_treasury_randomize(self):
        rand = (self.np_random.random() - 0.5) * self.random_magnitude_pct
        return self.init_treasury_nominal * (1 + rand)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.amm.reset()
        init_treasury = self._mango_treasury_randomize()
        self.mango.reset(init_treasury)
        self.mngo_user_balance = 0

        self._ext_mango_price_randomize()
        self._usdc_user_balance_randomize()
        self.arb.set_mango_price(self.ext_mngo_price)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def get_user_balance(self):
        return self.usdc_user_balance + self.mngo_user_balance * self.ext_mngo_price
    
    def convert_action_rl_to_human(self, rl_action):
        if REDUCED:
            if rl_action == 12:
                action_type, log_amount = 3, 0 # repay
            elif rl_action == 13:
                action_type, log_amount = 4, 0 # nop
            else:
                action_type, log_amount = divmod(rl_action, AMOUNT_REDUCED)
                if action_type == 3:
                    action_type = 5
                if log_amount == 1:
                    log_amount = 6
                elif log_amount == 2:
                    log_amount = 7
        else:
            if rl_action == (ACTIONS_NUM - 2) * AMOUNT_ORDERS_HIGH:
                action_type, log_amount = 3, 0 # repay
            elif rl_action == (ACTIONS_NUM - 2) * AMOUNT_ORDERS_HIGH + 1:
                action_type, log_amount = 4, 0 # nop
            else:
                action_type, log_amount = divmod(rl_action, AMOUNT_ORDERS_HIGH)
                if action_type == 3:
                    action_type = 5                
        amount = 10.**(log_amount + 1) / MAX_AMOUNT
        return (action_type, amount)

    def step(self, action):
        if isinstance(action, tuple):
            action_type, amount = action
        else:
            action_type, amount = self.convert_action_rl_to_human(action)            

        penalty = 0
        match Action(action_type):
            case Action.SWAP_USDC_TO_MNGO:
                if self.usdc_user_balance >= amount:
                    ret_mngo = self.amm.swap_usdc_to_mngo(amount)
                    self.usdc_user_balance -= amount
                    self.mngo_user_balance += ret_mngo
                else:
                    penalty = PENALTY
            case Action.PUT_COLLATERAL:
                if self.mngo_user_balance >= amount:
                    self.mango.put_collateral(amount)
                    self.mngo_user_balance -= amount
                else:
                    penalty = PENALTY
            case Action.BORROW:
                ret_usdc = self.mango.borrow(amount)
                self.usdc_user_balance += ret_usdc
                if ret_usdc == 0:
                    penalty = PENALTY
            case Action.REPAY:
                usdc_required = self.mango.user_debt_usdc
                if self.usdc_user_balance >= usdc_required and usdc_required > 0:
                    ret_mngo, usdc_repayed = self.mango.repay()
                    self.usdc_user_balance -= usdc_repayed
                    self.mngo_user_balance += ret_mngo
                else:
                    penalty = PENALTY                    
            case Action.NOP:
                penalty = PENALTY
            case Action.BUY_MNGO:
                usdc_required = amount * self.ext_mngo_price
                if self.usdc_user_balance >= usdc_required:                    
                    self.usdc_user_balance -= usdc_required
                    self.mngo_user_balance += amount
                else:
                    penalty = PENALTY

        self.arb.perform_arbitration()
        reward = 0
        terminated = False
        health = self.mango.get_user_health_factor()
        if 1e1 * THRESHOLD_HEALTH <= health < THRESHOLD_HEALTH:
            reward = 0.05
            terminated = True
        elif 1e2 * THRESHOLD_HEALTH <= health < 1e1 * THRESHOLD_HEALTH:
            reward = 0.2
            terminated = True
        elif 1e3 * THRESHOLD_HEALTH <= health < 1e2 * THRESHOLD_HEALTH:
            reward = 2.0
            terminated = True
        elif health < 1e3 * THRESHOLD_HEALTH:
            reward = 10.0
            terminated = True

        reward += penalty          
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass