class Protocol:
    def __init__(self, init_treasury_size_usdc, mngo_collateral_factor, amm):
        self.default_init_treasury_usdc = init_treasury_size_usdc
        self.mngo_factor = mngo_collateral_factor
        self.amm = amm
        self.reset()

    def reset(self, init_treasury_size_usdc=None):
        if init_treasury_size_usdc is None:
            self.treasury_usdc = self.default_init_treasury_usdc
        else:
            self.treasury_usdc = init_treasury_size_usdc
        self.user_collateral_mngo = 0
        self.user_debt_usdc = 0
        self._user_health_factor = 0        

    def put_collateral(self, mngo_amount):
        self.user_collateral_mngo += mngo_amount

    def borrow(self, usdc_amount):
        health = self.get_user_health_factor()
        if health >= usdc_amount and self.treasury_usdc >= usdc_amount:
            self.treasury_usdc -= usdc_amount
            self.user_debt_usdc += usdc_amount
            ret_usdc = usdc_amount
        else:
            ret_usdc = 0
        return ret_usdc

    def get_user_health_factor(self):
        mngo_price = self.amm.get_price()
        health = mngo_price * self.user_collateral_mngo / self.mngo_factor
        health -= self.user_debt_usdc
        return health
    
    def debt_is_bad(self):
        mngo_price = self.amm.get_price()
        return mngo_price * self.user_collateral_mngo - self.user_debt_usdc < 0

    def treasury_is_empty(self):
        return self.treasury_usdc == 0

    def repay(self):
        ret_mngo, usdc_repayed = self.user_collateral_mngo, self.user_debt_usdc
        self.user_collateral_mngo, self.user_debt_usdc = 0, 0
        return ret_mngo, usdc_repayed