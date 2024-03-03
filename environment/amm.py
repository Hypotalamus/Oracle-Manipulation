class AMM:
    def __init__(self, init_mngo_pool_balance, init_usdc_pool_balance):
        self.default_mngo_balance = init_mngo_pool_balance
        self.default_usdc_balance = init_usdc_pool_balance
        self.reset()

    def reset(self, mngo_pool_balance=None, usdc_pool_balance=None):
        if mngo_pool_balance is None:
            self.mngo_balance = self.default_mngo_balance
        else:
            self.mngo_balance = mngo_pool_balance
        if usdc_pool_balance is None:
            self.usdc_balance = self.default_usdc_balance
        else:
            self.usdc_balance = usdc_pool_balance

    def swap_usdc_to_mngo(self, usdc_amount):
        k = self.usdc_balance * self.mngo_balance
        self.usdc_balance += usdc_amount
        mngo_amount = self.mngo_balance - k / self.usdc_balance
        self.mngo_balance -= mngo_amount
        return mngo_amount
    
    def swap_mngo_to_usdc(self, mngo_amount):
        k = self.usdc_balance * self.mngo_balance
        self.mngo_balance += mngo_amount
        usdc_amount = self.usdc_balance - k / self.mngo_balance
        self.usdc_balance -= usdc_amount
        return usdc_amount

    def get_price(self):
        try:
            return self.usdc_balance / self.mngo_balance
        except ZeroDivisionError:
            return 0        