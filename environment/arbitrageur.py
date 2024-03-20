import math

class Arbitrageur:
    def __init__(self, amm, ext_mngo_price, efficiency_factor=0.6):
        self.set_mango_price(ext_mngo_price)
        self.amm = amm
        self.efficiency = efficiency_factor

    def perform_arbitration(self):
        mngo_price = self.amm.get_price()
        delta_price = mngo_price - self.ext_mngo_price
        new_price = self.ext_mngo_price + (1 - self.efficiency) * delta_price
        mngo_balance = self.amm.mngo_balance
        usdc_balance = self.amm.usdc_balance
        k = mngo_balance * usdc_balance 

        if delta_price > 0:
            # sell MNGO to pool
            mngo_amount = math.sqrt(k / new_price) - mngo_balance
            _ = self.amm.swap_mngo_to_usdc(mngo_amount)
        else:
            # sell USDC to pool
            usdc_amount = math.sqrt(k * new_price) - usdc_balance
            _ = self.amm.swap_usdc_to_mngo(usdc_amount)

    def set_mango_price(self, price):
         self.ext_mngo_price = price 
