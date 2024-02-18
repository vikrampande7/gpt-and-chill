class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        for i in range(0,iterations,1):
            d = 2 * (init) # get the derivative: 2x
            init = init - learning_rate * d # update the current guess to new value which will be minimized
        return round(init,5)

