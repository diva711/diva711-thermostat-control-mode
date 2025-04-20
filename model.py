import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class HomeHeatingSystem:
    def __init__(self, a, b, c, target_temp):
        self.a = a  # coefficient for d²y/dt²
        self.b = b  # coefficient for dy/dt
        self.c = c  # coefficient for y
        self.target_temp = target_temp
        self.heating_on = False

    def heating_input(self, t, current_temp):
        # Control logic: turn heating on if temperature below target - 0.5, off if above target + 0.5
        if current_temp < self.target_temp - 0.5:
            self.heating_on = True
        elif current_temp > self.target_temp + 0.5:
            self.heating_on = False
        return 1.0 if self.heating_on else 0.0

    def model_ode(self, y, t):
        # y[0] = temperature y(t)
        # y[1] = temperature derivative dy/dt
        temp = y[0]
        temp_dot = y[1]
        f_t = self.heating_input(t, temp)  # heating input f(t)

        # Second order ODE: a*d²y/dt² + b*dy/dt + c*y = f(t)
        # Rearranged: d²y/dt² = (f(t) - b*dy/dt - c*y) / a
        temp_ddot = (f_t - self.b * temp_dot - self.c * temp) / self.a

        return [temp_dot, temp_ddot]

    def simulate(self, y0, t):
        # y0: initial conditions [initial temperature, initial temperature rate]
        # t: time points array
        sol = odeint(self.model_ode, y0, t)
        return sol

if __name__ == "__main__":
    # System parameters (example values)
    a = 1.0
    b = 0.5
    c = 0.2
    target_temp = 22.0

    # Initial conditions: initial temperature 20°C, initial rate 0
    y0 = [20.0, 0.0]

    # Time points (0 to 100 seconds)
    t = np.linspace(0, 100, 1000)

    heating_system = HomeHeatingSystem(a, b, c, target_temp)
    solution = heating_system.simulate(y0, t)

    temperatures = solution[:, 0]

    # Plot results
    plt.plot(t, temperatures, label='Temperature (y(t))')
    plt.axhline(y=target_temp, color='r', linestyle='--', label='Target Temperature')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Temperature (°C)')
    plt.title('Home Heating System Temperature Control')
    plt.legend()
    plt.grid(True)
    plt.show()
