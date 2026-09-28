def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """

    LR = 0.09
    x = start_x
    gradient = 4 * x**3 - 9 * x**2

    while True:
      x = x - LR * gradient
      gradient = 4 * x**3 - 9 * x**2
      if abs(gradient) <= 0.001:
        break

    return x