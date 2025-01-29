import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


def plot_beta_distributions(alpha_beta_pairs: list[tuple], x_range: tuple=(0, 1), log_scale: bool=False) -> None:
    """Generate plot for beta distribution for mutliply alpha_beta pairs and range of x. 

    Args:
        alpha_beta_pairs (list[tuple]): Alpha beta pairs, first is alpha and second is beta -> (alpha, beta). This params control shape of beta distribution.
        x_range (tuple, optional): Range of x, it depends on your search space. Defaults to (0, 1).
        log_space (bool): Scaled data into logarithmic scale.
    """
    x = np.linspace(0, 1, 500)
    plt.figure(figsize=(8, 6))

    for alpha, beta_param in alpha_beta_pairs:
        y = beta.pdf(x, alpha, beta_param)
        
        if log_scale:
            x_scaled = np.power(10, np.log10(x_range[0]) + (np.log10(x_range[1]) - np.log10(x_range[0])) * x)
            plt.plot(x_scaled, y, label=f'α={alpha}, β={beta_param}')
        else:
            plt.plot(x, y, label=f'α={alpha}, β={beta_param}')
    
    plt.title("Rozkłady Beta dla różnych parametrów α i β")
    plt.xlabel("Wartość hiperparametru" if log_scale else "x")
    plt.ylabel("P(x)")
    plt.xscale("log") if log_scale else None
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.show()


if __name__ == '__main__':
    low = 5e-7
    high = 5e-2
    
    alpha_beta_examples = [(2, 2), (2, 5), (3, 6), (3, 9), (5, 2), (8, 2)]
    plot_beta_distributions(alpha_beta_examples)
    plot_beta_distributions(alpha_beta_examples, x_range=(low, high), log_scale=True)