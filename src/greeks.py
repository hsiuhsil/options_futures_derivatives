from black_scholes_merton import cumulative_prob, calculate_d1_d2

def delta(option_type, s, k, r, sigma, T):

    d1, d2 = calculate_d1_d2(s, k, r, sigma, T, q=0)

    if option_type=='call':
        return cumulative_prob(d1)
    elif option_type=='put':
        return cumulative_prob(-d1) - 1


