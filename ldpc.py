import numpy as np
import algebra as alg
from scipy import stats


class DegenerateMatrixError(Exception):
    pass
    #print("DEGENERATE MATRIX ERROR")


def make_generator_matrix(H):
    """
    Create code generator matrix using given check matrix. The function must raise DegenerateMatrixError exception,
    if given check matrix is degenerate

    :param H: check matrix of size (m,n), np.array
    :return G: generator matrix of size (n,k), np.array
    :return ind: indices for systematic coding, i.e. G[ind,:] is unity matrix
    """
    m, n = H.shape
    A, ind = alg.gauss(H)
    k = n - m
    G = np.zeros((n, k), dtype=int)
    P = A[:, ~ind]
    G[ind, :] = P
    G[~ind, :] = np.identity(k)
    return G, np.where(~ind)


def update_messages_h_to_e(mu_h_to_e, mu_e_to_h, s, factor_index, var_indices, output_var_indices, non_converged_indices, trim=1e-8):
    """
    Updates messages (in place) from one factor to a set of variables.

    :param mu_h_to_e: all messages from factors to variables, 3D numpy array of size (m, n, num_syndromes)
    :param mu_e_to_h: all messages from variables to factors, 3D numpy array of size (m, n, num_syndromes)
    :param s: input syndroms, numpy array of size (m, num_syndromes)
    :param factor_index: index of selected factor, a number
    :param var_indices: indices of all variables than are connected to factor
    :param output_var_indices: indices of variable for updated messages
    :param non_converged_indices: indices of syndromes for updated messages
    :param trim: trim value for updated messages
    """
    eh = mu_e_to_h[:, :, :]
    mu_factor = eh[factor_index, var_indices, :]
    t_deltas = 1 - 2 * mu_factor
    zeros_cnt = np.sum(t_deltas == 0, axis=0)
    w0 = zeros_cnt == 0
    w1 = zeros_cnt == 1
    deltas = np.zeros(t_deltas.shape)
    deltas[:, w0] = np.prod(t_deltas[:, w0], axis=0)/t_deltas[:, w0]

    if np.sum(w1) != 0:
        for i in np.where(w1):
            ind = np.where(t_deltas[i, :] == 0)
            deltas[i, ind] = np.prod(deltas[i, :ind]) * np.prod(deltas[i, ind + 1:])

    pl = 0.5 * (1 + deltas)
    mu = pl
    w = s[factor_index, :] == 0
    mu[:, w] = 1 - pl[:, w]
    mu = alg.trimmer(mu, trim)
    t_mu = np.copy(mu_e_to_h[factor_index, :, :])
    t_mu[var_indices, :] = mu
    mu_h_to_e[factor_index, output_var_indices[None, :], non_converged_indices[:, None]] = \
        t_mu[output_var_indices[None, :], non_converged_indices[:, None]]


def update_messages_e_to_h_and_beliefs(mu_e_to_h, beliefs, mu_h_to_e, q, var_index, factor_indices, output_factor_indices, non_converged_indices, trim=1e-8):
    """
    Updates messages (in place) from one variable to a set of factors and updates belief for this variable.
    :param mu_e_to_h: all messages from variables to factors, 3D numpy array of size (m, n, num_syndromes)
    :param beliefs: all beliefs, numpy array of size (n, num_syndromes)
    :param mu_h_to_e: all messages from factors to variables, 3D numpy array of size (m, n, num_syndromes)
    :param q: channel error probability
    :param var_index: index of selected variable, a number
    :param factor_indices: indices of all factors that are connected to selected variable
    :param output_factor_indices: indices of factors for updated messages
    :param non_converged_indices: indices of syndromes for updated messages
    :param trim: trim value for updated messages
    """
    output_factor_indices = np.array(output_factor_indices)
    n_fact = factor_indices.size
    n_synd = mu_h_to_e.shape[2]
    log_mu_factor = np.empty((n_fact, n_synd, 2))
    log_mu_factor[:, :, 0] = np.log(1 - mu_h_to_e[factor_indices, var_index])
    log_mu_factor[:, :, 1] = np.log(mu_h_to_e[factor_indices, var_index])

    log_b = np.sum(log_mu_factor, axis=0)
    log_b[:, 0] += np.log(1 - q)
    log_b[:, 1] += np.log(q)

    t_log_mu_var = log_b[None, :, :] - log_mu_factor
    log_mu_var = np.copy(mu_e_to_h[:, var_index, :])
    log_mu_var[factor_indices, :] = alg.trimmer(np.exp(t_log_mu_var[:, :, 1])/np.sum(np.exp(t_log_mu_var), axis=2), trim)

    t_beliefs = alg.trimmer(np.exp(log_b[:, 1])/np.sum(np.exp(log_b), axis=1), trim)
    beliefs[var_index, non_converged_indices] = t_beliefs[non_converged_indices]

    mu_e_to_h[output_factor_indices[:, None], var_index, non_converged_indices[None, :]] = \
        log_mu_var[output_factor_indices[:, None], non_converged_indices[None, :]]


def decode(s, H, q, schedule='parallel', max_iter=200, tol_beliefs=1e-4, display=False):
    """
    LDPC decoding procedure for syndrome probabilistic model.
    :param s: a set of syndromes, numpy array of size (m, num_syndromes)
    :param H: LDPC check matrix, numpy array of size (m, n)
    :param q: channel error probability
    :param schedule: a schedule for updating messages, possible values are 'parallel' and 'sequential'
    :param max_iter: maximal number of iterations
    :param tol_beliefs: tolerance for beliefs stabilization
    :param display: verbosity level
    :return hat_e: decoded error vectors, numpy array of size (n, num_syndromes)
    :return results: additional results, a dictionary with fields:
        'num_iter': number of iterations for each syndrome decoding, numpy array of length num_syndromes
        'status': status (0, 1, 2) for each syndrome decoding, numpy array of length num_syndromes
    """
    m, n = H.shape
    n_synd = s.shape[1]
    mu_e_to_h = q * np.ones((m, n, n_synd))
    mu_h_to_e = np.zeros((m, n, n_synd))
    beliefs = np.zeros((n, n_synd))
    status = 2 * np.ones(n_synd)
    num_iter = max_iter * np.ones(n_synd)
    non_converged_indices = np.array(range(n_synd))

    ons = np.where(H)
    var_to_fact_indeces = []
    fact_to_var_indeces = []
    for i in range(m):
        w = ons[0] == i
        var_to_fact_indeces.append(ons[1][w])
    for i in range(n):
        w = ons[1] == i
        fact_to_var_indeces.append(ons[0][w])

    old_beliefs = np.copy(beliefs)
    old_mu_h_to_e = np.copy(mu_h_to_e)
    old_mu_e_to_h = np.copy(mu_e_to_h)
    e_list = []
    for i in range(max_iter):
        if schedule == 'parallel':
            for j in range(m):
                update_messages_h_to_e(mu_h_to_e, mu_e_to_h, s, j, var_to_fact_indeces[j], var_to_fact_indeces[j], non_converged_indices)
            for j in range(n):
                update_messages_e_to_h_and_beliefs(mu_e_to_h, beliefs, mu_h_to_e, q, j, fact_to_var_indeces[j], fact_to_var_indeces[j], non_converged_indices)
        if schedule == 'sequential':
            for j in range(n):
                for k in fact_to_var_indeces[j]:
                    update_messages_h_to_e(mu_h_to_e, mu_e_to_h, s, k, var_to_fact_indeces[k], np.array([j]), non_converged_indices)
                update_messages_e_to_h_and_beliefs(mu_e_to_h, beliefs, mu_h_to_e, q, j, fact_to_var_indeces[j], fact_to_var_indeces[j], non_converged_indices)

        stabilized = (np.sum(np.abs(beliefs - old_beliefs) >= tol_beliefs, axis=0)) == 0
        status[stabilized] = 1
        t_e = beliefs > 0.5
        found = (np.sum(H.dot(t_e) % 2 != s, axis=0)) == 0
        status[found] = 0
        non_converged_indices = np.where(~found & ~stabilized)[0]
        num_iter[(found | stabilized) & (num_iter == max_iter)] = i + 1

        if display:
            print(np.mean(np.abs(old_beliefs - beliefs)))
            print(np.mean(np.abs(old_mu_h_to_e - mu_h_to_e)))
            print(np.mean(np.abs(old_mu_e_to_h - mu_e_to_h )))
            print('-' * 20)

        if np.sum(~stabilized) == 0:
            break

        if np.sum(~found) == 0:
            break

        if np.sum(np.abs(old_beliefs - beliefs) >= tol_beliefs) == 0:
            break

        old_beliefs = np.copy(beliefs)
        old_mu_h_to_e = np.copy(mu_h_to_e)
        old_mu_e_to_h = np.copy(mu_e_to_h)
        e_list.append(t_e)
    #print(status)
    return t_e, {'num_iter': num_iter, 'status': status, 'e_list': e_list}


def estimate_errors(H, q, num_syndromes=200, display=False, schedule='parallel'):
    """
    Estimate error characteristics for given LDPC code
    :param H: LDPC check matrix, numpy array of size (m, n)
    :param q: channel error probability
    :param num_syndromes: number of Monte Carlo simulations
    :param display: verbosity level
    :param schedule: message schedule for decoding procedure, possible values are 'sequential' and 'parallel'
    :return err_bit: mean bit error, a number
    :return err_block: mean block error, a number
    :return diver: mean divergence, a number
    """
    true_e = stats.bernoulli.rvs(q, size=(H.shape[1], num_syndromes))
    decoded_e, info = decode(H.dot(true_e) % 2, H, q, schedule, display=display)
    correct = (info['status'] == 1) | (info['status'] == 0)

    err_bit = np.mean(true_e[:, correct] != decoded_e[:, correct])
    block_err_w = (np.sum(true_e[:, correct] != decoded_e[:, correct], axis=0)) != 0
    err_block = np.mean(block_err_w)
    diver = np.mean(info['status'] == 2)
    return err_bit, err_block, diver
