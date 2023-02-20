import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np


def generate_sequence():
    # [B, C, D, E, F]
    state = 3
    x = []
    while state != 0 and state != 6:
        xt = np.array([0, 0, 0, 0, 0])
        xt[state - 1] = 1
        x.append(xt)
        r = np.random.random()
        if r >= 0.5:
            state += 1
        else:
            state -= 1

    z = 0
    if state == 6:
        z = 1
    return np.array(x), z


def fig_3():
    num_sets = 100
    num_sequences = 10
    # Sutten uses 100 training sets with 10 sequences each
    training_sets = []
    reward_sets = []
    for _ in range(num_sets):
        training_set = []
        reward_set = []
        for _ in range(num_sequences):
            sequence, z = generate_sequence()
            training_set.append(sequence)
            reward_set.append(z)
        training_set = np.array(training_set)
        training_sets.append(training_set)
        reward_sets.append(reward_set)

    training_sets = np.array(training_sets)
    true_prob = np.array([1/6, 1/3, 1/2, 2/3, 5/6])
    for alpha in [0.001]:
        rms_errs = []
        lams = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        for lam in lams:
            rms_err = 0
            for i in range(num_sets):
                w = np.array([1/2, 1/2, 1/2, 1/2, 1/2])
                delta_w = np.array([1., 0., 0., 0., 0.])  # setting some value so while loop happens
                while sum(np.abs(delta_w)) > alpha:
                    delta_w = np.array([0., 0., 0., 0., 0.])
                    for j in range(num_sequences):
                        et = np.array([0., 0., 0., 0., 0.])
                        sequence = training_sets[i][j]
                        z = reward_sets[i][j]
                        for k in range(len(sequence)):
                            et = sequence[k] + lam * et
                            pt = np.dot(w, sequence[k])
                            pt_plus_1 = z if k == len(sequence) - 1 else np.dot(w, sequence[k+1])
                            delta_w += alpha * (pt_plus_1 - pt) * et
                    w += delta_w
                err = true_prob - w
                rms_err += np.sqrt(np.sum(np.power(err, 2))/len(w))
            rms_err /= num_sets
            rms_errs.append(rms_err)
        fig, ax = plt.subplots()
        ax.plot(lams, rms_errs)
        fig.savefig(f'Curve_{alpha*1000}.jpeg')
        print("Done")

def fig_4_and_5():
    num_sets = 100
    num_sequences = 10
    # Sutten uses 100 training sets with 10 sequences each
    training_sets = []
    reward_sets = []
    for _ in range(num_sets):
        training_set = []
        reward_set = []
        for _ in range(num_sequences):
            sequence, z = generate_sequence()
            training_set.append(sequence)
            reward_set.append(z)
        training_set = np.array(training_set)
        training_sets.append(training_set)
        reward_sets.append(reward_set)

    training_sets = np.array(training_sets)
    true_prob = np.array([1/6, 1/3, 1/2, 2/3, 5/6])
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    lams = [0, 0.3, 0.8, 1]
    rms_errs = {}
    for lam in lams:
        rms_errs[lam] = []
        for alpha in alphas:
            rms_err = 0
            for i in range(num_sets):
                w = np.array([1/2, 1/2, 1/2, 1/2, 1/2])
                for j in range(num_sequences):
                    delta_w = np.array([0., 0., 0., 0., 0.])
                    et = np.array([0., 0., 0., 0., 0.])
                    sequence = training_sets[i][j]
                    z = reward_sets[i][j]
                    for k in range(len(sequence)):
                        et = sequence[k] + lam * et
                        pt = np.dot(w, sequence[k])
                        pt_plus_1 = z if k == len(sequence) - 1 else np.dot(w, sequence[k+1])
                        delta_w += alpha * (pt_plus_1 - pt) * et
                    w += delta_w
                err = true_prob - w
                rms_err += np.sqrt(np.sum(np.power(err, 2))/len(w))
            rms_err /= num_sets
            rms_errs[lam].append(rms_err)
    fig, ax = plt.subplots()
    for lam in lams:
        ax.plot(alphas, rms_errs[lam], label=f'lambda = {lam}')
    ax.legend()
    fig.savefig(f'fig4.jpeg')
    print("Done")
    best_rms_errs = []
    for lam in lams:
        best_rms_errs.append(min(rms_errs[lam]))
    fig, ax = plt.subplots()
    ax.plot(lams, best_rms_errs)
    fig.savefig(f'fig5.jpeg')

fig_3()
fig_4_and_5()
