import audio_processing as ap
from audio_processing import *

import snntorch as sn
from snntorch import spikegen
def spike_fn(x):
    """
    Produces spikes as a function of membrane threshold
    :param x: membrane potential reduced by threshold
    :return: out, where returns 1 if membrane - threshold is still positive, and 0 if less
    """
    out = torch.zeros_like(x)
    out = torch.where(x > 0, 1.0, out)
    return out


def tde(tau_fac, tau_trg, tau_mem, time_step, n_time_steps, fac_in, trg_in):
    """

    :param tau_fac: time constant for fac input
    :param tau_trg: time constant for trig input
    :param tau_mem: time constant for membrane voltage
    :param time_step: sampling rate of the audio file taken
    :param n_time_steps:
    :param fac_in:
    :param trg_in:
    :return: all membrane, facilitatory, trigger and spike recordings
    """

    # adjusting time constants for
    alpha   = torch.exp(-time_step/tau_fac)
    beta    = torch.exp(-time_step/tau_trg)
    gamma   = torch.exp(-time_step/tau_mem)

    # create file of zeros to store all spiking action.
    fac = torch.zeros(fac_in.shape[0])
    trg = torch.zeros(fac_in.shape[0])
    mem = torch.zeros(fac_in.shape[0])

    # create recording arrays to save each timestep
    mem_rec = []
    spk_rec = []
    fac_rec = []
    trg_rec = []

    for t in range(n_time_steps):
        mthr = mem-1.0  # membrane threshold
        out = spike_fn(mthr)  # all values above threshold, return spike

        new_fac = alpha*fac + fac_in[t]  # generate potential for facilitatory input
        new_trg = beta*trg + new_fac*trg_in[t]  # generate potential for trigger input
        new_mem = (gamma*mem + new_trg)*(1.0-out)  # generate potential for membrane voltage that is passed into LiF

        # save all values
        mem_rec.append(mem)
        spk_rec.append(out)
        fac_rec.append(fac)
        trg_rec.append(trg)

        mem = new_mem
        fac = new_fac
        trg = new_trg

    return torch.stack(mem_rec, axis=1), torch.stack(spk_rec, axis=1), torch.stack(fac_rec, axis=1), torch.stack(trg_rec, axis=1)


sr = 5000
start = 0
end = 1
timestep = sr * (end - start)
left, right = ap.generate_test_waves(30, 500, sr, 1)


# generate spikes with zero crossings
itd_x, itd_y = ap.zero_crossing([left, right], sr)
itd_x, itd_y = ap.fix_broadcasting(itd_x, itd_y)

# itd_x, itd_y = ap.set_recording(itd_x, itd_y, start, end)
plt.plot(np.arange(timestep), left, color="red")
plt.plot(np.arange(timestep), right, color="blue")
plt.scatter(itd_x[0] * sr, itd_y[0], color='red')
plt.scatter(itd_x[1] * sr, itd_y[1], color="blue")
fac_in = torch.Tensor((itd_x[0] - start) * sr)
trg_in = torch.Tensor((itd_x[1] - start) * sr)

# create array of timesteps to send spikes and create spike train for facilitatory input
current_f = torch.zeros(timestep)
for j in fac_in:
    current_f[int(j)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

current_t = torch.zeros(timestep)
for j in trg_in:
    current_t[int(j)] = torch.ones(1)


# tau = 100000
# mem, spk, fac, trg = tde(torch.tensor(tau), torch.tensor(tau), torch.tensor(tau), torch.tensor(tau), torch.tensor(timestep), current_f, current_t)
#
# time = np.arange(timestep)
# fig, ax = plt.subplots(2, 1)
#
# ax[0].plot(time, fac[0], label="facilitatory")
# ax[0].plot(time, trg[0], label="trigger")
# ax[0].legend()
#
# ax[1].plot(time, mem[0], label="membrane")
# ax[1].plot(time, spk[0], label="spike")
# ax[1].legend()
#
# plt.show()
