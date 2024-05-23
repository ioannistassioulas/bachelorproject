import audio_processing
from audio_processing import *

# start creating TDE model


def forward_euler_lif(volt, current, threshold, resist=5e7, cap=1e-10, time_step=1e-3):
    """
    Numerically evaluate the ODE of the passive membrane. Save results onto volt_mem
    array.
    :param volt: rest membrane voltage
    :param current: input current - taken as spikes
    :param resist: resistance of the membrane
    :param cap: capacitance of the membrane
    :param threshold: voltage threshold for when spike will fire
    :param time_step: timestep to evaluate potential

    :return v: voltage at following timestep
    :return spike: 1 if spike is sent, 0 otherwise
    """

    # define time constant and generate spike
    time_constant = resist * cap
    spike = (volt > threshold)

    # reset value for if spike is produced, always evaluate to 0
    reset = spike * threshold

    # evaluate next step of voltage
    volt = volt + (time_step / time_constant) * (current * resist - volt) - reset

    return volt, spike


def tde_neuron(start, time, facilitatory, trigger, theta, sr, refractory=500):
    """
    TDE model for receiving 2 spike trains and then returning corresponding
    EPSC.
    :param start: timestamp of beginning of sound event
    :param time: timestep of
    :param facilitatory: spike train of first input that is received
    :param trigger: spike train of second input that is received
    :param theta: the cutoff threshold for the LiF neuron
    :param refractory: how many timesteps to prevent before next fire
    :param sr: sampling rate of audiofile

    :return espc: excitatory post synpatic current
    :return voltage, the recording of the voltage change of the neuron
    :return spikes, the recording of the spike output of the neuron
    """

    # define amount of timesteps to be looped over
    timestep = int(time * sr)

    # create array of timesteps to send spikes and create spike train for facilitatory input
    facilitatory = (facilitatory - start) * sr  # subtract by start time such that length is consistent with timestep length
    current_f = torch.zeros(timestep)
    for j in facilitatory:
        current_f[int(j)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

    # repeating for trigger input
    trigger = (trigger - start) * sr
    current_t = torch.zeros(timestep)
    for j in trigger:
        current_t[int(j)] = torch.ones(1)

    # define initial conditions of voltage
    mem_voltage = torch.zeros(1)
    record = False

    # define recording arrays
    voltage_record = []
    spike_record = []
    epsc = []
    refrac = 0

    # integrate over IDE to produce voltage and spike evolution
    for step in range(timestep):

        # start simulating the TDE, change setting of threshold
        mem_voltage, current = forward_euler_lif(mem_voltage, current_f[step], theta)
        spike_record.append(current)

        # adds refractory time depending on if a spike was released or not
        refrac += current * refractory
        # checks if refractory time is still active
        if refrac > 0:
            mem_voltage = torch.zeros(1)
            voltage_record.append(mem_voltage)
            refrac -= 1
        else:
            voltage_record.append(mem_voltage)

        # only record in time distance between both crossings
        if current_f[step] > 0:  # stop record
            record = False
        if current_t[step] > 0:  # start record
            record = True

        if record:
            epsc.append(mem_voltage)
        else:
            epsc.append(torch.zeros(1))

    return epsc, voltage_record, spike_record


def tde_model(start, time, facilitatory, trigger, theta_input, theta_evaluate, sr):
    """
    Generate spikes via potential
    :param start: timestamp of beginning of sound event
    :param time: how long the recording for spike input takes place
    :param facilitatory:  the spike chain for the facilitatory input
    :param trigger: The spike chain for the trigger input
    :param theta_input: threshold value for parsing through overlapping inputs
    :param theta_evaluate: threshold value for EPSC evaluation
    :param sr: sampling rate of the audio, taken via SciPy

    :return: final_spikes, the spike rate for the two input currents
    """

    # define timesteps where model will be defined over
    timesteps = int(sr * time)

    # Generate voltage for LiF model
    epsc, membrane_volt, spike = tde_neuron(start, time, facilitatory, trigger, theta_input, sr)

    # feed membrane recording as input for new neuron
    final_spikes = []
    for step in range(timesteps):
        throwaway, spike = forward_euler_lif(epsc[step], torch.zeros(1), theta_evaluate)
        final_spikes.append(spike)

    return epsc, spike, membrane_volt, final_spikes

#
# # current_fac = torch.cat((torch.zeros(10, 1), torch.ones(1, 1), torch.zeros(189, 1), torch.ones(1, 1), torch.zeros(199, 1)), 0)
# # current_trig = torch.cat((torch.zeros(15, 1), torch.ones(1, 1), torch.zeros(189, 1), torch.ones(1, 1), torch.zeros(194, 1)), 0)
# time1 = 100
# current_fac = np.array([11, 25, 71])
# current_trig = np.array([13, 27, 73])
# # model the TDE reaction to above inputs
# # post_synaptic, v, i = tde_neuron(time1, current_fac, current_trig, 1e20)
#
# # Final LiF neuron modeling the voltage
# post_synaptic, i, v, spikes = tde_model(time1, current_fac, current_trig, 1e20, 2e6)
#
# # it works!
# # plot out the dataset from what was created
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(np.arange(len(post_synaptic)), post_synaptic, color="blue", label="recording")
# ax[0].scatter(current_fac, [1e7] * len(current_fac), color="red", marker="x", label="current")
# ax[0].scatter(current_trig, [1e7] * len(current_trig), color="blue", marker="x", label="trigger")
# ax[0].plot(np.arange(len(i)), i, label="tde triggering")
# ax[0].plot(np.arange(len(v)), v, color="red", label="voltage")
# # ax[0].axhline(thresh, color="black", label=f"threshold={thresh}")
# ax[0].legend()
#
# ax[1].plot(np.arange(len(post_synaptic)), np.array(post_synaptic), label="voltage")
# ax[1].axhline(2e6, color="black", label="threshold")
# ax[1].scatter(np.arange(len(spikes)), np.array(spikes) * 2e6, label="spikes", marker="|", color="grey")
#
# fig.suptitle("Spiking behavior of TDE model")
# plt.legend()
# plt.show()
