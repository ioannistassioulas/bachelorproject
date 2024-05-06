# #
# from brian2 import *
#
# start_scope()
#
# N = 10
# # eqs = '''
# # dv/dt = (current-v)/tau : 1
# # current : 1
# # tau : second
# # '''
#
# G = NeuronGroup(N, 'v:1')
# # G.current = [2, 0, 0]
# # G.tau = [10, 100, 100] * ms
#
# synapse = Synapses(G, G)
# synapse.connect(condition='abs(i-j)<4 and i!=j')
#
# synapse1 = Synapses(G, G)
# synapse1.connect(j='i')
# M = StateMonitor(G, 'v', record=True)
#
# run(100 * ms)
#
#
# def visualize_connections(s):
#     number_source = len(s.source)
#     number_target = len(s.target)
#
#     figure(figsize=(10, 4))
#     subplot(121)
#
#     plot(zeros(number_source), arange(number_source), 'ok', ms=10)  # put points of sources on the map
#     plot(ones(number_target), arange(number_target), 'ok', ms=10)  # put points of targets on the map
#
#     for i, j in zip(s.i, s.j):
#         plot([0, 1], [i, j],
#              '-k')  # for each connection created by the synapse, plot a line from the source(0, source) to target
#         # (1, target)
#
#     xticks([0, 1], ['Source', 'Target'])  # label of sources and targets
#     ylabel('Neuron index')  #
#     xlim(-0.1, 1.1)
#     ylim(-1, max(number_source, number_target))
#
#     subplot(222)
#     plot(s.i, s.j, 'ok')
#     xlim(-1, number_source)
#     ylim(-1, number_target)
#     xlabel('Source neurons')
#     ylabel('Target neurons')
#
#
# visualize_connections(synapse)
# visualize_connections(synapse1)
# show()
#
# # plot(M.t/ms, M.v[0], label="Neuron 0")
# # plot(M.t/ms, M.v[1], label="Neuron 1")
# # plot(M.t/ms, M.v[2], label="Neuron 2")
# # xlabel("Time [ms]")
# # ylabel("v")
# # legend()
#
# # show()

import os
import audio_processing as ap

home = os.path.expanduser("~")
test_dir = home + "/PycharmProjects/pythonProject/bachelorproject/datasets/synthetic_data"
print(test_dir)
test = os.path.isdir(test_dir)
loc = ap.name_parse(test_dir, 30, 200)
print(loc)
print(os.path.isfile(loc))
print(home)