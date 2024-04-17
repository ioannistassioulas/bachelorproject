#


from brian2 import *

start_scope()

N = 100
tau = 10*ms
eqs ='''
dv/dt = (2-v)/tau : 1
'''

G = NeuronGroup(N, eqs, threshold='v > 1', reset = 'v = 0', refractory = 5*ms, method = 'exact')
G.v = 'rand()'

S = SpikeMonitor(G)

run(20*ms)

plot(S.t/ms, S.i, '.k')
xlabel('Time(ms)')
ylabel('Neuron index')
show()
