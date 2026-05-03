import cudaq
cudaq.set_target('nvidia')
@cudaq.kernel
def ghz():
    q = cudaq.qvector(4)
    h(q[0])
    cx(q[0],q[1])
    cx(q[0],q[2])
    cx(q[0],q[3])
    mz(q)
result = cudaq.sample(ghz)
print(result)
