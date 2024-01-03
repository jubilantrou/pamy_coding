import numpy as np 
import sysv_ipc

key = 123456

mem = sysv_ipc.SharedMemory(key=123456, flags=sysv_ipc.IPC_CREX, size=sysv_ipc.PAGE_SIZE)
# mem = sysv_ipc.SharedMemory(key=123456)

a = bytes([11, 478, 3.0122])

mem.write( a )

mv = memoryview( mem )


mem.detach()
mem.remove()

