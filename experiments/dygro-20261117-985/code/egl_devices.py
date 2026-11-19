import ctypes
import json
import uuid
from pathlib import Path
from mujoco.egl import egl_ext as EGL


def query_devices():
    query = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(EGL.eglGetProcAddress(b'eglQueryDeviceStringEXT'))
    binary = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(EGL.eglGetProcAddress(b'eglQueryDeviceBinaryEXT'))
    rows = []
    for index, device in enumerate(EGL.eglQueryDevicesEXT()):
        row = dict(egl_index=index)
        value = (ctypes.c_ubyte * 16)()
        size = ctypes.c_int()
        if binary(device, 0x335C, 16, value, ctypes.byref(size)) and size.value == 16:
            row['uuid'] = 'GPU-' + str(uuid.UUID(bytes=bytes(value)))
        for key, attribute in [('extensions', 0x3055), ('drm_device', 0x3233), ('drm_render_node', 0x3377)]:
            value = query(device, attribute)
            row[key] = value.decode() if value else None
        node = row['drm_render_node'] or row['drm_device']
        if node:
            path = Path('/sys/class/drm') / Path(node).name / 'device'
            row['pci_bus_id'] = path.resolve().name
        rows.append(row)
    return rows


if __name__ == '__main__':
    print(json.dumps(query_devices(), indent=2))
