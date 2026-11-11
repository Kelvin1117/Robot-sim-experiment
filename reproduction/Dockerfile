FROM python:3.11.15-slim-bookworm AS python
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libbz2-1.0 libcrypt1 libexpat1 libffi8 libgdbm6t64 \
    libdb5.3t64 libgdbm-compat4t64 liblzma5 libncursesw6 libreadline8t64 libsqlite3-0 libssl3t64 \
    libuuid1 zlib1g libstdc++6 libgomp1 libglib2.0-0t64 \
    libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libglx-mesa0 libosmesa6 libglu1-mesa \
    libsm6 libxext6 libxrender1 libxrandr2 libxcursor1 libxi6 libxinerama1 \
    xvfb xauth xfonts-base zstd && rm -rf /var/lib/apt/lists/*
COPY --from=python /usr/local /usr/local
RUN ln -s /usr/local/bin/python3.11 /usr/bin/python3.11 \
    && ln -s /usr/local/lib/python3.11 /usr/lib/python3.11 \
    && ldconfig \
    && mkdir -p /common/home/cs2202 /common/users/cs2202 /reproduction /outputs /cache \
    && python3.11 -c 'import ssl, sqlite3, bz2, lzma, ctypes'
RUN python3.11 - <<'PYTHON'
from pathlib import Path
import shutil, subprocess
p = Path('/tmp/frozen-venv-check')
(p / 'bin').mkdir(parents=True)
(p / 'pyvenv.cfg').write_text('home = /usr/bin\ninclude-system-site-packages = false\nversion_info = 3.11.15\n')
(p / 'bin/python').symlink_to('/usr/bin/python3.11')
subprocess.run([str(p / 'bin/python'), '-c', 'import ssl, sqlite3, bz2, lzma, ctypes, sys; assert sys.prefix == "/tmp/frozen-venv-check"'], check=True)
shutil.rmtree(p)
PYTHON
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/local/bin/python3.11", "/reproduction/reproduction/container_entry.py"]
