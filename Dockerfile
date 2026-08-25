# Development / CI environment for dense-evolution -- reproduces the
# environment the test suite and docs build are validated against
# (see requirements-lock.txt). Not an image meant to be published or run
# as a service: there's no server here, just a container to `docker run
# --rm -it dense-evolution-dev pytest tests/ -q` (or `bash`) in, so CI
# failures reported by contributors on other machines/OSes can be
# reproduced exactly instead of chased through "works on my machine".
FROM python:3.12-slim

WORKDIR /workspace

# git: setuptools_scm / editable installs sometimes shell out to it;
# build-essential: a couple of extras (pymatching, stim) ship no prebuilt
# wheel for every platform and fall back to compiling from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-lock.txt

COPY . .
RUN pip install --no-cache-dir -e . --no-deps

CMD ["pytest", "tests/", "-q"]
