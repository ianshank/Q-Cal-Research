# syntax=docker/dockerfile:1
# Reproducible environment for the qcal integrity tooling and its test suite.
#
#   make docker-build              # docker build -t qcal:dev .
#   make docker-test               # the full suite inside the image
#   docker run --rm qcal:dev qcal --help
#
# Behind a TLS-intercepting proxy, pass its CA as a build secret (never baked into a layer):
#   make docker-build DOCKER_BUILD_ARGS="--network host --secret id=build_ca,src=ca.crt \
#     --build-arg HTTPS_PROXY=$HTTPS_PROXY --build-arg HTTP_PROXY=$HTTP_PROXY"
#
# CPU only: the science stack (CUDA, TensorRT) arrives with Phase 1 in its own image.
ARG PYTHON_VERSION=3.12
# Any Debian-based Python image; tools it already ships are not reinstalled.
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# git and ssh-keygen back the signature checks and their tests; make runs the targets.
# Packages come over HTTPS; an optional build_ca secret serves TLS-intercepting proxies.
RUN --mount=type=secret,id=build_ca,required=false \
    if command -v git && command -v ssh-keygen && command -v make; then exit 0; fi \
 && sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
 && if [ -f /run/secrets/build_ca ]; then \
      echo 'Acquire::https::CAInfo "/run/secrets/build_ca";' > /etc/apt/apt.conf.d/99build-ca; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends git openssh-client make \
 && rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/99build-ca

ARG APP_USER=qcal
ARG APP_UID=10001
RUN useradd --create-home --uid "${APP_UID}" "${APP_USER}"

WORKDIR /app
# Install dependencies from the metadata first so source edits keep this layer cached.
COPY pyproject.toml README.md LICENSE ./
COPY src/qcal/__init__.py src/qcal/__init__.py
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi \
 && python -m pip install "setuptools>=68" wheel ".[dev]" \
 && python -m pip uninstall -y qcal

COPY --chown=${APP_USER}:${APP_USER} . .
# Offline from here: the build backend was installed with the dependencies above.
RUN python -m pip install --no-deps --no-build-isolation -e .

USER ${APP_USER}
# The suite needs a git identity and treats the tree as untrusted-but-ours.
RUN git config --global user.name "qcal container" \
 && git config --global user.email "qcal@example.invalid" \
 && git config --global --add safe.directory /app

CMD ["qcal", "--help"]
