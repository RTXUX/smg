# Install local SMG wheels into an engine image

From any directory, run `scripts/build_wheel_image.sh` using its path in this
checkout:

```bash
./scripts/build_wheel_image.sh your-engine:base your-engine:smg

# Select an engine's virtual environment and adjust Rust build parallelism.
./scripts/build_wheel_image.sh your-engine:base your-engine:smg \
  --build-arg RUNTIME_PYTHON=/opt/venv/bin/python \
  --build-arg CARGO_BUILD_JOBS=32
```

`Dockerfile.wheels` builds `smg-grpc-proto` and `smg-grpc-servicer` in the
`grpc-wheels` stage and the Rust-backed `smg` wheel in the independent `smg-wheel`
stage. All three come from this checkout, including regenerated protobuf stubs.
Rust uses `rust-toolchain.toml` and `Cargo.lock`. BuildKit caches Cargo and Zig
artifacts across builds; changing only `BASE_IMAGE` reuses the wheels.

The final stage installs the wheels and their required Python dependencies into
`BASE_IMAGE`. It replaces existing copies of the same package versions without
forcing unrelated dependencies to reinstall. Engine-specific extras are not
installed: the base image should already contain the engine and its dependencies.
As in CI, gRPC health/reflection packages use the protobuf-6-compatible 1.81.x
line. Any preinstalled gRPC companion packages participate in dependency
resolution so exact pins such as `grpcio-observability` stay consistent.
The base image's startup command, environment, working directory, and user are
inherited. For example, to run the gateway explicitly:

```bash
docker run --rm -p 30000:30000 --entrypoint smg your-engine:smg \
  launch --worker-urls http://host.docker.internal:8000
```

Requirements: Docker with BuildKit, a Linux base image with glibc >=2.28,
CPython >=3.10, and pip for the selected `RUNTIME_PYTHON`. The inherited user must
have permission to install into that Python environment. Its dependencies must
also allow the servicer's `grpcio>=1.81.1` requirement and protobuf 6. For example,
older LMCache builds that pin `grpcio-observability==1.78.0` transitively require
`grpcio==1.78.0`; use a base with compatible dependency versions before layering
these wheels. Pip can report conflicts with preinstalled engine packages even
when installation and the import check succeed.

Wheels are built for the selected Docker platform; use `--platform linux/arm64`
with a native builder
or QEMU when targeting ARM. The default SMG wheel uses vendored OpenSSL and the
same manylinux 2.28/Zig approach as CI. It omits the optional `opencv-video`
feature; use the gateway's main `docker/Dockerfile` for its OpenCV/FFmpeg build.
JPEG/WebP acceleration uses the runtime libraries already in the base image.

| Build argument | Default | Purpose |
| --- | --- | --- |
| `BASE_IMAGE` | `python:3.12-slim-bookworm` | Final image to extend |
| `BUILDER_IMAGE` | `python:3.12-bookworm` | Debian-based Python image for both builders |
| `RUNTIME_PYTHON` | `python3` | Interpreter used to install and verify the wheels |
| `BUILD_PROFILE` | `release` | Cargo profile (`ci` for a faster build) |
| `CARGO_BUILD_JOBS` | `16` | Maximum parallel Cargo jobs |
| `PIP_BREAK_SYSTEM_PACKAGES` | `1` | Allow installation into distro-managed Python during the build |

The wrapper forwards extra arguments to `docker build`. The equivalent direct
build command, run at the repository root, is:

```bash
docker build -f docker/Dockerfile.wheels \
  --build-arg BASE_IMAGE=your-engine:base -t your-engine:smg .
```

To export just the three wheels without installing them into a runtime image:

```bash
docker build -f docker/Dockerfile.wheels --target wheels \
  --output type=local,dest=dist .
```
