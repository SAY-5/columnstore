# Multi-stage Dockerfile for columnstore. Build stage uses a full toolchain;
# runtime stage carries only the binary + glibc.

# ---------------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCOLUMNSTORE_BUILD_TESTS=OFF \
        -DCOLUMNSTORE_BUILD_BENCH=OFF \
    && cmake --build build -j

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/columnstore /usr/local/bin/columnstore

ENTRYPOINT ["/usr/local/bin/columnstore"]
CMD ["--rows", "1000000", "--threshold", "1000"]
