#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// Writes a deterministic int32 column to `out_path` as a raw binary stream
// of little-endian int32 values.
int main(int argc, char** argv) {
    std::size_t rows = 100'000'000;
    std::string out_path = "bench/data/col.bin";
    uint64_t seed = 0xC01D5EEDu;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--rows" && i + 1 < argc) {
            rows = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return 2;
        }
    }

    std::ofstream of(out_path, std::ios::binary);
    if (!of) {
        std::fprintf(stderr, "cannot open %s for writing\n", out_path.c_str());
        return 1;
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, 100'000);
    constexpr std::size_t kChunk = 1 << 16;
    std::vector<int32_t> buf(kChunk);
    std::size_t remaining = rows;
    while (remaining > 0) {
        const std::size_t take = remaining < kChunk ? remaining : kChunk;
        for (std::size_t i = 0; i < take; ++i) {
            buf[i] = dist(rng);
        }
        of.write(reinterpret_cast<const char*>(buf.data()),
                 static_cast<std::streamsize>(take * sizeof(int32_t)));
        remaining -= take;
    }
    std::fprintf(stdout, "wrote %zu int32 rows to %s\n", rows, out_path.c_str());
    return 0;
}
