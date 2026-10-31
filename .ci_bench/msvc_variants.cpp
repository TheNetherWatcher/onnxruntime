// Which loop shapes does each compiler vectorize? 16M int4 elements.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
constexpr size_t C = 512;
constexpr size_t NEL = size_t(1) << 24;
using Clock = std::chrono::steady_clock;
float* g_out = nullptr;
volatile float g_sink;
template <class F> double best_ms(F&& f) {
  double best = 1e30;
  for (int r = 0; r < 9; ++r) { auto t0 = Clock::now(); f(); best = std::min(best, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    float acc = 0; for (size_t i = 0; i < NEL; i += 4099) acc += g_out[i]; g_sink = acc; }
  return best;
}

// unpack variants (signed int4) --------------------------------------------
inline void unpack_i8_interleave(const uint8_t* p, size_t nbytes, int8_t* d) {
  for (size_t b = 0; b < nbytes; ++b) {
    const uint8_t v = p[b];
    d[2 * b] = static_cast<int8_t>(static_cast<int8_t>(v << 4) >> 4);
    d[2 * b + 1] = static_cast<int8_t>(static_cast<int8_t>(v) >> 4);
  }
}
inline void unpack_i32_interleave(const uint8_t* p, size_t nbytes, int32_t* d) {
  for (size_t b = 0; b < nbytes; ++b) {
    const int32_t v = static_cast<int32_t>(static_cast<int8_t>(p[b]));
    d[2 * b] = (v << 28) >> 28;
    d[2 * b + 1] = v >> 4;
  }
}
inline void unpack_i32_lut(const uint8_t* p, size_t nbytes, int32_t* d) {
  static const struct T { int32_t t[256][2]; T() { for (int v = 0; v < 256; ++v) { t[v][0] = int8_t(v << 4) >> 4; t[v][1] = int8_t(v) >> 4; } } } lut;
  for (size_t b = 0; b < nbytes; ++b) { std::memcpy(d + 2 * b, lut.t[p[b]], 8); }
}

int main() {
  std::vector<uint8_t> in(NEL / 2);
  for (size_t i = 0; i < in.size(); ++i) in[i] = uint8_t(i * 2654435761u >> 13);
  std::vector<float> out(NEL), sc(NEL);
  g_out = out.data();
  for (size_t i = 0; i < NEL; ++i) sc[i] = 0.001f * float(i % 97);
  std::vector<uint8_t> zpb(NEL / 2, 0x35);
  alignas(64) int8_t v8[C], z8[C];
  alignas(64) int32_t v32[C], z32[C];
  const float s = 0.0123f; const int32_t zp = 3;
  double t;
  // fixed scale/zp
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_i8_interleave(&in[c / 2], C / 2, v8);
      for (size_t i = 0; i < C; ++i) out[c + i] = static_cast<float>(static_cast<int32_t>(v8[i]) - zp) * s; } });
  std::printf("fixed  i8 interleave          %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_i32_interleave(&in[c / 2], C / 2, v32);
      for (size_t i = 0; i < C; ++i) out[c + i] = static_cast<float>(v32[i] - zp) * s; } });
  std::printf("fixed  i32 interleave         %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_i32_lut(&in[c / 2], C / 2, v32);
      for (size_t i = 0; i < C; ++i) out[c + i] = static_cast<float>(v32[i] - zp) * s; } });
  std::printf("fixed  i32 lut                %7.3f ms\n", t);
  t = best_ms([&] { float tab[16]; for (int q = -8; q < 8; ++q) tab[q & 15] = static_cast<float>(q - zp) * s;
      for (size_t b = 0; b < NEL / 2; ++b) { const uint8_t v = in[b]; out[2 * b] = tab[v & 15]; out[2 * b + 1] = tab[v >> 4]; } });
  std::printf("fixed  float table[16]        %7.3f ms\n", t);
  // per-element scale/zp
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_i8_interleave(&in[c / 2], C / 2, v8); unpack_i8_interleave(&zpb[c / 2], C / 2, z8);
      for (size_t i = 0; i < C; ++i) out[c + i] = static_cast<float>(static_cast<int32_t>(v8[i]) - z8[i]) * sc[c + i]; } });
  std::printf("vector i8 interleave          %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_i32_interleave(&in[c / 2], C / 2, v32); unpack_i32_interleave(&zpb[c / 2], C / 2, z32);
      for (size_t i = 0; i < C; ++i) out[c + i] = static_cast<float>(v32[i] - z32[i]) * sc[c + i]; } });
  std::printf("vector i32 interleave         %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_i32_lut(&in[c / 2], C / 2, v32); unpack_i32_lut(&zpb[c / 2], C / 2, z32);
      for (size_t i = 0; i < C; ++i) out[c + i] = static_cast<float>(v32[i] - z32[i]) * sc[c + i]; } });
  std::printf("vector i32 lut                %7.3f ms\n", t);
  // unpack alone
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_i8_interleave(&in[c / 2], C / 2, v8); out[c] = v8[5]; } });
  std::printf("unpack only i8                %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_i32_interleave(&in[c / 2], C / 2, v32); out[c] = float(v32[5]); } });
  std::printf("unpack only i32               %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_i32_lut(&in[c / 2], C / 2, v32); out[c] = float(v32[5]); } });
  std::printf("unpack only i32 lut           %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { for (size_t i = 0; i < C; ++i) out[c + i] = static_cast<float>(v32[i] - zp) * s; } });
  std::printf("float loop only (i32)         %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { for (size_t i = 0; i < C; ++i) out[c + i] = static_cast<float>(static_cast<int32_t>(v8[i]) - zp) * s; } });
  std::printf("float loop only (i8)          %7.3f ms\n", t);
}
