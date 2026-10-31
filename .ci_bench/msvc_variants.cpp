#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#if defined(_MSC_VER) && !defined(__clang__)
#define IVDEP __pragma(loop(ivdep))
#else
#define IVDEP
#endif
constexpr size_t C = 512;
constexpr size_t NEL = size_t(1) << 24;
using Clock = std::chrono::steady_clock;
float* g_out = nullptr;
volatile float g_sink;
template <class F> double best_ms(F&& f) {
  double best = 1e30;
  for (int r = 0; r < 9; ++r) {
    auto t0 = Clock::now(); f();
    best = std::min(best, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    float acc = 0; for (size_t i = 0; i < NEL; i += 4099) acc += g_out[i]; g_sink = acc;
  }
  return best;
}
// signed int4 unpack to int8
void unpack_a(const uint8_t* __restrict p, size_t nbytes, int8_t* __restrict d) {
  IVDEP for (size_t b = 0; b < nbytes; ++b) {
    const uint8_t v = p[b];
    d[2 * b] = static_cast<int8_t>(static_cast<int8_t>(v << 4) >> 4);
    d[2 * b + 1] = static_cast<int8_t>(static_cast<int8_t>(v) >> 4);
  }
}
// split into low/high halves (non-interleaved), then interleave via a second pass
void unpack_b(const uint8_t* __restrict p, size_t nbytes, int8_t* __restrict lo, int8_t* __restrict hi) {
  IVDEP for (size_t b = 0; b < nbytes; ++b) lo[b] = static_cast<int8_t>(static_cast<int8_t>(p[b] << 4) >> 4);
  IVDEP for (size_t b = 0; b < nbytes; ++b) hi[b] = static_cast<int8_t>(static_cast<int8_t>(p[b]) >> 4);
}
void fixed_loop(const int8_t* __restrict v, size_t n, int32_t zp, float s, float* __restrict out) {
  IVDEP for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(static_cast<int32_t>(v[i]) - zp) * s;
}
void fixed_loop32(const int32_t* __restrict v, size_t n, int32_t zp, float s, float* __restrict out) {
  IVDEP for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(v[i] - zp) * s;
}
void vec_loop(const int8_t* __restrict v, const int8_t* __restrict z, const float* __restrict sc, size_t n, float* __restrict out) {
  IVDEP for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(static_cast<int32_t>(v[i]) - z[i]) * sc[i];
}
void widen(const int8_t* __restrict v, size_t n, int32_t* __restrict w) {
  IVDEP for (size_t i = 0; i < n; ++i) w[i] = v[i];
}
int main() {
  std::vector<uint8_t> in(NEL / 2), zpb(NEL / 2, 0x35);
  for (size_t i = 0; i < in.size(); ++i) in[i] = uint8_t(i * 2654435761u >> 13);
  std::vector<float> out(NEL), sc(NEL);
  g_out = out.data();
  for (size_t i = 0; i < NEL; ++i) sc[i] = 0.001f * float(i % 97);
  alignas(64) int8_t v8[C], z8[C], lo[C / 2], hi[C / 2];
  alignas(64) int32_t v32[C];
  const float s = 0.0123f; const int32_t zp = 3;
  double t;
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_a(&in[c / 2], C / 2, v8); fixed_loop(v8, C, zp, s, &out[c]); } });
  std::printf("restrict fixed (i8)             %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_a(&in[c / 2], C / 2, v8); widen(v8, C, v32); fixed_loop32(v32, C, zp, s, &out[c]); } });
  std::printf("restrict fixed (widen i32)      %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_a(&in[c / 2], C / 2, v8); unpack_a(&zpb[c / 2], C / 2, z8); vec_loop(v8, z8, &sc[c], C, &out[c]); } });
  std::printf("restrict vector (i8)            %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_a(&in[c / 2], C / 2, v8); out[c] = v8[3]; } });
  std::printf("unpack interleaved only         %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) { unpack_b(&in[c / 2], C / 2, lo, hi); out[c] = lo[3] + hi[2]; } });
  std::printf("unpack split only               %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) fixed_loop(v8, C, zp, s, &out[c]); });
  std::printf("float loop only (i8)            %7.3f ms\n", t);
  t = best_ms([&] { for (size_t c = 0; c < NEL; c += C) fixed_loop32(v32, C, zp, s, &out[c]); });
  std::printf("float loop only (i32)           %7.3f ms\n", t);
  t = best_ms([&] { float tab[16]; for (int q = -8; q < 8; ++q) tab[q & 15] = static_cast<float>(q - zp) * s;
      for (size_t b = 0; b < NEL / 2; ++b) { const uint8_t v = in[b]; out[2 * b] = tab[v & 15]; out[2 * b + 1] = tab[v >> 4]; } });
  std::printf("float table[16]                 %7.3f ms\n", t);
}
