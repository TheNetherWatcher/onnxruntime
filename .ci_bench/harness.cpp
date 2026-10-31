// Runs the exact new ORT code (newcode.cc) against the old scalar loops, with mocked ORT types.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

struct TensorOpCost { double a, b, c; };
namespace concurrency {
struct ThreadPool {
  int n;
  template <class F>
  static void TryParallelFor(ThreadPool* tp, std::ptrdiff_t total, const TensorOpCost&, const F& f) {
    if (!tp || tp->n <= 1) { f(0, total); return; }
    std::vector<std::thread> th;
    for (int t = 0; t < tp->n; ++t) {
      std::ptrdiff_t b = total * t / tp->n, e = total * (t + 1) / tp->n;
      th.emplace_back([&, b, e] { f(b, e); });
    }
    for (auto& x : th) x.join();
  }
};
}  // namespace concurrency

template <int Bits, bool Signed>
struct Packed {
  using UnpackedType = std::conditional_t<Signed, int8_t, uint8_t>;
  std::byte bits_;
  std::byte ToBits() const { return bits_; }
  UnpackedType GetElem(size_t index) const {
    const uint8_t shift = Bits * static_cast<uint8_t>(index);
    const std::byte val = (bits_ >> shift) & std::byte((1 << Bits) - 1);
    if constexpr (Signed) {
      constexpr int s = 32 - Bits;
      return static_cast<int8_t>((static_cast<int32_t>(val) << s) >> s);
    } else {
      return static_cast<UnpackedType>(val);
    }
  }
};

template <typename T, typename ScaleT, typename OutT, bool is_sub_byte, int elements_per_byte = 0>
struct DequantizeLinearApply;
#include "newcode.cc"

template <typename T, typename OutT, int elements_per_byte>
struct Ref {
  void op(size_t M, size_t K, size_t N, const T* input, const float* scale, OutT* output, const T* zero_point) {
    size_t input_index = 0;
    constexpr size_t shift_bits = (elements_per_byte == 2) ? 1 : 2;
    constexpr size_t mask = elements_per_byte - 1;
    for (size_t m = 0; m < M; m++)
      for (size_t bd = 0; bd < K; bd++) {
        auto zp = zero_point ? static_cast<int32_t>(zero_point[bd >> shift_bits].GetElem(bd & mask)) : 0;
        auto sc = static_cast<float>(scale[bd]);
        for (size_t bs = 0; bs < N; bs++, input_index++) {
          int32_t val = static_cast<int32_t>(input[input_index >> shift_bits].GetElem(input_index & mask));
          *output++ = static_cast<OutT>(static_cast<float>(val - zp) * sc);
        }
      }
  }
  void op(size_t M, size_t K, size_t N, size_t B, const T* input, const float* scale, OutT* output,
          const T* zero_point) {
    size_t input_index = 0, zp_index = 0;
    constexpr size_t shift_bits = (elements_per_byte == 2) ? 1 : 2;
    constexpr size_t mask = elements_per_byte - 1;
    for (size_t n = 0; n < M; n++)
      for (size_t bd = 0; bd < K; bd += B) {
        for (size_t qb = 0, qb_end = std::min(B, K - bd); qb < qb_end; ++qb) {
          auto q_zp_index = zp_index;
          for (size_t bs = 0; bs < N; ++bs, ++input_index, ++q_zp_index) {
            auto zp = zero_point ? static_cast<int32_t>(zero_point[q_zp_index >> shift_bits].GetElem(q_zp_index & mask)) : 0;
            auto sc = static_cast<float>(scale[bs]);
            int32_t val = static_cast<int32_t>(input[input_index >> shift_bits].GetElem(input_index & mask));
            *output++ = static_cast<OutT>(static_cast<float>(val - zp) * sc);
          }
        }
        scale += N;
        zp_index += N;
      }
  }
};

using Clock = std::chrono::steady_clock;
template <class F> double best_ms(F&& f, int reps = 7) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = Clock::now(); f();
    best = std::min(best, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  return best;
}

template <int Bits, bool Signed>
bool check(std::mt19937& rng, size_t M, size_t K, size_t N, size_t B, bool with_zp, int threads, bool timeit) {
  using T = Packed<Bits, Signed>;
  constexpr int epb = 8 / Bits;
  const size_t total = M * K * N;
  const size_t nscale = B ? M * ((K + B - 1) / B) * N : K;
  std::vector<T> in((total + epb - 1) / epb), zp((nscale + epb - 1) / epb);
  for (auto& x : in) x.bits_ = std::byte(rng() & 0xFF);
  for (auto& x : zp) x.bits_ = std::byte(rng() & 0xFF);
  std::vector<float> sc(nscale);
  std::uniform_real_distribution<float> ud(-0.1f, 0.1f);
  for (auto& x : sc) x = ud(rng);
  std::vector<float> o1(total, 1.f), o2(total, 2.f);
  const T* z = with_zp ? zp.data() : nullptr;
  Ref<T, float, epb> ref;
  DequantizeLinearApply<T, float, float, true, epb> nw;
  concurrency::ThreadPool tp{threads};
  auto r = [&] { B ? ref.op(M, K, N, B, in.data(), sc.data(), o1.data(), z) : ref.op(M, K, N, in.data(), sc.data(), o1.data(), z); };
  auto w = [&] { B ? nw.op(M, K, N, B, in.data(), sc.data(), o2.data(), z, &tp) : nw.op(M, K, N, in.data(), sc.data(), o2.data(), z, &tp); };
  double t1 = timeit ? best_ms(r) : (r(), 0), t2 = timeit ? best_ms(w) : (w(), 0);
  const bool same = std::memcmp(o1.data(), o2.data(), total * sizeof(float)) == 0;
  if (timeit || !same)
    std::printf("%s%d M=%-5zu K=%-6zu N=%-5zu B=%-4zu zp=%d thr=%-2d ref %8.3f ms new %7.3f ms %6.1fx %s\n",
                Signed ? "int" : "uint", Bits, M, K, N, B, with_zp, threads, t1, t2, t1 / t2,
                same ? "bitwise-equal" : "MISMATCH");
  return same;
}

int main() {
  std::mt19937 rng(42);
  int bad = 0, cnt = 0;
  // fuzz: random small shapes, all type/zp/blocking combos, threads 1 and 3
  for (int it = 0; it < 3000; ++it) {
    size_t M = 1 + rng() % 6, K = 1 + rng() % 1200, N = 1 + (rng() % 3 == 0 ? 0 : rng() % 40);
    size_t B = rng() % 3 == 0 ? 0 : 1 + rng() % 700;
    bool zp = rng() & 1; int thr = 1 + 2 * (rng() & 1);
    switch (rng() % 4) {
      case 0: bad += !check<4, true>(rng, M, K, N, B, zp, thr, false); break;
      case 1: bad += !check<4, false>(rng, M, K, N, B, zp, thr, false); break;
      case 2: bad += !check<2, true>(rng, M, K, N, B, zp, thr, false); break;
      case 3: bad += !check<2, false>(rng, M, K, N, B, zp, thr, false); break;
    }
    ++cnt;
  }
  std::printf("fuzz: %d/%d mismatches\n", bad, cnt);
  for (int thr : {1, static_cast<int>(std::thread::hardware_concurrency())}) {
    check<4, true>(rng, 1, 4096, 4096, 128, true, thr, true);
    check<4, false>(rng, 1, 4096, 4096, 32, true, thr, true);
    check<4, true>(rng, 4096, 4096, 1, 32, true, thr, true);
    check<4, false>(rng, 4096, 4096, 1, 128, false, thr, true);
    check<4, true>(rng, 1, 4096, 4096, 0, true, thr, true);
    check<4, true>(rng, 4096, 4096, 1, 0, true, thr, true);
    check<2, true>(rng, 1, 4096, 4096, 64, true, thr, true);
    check<2, false>(rng, 4096, 4096, 1, 64, true, thr, true);
    check<4, true>(rng, 1, 1 << 22, 2, 0, true, thr, true);   // tiny rows
    check<4, true>(rng, 1, 1 << 21, 4, 32, true, thr, true);  // tiny rows, blocked
  }
}
