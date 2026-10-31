namespace {

// Number of sub-byte elements unpacked into a stack buffer at a time.
constexpr size_t kSubByteChunk = 512;

// Unpacks `len` sub-byte elements starting at element index `first` into int8 values.
template <typename T, int elements_per_byte>
void UnpackSubByteElements(const T* src, size_t first, size_t len, int8_t* dst) {
  constexpr int bits = 8 / elements_per_byte;
  constexpr uint8_t mask = (1 << bits) - 1;
  constexpr bool is_signed = std::is_signed_v<typename T::UnpackedType>;

  size_t i = 0;
  for (; i < len && (first + i) % elements_per_byte != 0; ++i) {
    dst[i] = static_cast<int8_t>(src[(first + i) / elements_per_byte].GetElem((first + i) % elements_per_byte));
  }

  const size_t num_bytes = (len - i) / elements_per_byte;
  const T* packed = src + (first + i) / elements_per_byte;
  int8_t* out = dst + i;
  for (size_t b = 0; b < num_bytes; ++b) {
    const uint8_t v = static_cast<uint8_t>(packed[b].ToBits());
    int8_t* o = out + b * elements_per_byte;
    // Unrolled over the elements of the byte so that the compiler vectorizes with 8-bit lanes.
    [&]<size_t... J>(std::index_sequence<J...>) {
      if constexpr (is_signed) {
        // Move element J to the top bits, then arithmetic-shift it back down to sign-extend it.
        ((o[J] = static_cast<int8_t>(static_cast<int8_t>(v << (8 - bits * (J + 1))) >> (8 - bits))), ...);
      } else {
        ((o[J] = static_cast<int8_t>((v >> (bits * J)) & mask)), ...);
      }
    }(std::make_index_sequence<elements_per_byte>{});
  }
  i += num_bytes * elements_per_byte;

  for (; i < len; ++i) {
    dst[i] = static_cast<int8_t>(src[(first + i) / elements_per_byte].GetElem((first + i) % elements_per_byte));
  }
}

// output[i] = (input[first + i] - zp) * scale for i in [0, len).
// With per_element_params, zp and scale advance with i, otherwise they are fixed.
// The zero point is read from packed element index zp_first.
template <typename T, typename ScaleT, typename OutT, int elements_per_byte, bool per_element_params>
void DequantizeSubByteRun(const T* input, size_t first, size_t len, const ScaleT* scale, const T* zero_point,
                          size_t zp_first, OutT* output) {
  alignas(64) int8_t values[kSubByteChunk];
  alignas(64) int8_t zps[kSubByteChunk];
  const int32_t fixed_zp = (!per_element_params && zero_point)
                               ? static_cast<int32_t>(zero_point[zp_first / elements_per_byte].GetElem(
                                     zp_first % elements_per_byte))
                               : 0;
  const float fixed_scale = per_element_params ? 0.0f : static_cast<float>(scale[0]);

  for (size_t c = 0; c < len; c += kSubByteChunk) {
    const size_t n = std::min(kSubByteChunk, len - c);
    UnpackSubByteElements<T, elements_per_byte>(input, first + c, n, values);
    if constexpr (per_element_params) {
      if (zero_point) {
        UnpackSubByteElements<T, elements_per_byte>(zero_point, zp_first + c, n, zps);
      } else {
        std::fill_n(zps, n, int8_t{0});
      }
      for (size_t i = 0; i < n; ++i) {
        output[c + i] = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(values[i]) - zps[i]) *
                                          static_cast<float>(scale[c + i]));
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        output[c + i] = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(values[i]) - fixed_zp) * fixed_scale);
      }
    }
  }
}

// Runs fn(unit) for unit in [0, num_units) on the thread pool. Each unit dequantizes about unit_size elements.
template <typename OutT, typename Fn>
void ParallelForSubByte(concurrency::ThreadPool* thread_pool, size_t num_units, size_t unit_size, const Fn& fn) {
  const double n = static_cast<double>(unit_size);
  const TensorOpCost unit_cost{n, n * sizeof(OutT), n * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, static_cast<std::ptrdiff_t>(num_units), unit_cost,
                                          [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
                                            for (std::ptrdiff_t u = begin; u < end; ++u) {
                                              fn(static_cast<size_t>(u));
                                            }
                                          });
}

}  // namespace

template <typename T, typename ScaleT, typename OutT, int elements_per_byte>
struct DequantizeLinearApply<T, ScaleT, OutT, true, elements_per_byte> {
  // per-tensor/layer or per-axis quantization for sub-byte types
  void op(size_t M, size_t K, size_t N,
          const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    if (N == 1) {
      // Scale and zero point change with every element. Split each row of K elements into chunks.
      const size_t chunks_per_row = (K + kSubByteChunk - 1) / kSubByteChunk;
      ParallelForSubByte<OutT>(thread_pool, M * chunks_per_row, std::min(K, kSubByteChunk), [&](size_t unit) {
        const size_t m = unit / chunks_per_row;
        const size_t k = (unit % chunks_per_row) * kSubByteChunk;
        const size_t len = std::min(kSubByteChunk, K - k);
        DequantizeSubByteRun<T, ScaleT, OutT, elements_per_byte, true>(input, m * K + k, len, scale + k, zero_point,
                                                                       k, output + m * K + k);
      });
      return;
    }

    // Each row of N elements shares one scale and zero point.
    if (N >= kSubByteChunk) {
      ParallelForSubByte<OutT>(thread_pool, M * K, N, [&](size_t row) {
        const size_t k = row % K;
        DequantizeSubByteRun<T, ScaleT, OutT, elements_per_byte, false>(input, row * N, N, scale + k, zero_point, k,
                                                                        output + row * N);
      });
      return;
    }

    // Short rows: unpack several rows at once.
    const size_t rows_per_unit = kSubByteChunk / N;
    const size_t num_rows = M * K;
    ParallelForSubByte<OutT>(
        thread_pool, (num_rows + rows_per_unit - 1) / rows_per_unit, rows_per_unit * N, [&](size_t unit) {
          const size_t row_begin = unit * rows_per_unit;
          const size_t row_end = std::min(num_rows, row_begin + rows_per_unit);
          alignas(64) int8_t values[kSubByteChunk];
          UnpackSubByteElements<T, elements_per_byte>(input, row_begin * N, (row_end - row_begin) * N, values);
          for (size_t row = row_begin; row < row_end; ++row) {
            const size_t k = row % K;
            const int32_t zp = zero_point ? static_cast<int32_t>(zero_point[k / elements_per_byte].GetElem(
                                                k % elements_per_byte))
                                          : 0;
            const float sc = static_cast<float>(scale[k]);
            const int8_t* v = values + (row - row_begin) * N;
            OutT* out = output + row * N;
            for (size_t i = 0; i < N; ++i) {
              out[i] = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(v[i]) - zp) * sc);
            }
          }
        });
  }

  // Blocked quantization
  void op(size_t M, size_t K, size_t N, size_t quant_block_size,
          const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    const size_t num_blocks = (K + quant_block_size - 1) / quant_block_size;

    if (N == 1) {
      // Each quantize block is a contiguous run of elements with one scale and zero point.
      // Short blocks are grouped so that every unit unpacks about kSubByteChunk elements at once.
      const size_t blocks_per_unit = std::max<size_t>(1, kSubByteChunk / quant_block_size);
      const size_t units_per_row = (num_blocks + blocks_per_unit - 1) / blocks_per_unit;
      ParallelForSubByte<OutT>(
          thread_pool, M * units_per_row, std::min(K, blocks_per_unit * quant_block_size), [&](size_t unit) {
            const size_t m = unit / units_per_row;
            const size_t first_block = (unit % units_per_row) * blocks_per_unit;
            const size_t k_begin = first_block * quant_block_size;
            const size_t k_end = std::min(K, k_begin + blocks_per_unit * quant_block_size);
            if (blocks_per_unit == 1) {
              const size_t s = m * num_blocks + first_block;
              DequantizeSubByteRun<T, ScaleT, OutT, elements_per_byte, false>(
                  input, m * K + k_begin, k_end - k_begin, scale + s, zero_point, s, output + m * K + k_begin);
              return;
            }

            alignas(64) int8_t values[kSubByteChunk];
            UnpackSubByteElements<T, elements_per_byte>(input, m * K + k_begin, k_end - k_begin, values);
            for (size_t k = k_begin, block = m * num_blocks + first_block; k < k_end; k += quant_block_size, ++block) {
              const int32_t zp = zero_point ? static_cast<int32_t>(zero_point[block / elements_per_byte].GetElem(
                                                  block % elements_per_byte))
                                            : 0;
              const float sc = static_cast<float>(scale[block]);
              const int8_t* v = values + (k - k_begin);
              OutT* out = output + m * K + k;
              for (size_t i = 0, n = std::min(quant_block_size, k_end - k); i < n; ++i) {
                out[i] = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(v[i]) - zp) * sc);
              }
            }
          });
      return;
    }

    // Each row of N elements uses N consecutive scales and zero points.
    if (N >= kSubByteChunk) {
      ParallelForSubByte<OutT>(thread_pool, M * K, N, [&](size_t row) {
        const size_t m = row / K;
        const size_t k = row % K;
        const size_t s = (m * num_blocks + k / quant_block_size) * N;
        DequantizeSubByteRun<T, ScaleT, OutT, elements_per_byte, true>(input, row * N, N, scale + s, zero_point, s,
                                                                       output + row * N);
      });
      return;
    }

    // Short rows: unpack several rows at once.
    const size_t rows_per_unit = kSubByteChunk / N;
    const size_t num_rows = M * K;
    ParallelForSubByte<OutT>(
        thread_pool, (num_rows + rows_per_unit - 1) / rows_per_unit, rows_per_unit * N, [&](size_t unit) {
          const size_t row_begin = unit * rows_per_unit;
          const size_t row_end = std::min(num_rows, row_begin + rows_per_unit);
          alignas(64) int8_t values[kSubByteChunk];
          alignas(64) int8_t zps[kSubByteChunk];
          UnpackSubByteElements<T, elements_per_byte>(input, row_begin * N, (row_end - row_begin) * N, values);
          if (!zero_point) {
            std::fill_n(zps, N, int8_t{0});
          }
          for (size_t row = row_begin; row < row_end; ++row) {
            const size_t s = (row / K * num_blocks + row % K / quant_block_size) * N;
            if (zero_point) {
              UnpackSubByteElements<T, elements_per_byte>(zero_point, s, N, zps);
            }
            const int8_t* v = values + (row - row_begin) * N;
            const ScaleT* sc = scale + s;
            OutT* out = output + row * N;
            for (size_t i = 0; i < N; ++i) {
              out[i] = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(v[i]) - zps[i]) *
                                         static_cast<float>(sc[i]));
            }
          }
        });
  }
};
