
//
// * This program is free software : you can redistribute itand /or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation, either version 3 of the License, or
// *(at your option) any later version.
// *
// *This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License
// * along with this program.If not, see < http://www.gnu.org/licenses/>.

//#define USE_SSE false
//#define USE_AVX2 true

#define USE_SSE false
#define USE_AVX2 true

#include <assert.h>
#include <stdlib.h>

#if USE_SSE
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#endif
#if USE_AVX2
#include <immintrin.h>
#endif

#define int128 __m128i
#define int256 __m256i
#define int32 int32_t
#define int16 int16_t
#define int8 int8_t
#define u8 uint8_t

#include "simd.h"

#if (USE_AVX2) || (USE_SSE)
static int32 hsum_4x32(int128 v)
{
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    return _mm_cvtsi128_si32(v);
}
#endif

#if (USE_AVX2)
static int32 hsum_8x32(int256 v)
{
    int128 sum128 = _mm_add_epi32(_mm256_castsi256_si128(v),
                                   _mm256_extracti128_si256(v, 1));
    return hsum_4x32(sum128);
}
#endif

void simd_linear_forward(u8 *input, int32 *output, int n_inputs,
                         int n_outputs, int32 *biases, int8 *weights)
{
#if (USE_AVX2)
    int i;
    int j;
    int n_iterations = n_inputs/32;

    int256 c1 = _mm256_set1_epi16(1);

    for (i=0; i < n_outputs; i++) 
    {
        int256 *pi = (int256*)input;
        int256 *pw = (int256*)&weights[i*n_inputs];

        int256 v1 = _mm256_load_si256(pi++);
        int256 v2 = _mm256_load_si256(pw++);
        int256 t1 = _mm256_maddubs_epi16(v1, v2);
        int256 t2 = _mm256_madd_epi16(t1, c1);
        int256 vsum = t2;

        for (j=1; j < n_iterations; j++) 
        {
            v1 = _mm256_load_si256(pi++);
            v2 = _mm256_load_si256(pw++);
            t1 = _mm256_maddubs_epi16(v1, v2);
            t2 = _mm256_madd_epi16(t1, c1);
            vsum = _mm256_add_epi32(vsum, t2);
        }

        output[i] = hsum_8x32(vsum) + biases[i];
    }
#elif (USE_SSE)
    int i;
    int j;
    int n_iterations = n_inputs/16;

    int128 c1 = _mm_set1_epi16(1);

    for (i=0; i < n_outputs; i++) 
    {
        int128 *pi = (int128*)input;
        int128 *pw = (int128*)&weights[i*n_inputs];

        int128 v1 = _mm_load_si128(pi++);
        int128 v2 = _mm_load_si128(pw++);
        int128 temp = _mm_maddubs_epi16(v1, v2);
        int128 vsum = _mm_madd_epi16(temp, c1);

        for (j=1; j < n_iterations; j++) 
        {
            v1 = _mm_load_si128(pi++);
            v2 = _mm_load_si128(pw++);
            temp = _mm_maddubs_epi16(v1, v2);
            temp = _mm_madd_epi16(temp, c1);
            vsum = _mm_add_epi32(vsum, temp);
        }

        output[i] = hsum_4x32(vsum) + biases[i];
    }
#else
    int i;
    int j;

    for (i=0; i < n_outputs; i++) 
    {
        output[i] = biases[i];
        for (j=0; j < n_inputs; j++) 
        {
            output[i] += (input[j]*weights[i*n_inputs+j]);
        }
    }
#endif
}

void simd_scale_and_clamp(int32 *input, u8 *output, int shift,
                          int n_values)
{
#if (USE_AVX2)
    int i;
    int n_iterations = n_values/8;

    int256 *pi = (int256*)input;
    int256 *po = (int256*)output;

    int256 c0 = _mm256_set1_epi8(0);
    int256 idx = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    for (i=0; i < n_iterations / 4; i++) 
    {
        int256 v1 = _mm256_load_si256(pi++);
        int256 v2 = _mm256_load_si256(pi++);
        int256 v16_1 = _mm256_packs_epi32(v1, v2);
        v16_1 = _mm256_srai_epi16(v16_1, shift);

        v1 = _mm256_loadu_si256(pi++);
        v2 = _mm256_loadu_si256(pi++);
        int256 v16_2 = _mm256_packs_epi32(v1, v2);
        v16_2 = _mm256_srai_epi16(v16_2, shift);

        int256 v8 = _mm256_packs_epi16(v16_1, v16_2);
        int256 s = _mm256_permutevar8x32_epi32(v8, idx);
        s = _mm256_max_epi8(s, c0);
        _mm256_store_si256(po++, s);
    }
#elif (USE_SSE)
    int i;
    int n_iterations = n_values / 4;

    int128 *pi = (int128*)input;
    int128 *po = (int128*)output;

    int128 c0 = _mm_set1_epi32(0);
    int128 c127 = _mm_set1_epi32(127);

    for (i=0; i < n_iterations / 4; i++) 
    {
        int128 v1 = _mm_load_si128(pi++);
        v1 = _mm_srai_epi32(v1, shift);
        v1 = _mm_max_epi32(v1, c0);
        v1 = _mm_min_epi32(v1, c127);

        int128 v2 = _mm_load_si128(pi++);
        v2 = _mm_srai_epi32(v2, shift);
        v2 = _mm_max_epi32(v2, c0);
        v2 = _mm_min_epi32(v2, c127);

        int128 o1 =  _mm_packs_epi32(v1, v2);
        v1 = _mm_load_si128(pi++);
        v1 = _mm_srai_epi32(v1, shift);
        v1 = _mm_max_epi32(v1, c0);
        v1 = _mm_min_epi32(v1, c127);

        v2 = _mm_load_si128(pi++);
        v2 = _mm_srai_epi32(v2, shift);
        v2 = _mm_max_epi32(v2, c0);
        v2 = _mm_min_epi32(v2, c127);

        int128 o2 =  _mm_packs_epi32(v1, v2);
        int128 o =  _mm_packs_epi16(o1, o2);
        _mm_store_si128(po++, o);
    }
#else
    int i;

    for (i=0; i < n_values; i++) 
    {
        output[i] = CLAMP((input[i]>>shift), 0, 127);
    }
#endif
}

void simd_clamp(int16 *input, u8 *output, int n_values)
{
#if (USE_AVX2)
    int i;
    int n_iterations = n_values/16;

    int256 *pi = (int256*)input;
    int256 *po = (int256*)output;

    int256 c0 = _mm256_set1_epi8(0);

    for (i=0; i < n_iterations / 2; i++) 
    {
        int256 v1 = _mm256_load_si256(pi++);
        int256 v2 = _mm256_load_si256(pi++);
        int256 v8 = _mm256_packs_epi16(v1, v2);
        int256 s = _mm256_permute4x64_epi64(v8, 0xD8);
        s = _mm256_max_epi8(s, c0);
        _mm256_store_si256(po++, s);
    }
#elif (USE_SSE)
    int i;
    int n_iterations = n_values/8;

    int128 *pi = (int128*)input;
    int128 *po = (int128*)output;

    int128 c0 = _mm_set1_epi16(0);

    for (i=0; i < n_iterations/2; i++) 
    {
        int128 v1 = _mm_load_si128(pi++);
        int128 v2 = _mm_load_si128(pi++);
        int128 v8 = _mm_packs_epi16(v1, v2);
        int128 s = _mm_max_epi8(v8, c0);
        _mm_store_si128(po++, s);
    }
#else
    int i;

    for (i=0; i < n_values; i++) 
    {
        output[i] = CLAMP(input[i], 0, 127);
    }
#endif
}

void simd_copy(int16 *from, int16 *to, int n_values)
{
#if (USE_AVX2)
    int i;
    int n_iterations = n_values/16;

    int256 *pf = (int256*)from;
    int256 *pt = (int256*)to;

    for (i=0; i < n_iterations; i++) 
    {
        pt[i] = _mm256_load_si256(pf++);
    }
#elif (USE_SSE)
    int i;
    int n_iterations = n_values/8;

    int128 *pf = (int128*)from;
    int128 *pt = (int128*)to;

    for (i=0; i < n_iterations; i++) 
    {
        pt[i] = _mm_load_si128(pf++);
    }
#else
    int i;

    for (i=0; i < n_values; i++)
    {
        to[i] = from[i];
    }
#endif
}

void simd_add(int16 *from, int16 *to, int n_values)
{
#if (USE_AVX2)
    int i;
    int n_iterations = n_values/16;

    int256 *pf = (int256*)from;
    int256 *pt = (int256*)to;

    for (i=0; i < n_iterations; i++) 
    {
        pt[i] = _mm256_add_epi16(pf[i], pt[i]);
    }
#elif (USE_SSE)
    int i;
    int n_iterations = n_values/8;

    int128 *pf = (int128*)from;
    int128 *pt = (int128*)to;

    for (i=0; i < n_iterations; i++)
    {
        pt[i] = _mm_add_epi16(pf[i], pt[i]);
    }
#else
    int i;

    for (i=0; i < n_values; i++) 
    {
        to[i] += from[i];
    }
#endif
}

void simd_sub(int16 *from, int16 *to, int n_values)
{
#if (USE_AVX2)
    int i;
    int n_iterations = n_values/16;

    int256 *pf = (int256*)from;
    int256 *pt = (int256*)to;

    for (i=0; i < n_iterations; i++) 
    {
        pt[i] = _mm256_sub_epi16(pt[i], pf[i]);
    }
#elif (USE_SSE)
    int i;
    int n_iterations = n_values/8;

    int128 *pf = (int128*)from;
    int128 *pt = (int128*)to;

    for (i=0 ;i < n_iterations; i++) 
    {
        pt[i] = _mm_sub_epi16(pt[i], pf[i]);
    }
#else
    int i;

    for (i=0; i < n_values; i++) 
    {
        to[i] -= from[i];
    }
#endif
}
