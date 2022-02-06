
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

#ifndef NNUE_H
#define NNUE_H

#include <stdint.h>
#include <stdbool.h>
#include <intrin.h>

#define WHITE 0
#define BLACK 1

// NNUE chess defines
#define NN_NPIECES 12
#define NN_NSQUARES 64

enum {
	NN_PAWN = 0,
	NN_KNIGHT = 2,
	NN_BISHOP = 4,
	NN_ROOK = 6,
	NN_QUEEN = 8,
	NN_KING = 10
};

enum {
	NN_WHITE_PAWN,
	NN_BLACK_PAWN,
	NN_WHITE_KNIGHT,
	NN_BLACK_KNIGHT,
	NN_WHITE_BISHOP,
	NN_BLACK_BISHOP,
	NN_WHITE_ROOK,
	NN_BLACK_ROOK,
	NN_WHITE_QUEEN,
	NN_BLACK_QUEEN,
	NN_WHITE_KING,
	NN_BLACK_KING,
	NN_NO_PIECE
};

#define NN_NO_SQUARE 64

// The color of a piece
#define COLOR(p) ((p)&BLACK)

// The value of a piece
#define VALUE(p) ((p)&(~BLACK))

// Change WHITE to BLACK and vice versa
#define FLIP_COLOR(c) ((c)^BLACK)
//

#define ONE 1ULL

#define CACHELINESIZE 64

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

typedef int64_t int64;
typedef int32_t int32;
typedef int16_t int16;
typedef int8_t int8;

struct layer {
	union {
		int8* i8;
		int16* i16;
	}   weights;
	union {
		int32* i32;
		int16* i16;
	}	biases;
};

// Definition of the network architecture
#if (0)
// HalfKA
#define NET_VERSION 0x00000003
#define HALFKX_LAYER_SIZE 256
#define NUM_PT 12
#define MAX_ACTIVE_FEATURES 32
#else
// HalfKA
#if (1)
#define NET_VERSION 0x00000005
#define HALFKX_LAYER_SIZE 256
#define NUM_PT 11					// org = 11
#define MAX_ACTIVE_FEATURES 32
#else
// HalfKP
#if (0)
#define NET_VERSION 0x00000002
#define HALFKX_LAYER_SIZE 128
#define NUM_PT 10
#define MAX_ACTIVE_FEATURES 30
#endif
#endif
#endif


#define NET_HEADER_SIZE 4
#define NUM_INPUT_FEATURES 64*64*NUM_PT

#define ACTIVATION_SCALE_BITS 6
#define OUTPUT_SCALE_FACTOR 16

#define NUM_LAYERS 4

static struct layer layers[NUM_LAYERS];
static int layer_sizes[NUM_LAYERS] = { HALFKX_LAYER_SIZE * 2, 32, 32, 1 };


struct net_data {
	alignas(64) int32 intermediate[HALFKX_LAYER_SIZE * 2];
	alignas(64) u8 output[HALFKX_LAYER_SIZE * 2];
};

struct feature_list {
	u8  size;
	u32 features[MAX_ACTIVE_FEATURES];
};
//


struct dirty_pieces {
	int ndirty;
	int piece[3];
	int from[3];
	int to[3];
};

// acc
struct nnue_input_state {
	// acc
	alignas(64) int16 data[2][HALFKX_LAYER_SIZE];
	// acc is up to date
	bool valid;
};

// Item in the evaluation stack
struct eval_item {
	// acc
	struct nnue_input_state state;
	// dirties
	struct dirty_pieces dirty_pieces;
	int score;
};

//
typedef struct POSITIONDATA
{
	u64 miniboard[4];
	u64 pieces[2];		// whites, blacks
	u64 pawns;
	u64 nites;
	u64 diagonals;
	u64 manhattans;
	int king_sq[2];

	int move;

	struct eval_item eval_stack;
} POSITIONDATA;
//


//
void InitNnue(void);
void NnueDestroy(void);
void NnueResetState(POSITIONDATA* pos);
bool NnueLoadNet(char *path);
bool NnueLoadEmbeddedNet();
int16 NnueEvaluate(POSITIONDATA* pos, int us, int nply);

#endif

inline int popcount64(u64 x)
{
	return ((int)(__popcnt64(x)));
}
//

//_tzcnt_u64(x)
// #include <intrin.h>
//_lzcnt_u64(x)
inline int scanforward64(u64 x)
{
	return ((int)_tzcnt_u64(x));
}
//
inline int scanbackward64(u64 x)
{
	return (63 - (int)__lzcnt64(x));
}

inline u64 flipvertical64(u64 x)
{
	return _byteswap_uint64(x);
}
// Mirror a bitboard horizontally about the center files.
// File a is mapped to file h and vice versa.
inline u64 mirror_horizontal(u64 x)
{
	const u64 k1 = (0x5555555555555555);
	const u64 k2 = (0x3333333333333333);
	const u64 k4 = (0x0f0f0f0f0f0f0f0f);
	x = ((x >> 1) & k1) + 2 * (x & k1);
	x = ((x >> 2) & k2) + 4 * (x & k2);
	x = ((x >> 4) & k4) + 16 * (x & k4);
	return x;
}

// return TRUE if morethanone bit
inline bool morethanone(u64 x)
{
	return x & (x - 1);
}

//inline bool exactly_one(u64 x)
//{
//}

// same concept as exactly one
inline bool onlyone(u64 x)
{
	return x && !morethanone(x);
}

inline bool onlytwo(u64 x)
{
	return (popcount64(x) == 2);
}

inline bool morethantwo(u64 x)
{
	return (popcount64(x) > 2);
}

// return whats left of x after stripping the lsbit
inline u64 lsbitstrip(u64 x)
{
	return (x & (x - 1));
	//return (x ^ (x & (-((int64)x))));
}

// return the lsbit
inline u64 lsbit(u64 x)
{
	//return (x & (-x));
	return (x & (-((int64)x)));
}

#define swaps(a,b) (a = a ^ b, b = a ^ b, a = a ^ b)
#define swapcontents(a,b) (*a = *a ^ *b, *b = *a ^ *b, *a = *a ^ *b)


inline int minint(int a, int b)
{
	return (a < b) ? a : b;
}

inline int maxint(int a, int b)
{
	return (a > b) ? a : b;
}

inline int64 maxint64(int64 a, int64 b)
{
	return (a > b) ? a : b;
}

inline int absint(int a)
{
	return (a < 0) ? -a : a;
}

inline int limit_inclusive_int(int x, int lo, int hi)
{
	return ((x < lo) ? lo : ((x > hi) ? hi : x));
}
//
// set x>=lo, x<=hi
inline u64 limit_inclusive_uint(u64 x, u64 lo, u64 hi)
{
	return ((x < lo) ? lo : ((x > hi) ? hi : x));
}
//
// returns 1 or -1
inline int signofint(int x)
{
	return((x > 0) - (x < 0));
}
//

//
inline u64 reverse_3_bits64(u64 x)
{
	return ((x & 1) << 2) | ((x >> 2) & 1) | (x & 2);
}
//

inline float clamp_float(float x, float min, float max)
{
	if (x < min)
		return min;
	if (x > max)
		return max;
	return x;
}


inline int clamp_int(int x, int min, int max)
{
	if (x < min)
		return min;
	if (x > max)
		return max;
	return x;
}



u16 read_u16_le(u8* buffer);
u16 read_u16_be(u8* buffer);
u32 read_u32_le(u8* buffer);
u32 read_u32_be(u8* buffer);
u64 read_u64_le(u8* buffer);
u64 read_u64_be(u8* buffer);

#if 0
u32 get_file_size(char* file);

#endif
