

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

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>

#include "board.h"
#include "move_do.h"

// switch me off please ...
#define DEBUG_NNUE 0

#define UPDATE_ALWAYS 0


#include "nnue.h"
#include "globals.h"
#include "simd.h"

POSITIONDATA posdata[256];


#define TO_SHIFT 6
#define PROMO_SHIFT 12
#define CAPTYPE_SHIFT 16	// NB empty = 0b0111 shifted
#define MOVETYPE_SHIFT 19

// lowest 16 bits form hash16
// 29 wasteful bits at the moment
#define FROM_MASK		0b0000000000000000000111111ULL
#define TO_MASK			0b0000000000000111111000000ULL
#define PROMO_MASK		0b0000000000111000000000000ULL
#define CAPTURE_FLAG	0b0000000001000000000000000ULL
// above this points exceeds 16 bits
#define CAPTYPE_MASK	0b0000001110000000000000000ULL
#define MOVETYPE_MASK	0b0001110000000000000000000ULL
#define EP_MOVE_FLAG	0b0010000000000000000000000ULL
#define CASTLES_FLAG	0b0100000000000000000000000ULL
#define PAWNMOVE_FLAG	0b1000000000000000000000000ULL

inline int from_sq(int move)
{
	return (move & FROM_MASK);
}
inline int to_sq(int move)
{
	return ((move & TO_MASK) >> TO_SHIFT);
}
inline int cap_type(int move)
{
	return ((move & CAPTYPE_MASK) >> CAPTYPE_SHIFT);
}
inline int move_type(int move)
{
	return ((move & MOVETYPE_MASK) >> MOVETYPE_SHIFT);
}
inline int promo_type(int move)
{
	return ((move & PROMO_MASK) >> PROMO_SHIFT);
}

inline bool IsCastles(int move)
{
	return (move & CASTLES_FLAG);
}
inline bool IsKingSideCastles(int move)
{
	return (IsCastles(move) && ((to_sq(move) & 7) == 2));
}
inline bool IsEpCapture(int move)
{
	return (move & EP_MOVE_FLAG);
}
inline bool IsCapture(int move)
{
	return (move & CAPTURE_FLAG);
}
inline bool IsPromo(int move)
{
	return (move & PROMO_MASK);
}


inline u64 get_queens(POSITIONDATA* pos)
{
	return (pos->diagonals & pos->manhattans);
}

inline u64 get_bishops(POSITIONDATA* pos)
{
	return (pos->diagonals & ~pos->manhattans);
}

inline u64 get_rooks(POSITIONDATA* pos)
{
	return (pos->manhattans & ~pos->diagonals);
}

// 4 bit nibbles
// write piece type to minboard entry
// for empty entry, type = 0b0111
#define EMPTY 0b0111ULL

inline void piece_to_square(POSITIONDATA* pos, u64 s1, u64 type)
{
	int ix = s1 & 0b11ULL;
	u64 entry = pos->miniboard[ix] & (~(EMPTY << (s1 & 0b111100ULL)));
	pos->miniboard[ix] = entry | (type << (s1 & 0b111100ULL));
}

inline void clear_square(POSITIONDATA* pos, u64 s1)
{
	pos->miniboard[s1 & 0b11ULL] |= (EMPTY << (s1 & 0b111100ULL));
}

inline u64 square_to_piece(POSITIONDATA* pos, u64 s1)
{
	u64 type = (pos->miniboard[s1 & 0b11ULL] >> (s1 & 0b111100ULL)) & EMPTY;
	return type;
}

#if DEBUG_NNUE
bool IntegrityCheckPosition(POSITIONDATA* pos)
{
	bool status = TRUE;
	u64 all_pieces = pos->pieces[0] | pos->pieces[1];
	u64 pawns = pos->pawns;
	u64 nites = pos->nites;
	u64 bishops = get_bishops(pos);
	u64 rooks = get_rooks(pos);
	u64 queens = get_queens(pos);
	u64 kings = ((ONE << pos->king_sq[0]) | (ONE << pos->king_sq[1]));

	// test miniboard and piece bitboards match up

	for (u64 s1 = 0; s1 < 64; s1++)
	{
		u64 type = square_to_piece(pos, s1);
		if (type != EMPTY)
		{
			all_pieces ^= (ONE << s1);
			switch (type)
			{
			case NN_PAWN/2:
				pawns ^= (ONE << s1);
				break;
			case NN_KNIGHT/2:
				nites ^= (ONE << s1);
				break;
			case NN_BISHOP/2:
				bishops ^= (ONE << s1);
				break;
			case NN_ROOK/2:
				rooks ^= (ONE << s1);
				break;
			case NN_QUEEN/2:
				queens ^= (ONE << s1);
				break;
			case NN_KING/2:
				kings ^= (ONE << s1);
				break;
			default:
				status = FALSE;
				assert(1 == 2);
			}
		}
	}
	status = !(all_pieces | pawns | nites | bishops | rooks | queens | kings);
	return status;
}
#endif

// ================================================


inline int transform_square(int s1, int side)
{
	return (side == BLACK) ? s1 ^ 56 : s1;
}

// piece = 2*type + stm=BLACK, eg black_king = 11 (2*5 + 1)
static int CalculateFeatureIndex(int sq, int piece, int king_sq, int side)
{
	sq = transform_square(sq, side);
	piece ^= side;
#if (NUM_PT ==11)
	if (piece == 11)
		piece -= 1;
#endif
	return (sq + (piece * NN_NSQUARES) + NUM_PT * NN_NSQUARES * king_sq);
}

static void FindActiveFeatures(POSITIONDATA* pos, int side,
								 struct feature_list *list)
{
	int king_sq;
	int sq;
	u32 index;
	u64 bb;

	// Initialize
	list->size = 0;

	// king
	king_sq = transform_square(pos->king_sq[side], side);

	// bitboard of all pieces excluding the two kings if HalfKP
	bb = (pos->pieces[WHITE] | pos->pieces[BLACK]);
	if (NUM_PT == 10)
	{
		bb &= (~((ONE << pos->king_sq[WHITE]) | (ONE << pos->king_sq[BLACK])));
	}

	// feature index
	while (bb)
	{
		sq = scanforward64(bb);
		bb ^= (ONE << sq);
		
		int t1 = (int)square_to_piece(pos, sq) * 2 + (!((ONE << sq) & pos->pieces[WHITE]));
		index = CalculateFeatureIndex(sq, t1, king_sq, side);
		list->features[list->size++] = index;
	}
}

static bool FindChangedFeatures(POSITIONDATA* pos, int side,
								  struct feature_list *added,
								  struct feature_list *removed)
{
	struct dirty_pieces *dp = &pos->eval_stack.dirty_pieces;
	int            king_sq;
	int            piece;
	u32            index;

	// Init
	added->size = 0;
	removed->size = 0;

	// king
	king_sq = transform_square(pos->king_sq[side], side);

	// dirties loop
	for (int i=0; i < dp->ndirty; i++) 
	{
		piece = dp->piece[i];

		// ignore the two kings for HalfKP
		if ((VALUE(piece) == NN_KING) && (NUM_PT==10))
		{
			continue;
		}

		// removed or added features
		if (dp->from[i] != NN_NO_SQUARE) 
		{
			index = CalculateFeatureIndex(dp->from[i], piece, king_sq, side);
			removed->features[removed->size++] = index;
		}
		if (dp->to[i] != NN_NO_SQUARE) 
		{
			index = CalculateFeatureIndex(dp->to[i], piece, king_sq, side);
			added->features[added->size++] = index;
		}
	}
	return false;
}

static void FullUpdate(POSITIONDATA* pos, int side)
{
	struct feature_list active_features;
	u32            offset;
	u32            index;
	int16          *data;

	FindActiveFeatures(pos, side, &active_features);

	data = &pos->eval_stack.state.data[side][0];

	simd_copy(layers[0].biases.i16, data, HALFKX_LAYER_SIZE);

	for (int i=0; i < active_features.size; i++) 
	{
		index = active_features.features[i];
		offset = HALFKX_LAYER_SIZE*index;
		simd_add(&layers[0].weights.i16[offset], data, HALFKX_LAYER_SIZE);
	}
}

static void IncrementalUpdate(POSITIONDATA *pos, int side)
{
	struct feature_list added;
	struct feature_list removed;
	u32            offset;
	u32            index;
	int16             *data;
	int16             *prev_data;

	FindChangedFeatures(pos, side, &added, &removed);

	data = &pos->eval_stack.state.data[side][0];
	prev_data = &(pos-1)->eval_stack.state.data[side][0];

	// copy up acc
	simd_copy(prev_data, data, HALFKX_LAYER_SIZE);

	// removed features
	for (int i=0; i < removed.size; i++) 
	{
		index = removed.features[i];
		offset = HALFKX_LAYER_SIZE*index;
		simd_sub(&layers[0].weights.i16[offset], data, HALFKX_LAYER_SIZE);
	}

	// added features
	for (int i=0; i < added.size; i++) 
	{
		index = added.features[i];
		offset = HALFKX_LAYER_SIZE*index;
		simd_add(&layers[0].weights.i16[offset], data, HALFKX_LAYER_SIZE);
	}
}

static bool IsIncrementalUpdatePossible(POSITIONDATA* pos, int side, int nply)
{	// if at root, or we don't have a valid accumalator from one below, then full update
	if ((nply == 0) || !(pos-1)->eval_stack.state.valid) 
	{
		return false;
	}

	// our king moves -> rebuild acc
	if (pos->eval_stack.dirty_pieces.piece[0] == (side + NN_KING)) 
	{
		return false;
	}
	return true;
}

static void HalfKxLayerForward(POSITIONDATA *pos, int move, struct net_data *data, int us, int nply)
{
	int      perspectives[2];
	int      side;
	uint32_t offset;
	uint8_t  *temp;
	int16_t  *features;

#if UPDATE_ALWAYS
	for (side = 0; side < 2; side++)
	{
		FullUpdate(pos, side);
	}
#else
	// is acc valid?
	if (pos->eval_stack.state.valid)
	{	// only valid because we just did a null move
		// in which case, we need to xfer up both sides of the acc and the psqt acc
		//assert(!move);
		memcpy(&pos->eval_stack.state.accumulator[0][0], &(pos - 1)->eval_stack.state.accumulator[0][0], 2 * HALFKX_LAYER_SIZE * sizeof(int16));
	}
	else
	{
		for (side=0; side < 2; side++) 
		{
			if (IsIncrementalUpdatePossible(pos, side, nply)) 
			{
				IncrementalUpdate(pos, side);
			} 
			else 
			{
				FullUpdate(pos, side);
			}
		}
		// acc is now valid
		pos->eval_stack.state.valid = true;
	}
#endif

	perspectives[0] = us;
	perspectives[1] = us ^ 1;
	for (side=0; side < 2; side++) 
	{
		offset = HALFKX_LAYER_SIZE*side;
		temp = &data->output[offset];
		features = pos->eval_stack.state.data[perspectives[side]];
		simd_clamp(features, temp, HALFKX_LAYER_SIZE);
	}
}

static void LinearLayerForward(int idx, struct net_data *data, bool output)
{
	simd_linear_forward(data->output, data->intermediate, layer_sizes[idx-1],
						layer_sizes[idx], layers[idx].biases.i32,
						layers[idx].weights.i8);
	if (!output) 
	{
		simd_scale_and_clamp(data->intermediate, data->output,
							 ACTIVATION_SCALE_BITS, layer_sizes[idx]);
	}
}

static void NetworkForward(POSITIONDATA* pos, int move, struct net_data *data, int us, int nply)
{
	int i;
	HalfKxLayerForward(pos, move, data, us, nply);
	for (i=1; i < NUM_LAYERS-1; i++) 
	{
		LinearLayerForward(i, data, false);
	}
	LinearLayerForward(i, data, true);
}

static bool ParseHeader(u8 **data)
{
	u8  *iter = *data;
	u32 version;

	// NNUE version number
	version = read_u32_le(iter);
	iter += 4;
	*data = iter;
	return (version == NET_VERSION);
}

static bool ParseNetwork(u8 **data)
{
	u8 *iter = *data;
	int     i,j;

	// biases and weights
	for (i=0; i < HALFKX_LAYER_SIZE; i++, iter+=2) 
	{
		layers[0].biases.i16[i] = read_u16_le(iter);
	}
	for (i=0; i < HALFKX_LAYER_SIZE*NUM_INPUT_FEATURES; i++, iter+=2) 
	{
		layers[0].weights.i16[i] = read_u16_le(iter);
	}

	// biases and weights linear layers
	for (i=1; i < NUM_LAYERS; i++) 
	{
		for (j=0; j < layer_sizes[i]; j++, iter+=4) 
		{
			layers[i].biases.i32[j] = read_u32_le(iter);
		}
		for (j=0; j < layer_sizes[i]*layer_sizes[i-1]; j++, iter++) 
		{
			layers[i].weights.i8[j] = (int8)*iter;
		}
	}
	*data = iter;
	return true;
}

static u32 CalculateNetSize(void)
{
	u32 size = 0;

	size += NET_HEADER_SIZE;
	size += HALFKX_LAYER_SIZE*sizeof(int16);
	size += HALFKX_LAYER_SIZE*NUM_INPUT_FEATURES*sizeof(int16);

	for (int i=1; i < NUM_LAYERS; i++) 
	{
		size += layer_sizes[i]*sizeof(int32);
		size += layer_sizes[i]*layer_sizes[i-1]*sizeof(int8);
	}
	return size;
}

void InitNnue(void)
{
	// mallocs
	layers[0].weights.i16 = (int16*) _aligned_malloc(HALFKX_LAYER_SIZE*NUM_INPUT_FEATURES*sizeof(int16), 64);
	layers[0].biases.i16 = (int16*)_aligned_malloc(HALFKX_LAYER_SIZE*sizeof(int16), 64);
	for (int i=1; i < NUM_LAYERS; i++) 
	{
		layers[i].weights.i8 = (int8*)_aligned_malloc(layer_sizes[i]*layer_sizes[i-1]*sizeof(int8), 64);
		layers[i].biases.i32 = (int32*)_aligned_malloc(layer_sizes[i]*sizeof(int32), 64);
	}
}

void NnueDestroy(void)
{
	_aligned_free(layers[0].weights.i16);
	_aligned_free(layers[0].biases.i16);
	for (int i=1; i < NUM_LAYERS; i++) 
	{
		_aligned_free(layers[i].weights.i8);
		_aligned_free(layers[i].biases.i32);
	}
}

#if 0
bool NnueLoadNet(char *path)
{
	u32		size;
	size_t	count;
	u8		*data = NULL;
	u8		*iter;
	FILE	*fh = NULL;
	bool	ret = true;

	size = get_file_size(path);
	if (size != CalculateNetSize()) 
	{
		ret = false;
	}
	else
	{
		errno_t err;
		err = fopen_s(&fh, path, "rb");
		if (fh == NULL) {
			ret = false;
		}
		else
		{
			data = (u8*)malloc(size);
			count = fread(data, 1, size, fh);
			if (count != size) {
				ret = false;
			}
			else
			{
				/* Parse network header */
				iter = data;
				if (!ParseHeader(&iter)) {
					ret = false;
				}
				else
				{
					/* Parse network */
					if (!ParseNetwork(&iter)) {
						ret = false;
					}
				}
			}
		}
	}

	free(data);
	if (fh != NULL) {
		fclose(fh);
	}
	return ret;
}
#endif

// #include "c:/users/chris/source/repos/growing_fruit-master-2/embedded-nnues/net-sf14-epoch-316.txt"
// #include "c:/rebel/growing_fruit-master-3/embedded-nnues/cv121_tal_100-iter=1-pos=0.9B-d9-lambda=0.85-lr=0.000184-epoch=132-arch=3-loss=0.04859.txt"
// #include "g:/Epoch-104-20Mb.txt"

// #include "g:/epoch-78x.txt"			// rebel-14

// best #include "c:/rebel/growing_fruit-master-3/embedded-nnues/Benjamin 1.1-iter=1-pos=1.2B-d6+d7-lambda=0.5-lr=0.000001-epoch=79-arch=3-loss=0.16672.txt"

//#include "embedded-nnues/resume-iter=1-pos=1.2B-d6+d7-lambda=0.5-lr=0.000004-epoch=147-arch=3-loss=0.16804.txt"
#include "embedded-nnues/ChrisW-NNUE-Tal-10-Minute-Blitzer.txt"
//#include "embedded-nnues/ChrisW-NNUE-Tal-5-Minute-Blitzer.txt"
//#include "embedded-nnues/ChrisW-NNUE-Tal-2-Minute-Blitzer.txt"
//#include "embedded-nnues/ChrisW-NNUE-Tal-1-Minute-Blitzer.txt"


bool NnueLoadEmbeddedNet()
{
	u8* iter;
	bool	ret = true;

	int size = 0;
	int i = 0;
	while (sizes[i])
	{
		size += sizes[i++];
	}
	void* mem = malloc((size_t)size);
	u8* embedded_weights = (u8*)mem;
	u8* dest = embedded_weights;

#define N_SPLITS (N_BUCKET+1)

#if (N_BUCKET == 5)
	u8* cpp[N_SPLITS] = { embedded_weights_0,
				embedded_weights_1,
				embedded_weights_2,
				embedded_weights_3,
				embedded_weights_4,
				embedded_weights_5 };
#elif (N_BUCKET == 7)
	u8* cpp[N_SPLITS] = { embedded_weights_0,
				embedded_weights_1,
				embedded_weights_2,
				embedded_weights_3,
				embedded_weights_4,
				embedded_weights_5,
				embedded_weights_6,
				embedded_weights_7};
#elif (N_BUCKET == 11)
	u8* cpp[N_SPLITS] = { embedded_weights_0,
				embedded_weights_1,
				embedded_weights_2,
				embedded_weights_3,
				embedded_weights_4,
				embedded_weights_5,
				embedded_weights_6,
				embedded_weights_7,
				embedded_weights_8,
				embedded_weights_9,
				embedded_weights_10,
				embedded_weights_11 };
#endif
	for (i = 0; i < N_SPLITS; i++)
	{
		u8* source = cpp[i];
		memcpy(dest, source, (size_t)sizes[i]);
		dest += (u64)sizes[i];
	}

	assert((dest - embedded_weights) == size);
	assert(NET_VERSION == *embedded_weights);

	// size = sizeof(embedded_weights);
	if (size != CalculateNetSize())
	{
		ret = false;
	}
	else
	{
		// Parse network header
		iter = embedded_weights;
		if (!ParseHeader(&iter)) 
		{
			ret = false;
		}
		else
		{
			// Parse network
			if (!ParseNetwork(&iter)) 
			{
				ret = false;
			}
		}
	}
	free(mem);
	return ret;
}



void NnueMakeMove(POSITIONDATA *pos, int move, int us, int nply)
{
	assert(pos != NULL);
	pos->eval_stack.state.valid = false;

	if (!move)
	{	// null move
		if ((nply > 0) && (pos - 1)->eval_stack.state.valid)
		{
			pos->eval_stack.state = (pos - 1)->eval_stack.state;
		}
		return;
	}

#if 1

	int	from = from_sq(move);
	int	to = to_sq(move);

	assert(pos->pieces[us] & (ONE << to));

	int	piece = move_type(move) * 2 + us;
	struct dirty_pieces *dp;

	dp = &(pos->eval_stack.dirty_pieces);
	dp->ndirty = 1;

	if (IsCastles(move))
	{
		dp->ndirty = 2;
		dp->piece[0] = NN_KING + us;
		dp->from[0] = from;
		dp->to[0] = to;
		dp->piece[1] = NN_ROOK + us;
		if (IsKingSideCastles(move))
		{
			dp->from[1] = to + 1;
			dp->to[1] = to - 1;
		}
		else
		{
			dp->from[1] = to - 2;
			dp->to[1] = to + 1;
		}
		return;
	}
	if (IsEpCapture(move))
	{
		dp->ndirty = 2;
		dp->piece[0] = piece;
		dp->from[0] = from;
		dp->to[0] = to;
		dp->piece[1] = NN_PAWN + (us^1);
		dp->from[1] = (us == WHITE) ? to-8 : to+8;
		dp->to[1] = NN_NO_SQUARE;
		return;
	} 

	dp->piece[0] = piece;
	dp->from[0] = from;
	dp->to[0] = to;

	if (IsCapture(move)) 
	{
		dp->ndirty = 2;
		dp->piece[1] = cap_type(move) * 2 + (us ^ 1);
		dp->from[1] = to;
		dp->to[1] = NN_NO_SQUARE;
	}
	if (IsPromo(move)) 
	{
		dp->to[0] = NN_NO_SQUARE;
		dp->piece[dp->ndirty] = promo_type(move) * 2 + us;
		dp->from[dp->ndirty] = NN_NO_SQUARE;
		dp->to[dp->ndirty] = to;
		dp->ndirty++;
	}
#endif
}

//POSITIONDATA posdata[256];

int16 NnueEvaluate(POSITIONDATA* pos, int us, int nply)
{
	assert(nply > 0);

	struct net_data data;
	
	int move = pos->move;

	//assert(pos->pieces[us] & (ONE << to_sq(move)));

	NnueMakeMove(pos, move, us ^ 1, nply);
	NetworkForward(pos, move, &data, us, nply);
	return data.intermediate[0] / OUTPUT_SCALE_FACTOR;
}
//

// 
// 
// at root (nply=0)
// build our posdata structure
// build accumulator
// acc_status=good
void Init_Nnue_Root(board_t* board, int nply)
{
	assert(nply == 0);
	POSITIONDATA* pos = &posdata[0];
	posdata[0] = { 0 };
	
	pos->pieces[WHITE] = pos->pieces[BLACK] = 0ull;		// whites, blacks
	pos->pawns = pos->nites = pos->diagonals = pos->manhattans = 0ull;
	pos->king_sq[WHITE] = pos->king_sq[BLACK] = -1;
	for (int s1 = 0; s1 < 64; s1++)
	{
		clear_square(pos, s1);
	}
	for (int r = 0; r < 8; r++)
	{
		for (int f = 0; f < 8; f++)
		{
			int fruit_sq = 0x44 + f + r*16;	// fruit format
			int fruit_pt = board->square[fruit_sq];
			if (fruit_pt == 0)
				continue;
			int pt12 = PIECE_TO_12(fruit_pt);
			int t1 = pt12 >> 1;
			int side = pt12 & 1;
			int s1 = r * 8 + f;				// 0-63
			u64 bb = (ONE << s1);
			pos->pieces[side] |= bb;
			switch (t1)
			{
			case (NN_PAWN >> 1):
				pos->pawns |= bb;
				piece_to_square(pos, s1, NN_PAWN/2);
				break;
			case (NN_KNIGHT >> 1):
				pos->nites |= bb;
				piece_to_square(pos, s1, NN_KNIGHT / 2);
				break;
			case (NN_BISHOP >> 1):
				pos->diagonals |= bb;
				piece_to_square(pos, s1, NN_BISHOP / 2);
				break;
			case (NN_ROOK >> 1):
				pos->manhattans |= bb;
				piece_to_square(pos, s1, NN_ROOK / 2);
				break;
			case (NN_QUEEN >> 1):
				pos->diagonals |= bb;
				pos->manhattans |= bb;
				piece_to_square(pos, s1, NN_QUEEN / 2);
				break;
			case (NN_KING >> 1):
				pos->king_sq[side] = s1;
				piece_to_square(pos, s1, NN_KING / 2);
				break;
			default:
				assert(1 == 2);
			}
		}
	}
	assert((pos->king_sq[WHITE] >= 0) && (pos->king_sq[BLACK] >= 0));
#if DEBUG_NNUE
	assert(IntegrityCheckPosition(pos));
#endif
	// full update both sides of accumulator
	for (int side = 0; side < 2; side++)
	{
		FullUpdate(pos, side);
	}
	// state is now valid
	pos->eval_stack.state.valid = true;
	return;
}
//
#include "move.h"
//#define MOVE_FROM(move)                (SQUARE_FROM_64(((move)>>6)&077))
//#define MOVE_TO(move)                  (SQUARE_FROM_64((move)&077))
//const int MoveNone = 0;  // HACK: a1a1 cannot be a legal move
//const int MoveNull = 11; // HACK: a1d2 cannot be a legal move

// file = (sq-0x44) & 7
// rank = ((sq-0x44) / 16) & 7
#define SQ64(sq) ((((sq)-0x44) & 7) | ((((sq)-0x44)>>1) & 0x38))


// just prior to move()
// can be nullmove=0
// modify posdata+1 for move
void Nnue_Housekeeping(board_t* board, int fruit_move, undo_t* undo, int nply)
{
	POSITIONDATA* pos = &posdata[nply];
	POSITIONDATA* newpos = pos + 1;
	memcpy(newpos, pos, sizeof(POSITIONDATA));
	
	int move = 0;
	if (!((fruit_move == MoveNone) || (fruit_move == MoveNull)))
	{
		int s1 = SQ64(MOVE_FROM(fruit_move));
		int s2 = SQ64(MOVE_TO(fruit_move));

		int fruit_t1 = PIECE_TO_12(board->square[MOVE_FROM(fruit_move)]);

		int us = fruit_t1 & 1;
		int t1 = fruit_t1 >> 1;
		move |= (s1 | (s2 << TO_SHIFT) | (t1 << MOVETYPE_SHIFT));

		int t2 = 0;
		if (move_is_capture(fruit_move, board))
		{
			int s3 = s2;
			t2 = PIECE_TO_12(move_capture(fruit_move, board)) >> 1;	// accounts for ep sq
			move |= (CAPTURE_FLAG | (t2 << CAPTYPE_SHIFT));
			if (MOVE_IS_EN_PASSANT(fruit_move))
			{
				move |= EP_MOVE_FLAG;
				// capture sq = fileof(s2), rankof(s1)
				s3 = (s2 & 7) | (s1 & (7<<3));
			}
			// remove piece on s3
			clear_square(newpos, s3);
			newpos->pieces[us ^ 1] ^= (ONE << s3);
			assert(t2 != NN_KING / 2);
			switch (t2)
			{
			case NN_PAWN / 2:
				newpos->pawns ^= (ONE << s3);
				break;
			case NN_KNIGHT / 2:
				newpos->nites ^= (ONE << s3);
				break;
			case NN_BISHOP / 2:
				newpos->diagonals ^= (ONE << s3);
				break;
			case NN_ROOK / 2:
				newpos->manhattans ^= (ONE << s3);
				break;
			case NN_QUEEN / 2:
				newpos->diagonals ^= (ONE << s3);
				newpos->manhattans ^= (ONE << s3);
				break;
			default:
				assert(1 == 2);
			}
		}

		if (MOVE_IS_CASTLE(fruit_move))
		{	// move the rook
			move |= CASTLES_FLAG;
			int s4, s5;
			s4 = s1 & 0x38;	// A1 or A8
			s5 = s4 + 3;	// D1 or D8
			if ((s2 & 7) == 6)
			{
				s4 += 7;	// H1 or H8
				s5 += 2;	// F1 or F8
			}
			clear_square(newpos, s4);
			newpos->pieces[us] ^= ((ONE << s4) | (ONE << s5));
			piece_to_square(newpos, s5, NN_ROOK / 2);
			newpos->manhattans ^= ((ONE << s4) | (ONE << s5));
		}
		// remove from piece
		clear_square(newpos, s1);
		newpos->pieces[us] ^= (ONE << s1);
		switch (t1)
		{
		case NN_PAWN / 2:
			newpos->pawns ^= (ONE << s1);
			break;
		case NN_KNIGHT / 2:
			newpos->nites ^= (ONE << s1);
			break;
		case NN_BISHOP / 2:
			newpos->diagonals ^= (ONE << s1);
			break;
		case NN_ROOK / 2:
			newpos->manhattans ^= (ONE << s1);
			break;
		case NN_QUEEN / 2:
			newpos->diagonals ^= (ONE << s1);
			newpos->manhattans ^= (ONE << s1);
			break;
		case NN_KING /2:
			break;
		default:
			assert(1 == 2);
		}

		if (MOVE_IS_PROMOTE(fruit_move))
		{	// make promo piece
			t1 = PIECE_TO_12(move_promote(fruit_move)) >> 1;
			move |= (t1 << PROMO_SHIFT);
		}
		// add moving piece
		newpos->pieces[us] ^= (ONE << s2);
		piece_to_square(newpos, s2, t1);
		switch (t1)
		{
		case NN_PAWN / 2:
			newpos->pawns ^= (ONE << s2);
			break;
		case NN_KNIGHT / 2:
			newpos->nites ^= (ONE << s2);
			break;
		case NN_BISHOP / 2:
			newpos->diagonals ^= (ONE << s2);
			break;
		case NN_ROOK / 2:
			newpos->manhattans ^= (ONE << s2);
			break;
		case NN_QUEEN / 2:
			newpos->diagonals ^= (ONE << s2);
			newpos->manhattans ^= (ONE << s2);
			break;
		case NN_KING / 2:
			newpos->king_sq[us] = s2;
			break;
		default:
			assert(1 == 2);
		}
	}
	newpos->move = move;
	newpos->eval_stack.state.valid = false;
#if DEBUG_NNUE
	assert(IntegrityCheckPosition(newpos));
#endif
	return;
}

