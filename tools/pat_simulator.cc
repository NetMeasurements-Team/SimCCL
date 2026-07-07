/**
 * PAT Algorithm Host-Side Simulator
 *
 * Extracted from NCCL 2.30 src/include/collectives.h (PatAGAlgorithm class).
 * This standalone program simulates the PAT (Parallel Aggregated Trees) algorithm
 * step sequence WITHOUT requiring CUDA or full NCCL build infrastructure.
 *
 * Purpose: Understand PAT's internal XOR-based peer selection for each step,
 * enabling accurate FlowModel generation in SimCCL.
 *
 * Build: g++ -std=c++17 -O2 -o pat_simulator pat_simulator.cc
 * Run:   ./pat_simulator [nranks] [count] [nChannels]
 *
 * Source: nccl-2.30/src/include/collectives.h L701-870 (PatAGAlgorithm)
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

// === Stubs for NCCL internal dependencies ===
#define __host__
#define __device__
#define __forceinline__
#define COMPILER_FFS(x) __builtin_ffs(x)
#define COMPILER_CLZ(x) __builtin_clz(x)
#define COMPILER_CLZL(x) __builtin_clzl(x)
#define COMPILER_CLZLL(x) __builtin_clzll(x)

// log2Up from bitops.h (host path only)
template <typename Int>
static int log2Up(Int x) {
  int w, n;
  if (x != 0) x -= 1;
  if (x == 0) return 0;
  if (sizeof(Int) <= sizeof(unsigned int)) {
    w = 8 * sizeof(unsigned int);
    n = COMPILER_CLZ((unsigned int)x);
  } else if (sizeof(Int) <= sizeof(unsigned long)) {
    w = 8 * sizeof(unsigned long);
    n = COMPILER_CLZL((unsigned long)x);
  } else {
    w = 8 * sizeof(unsigned long long);
    n = COMPILER_CLZLL((unsigned long long)x);
  }
  return w - n;
}

// ncclPatStep struct from collectives.h L415-418
struct ncclPatStep {
  int recvDim, sendDim, recvOffset, sendOffset, stepOffset, postRecv, postSend, nelem, last, flags;
  size_t inpIx, outIx;
};

static constexpr int PatUsed = 0x1, PatSkipped = 0x2;

// === PatAGAlgorithm class (extracted from collectives.h L701-870) ===
template <typename T>
class PatAGAlgorithm {
  size_t offset;
  size_t end;
  size_t count;
  int chunkCount;
  int nelem;
  int rank;
  int nranks;
  int nrPow2;
  int postFreq;
  int lastA;
  int parallelFactor;
  int aggFactor;
  int as;
  int a;
  int aggDelta;
  int scale;
  int phase;

  int asDim;
  int v;
  int bitCount[32];
  int bitZeroStep[32];

  ssize_t min(ssize_t a, ssize_t b) { return (a < b) ? a : b; }

  int getNelem() { return min(chunkCount, end - offset); }

  int mirror(int i, int max) {
    int ret = 0;
    for (int mask = 1, imask = max / 2; mask < max; mask <<= 1, imask >>= 1) {
      if ((i & mask)) ret += imask;
    }
    return ret;
  }

  int firstBitSet(int i, int max) {
    int ffs = COMPILER_FFS(i);
    return ffs ? ffs - 1 : max;
  }

  void resetA() {
    a = 0;
    lastA = aggFactor;
    if (phase >= 2) lastA /= 2 * scale;
  }

  void reset() {
    nelem = getNelem();
    scale = aggFactor / 2;
    phase = scale ? 2 : 1;
    v = 0;
    for (int i = 0; i < asDim; i++) {
      bitCount[i] = asDim - i;
      bitZeroStep[i] = 1;
    }
    as = nextAs();
    resetA();
  }

  int nextAs() {
    for (int d = 0; d < asDim; d++) {
      int p = 1 << d;
      bitCount[d]--;
      if (bitCount[d] == 0) {
        v ^= p;
        bitCount[d] = p;
        if ((v & p) == 0) {
          bitCount[d] += firstBitSet(bitZeroStep[d], asDim) - 1;
          if (bitCount[d] == 0) {
            v ^= p;
            bitCount[d] = p;
          }
          bitZeroStep[d]++;
        }
      }
    }
    return v;
  }

public:
  PatAGAlgorithm(int stepSize, int stepDepth, int maxParallelFactor, size_t offset, size_t end,
                 size_t count, int chunkCount, int rank, int nranks)
    : offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    parallelFactor = maxParallelFactor;
    aggDelta = nrPow2 = (1 << log2Up(nranks));

    aggFactor = 1;
    size_t channelSize = end - offset;
    while (stepSize / (channelSize * sizeof(T) * aggFactor) >= 2 && aggFactor < nranks / 2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    if (postFreq < parallelFactor) parallelFactor = postFreq;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks / 2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }
    asDim = log2Up(aggDelta);
    reset();
  }

  int getParallelFactor() { return parallelFactor; }

  void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0;
    ps->nelem = nelem;
    ps->inpIx = offset;
    int skip = 0;
    if (a >= lastA) {
      skip = 1;
    } else if (phase == 0) {
      int s = a * aggDelta + as;
      if (s >= nranks) skip = 1;
      int recvDataRank = (rank + s) % nranks;
      ps->outIx = recvDataRank * count + offset;
      ps->sendDim = -1;
      ps->recvDim = 0;
      ps->inpIx = 0;
      ps->sendOffset = -1;
      ps->recvOffset = (a % postFreq) * nelem;
      ps->stepOffset = 0;
      ps->postRecv = (a % postFreq == postFreq - 1) || ((a + 1) * aggDelta + as >= nranks) ? 1 : 0;
      ps->postSend = 0;
    } else if (phase == 1) {
      int s = a * aggDelta + as;
      if (s >= nranks) skip = 1;
      ps->sendDim = firstBitSet(s, nrPow2);
      int sMinusDim = s - (1 << ps->sendDim);
      int sendDataRank = (rank + nranks + sMinusDim) % nranks;
      ps->outIx = sendDataRank * count + offset;
      ps->recvDim = sMinusDim ? firstBitSet(sMinusDim, nrPow2) : -1;
      ps->sendOffset = ps->recvOffset = (a % postFreq) * nelem;
      ps->postSend = (a % postFreq == postFreq - 1) || ((a + 1) * aggDelta + as >= nranks) ? 1 : 0;
      ps->postRecv =
        (ps->sendDim == 0) && ((a % postFreq == postFreq - 1) || ((a + 1) * aggDelta + as - 1 >= nranks)) ? 1 : 0;
      ps->stepOffset = (ps->sendDim == 0) ? 0 : a / postFreq;
      if (ps->recvDim == -1) {
        ps->recvOffset = -1;
        ps->postRecv = 0;
      } else if (as - (1 << ps->sendDim) == 0) {
        int foffset = (a * aggDelta) >> (ps->recvDim + 1);
        ps->recvOffset = (foffset % postFreq) * nelem;
        ps->postRecv = (ps->sendDim == 0) && ((foffset % postFreq == postFreq - 1) ||
                                              ((((foffset + 1) * 2) + 1) << ps->recvDim) >= nranks) ? 1 : 0;
        ps->stepOffset = (ps->sendDim == 0) ? 0 : foffset / postFreq;
      }
      if (sMinusDim < nranks && ps->sendDim == 0 && skip) {
        ps->sendDim = -1;
        ps->sendOffset = -1;
        ps->postSend = 0;
        skip = 0;
      }
    } else { // phase >= 2
      int s = (a * aggDelta + as) * 2 * scale;
      if (s >= nranks) skip = 1;
      ps->sendDim = firstBitSet(s, nrPow2);
      int sMinusDim = s - (1 << ps->sendDim);
      int sendDataRank = (rank + nranks + sMinusDim) % nranks;
      ps->outIx = sendDataRank * count + offset;
      ps->recvDim = sMinusDim ? firstBitSet(sMinusDim, nrPow2) : -1;
      ps->sendOffset = ps->recvOffset = (a % postFreq) * nelem;
      ps->postSend = (a % postFreq == postFreq - 1) || ((a + 1) * aggDelta + as >= nranks / (2 * scale)) ? 1 : 0;
      ps->postRecv = 0;
      ps->stepOffset = (ps->sendDim == 0) ? 0 : a / postFreq;
      if (ps->recvDim == -1) {
        ps->recvOffset = -1;
      } else if (as - (1 << (ps->sendDim - log2Up(2 * scale))) == 0) {
        int foffset = (a * aggDelta) >> (ps->recvDim + 1 - log2Up(2 * scale));
        ps->recvOffset = (foffset % postFreq) * nelem;
        ps->stepOffset = (ps->sendDim == 0) ? 0 : foffset / postFreq;
      }
    }

    // Advance state
    a++;
    if (a >= lastA) {
      if (phase == 0) {
        // Done
        ps->last = skip ? 1 : 2;
      } else if (phase == 1) {
        phase = 0;
        resetA();
      } else {
        scale /= 2;
        if (scale == 0) {
          phase = 1;
        }
        resetA();
      }
      if (!skip && phase > 0) {
        as = nextAs();
        resetA();
      }
    }

    // Mark flags
    int flags = PatUsed | (skip ? PatSkipped : 0);
    ps->flags = flags;
  }
};

// === Main: Simulate PAT for various configurations ===
int main(int argc, char** argv) {
  int nranks = argc > 1 ? atoi(argv[1]) : 2;
  int count = argc > 2 ? atoi(argv[2]) : 131072;  // 512KB / sizeof(float)
  int nChannels = argc > 3 ? atoi(argv[3]) : 4;

  int chunkCount = count / nranks;
  int stepSize = 8 * 1024 * 1024;  // 8MB (NCCL_STEPS * stepDepth)
  int stepDepth = 4;  // NCCL_PAT_NWORKERS / WARP_SIZE typical

  printf("=== PAT AllGather Simulator ===\n");
  printf("nranks=%d, count=%d (%.0f KB), nChannels=%d\n",
         nranks, count, count * sizeof(float) / 1024.0, nChannels);
  printf("chunkCount=%d, stepSize=%d, stepDepth=%d\n\n", chunkCount, stepSize, stepDepth);

  for (int r = 0; r < nranks; r++) {
    printf("--- Rank %d ---\n", r);
    size_t channelCount = count / nChannels;
    PatAGAlgorithm<float> algo(stepSize, stepDepth, stepDepth,
                               0, channelCount, count, chunkCount, r, nranks);
    printf("  parallelFactor=%d\n", algo.getParallelFactor());

    int step = 0;
    while (1) {
      ncclPatStep ps = {};
      algo.getNextOp(&ps);
      int skipped = (ps.flags & PatSkipped) ? 1 : 0;
      if (!skipped) {
        printf("  step=%2d: sendDim=%2d recvDim=%2d outIx=%6ld nelem=%d"
               " sendOff=%d recvOff=%d postS=%d postR=%d",
               step, ps.sendDim, ps.recvDim, (long)ps.outIx, ps.nelem,
               ps.sendOffset, ps.recvOffset, ps.postSend, ps.postRecv);
        // Compute actual peer rank from sendDim
        if (ps.sendDim >= 0) {
          int sendPeer = (r + (1 << ps.sendDim)) % nranks;
          printf(" → send_to_rank=%d", sendPeer);
        }
        if (ps.recvDim >= 0) {
          int recvPeer = (r + nranks - (1 << ps.recvDim)) % nranks;
          printf(" ← recv_from_rank=%d", recvPeer);
        }
        printf("\n");
      }
      if (ps.last) break;
      step++;
    }
    printf("\n");
  }
  return 0;
}
