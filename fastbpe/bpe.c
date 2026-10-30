/* 星辰 · 快速 BPE 合并引擎（纯 C，无依赖，供 ctypes 调用）
 *
 * 语义必须与 xingchen/tokenizer.py 的 ByteBPETokenizer._encode_piece 完全一致：
 *   1. 初始序列 = UTF-8 字节值（0..255）
 *   2. 每轮在相邻对里找 rank 最小的那个 pair，把它的所有不重叠出现（从左到右）合并成
 *      id = 256 + rank
 *   3. 直到没有任何可合并的 pair
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define XC_TBL_BITS 17
#define XC_TBL_SIZE (1u << XC_TBL_BITS)
#define XC_TBL_MASK (XC_TBL_SIZE - 1u)

typedef struct {
    uint32_t key;   /* (a << 14) | b */
    int32_t rank;
    uint8_t used;
} xc_slot;

typedef struct {
    xc_slot *tbl;
} xc_bpe;

static uint32_t xc_hash(uint32_t k) {
    k *= 2654435761u;
    return k >> (32 - XC_TBL_BITS);
}

/* a/b: 每次合并的左右 id；rank: 合并顺序（0 起）；n: 合并条数 */
__declspec(dllexport) xc_bpe *xc_bpe_create(const int32_t *a, const int32_t *b,
                                            const int32_t *rank, int32_t n) {
    xc_bpe *h = (xc_bpe *)calloc(1, sizeof(xc_bpe));
    if (!h) return NULL;
    h->tbl = (xc_slot *)calloc(XC_TBL_SIZE, sizeof(xc_slot));
    if (!h->tbl) { free(h); return NULL; }
    for (int32_t i = 0; i < n; i++) {
        uint32_t key = ((uint32_t)a[i] << 14) | (uint32_t)b[i];
        uint32_t pos = xc_hash(key);
        while (h->tbl[pos].used && h->tbl[pos].key != key) pos = (pos + 1) & XC_TBL_MASK;
        if (!h->tbl[pos].used) {
            h->tbl[pos].used = 1;
            h->tbl[pos].key = key;
            h->tbl[pos].rank = rank[i];
        }
    }
    return h;
}

__declspec(dllexport) void xc_bpe_destroy(xc_bpe *h) {
    if (!h) return;
    free(h->tbl);
    free(h);
}

static inline int32_t xc_lookup(const xc_bpe *h, int32_t a, int32_t b) {
    uint32_t key = ((uint32_t)a << 14) | (uint32_t)b;
    uint32_t pos = xc_hash(key);
    while (h->tbl[pos].used) {
        if (h->tbl[pos].key == key) return h->tbl[pos].rank;
        pos = (pos + 1) & XC_TBL_MASK;
    }
    return -1;
}

/* 返回写入 out 的 id 个数；out 容量需 >= nbytes */
__declspec(dllexport) int32_t xc_bpe_encode(const xc_bpe *h, const unsigned char *bytes,
                                            int32_t nbytes, int32_t *out, int32_t cap) {
    if (nbytes <= 0) return 0;
    if (nbytes > cap) nbytes = cap;
    int32_t n = nbytes;
    for (int32_t i = 0; i < n; i++) out[i] = (int32_t)bytes[i];

    while (n >= 2) {
        int32_t best_rank = 0x7fffffff, best_a = -1, best_b = -1;
        for (int32_t i = 0; i + 1 < n; i++) {
            int32_t r = xc_lookup(h, out[i], out[i + 1]);
            if (r >= 0 && r < best_rank) { best_rank = r; best_a = out[i]; best_b = out[i + 1]; }
        }
        if (best_a < 0) break;
        int32_t new_id = 256 + best_rank;
        int32_t w = 0;
        for (int32_t i = 0; i < n; ) {
            if (i + 1 < n && out[i] == best_a && out[i + 1] == best_b) {
                out[w++] = new_id;
                i += 2;
            } else {
                out[w++] = out[i];
                i += 1;
            }
        }
        n = w;
    }
    return n;
}
