/*
 * Odd-only Collatz descent — **must** match ``_collatz_sieve_parallel_odd`` (services.py).
 * Build: see build_c.sh
 *
 * Modes:
 *   ./sieve_descent verify <first_odd> <odd_count>   -> stderr human summary; exit 1 on mismatch vs embedded self-check
 *   ./sieve_descent bench <first_odd> <odd_count>     -> stdout "wall_s <seconds>\n"
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define COLLATZ_INT64_ODD_STEP_LIMIT ((INT64_MAX - 1) / 3)
#define MAX_KERNEL_STEPS 100000

typedef struct {
    int32_t steps;
    int32_t stopping;
    int64_t max_exc;
} triple_t;

static triple_t odd_sieve_descent_one(int64_t seed) {
    triple_t out;
    if (seed <= 1) {
        out.steps = 0;
        out.stopping = 0;
        out.max_exc = seed;
        return out;
    }
    int64_t current = seed;
    int32_t steps = 0;
    int64_t max_excursion = current;

    while (current >= seed && steps < MAX_KERNEL_STEPS) {
        if ((current & 1) == 0) {
            current >>= 1;
            steps++;
        } else {
            if (current > COLLATZ_INT64_ODD_STEP_LIMIT) {
                out.steps = -1;
                out.stopping = -1;
                out.max_exc = -1;
                return out;
            }
            current = 3 * current + 1;
            steps++;
            if (current > max_excursion) {
                max_excursion = current;
            }
            while ((current & 1) == 0 && current >= seed) {
                current >>= 1;
                steps++;
            }
        }
    }

    if (current >= seed && current != 1 && steps >= MAX_KERNEL_STEPS) {
        out.steps = -1;
        out.stopping = -1;
        out.max_exc = -1;
        return out;
    }

    out.steps = steps;
    out.stopping = steps;
    out.max_exc = max_excursion;
    return out;
}

#ifndef COLLATZ_NO_MAIN
static void range_aggregate(int64_t first_odd, int32_t odd_count,
                            int64_t *best_tst_n, int32_t *best_tst_v,
                            int64_t *best_exc_n, int64_t *best_exc_v) {
    *best_tst_v = -1;
    *best_exc_v = -1;
    *best_tst_n = first_odd;
    *best_exc_n = first_odd;

    for (int32_t i = 0; i < odd_count; i++) {
        int64_t seed = first_odd + 2 * (int64_t)i;
        triple_t t = odd_sieve_descent_one(seed);
        if (t.steps > *best_tst_v) {
            *best_tst_v = t.steps;
            *best_tst_n = seed;
        }
        if (t.max_exc > *best_exc_v) {
            *best_exc_v = t.max_exc;
            *best_exc_n = seed;
        }
    }
}
#endif /* !COLLATZ_NO_MAIN */

/*
 * Shared library export for the lab worker (``cpu_sieve`` backend: auto / native).
 * Build: ``build_native_cpu_sieve_lib.sh`` (tries OpenMP when available, else sequential).
 * Fills per-odd-seed arrays matching Numba ``_collatz_sieve_parallel_odd`` outputs
 * (overflow seeds: steps/stopping/max_exc = -1 until Python patches with bigint).
 */
#ifdef COLLATZ_CPU_SIEVE_NATIVE_EXPORT
#  if defined(_WIN32) || defined(__CYGWIN__)
#    define COLLATZ_ABI __declspec(dllexport)
#  else
#    define COLLATZ_ABI __attribute__((visibility("default")))
#  endif
#  ifdef COLLATZ_CPU_SIEVE_OPENMP
#    include <omp.h>
#  endif

/** 1 = linked with OpenMP for ``collatz_lab_cpu_sieve_odd_fill``; 0 = sequential build. */
COLLATZ_ABI int32_t collatz_lab_cpu_sieve_build_info(void) {
#  ifdef COLLATZ_CPU_SIEVE_OPENMP
    return 1;
#  else
    return 0;
#  endif
}

COLLATZ_ABI void collatz_lab_cpu_sieve_odd_fill(
    int64_t first_odd,
    int32_t odd_count,
    int32_t *out_total_steps,
    int32_t *out_stopping_steps,
    int64_t *out_max_exc)
{
#  ifdef COLLATZ_CPU_SIEVE_OPENMP
    /* dynamic schedule: seeds with long orbits won't starve other threads.
     * Chunk of 4096 amortises scheduler overhead while keeping load balanced. */
#    pragma omp parallel for schedule(dynamic, 4096)
#  endif
    for (int32_t i = 0; i < odd_count; i++) {
        int64_t seed = first_odd + 2 * (int64_t)i;
        triple_t t = odd_sieve_descent_one(seed);
        out_total_steps[i] = t.steps;
        out_stopping_steps[i] = t.stopping;
        out_max_exc[i] = t.max_exc;
    }
}
#endif

#ifndef COLLATZ_NO_MAIN
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s verify|bench <first_odd> <odd_count>\n", argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    int64_t first_odd = (int64_t)strtoll(argv[2], NULL, 10);
    int32_t odd_count = (int32_t)strtol(argv[3], NULL, 10);
    if ((first_odd & 1) == 0 || first_odd < 1 || odd_count < 1) {
        fprintf(stderr, "invalid first_odd or odd_count\n");
        return 2;
    }

    if (strcmp(mode, "bench") == 0) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int64_t bn, en;
        int32_t bv;
        int64_t ev;
        range_aggregate(first_odd, odd_count, &bn, &bv, &en, &ev);
        (void)bn;
        (void)en;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double wall = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("wall_s %.9f\n", wall);
        printf("odd_count %" PRId32 " max_tst_seed %" PRId64 " max_tst %" PRId32 " max_exc %" PRId64 "\n",
               odd_count, bn, bv, ev);
        return 0;
    }

    if (strcmp(mode, "verify") != 0) {
        fprintf(stderr, "unknown mode\n");
        return 2;
    }

    /* Spot-check a few seeds against known small values */
    static const struct {
        int64_t seed;
        int32_t expect_steps;
    } spots[] = {
        {1, 0},
        {3, 6},
        {27, 96},
    };
    for (size_t i = 0; i < sizeof(spots) / sizeof(spots[0]); i++) {
        triple_t t = odd_sieve_descent_one(spots[i].seed);
        if (t.steps != spots[i].expect_steps) {
            fprintf(stderr, "self-check fail seed=%" PRId64 " got steps=%" PRId32 " want %" PRId32 "\n",
                    spots[i].seed, t.steps, spots[i].expect_steps);
            return 1;
        }
    }

    int64_t bn, en;
    int32_t bv;
    int64_t ev;
    range_aggregate(first_odd, odd_count, &bn, &bv, &en, &ev);
    int64_t last_linear = first_odd + 2 * ((int64_t)odd_count - 1);
    fprintf(stderr, "verify ok first_odd=%" PRId64 " odd_count=%" PRId32 " max_tst=(n=%" PRId64 ",v=%" PRId32 ") max_exc=(n=%" PRId64 ",v=%" PRId64 ")\n",
            first_odd, odd_count, bn, bv, en, ev);
    /* Machine-readable line for scripts (stdout): */
    printf(
        "{\"processed\":%" PRId32 ",\"last_processed\":%" PRId64
        ",\"max_total_stopping_time\":{\"n\":%" PRId64 ",\"value\":%" PRId32 "}"
        ",\"max_stopping_time\":{\"n\":%" PRId64 ",\"value\":%" PRId32 "}"
        ",\"max_excursion\":{\"n\":%" PRId64 ",\"value\":%" PRId64 "}}\n",
        odd_count, last_linear, bn, bv, bn, bv, en, ev);
    return 0;
}
#endif /* !COLLATZ_NO_MAIN */
