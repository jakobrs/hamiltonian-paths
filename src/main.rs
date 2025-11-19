#![feature(portable_simd)]

use std::{io::Read, simd::prelude::*};

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

const L: usize = 8;

type T = u64;
type ST = i64;
type V = Simd<T, L>;
type M = Mask<ST, L>;

#[inline(always)]
fn test_bit(x: V, u: u32) -> M {
    (x << (T::BITS as T - 1 - u as T))
        .cast::<ST>()
        .is_negative()
}

#[inline(always)]
fn add<const M: T>(x: V, y: V) -> V {
    let z = x + y;
    if M == 0 {
        return z;
    }
    let s = z.simd_ge(V::splat(M));
    s.select(z - V::splat(M), z)
}

fn count_hamiltonian_paths<const N: usize, const M: T>(edges: &[Vec<u32>; N]) -> T {
    if N == 1 {
        return edges.len() as T;
    }
    assert!(N != 0);
    assert!(N <= 31);

    let sum: V = (0..1u32 << N)
        .into_par_iter()
        .step_by(L * 2)
        .map(|mask_idx| {
            let mask = V::from_array({
                let mut mask_data = [mask_idx as T; L];
                for i in 0..L {
                    mask_data[i] += 2 * i as T;
                }
                mask_data
            });

            let mut old = [V::splat(0); N];
            let mut new = [V::splat(0); N];
            old[0] = V::splat(1);

            for _ in 0..N {
                for (u, vs) in edges.iter().enumerate() {
                    if (1 << u) < 2 * L {
                        let mut acc = V::splat(0);

                        for &v in vs {
                            let mut count = acc + old[v as usize];
                            if M != 0 {
                                let reducible = count.simd_ge(V::splat(M));
                                count = reducible.select(count - V::splat(M), count);
                            }
                            let cleared = test_bit(mask, u as u32);
                            acc = cleared.select(acc, count);
                        }

                        new[u as usize] = acc;
                    } else {
                        // can this be merged with the above?
                        let mut acc = V::splat(0);

                        if (mask_idx >> u) & 1 == 0 {
                            for &v in vs {
                                acc = acc + old[v as usize];
                                if M != 0 {
                                    let reducible = acc.simd_ge(V::splat(M));
                                    acc = reducible.select(acc - V::splat(M), acc);
                                }
                            }
                        }

                        new[u as usize] = acc;
                    }
                }

                old = new;
            }

            let c = (mask.count_ones() & V::splat(1)).simd_eq(V::splat(0));
            c.select(old[0], V::splat(M) - old[0])
        })
        .reduce(|| V::splat(0), add::<{ M }>);

    if M == 0 {
        sum.reduce_sum()
    } else if M.checked_mul(L as T).is_some() {
        sum.reduce_sum() % M
    } else {
        sum.to_array()
            .into_iter()
            .reduce(|x, y| (x + y) % M)
            .unwrap_or(0)
    }
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let mut words = input.split_ascii_whitespace();

    const N: usize = 30;

    let n: usize = words.next().unwrap().parse().unwrap();
    assert!(n == N);
    let m: usize = words.next().unwrap().parse().unwrap();

    let mut edges = [const { vec![] }; N];
    for _ in 0..m {
        let u: usize = words.next().unwrap().parse().unwrap();
        let v = words.next().unwrap().parse().unwrap();

        edges[u].push(v);
    }

    let c = count_hamiltonian_paths::<N, 1_000_000_007>(&edges);
    println!("{c} mod 1e9+7");
    let c = count_hamiltonian_paths::<N, 1_000_000_009>(&edges);
    println!("{c} mod 1e9+9");
    let c = count_hamiltonian_paths::<N, 1_000_000_021>(&edges);
    println!("{c} mod 1e9+21");
    let c = count_hamiltonian_paths::<N, 0>(&edges);
    println!("{c} mod 2^32");
}
