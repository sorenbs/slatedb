//! Isolated A/B/C for the yield points in PR #1964, on the shape that
//! motivates the compactor merge-loop yield: dedup-heavy compaction
//! (many entries consumed per emitted block).
//!
//! Arms (env): YIELD_SST=0|1, YIELD_MERGE=0|1
//! Runtime: ONE worker, so any long poll shows directly as sentinel
//! overshoot. Store: in-memory, so merge-loop next()/add() are always
//! ready (the pathological ready-to-ready run).
//!
//! Workload: KEYS keys overwritten ROUNDS times (32 B values), one L0
//! flush per round -> size-tiered compactions repeatedly merge R
//! versions down to 1: entries consumed per emitted block ~ R * (block
//! entries), so finish_block yields are rare relative to consumed work.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const KEYS: usize = 2_000;
const ROUNDS: usize = 150;

fn pct(v: &mut Vec<u64>, p: f64) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() as f64 - 1.0) * p) as usize]
}

#[tokio::main(flavor = "multi_thread", worker_threads = 1)]
async fn main() {
    let sst = std::env::var("YIELD_SST").unwrap_or_else(|_| "1".into());
    let merge = std::env::var("YIELD_MERGE").unwrap_or_else(|_| "1".into());
    let arm = format!("sst={sst} merge={merge}");

    // Sentinel: 1 ms sleeps; overshoot = scheduling delay on this worker.
    let overshoots: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let o2 = overshoots.clone();
    tokio::spawn(async move {
        loop {
            let t0 = Instant::now();
            tokio::time::sleep(Duration::from_millis(1)).await;
            let over = t0.elapsed().as_micros().saturating_sub(1_000) as u64;
            o2.lock().unwrap().push(over);
        }
    });

    let store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::memory::InMemory::new());
    let settings = slatedb::config::Settings {
        flush_interval: Some(Duration::from_millis(50)),
        l0_sst_size_bytes: 128 * 1024,
        max_unflushed_bytes: 4 * 1024 * 1024,
        ..Default::default()
    };
    let db = slatedb::Db::builder("exp", store)
        .with_settings(settings)
        .build()
        .await
        .expect("open");

    let t0 = Instant::now();
    let val = vec![7u8; 32];
    let popts = slatedb::config::PutOptions::default();
    let wopts = slatedb::config::WriteOptions {
        await_durable: false,
        ..Default::default()
    };
    for r in 0..ROUNDS {
        for k in 0..KEYS {
            let key = format!("k{:06}", k);
            db.put_with_options(key.as_bytes(), val.as_slice(), &popts, &wopts)
                .await
                .expect("put");
        }
        db.flush().await.expect("flush");
        if r % 20 == 0 {
            eprintln!("[{arm}] round {r}/{ROUNDS} t={:?}", t0.elapsed());
        }
    }
    let write_done = t0.elapsed();
    // Let in-flight compactions finish while the sentinel keeps sampling.
    tokio::time::sleep(Duration::from_secs(10)).await;
    let total = t0.elapsed();

    let mut o = overshoots.lock().unwrap().clone();
    let n = o.len();
    let p50 = pct(&mut o, 0.50);
    let p99 = pct(&mut o, 0.99);
    let p999 = pct(&mut o, 0.999);
    let max = *o.iter().max().unwrap_or(&0);
    println!(
        "ARM[{arm}] entries={} write={:.1}s total={:.1}s sentinel(n={n}): p50={}us p99={}us p99.9={}us max={}us",
        KEYS * ROUNDS,
        write_done.as_secs_f64(),
        total.as_secs_f64(),
        p50, p99, p999, max
    );
}
