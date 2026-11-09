//! Low-latency fork/join thread pool.
//!
//! Token generation issues hundreds of small parallel regions per token (one per matrix
//! multiply), so wake-up latency dominates over throughput. Workers spin briefly on an
//! epoch counter after each job (sub-microsecond hand-off) and fall back to a condition
//! variable when idle, so the pool costs no battery between requests.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

type Trampoline = unsafe fn(*const (), usize);

struct Job {
    data: *const (),
    call: Trampoline,
}

struct Inner {
    job: UnsafeCell<Job>,
    epoch: AtomicU64,
    pending: AtomicUsize,
    panicked: AtomicBool,
    shutdown: AtomicBool,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
    spin: Duration,
}

// SAFETY: `job` is only written by the owner thread while no job is in flight
// (all workers have decremented `pending`) and read by workers after observing a new epoch.
unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

pub struct ThreadPool {
    inner: Arc<Inner>,
    handles: Vec<JoinHandle<()>>,
    n_threads: usize,
    busy: AtomicBool,
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub n_threads: usize,
    /// Logical cores to pin worker threads to (empty: no pinning).
    pub cores: Vec<usize>,
    /// How long idle workers spin before sleeping.
    pub spin: Duration,
}

unsafe fn trampoline<F: Fn(usize) + Sync>(data: *const (), tid: usize) {
    (*(data as *const F))(tid)
}

impl ThreadPool {
    pub fn new(cfg: PoolConfig) -> Self {
        let n = cfg.n_threads.max(1);
        let inner = Arc::new(Inner {
            job: UnsafeCell::new(Job { data: std::ptr::null(), call: trampoline::<fn(usize)> }),
            epoch: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            spin: cfg.spin,
        });
        let mut handles = Vec::with_capacity(n - 1);
        for tid in 1..n {
            let inner = inner.clone();
            let cores = cfg.cores.clone();
            let h = std::thread::Builder::new()
                .name(format!("rie-worker-{tid}"))
                .stack_size(256 * 1024)
                .spawn(move || {
                    if !cores.is_empty() {
                        crate::cpu::pin_current_thread(&cores);
                    }
                    crate::cpu::boost_current_thread();
                    worker_loop(&inner, tid)
                })
                .expect("failed to spawn worker thread");
            handles.push(h);
        }
        ThreadPool { inner, handles, n_threads: n, busy: AtomicBool::new(false) }
    }

    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Run `f(tid)` on every thread of the pool (the caller acts as thread 0) and wait.
    pub fn run<F: Fn(usize) + Sync>(&self, f: F) {
        if self.n_threads == 1 {
            f(0);
            return;
        }
        if self.busy.swap(true, Ordering::Acquire) {
            // Re-entrant or concurrent use: degrade gracefully to serial execution.
            f(0);
            return;
        }
        let inner = &*self.inner;
        unsafe {
            *inner.job.get() = Job { data: &f as *const F as *const (), call: trampoline::<F> };
        }
        inner.pending.store(self.n_threads - 1, Ordering::SeqCst);
        inner.epoch.fetch_add(1, Ordering::SeqCst);
        if inner.sleepers.load(Ordering::SeqCst) > 0 {
            let _g = inner.lock.lock().unwrap();
            inner.cv.notify_all();
        }
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(0)));
        let mut spins = 0u32;
        while inner.pending.load(Ordering::Acquire) != 0 {
            spins += 1;
            if spins < 1 << 16 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        self.busy.store(false, Ordering::Release);
        if let Err(e) = res {
            std::panic::resume_unwind(e);
        }
        if inner.panicked.swap(false, Ordering::SeqCst) {
            panic!("a worker thread panicked");
        }
    }

    /// Parallel loop over `n` items with dynamic (work-stealing style) scheduling:
    /// threads grab chunks of `grain` items from a shared counter, so faster cores
    /// automatically process more of the work.
    pub fn for_each<F: Fn(usize, usize) + Sync>(&self, n: usize, grain: usize, f: F) {
        let grain = grain.max(1);
        if n == 0 {
            return;
        }
        if self.n_threads == 1 || n <= grain {
            for i in 0..n {
                f(i, 0);
            }
            return;
        }
        let next = AtomicUsize::new(0);
        self.run(|tid| loop {
            let start = next.fetch_add(grain, Ordering::Relaxed);
            if start >= n {
                break;
            }
            for i in start..(start + grain).min(n) {
                f(i, tid);
            }
        });
    }
}

fn worker_loop(inner: &Inner, tid: usize) {
    let mut seen = 0u64;
    loop {
        // Spin phase.
        let start = Instant::now();
        let mut iters = 0u32;
        loop {
            if inner.epoch.load(Ordering::Acquire) != seen || inner.shutdown.load(Ordering::Relaxed) {
                break;
            }
            std::hint::spin_loop();
            iters += 1;
            if iters % 256 == 0 && start.elapsed() > inner.spin {
                // Sleep phase.
                let mut g = inner.lock.lock().unwrap();
                inner.sleepers.fetch_add(1, Ordering::SeqCst);
                while inner.epoch.load(Ordering::SeqCst) == seen && !inner.shutdown.load(Ordering::SeqCst) {
                    g = inner.cv.wait(g).unwrap();
                }
                inner.sleepers.fetch_sub(1, Ordering::SeqCst);
                drop(g);
                break;
            }
        }
        if inner.shutdown.load(Ordering::SeqCst) {
            return;
        }
        seen = inner.epoch.load(Ordering::Acquire);
        let (data, call) = unsafe {
            let j = &*inner.job.get();
            (j.data, j.call)
        };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { call(data, tid) }));
        if r.is_err() {
            inner.panicked.store(true, Ordering::SeqCst);
        }
        inner.pending.fetch_sub(1, Ordering::Release);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        {
            let _g = self.inner.lock.lock().unwrap();
            self.inner.cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_on_all_threads_repeatedly() {
        let pool = ThreadPool::new(PoolConfig { n_threads: 4, cores: vec![], spin: Duration::from_micros(50) });
        for round in 0..2000 {
            let hits = AtomicUsize::new(0);
            pool.run(|_| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(hits.load(Ordering::Relaxed), 4, "round {round}");
            if round % 500 == 0 {
                std::thread::sleep(Duration::from_millis(2)); // exercise the sleep path
            }
        }
    }

    #[test]
    fn for_each_covers_every_index_once() {
        let pool = ThreadPool::new(PoolConfig { n_threads: 3, cores: vec![], spin: Duration::from_micros(10) });
        let counts: Vec<AtomicUsize> = (0..1003).map(|_| AtomicUsize::new(0)).collect();
        pool.for_each(counts.len(), 7, |i, _| {
            counts[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }
}
