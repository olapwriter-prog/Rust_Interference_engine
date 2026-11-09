//! SoC topology discovery and thread placement for heterogeneous (big.LITTLE / DynamIQ)
//! mobile CPUs.
//!
//! Modern phone SoCs mix 2–4 kinds of cores, e.g.
//!   * Snapdragon 8 Elite: 2× Oryon Prime @4.32 GHz + 6× Oryon Perf @3.53 GHz
//!   * Snapdragon 8 Gen 3:  1× X4 + 5× A720 + 2× A520
//!   * Dimensity 9400:      1× X925 + 3× X4 + 4× A720
//!   * Tensor G4:           1× X4 + 3× A720 + 4× A520
//!   * Exynos 2400:         1× X4 + 5× A720 + 4× A520
//!   * Apple A17 Pro/A18:   2× P + 4× E
//!
//! Running matrix kernels on efficiency cores (A5xx) makes every barrier wait for the
//! slowest thread, so by default we use only the performance cores, pin worker threads to
//! them, and hand out work dynamically so faster cores take more chunks.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreKind {
    Prime,
    Performance,
    Efficiency,
}

#[derive(Debug, Clone)]
pub struct Cluster {
    pub kind: CoreKind,
    pub cores: Vec<usize>,
    pub max_freq_khz: u64,
    pub capacity: Option<u32>,
    pub uarch: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct CpuTopology {
    pub soc: Option<String>,
    pub n_logical: usize,
    pub clusters: Vec<Cluster>,
}

impl fmt::Display for CpuTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(s) = &self.soc {
            writeln!(f, "SoC: {s}")?;
        }
        writeln!(f, "logical CPUs: {}", self.n_logical)?;
        for c in &self.clusters {
            write!(f, "  {:?}: {} core(s) {:?}", c.kind, c.cores.len(), c.cores)?;
            if c.max_freq_khz > 0 {
                write!(f, " @ {:.2} GHz", c.max_freq_khz as f64 / 1e6)?;
            }
            if let Some(u) = c.uarch {
                write!(f, " [{u}]")?;
            }
            if let Some(cap) = c.capacity {
                write!(f, " capacity={cap}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl CpuTopology {
    pub fn detect() -> Self {
        let n_logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if let Some(t) = linux::detect(n_logical) {
            return t;
        }
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        if let Some(t) = apple::detect(n_logical) {
            return t;
        }
        CpuTopology {
            soc: None,
            n_logical,
            clusters: vec![Cluster {
                kind: CoreKind::Performance,
                cores: (0..n_logical).collect(),
                max_freq_khz: 0,
                capacity: None,
                uarch: None,
            }],
        }
    }

    /// Logical core ids of all non-efficiency cores, fastest first.
    pub fn performance_cores(&self) -> Vec<usize> {
        let v: Vec<usize> = self
            .clusters
            .iter()
            .filter(|c| c.kind != CoreKind::Efficiency)
            .flat_map(|c| c.cores.iter().copied())
            .collect();
        if v.is_empty() {
            (0..self.n_logical).collect()
        } else {
            v
        }
    }

    /// Default thread count: the performance cores, capped at 8 — beyond that LPDDR5X
    /// bandwidth is saturated during token generation and extra threads only add heat.
    pub fn recommended_threads(&self) -> usize {
        self.performance_cores().len().clamp(1, 8)
    }

    /// Whether explicit core pinning is possible on this OS.
    pub fn can_pin(&self) -> bool {
        cfg!(any(target_os = "linux", target_os = "android")) && self.clusters.iter().any(|c| c.max_freq_khz > 0)
    }

    /// Build clusters from per-core (id, max_freq_khz, capacity, uarch) tuples.
    pub fn from_cores(soc: Option<String>, n_logical: usize, cores: &[(usize, u64, Option<u32>, Option<&'static str>)]) -> Self {
        let mut clusters: Vec<Cluster> = Vec::new();
        for &(id, freq, cap, uarch) in cores {
            if let Some(c) = clusters.iter_mut().find(|c| c.max_freq_khz == freq && c.capacity == cap && c.uarch == uarch) {
                c.cores.push(id);
            } else {
                clusters.push(Cluster { kind: CoreKind::Performance, cores: vec![id], max_freq_khz: freq, capacity: cap, uarch });
            }
        }
        clusters.sort_by(|a, b| {
            (b.capacity.unwrap_or(0), b.max_freq_khz).cmp(&(a.capacity.unwrap_or(0), a.max_freq_khz))
        });
        let top_freq = clusters.first().map(|c| c.max_freq_khz).unwrap_or(0);
        let top_cap = clusters.iter().filter_map(|c| c.capacity).max();
        let n_clusters = clusters.len();
        for (i, c) in clusters.iter_mut().enumerate() {
            let little_uarch = matches!(c.uarch, Some(u) if u.contains("A53") || u.contains("A55") || u.contains("A510") || u.contains("A520") || u.contains("A35"));
            let low_cap = match (c.capacity, top_cap) {
                (Some(cap), Some(top)) => (cap as f32) < 0.45 * top as f32,
                _ => false,
            };
            let low_freq = top_freq > 0 && (c.max_freq_khz as f32) < 0.62 * top_freq as f32;
            if n_clusters > 1 && (little_uarch || low_cap || (low_freq && i == n_clusters - 1)) {
                c.kind = CoreKind::Efficiency;
            } else if i == 0 && n_clusters >= 3 && c.cores.len() <= 2 {
                c.kind = CoreKind::Prime;
            }
        }
        CpuTopology { soc, n_logical, clusters }
    }
}

/// Decode an Arm MIDR_EL1 value into a micro-architecture name.
pub fn uarch_from_midr(midr: u64) -> Option<&'static str> {
    let implementer = (midr >> 24) & 0xff;
    let part = (midr >> 4) & 0xfff;
    Some(match (implementer, part) {
        (0x41, 0xd03) => "Cortex-A53",
        (0x41, 0xd04) => "Cortex-A35",
        (0x41, 0xd05) => "Cortex-A55",
        (0x41, 0xd07) => "Cortex-A57",
        (0x41, 0xd08) => "Cortex-A72",
        (0x41, 0xd09) => "Cortex-A73",
        (0x41, 0xd0a) => "Cortex-A75",
        (0x41, 0xd0b) => "Cortex-A76",
        (0x41, 0xd0d) => "Cortex-A77",
        (0x41, 0xd41) => "Cortex-A78",
        (0x41, 0xd44) => "Cortex-X1",
        (0x41, 0xd46) => "Cortex-A510",
        (0x41, 0xd47) => "Cortex-A710",
        (0x41, 0xd48) => "Cortex-X2",
        (0x41, 0xd4b) => "Cortex-A78C",
        (0x41, 0xd4d) => "Cortex-A715",
        (0x41, 0xd4e) => "Cortex-X3",
        (0x41, 0xd80) => "Cortex-A520",
        (0x41, 0xd81) => "Cortex-A720",
        (0x41, 0xd82) => "Cortex-X4",
        (0x41, 0xd85) => "Cortex-X925",
        (0x41, 0xd87) => "Cortex-A725",
        (0x41, 0xd8a) => "C1-Nano",
        (0x41, 0xd8b) => "C1-Pro",
        (0x41, 0xd8c) => "C1-Ultra",
        (0x41, 0xd8e) => "C1-Premium",
        (0x51, 0x001) => "Qualcomm Oryon",
        (0x51, 0x002) => "Qualcomm Oryon",
        (0x51, 0x802) | (0x51, 0x804) => "Kryo Gold (A7x)",
        (0x51, 0x803) | (0x51, 0x805) => "Kryo Silver (A55)",
        (0x53, _) => "Samsung Exynos-M",
        (0x61, _) => "Apple",
        _ => return None,
    })
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux {
    use super::*;
    use std::fs;

    fn read_u64(path: &str) -> Option<u64> {
        let s = fs::read_to_string(path).ok()?;
        let s = s.trim();
        if let Some(hex) = s.strip_prefix("0x") {
            u64::from_str_radix(hex, 16).ok()
        } else {
            s.parse().ok()
        }
    }

    fn parse_range(s: &str) -> Vec<usize> {
        let mut v = Vec::new();
        for part in s.trim().split(',') {
            if let Some((a, b)) = part.split_once('-') {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    v.extend(a..=b);
                }
            } else if let Ok(a) = part.parse() {
                v.push(a);
            }
        }
        v
    }

    fn soc_name() -> Option<String> {
        #[cfg(target_os = "android")]
        {
            for key in ["ro.soc.model", "ro.hardware.chipname", "ro.board.platform"] {
                if let Some(v) = android_prop(key) {
                    if !v.is_empty() {
                        let manuf = android_prop("ro.soc.manufacturer").unwrap_or_default();
                        return Some(if manuf.is_empty() { v } else { format!("{manuf} {v}") });
                    }
                }
            }
        }
        let info = fs::read_to_string("/proc/cpuinfo").ok()?;
        info.lines()
            .find(|l| l.starts_with("Hardware") || l.starts_with("model name"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
    }

    #[cfg(target_os = "android")]
    fn android_prop(key: &str) -> Option<String> {
        let k = std::ffi::CString::new(key).ok()?;
        let mut buf = [0u8; 92]; // PROP_VALUE_MAX
        let n = unsafe { libc::__system_property_get(k.as_ptr(), buf.as_mut_ptr() as *mut libc::c_char) };
        if n <= 0 {
            return None;
        }
        Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned())
    }

    pub fn detect(n_logical: usize) -> Option<CpuTopology> {
        let possible = fs::read_to_string("/sys/devices/system/cpu/possible").ok()?;
        let ids = parse_range(&possible);
        if ids.is_empty() {
            return None;
        }
        let mut cores = Vec::new();
        for id in ids {
            let base = format!("/sys/devices/system/cpu/cpu{id}");
            let freq = read_u64(&format!("{base}/cpufreq/cpuinfo_max_freq")).unwrap_or(0);
            let cap = read_u64(&format!("{base}/cpu_capacity")).map(|c| c as u32);
            let uarch = read_u64(&format!("{base}/regs/identification/midr_el1")).and_then(uarch_from_midr);
            cores.push((id, freq, cap, uarch));
        }
        Some(CpuTopology::from_cores(soc_name(), n_logical, &cores))
    }
}

#[cfg(any(target_os = "ios", target_os = "macos"))]
mod apple {
    use super::*;

    fn sysctl_u32(name: &str) -> Option<u32> {
        let c = std::ffi::CString::new(name).ok()?;
        let mut v: u32 = 0;
        let mut len = std::mem::size_of::<u32>();
        let r = unsafe { libc::sysctlbyname(c.as_ptr(), &mut v as *mut u32 as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0) };
        (r == 0).then_some(v)
    }

    fn sysctl_str(name: &str) -> Option<String> {
        let c = std::ffi::CString::new(name).ok()?;
        let mut buf = [0u8; 128];
        let mut len = buf.len();
        let r = unsafe { libc::sysctlbyname(c.as_ptr(), buf.as_mut_ptr() as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0) };
        if r != 0 {
            return None;
        }
        let s = String::from_utf8_lossy(&buf[..len]).trim_end_matches('\0').to_string();
        Some(s)
    }

    pub fn detect(n_logical: usize) -> Option<CpuTopology> {
        let p = sysctl_u32("hw.perflevel0.physicalcpu")? as usize;
        let e = sysctl_u32("hw.perflevel1.physicalcpu").unwrap_or(0) as usize;
        let soc = sysctl_str("machdep.cpu.brand_string").or_else(|| sysctl_str("hw.machine"));
        let mut clusters = vec![Cluster {
            kind: CoreKind::Performance,
            cores: (0..p).collect(),
            max_freq_khz: 0,
            capacity: None,
            uarch: Some("Apple P-core"),
        }];
        if e > 0 {
            clusters.push(Cluster {
                kind: CoreKind::Efficiency,
                cores: (p..p + e).collect(),
                max_freq_khz: 0,
                capacity: None,
                uarch: Some("Apple E-core"),
            });
        }
        Some(CpuTopology { soc, n_logical, clusters })
    }
}

/// Restrict the calling thread to the given logical cores (Linux/Android only).
/// Returns `false` when unsupported or refused by the OS.
pub fn pin_current_thread(cores: &[usize]) -> bool {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    unsafe {
        if cores.is_empty() {
            return false;
        }
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cores {
            if c < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(c, &mut set);
            }
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = cores;
        false
    }
}

/// Ask the scheduler to treat the calling thread as latency-critical
/// (QoS USER_INTERACTIVE on Apple platforms, which keeps it on P-cores).
pub fn boost_current_thread() {
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_snapdragon_8_gen3() {
        let a720 = Some("Cortex-A720");
        let cores = [
            (0, 2265600, Some(325), Some("Cortex-A520")),
            (1, 2265600, Some(325), Some("Cortex-A520")),
            (2, 3148800, Some(870), a720),
            (3, 3148800, Some(870), a720),
            (4, 3148800, Some(870), a720),
            (5, 2956800, Some(830), a720),
            (6, 2956800, Some(830), a720),
            (7, 3302400, Some(1024), Some("Cortex-X4")),
        ];
        let t = CpuTopology::from_cores(None, 8, &cores);
        assert_eq!(t.performance_cores(), vec![7, 2, 3, 4, 5, 6]);
        assert_eq!(t.clusters[0].kind, CoreKind::Prime);
        assert_eq!(t.recommended_threads(), 6);
    }

    #[test]
    fn classify_snapdragon_8_elite_all_big() {
        let cores: Vec<_> = (0..8)
            .map(|i| if i < 6 { (i, 3532800, None, Some("Qualcomm Oryon")) } else { (i, 4320000, None, Some("Qualcomm Oryon")) })
            .collect();
        let t = CpuTopology::from_cores(None, 8, &cores);
        assert_eq!(t.performance_cores().len(), 8);
        assert_eq!(t.performance_cores()[0], 6);
    }

    #[test]
    fn classify_tensor_g4() {
        let mut cores = Vec::new();
        for i in 0..4 {
            cores.push((i, 1950000, None, Some("Cortex-A520")));
        }
        for i in 4..7 {
            cores.push((i, 2600000, None, Some("Cortex-A720")));
        }
        cores.push((7, 3100000, None, Some("Cortex-X4")));
        let t = CpuTopology::from_cores(None, 8, &cores);
        assert_eq!(t.performance_cores(), vec![7, 4, 5, 6]);
    }

    #[test]
    fn midr_decode() {
        assert_eq!(uarch_from_midr(0x410fd820), Some("Cortex-X4"));
        assert_eq!(uarch_from_midr(0x511f0011), Some("Qualcomm Oryon"));
    }
}
