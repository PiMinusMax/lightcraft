//! How much device memory is free right now, where that can be asked without `unsafe`: the NVIDIA
//! driver's `nvidia-smi` (Linux and Windows) and the amdgpu driver's sysfs counters (Linux). Other
//! devices (AMD/Intel on Windows needs DXGI; integrated GPUs share system memory) report `None`, and the
//! caller keeps its fixed budget. The answer is cached for a few seconds: renders ask often, and the
//! query starts a process on NVIDIA. Only the first question for an adapter waits for the query; later
//! ones get the last answer while a background thread asks again, so no render (and no lock it holds)
//! waits for `nvidia-smi`.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

/// How long an answer is fresh; an older one is still returned while a newer one is asked for.
const TTL: Duration = Duration::from_secs(3);
/// How long `nvidia-smi` may take before it is given up on.
const QUERY_TIMEOUT: Duration = Duration::from_secs(2);

const NVIDIA: u32 = 0x10de;
const AMD: u32 = 0x1002;

static CACHE: Cache = Cache::new(TTL);

/// Free device memory of the adapter `info` describes, in bytes, or `None` when it cannot be asked.
pub fn free_bytes(info: &wgpu::AdapterInfo) -> Option<u64> {
    CACHE.get((info.vendor, info.device), query)
}

/// The last answer for one adapter (vendor, device): when it was asked, the answer, and whether a
/// newer one is being asked for.
struct Answer {
    adapter: (u32, u32),
    at: Instant,
    free: Option<u64>,
    asking: bool,
}

struct Cache {
    ttl: Duration,
    last: Mutex<Option<Answer>>,
}

impl Cache {
    const fn new(ttl: Duration) -> Self {
        Cache { ttl, last: Mutex::new(None) }
    }

    /// The answer of `ask` for `adapter`: cached, refreshed in the background once it is older than the
    /// TTL (in the caller where no thread can be started), and asked for directly the first time.
    fn get(&'static self, adapter: (u32, u32), ask: fn(u32, u32) -> Option<u64>) -> Option<u64> {
        {
            let mut last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(a) = last.as_mut()
                && a.adapter == adapter
            {
                if a.at.elapsed() >= self.ttl && !a.asking {
                    let refresh = std::thread::Builder::new().name("vram-query".into()).spawn(move || self.store(adapter, ask(adapter.0, adapter.1)));
                    if refresh.is_err() {
                        drop(last);
                        return self.ask_now(adapter, ask);
                    }
                    a.asking = true;
                }
                return a.free;
            }
        }
        self.ask_now(adapter, ask)
    }

    fn ask_now(&self, adapter: (u32, u32), ask: fn(u32, u32) -> Option<u64>) -> Option<u64> {
        let free = ask(adapter.0, adapter.1);
        self.store(adapter, free);
        free
    }

    fn store(&self, adapter: (u32, u32), free: Option<u64>) {
        *self.last.lock().unwrap_or_else(PoisonError::into_inner) = Some(Answer { adapter, at: Instant::now(), free, asking: false });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn query(vendor: u32, device: u32) -> Option<u64> {
    match vendor {
        NVIDIA => nvidia(device),
        AMD => amdgpu(vendor, device),
        _ => None,
    }
}

/// `nvidia-smi --query-gpu=pci.device_id,memory.free`: one line per GPU, `0xDDDDVVVV, <MiB>`.
#[cfg(not(target_arch = "wasm32"))]
fn nvidia(device: u32) -> Option<u64> {
    let out = run_with_timeout("nvidia-smi", &["--query-gpu=pci.device_id,memory.free", "--format=csv,noheader,nounits"])?;
    parse_nvidia(&out, device)
}

/// The free MiB of the GPU whose PCI id names `device`, or of the only GPU listed; in bytes.
fn parse_nvidia(out: &str, device: u32) -> Option<u64> {
    let gpus: Vec<(u32, u64)> = out
        .lines()
        .filter_map(|l| {
            let (id, free) = l.split_once(',')?;
            let id = u32::from_str_radix(id.trim().trim_start_matches("0x").trim_start_matches("0X"), 16).ok()?;
            let free: u64 = free.trim().parse().ok()?;
            Some((id >> 16, free))
        })
        .collect();
    let free = match gpus.iter().find(|g| g.0 == device) {
        Some(g) => g.1,
        None if gpus.len() == 1 => gpus.first()?.1,
        None => return None,
    };
    free.checked_mul(1 << 20)
}

/// `mem_info_vram_total - mem_info_vram_used` of the amdgpu card with this vendor and device id.
#[cfg(not(target_arch = "wasm32"))]
fn amdgpu(vendor: u32, device: u32) -> Option<u64> {
    let read_hex = |p: &std::path::Path| -> Option<u32> {
        let s = std::fs::read_to_string(p).ok()?;
        u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
    };
    let read_u64 = |p: &std::path::Path| -> Option<u64> { std::fs::read_to_string(p).ok()?.trim().parse().ok() };
    for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let dev = entry.path().join("device");
        if read_hex(&dev.join("vendor")) != Some(vendor) || read_hex(&dev.join("device")) != Some(device) {
            continue;
        }
        let (Some(total), Some(used)) = (read_u64(&dev.join("mem_info_vram_total")), read_u64(&dev.join("mem_info_vram_used"))) else { continue };
        return Some(total.saturating_sub(used));
    }
    None
}

/// Run a program and return its standard output, or `None` if it is missing, fails or takes too long.
#[cfg(not(target_arch = "wasm32"))]
fn run_with_timeout(program: &str, args: &[&str]) -> Option<String> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    let mut cmd = Command::new(program);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::null()).stdin(Stdio::null());
    // the release app has no console: without this, Windows opens one for each query
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break,
            Ok(Some(_)) | Err(_) => return None,
            Ok(None) if start.elapsed() > QUERY_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
        }
    }
    let mut out = String::new();
    child.stdout.take()?.read_to_string(&mut out).ok()?;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stale_answer_is_returned_while_a_new_one_is_asked_for() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CACHE: Cache = Cache::new(Duration::from_millis(50));
        static ASKED: AtomicU64 = AtomicU64::new(0);
        fn slow(_: u32, _: u32) -> Option<u64> {
            std::thread::sleep(Duration::from_millis(300));
            Some(ASKED.fetch_add(1, Ordering::SeqCst) + 1)
        }
        // the first question waits for the answer
        assert_eq!(CACHE.get((NVIDIA, 1), slow), Some(1));
        std::thread::sleep(Duration::from_millis(80));
        // stale: the old answer at once, a new one in the background
        let t = Instant::now();
        assert_eq!(CACHE.get((NVIDIA, 1), slow), Some(1));
        assert_eq!(CACHE.get((NVIDIA, 1), slow), Some(1));
        assert!(t.elapsed() < Duration::from_millis(200), "waited {:?} for the query", t.elapsed());
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(CACHE.get((NVIDIA, 1), slow), Some(2), "the background answer is kept");
        assert_eq!(ASKED.load(Ordering::SeqCst), 2, "one query in the background, not one per question");
        // another adapter is asked for directly
        assert_eq!(CACHE.get((AMD, 2), slow), Some(3));
    }

    #[test]
    fn nvidia_output_is_matched_by_device_id() {
        let two = "0x179B10DE, 3990\n0x1F0810DE, 5120\n";
        assert_eq!(parse_nvidia(two, 0x1f08), Some(5120 << 20));
        assert_eq!(parse_nvidia(two, 0x179b), Some(3990 << 20));
        // several GPUs and none matches: unknown
        assert_eq!(parse_nvidia(two, 0x2204), None);
        // a single GPU is taken even if its id is reported differently
        assert_eq!(parse_nvidia("0x179B10DE, 3990\n", 0x1234), Some(3990 << 20));
        // garbage is not a number
        assert_eq!(parse_nvidia("No devices were found\n", 0x179b), None);
        assert_eq!(parse_nvidia("", 0x179b), None);
    }
}
