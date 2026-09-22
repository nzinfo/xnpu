//! Device discovery and GET_INFO queries.

use std::fs;
use std::io;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};

use crate::ioctl::{amdxdna_ioctl, raw_ioctl, AmdxdnaCmd};

/// enum amdxdna_drm_get_param (Linux 7.0 UAPI).
const QUERY_AIE_METADATA: u32 = 1;
const QUERY_CLOCK_METADATA: u32 = 3;
const QUERY_FIRMWARE_VERSION: u32 = 8;
const GET_POWER_MODE: u32 = 9;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct TileMetadataRaw {
    row_count: u16,
    row_start: u16,
    dma_channel_count: u16,
    lock_count: u16,
    event_reg_count: u16,
    pad: [u16; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct AieMetadataRaw {
    col_size: u32,
    cols: u16,
    rows: u16,
    version_major: u32,
    version_minor: u32,
    core: TileMetadataRaw,
    mem: TileMetadataRaw,
    shim: TileMetadataRaw,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ClockRaw {
    name: [u8; 16],
    freq_mhz: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ClockMetadataRaw {
    mp_npu_clock: ClockRaw,
    h_clock: ClockRaw,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct FirmwareVersionRaw {
    major: u32,
    minor: u32,
    patch: u32,
    build: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct PowerModeRaw {
    power_mode: u8,
    pad: [u8; 7],
}

/// Per-tile-class AIE grid metadata (core / mem / shim rows).
#[derive(Debug, Clone, Copy)]
pub struct TileMetadata {
    pub row_count: u16,
    pub row_start: u16,
    pub dma_channel_count: u16,
    pub lock_count: u16,
    pub event_reg_count: u16,
}

/// AIE array geometry and versions, from `DRM_AMDXDNA_QUERY_AIE_METADATA`.
#[derive(Debug, Clone, Copy)]
pub struct AieMetadata {
    pub col_size: u32,
    pub cols: u16,
    pub rows: u16,
    pub version: (u32, u32),
    pub core: TileMetadata,
    pub mem: TileMetadata,
    pub shim: TileMetadata,
}

impl From<TileMetadataRaw> for TileMetadata {
    fn from(r: TileMetadataRaw) -> Self {
        TileMetadata {
            row_count: r.row_count,
            row_start: r.row_start,
            dma_channel_count: r.dma_channel_count,
            lock_count: r.lock_count,
            event_reg_count: r.event_reg_count,
        }
    }
}

/// An open amdxdna accel device.
pub struct Device {
    fd: fs::File,
    pub path: PathBuf,
    /// The client's device-memory heap plus its userspace mapping. A hwctx
    /// requires both: the BO ("dev heap object not exist" -> ENOENT) and an
    /// mmap of it ("Invalid dev heap userptr" -> EINVAL, because the kernel
    /// only records the heap's uva on the mmap path). The shim allocates and
    /// maps 64MB up front; the mapping stays alive for the Device lifetime.
    dev_heap: Option<(crate::bo::BufferObject, crate::bo::Mapping)>,
}

/// Device memory heap must sit inside one 64MB page; max size is 64MB.
const DEV_HEAP_SIZE: usize = 64 << 20;

impl Device {
    /// Find the first /dev/accel/accel* node that answers the amdxdna
    /// AIE-metadata query (other accel devices, e.g. GPUs exposing accel
    /// nodes, will fail it).
    pub fn open_default() -> io::Result<Device> {
        let mut candidates: Vec<PathBuf> = fs::read_dir("/dev/accel")?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        candidates.sort();
        let mut last_err = io::Error::other("no /dev/accel candidates");
        for path in candidates {
            if !path.to_string_lossy().contains("accel") {
                continue;
            }
            match Device::open(&path) {
                Ok(dev) => {
                    if dev.aie_metadata().is_ok() {
                        return Ok(dev);
                    }
                }
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }

    pub fn open(path: &Path) -> io::Result<Device> {
        let fd = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        let mut dev = Device {
            fd,
            path: path.to_path_buf(),
            dev_heap: None,
        };
        let heap =
            crate::bo::BufferObject::new(&dev, crate::bo::BoType::DevHeap, DEV_HEAP_SIZE)?;
        // The heap's user VA must be 64MB aligned: this kernel runs the fw in
        // SVM/PASID mode and MAP_HOST_BUFFER hands the firmware the heap's
        // uva, which rejects unaligned addresses (fw status 0x4000003).
        let heap_map = heap.map_owned_aligned(DEV_HEAP_SIZE)?;
        dev.dev_heap = Some((heap, heap_map));
        Ok(dev)
    }

    pub(crate) fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    fn get_info(&self, param: u32, buf: &mut [u8]) -> io::Result<()> {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct GetInfoArg {
            param: u32,
            buffer_size: u32,
            buffer: u64,
        }
        let mut arg = GetInfoArg {
            param,
            buffer_size: buf.len() as u32,
            buffer: buf.as_mut_ptr() as u64,
        };
        let cmd = amdxdna_ioctl(AmdxdnaCmd::GetInfo, std::mem::size_of::<GetInfoArg>());
        // SAFETY: arg is a plain repr(C) struct of the exact UAPI size; the
        // kernel reads/writes it (and the pointed-to buffer) in place.
        unsafe { raw_ioctl(self.raw_fd(), cmd, &mut arg as *mut GetInfoArg as *mut u8) }
    }

    pub fn aie_metadata(&self) -> io::Result<AieMetadata> {
        let mut raw = AieMetadataRaw::default();
        // The driver updates buffer_size to the size it wrote; passing an
        // exact-sized struct for the query is the standard usage.
        let mut bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut raw as *mut AieMetadataRaw as *mut u8,
                std::mem::size_of::<AieMetadataRaw>(),
            )
        };
        self.get_info(QUERY_AIE_METADATA, bytes)?;
        Ok(AieMetadata {
            col_size: raw.col_size,
            cols: raw.cols,
            rows: raw.rows,
            version: (raw.version_major, raw.version_minor),
            core: raw.core.into(),
            mem: raw.mem.into(),
            shim: raw.shim.into(),
        })
    }

    pub fn clock_metadata(&self) -> io::Result<((String, u32), (String, u32))> {
        let mut raw = ClockMetadataRaw::default();
        let mut bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut raw as *mut ClockMetadataRaw as *mut u8,
                std::mem::size_of::<ClockMetadataRaw>(),
            )
        };
        self.get_info(QUERY_CLOCK_METADATA, bytes)?;
        let name = |c: &ClockRaw| -> String {
            let end = c.name.iter().position(|&b| b == 0).unwrap_or(c.name.len());
            String::from_utf8_lossy(&c.name[..end]).into_owned()
        };
        Ok((
            (name(&raw.mp_npu_clock), raw.mp_npu_clock.freq_mhz),
            (name(&raw.h_clock), raw.h_clock.freq_mhz),
        ))
    }

    pub fn firmware_version(&self) -> io::Result<(u32, u32, u32, u32)> {
        let mut raw = FirmwareVersionRaw::default();
        let mut bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut raw as *mut FirmwareVersionRaw as *mut u8,
                std::mem::size_of::<FirmwareVersionRaw>(),
            )
        };
        self.get_info(QUERY_FIRMWARE_VERSION, bytes)?;
        Ok((raw.major, raw.minor, raw.patch, raw.build))
    }

    pub fn power_mode(&self) -> io::Result<u8> {
        let mut raw = PowerModeRaw::default();
        let mut bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut raw as *mut PowerModeRaw as *mut u8,
                std::mem::size_of::<PowerModeRaw>(),
            )
        };
        self.get_info(GET_POWER_MODE, bytes)?;
        Ok(raw.power_mode)
    }
}
