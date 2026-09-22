//! Raw DRM ioctl plumbing.
//!
//! DRM ioctl numbers are `_IOWR('d', DRM_COMMAND_BASE + id, struct)`, i.e.
//! `(2 << 30) | (size << 16) | (0x64 << 8) | (0x40 + id)` for the driver
//! commands used here (all defined `_IOWR` in the amdxdna UAPI).

use std::io;
use std::os::unix::io::RawFd;

const DRM_IOC_WRITE: u32 = 1 << 30;
const DRM_IOC_READ: u32 = 2 << 30;
const DRM_IOCTL_TYPE: u32 = b'd' as u32;
/// DRM_COMMAND_BASE: driver-private ioctl numbers start here.
const DRM_COMMAND_BASE: u32 = 0x40;

/// amdxdna ioctl ids (enum amdxdna_drm_ioctl_id, Linux 7.0 UAPI).
#[repr(u32)]
pub enum AmdxdnaCmd {
    CreateHwctx = 0,
    DestroyHwctx = 1,
    ConfigHwctx = 2,
    CreateBo = 3,
    GetBoInfo = 4,
    SyncBo = 5,
    ExecCmd = 6,
    GetInfo = 7,
    SetState = 8,
}

pub(crate) fn amdxdna_ioctl(cmd: AmdxdnaCmd, size: usize) -> u64 {
    let nr = DRM_COMMAND_BASE + cmd as u32;
    (DRM_IOC_READ | DRM_IOC_WRITE) as u64 | ((size as u64) << 16) | ((DRM_IOCTL_TYPE as u64) << 8) | nr as u64
}

/// `DRM_IOCTL_GEM_CLOSE` = _IOW('d', 0x09, drm_gem_close{handle u32, pad u32}).
pub(crate) const DRM_IOCTL_GEM_CLOSE: u64 =
    (DRM_IOC_WRITE as u64) | (8 << 16) | ((DRM_IOCTL_TYPE as u64) << 8) | 0x09;

/// `DRM_IOCTL_SYNCOBJ_DESTROY` = _IOWR('d', 0xc0, drm_syncobj_destroy{handle u32, pad u32}).
pub(crate) const DRM_IOCTL_SYNCOBJ_DESTROY: u64 = ((DRM_IOC_READ | DRM_IOC_WRITE) as u64)
    | (8 << 16)
    | ((DRM_IOCTL_TYPE as u64) << 8)
    | 0xc0;

/// `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` = _IOWR('d', 0xca,
/// drm_syncobj_timeline_wait{handles, points, timeout_nsec, count, flags,
/// first_signaled, pad} = 40 bytes).
pub(crate) const DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT: u64 = ((DRM_IOC_READ | DRM_IOC_WRITE) as u64)
    | (40 << 16)
    | ((DRM_IOCTL_TYPE as u64) << 8)
    | 0xca;

/// Perform a raw ioctl. `arg` must be a plain-old-data struct (or slice of
/// u32s) matching `cmd`'s expected size exactly.
///
/// # Safety
/// `arg` must point to a buffer that is valid for the whole call and whose
/// layout matches what the driver expects for `cmd`.
pub(crate) unsafe fn raw_ioctl(fd: RawFd, cmd: u64, arg: *mut u8) -> io::Result<()> {
    let ret = unsafe { libc::ioctl(fd, cmd as libc::c_ulong, arg) };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}
