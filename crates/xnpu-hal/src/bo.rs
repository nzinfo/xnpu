//! Buffer objects: create / map / sync / close.

use std::io;
use std::slice;

use crate::accel::Device;
use crate::ioctl::{amdxdna_ioctl, raw_ioctl, AmdxdnaCmd, DRM_IOCTL_GEM_CLOSE};

/// enum amdxdna_bo_type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoType {
    /// DRM GEM SHMEM bo (host-coherent; the general-purpose kind).
    Shmem = 1,
    /// Shared host memory exposed to the device as heap.
    DevHeap = 2,
    /// Carved out of a DevHeap; device-visible (AMDXTLA) memory.
    Dev = 3,
    /// Command buffer holding an ERT packet.
    Cmd = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    ToDevice = 0,
    FromDevice = 1,
}

/// A device buffer object. Holds the raw device fd (the BO stays valid as
/// long as the owning Device is open; dropping the BO closes the GEM handle).
pub struct BufferObject {
    fd: std::os::unix::io::RawFd,
    handle: u32,
    size: usize,
    map_offset: u64,
    xdna_addr: u64,
}

impl BufferObject {
    pub fn new(device: &Device, ty: BoType, size: usize) -> io::Result<BufferObject> {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct CreateBoArg {
            flags: u64,
            vaddr: u64,
            size: u64,
            ty: u32,
            handle: u32,
        }
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct BoInfoArg {
            ext: u64,
            ext_flags: u64,
            handle: u32,
            pad: u32,
            map_offset: u64,
            vaddr: u64,
            xdna_addr: u64,
        }

        let mut create = CreateBoArg {
            flags: 0,
            vaddr: 0,
            size: size as u64,
            ty: ty as u32,
            handle: 0,
        };
        let cmd = amdxdna_ioctl(AmdxdnaCmd::CreateBo, std::mem::size_of::<CreateBoArg>());
        // SAFETY: exact-size repr(C) UAPI struct, passed in place.
        unsafe {
            raw_ioctl(
                device.raw_fd(),
                cmd,
                &mut create as *mut CreateBoArg as *mut u8,
            )?;
        }

        let mut info = BoInfoArg {
            ext: 0,
            ext_flags: 0,
            handle: create.handle,
            pad: 0,
            map_offset: 0,
            vaddr: 0,
            xdna_addr: 0,
        };
        let cmd = amdxdna_ioctl(AmdxdnaCmd::GetBoInfo, std::mem::size_of::<BoInfoArg>());
        unsafe {
            raw_ioctl(device.raw_fd(), cmd, &mut info as *mut BoInfoArg as *mut u8)?;
        }

        Ok(BufferObject {
            fd: device.raw_fd(),
            handle: create.handle,
            size,
            map_offset: info.map_offset,
            xdna_addr: info.xdna_addr,
        })
    }

    pub fn handle(&self) -> u32 {
        self.handle
    }

    /// mmap the whole BO and return an owned mapping (unmapped on drop).
    pub fn map_owned(&self) -> io::Result<Mapping> {
        // SAFETY: the kernel keeps [ptr, ptr+len) mapped until munmap; the
        // Mapping owns exactly that range.
        unsafe { Mapping::new(self.fd, self.map_offset, self.size) }
    }

    /// mmap the whole BO at a VA aligned to `align` (power of two), using the
    /// reserve-then-MAP_FIXED trick. The firmware's MAP_HOST_BUFFER rejects a
    /// DEV_HEAP whose user VA is not 64MB aligned (status 0x4000003
    /// INVALID_PARAM), which is why the XRT shim maps the heap this way.
    pub fn map_owned_aligned(&self, align: usize) -> io::Result<Mapping> {
        debug_assert!(align.is_power_of_two());
        // SAFETY: see Mapping::new_aligned.
        unsafe { Mapping::new_aligned(self.fd, self.map_offset, self.size, align) }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Device-visible (XDNA) address to hand to ERT/instruction streams.
    pub fn xdna_addr(&self) -> u64 {
        self.xdna_addr
    }

    pub fn sync(&self, dir: SyncDirection, offset: u64, size: u64) -> io::Result<()> {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct SyncBoArg {
            handle: u32,
            direction: u32,
            offset: u64,
            size: u64,
        }
        let mut arg = SyncBoArg {
            handle: self.handle,
            direction: dir as u32,
            offset,
            size,
        };
        let cmd = amdxdna_ioctl(AmdxdnaCmd::SyncBo, std::mem::size_of::<SyncBoArg>());
        unsafe {
            raw_ioctl(
                self.fd,
                cmd,
                &mut arg as *mut SyncBoArg as *mut u8,
            )
        }
    }
}

impl Drop for BufferObject {
    fn drop(&mut self) {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct GemCloseArg {
            handle: u32,
            pad: u32,
        }
        let mut arg = GemCloseArg {
            handle: self.handle,
            pad: 0,
        };
        // SAFETY: exact-size DRM GEM_CLOSE struct.
        unsafe {
            let _ = raw_ioctl(
                self.fd,
                DRM_IOCTL_GEM_CLOSE,
                &mut arg as *mut GemCloseArg as *mut u8,
            );
        }
    }
}

/// An owned mmap of a BO, unmapped on drop. Beyond plain CPU access, the
/// kernel's mmap path registers the mapping with its MMU notifier and records
/// the VA as the BO's `uva` — required, e.g., before a hwctx can carve DEV
/// BOs out of a DEV_HEAP ("Invalid dev heap userptr" otherwise).
pub struct Mapping {
    ptr: *mut u8,
    len: usize,
    /// The anonymous reservation used to place an aligned mapping; unmapped
    /// together with the mapping on drop.
    parent: Option<(*mut u8, usize)>,
}

impl Mapping {
    /// SAFETY: caller must guarantee the BO outlives (or at least is not
    /// closed before) this Mapping.
    unsafe fn new(fd: std::os::unix::io::RawFd, offset: u64, len: usize) -> io::Result<Mapping> {
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        // MAP_LOCKED pins the pages now; it needs the memlock rlimit (IRON
        // runs with unlimited). Fall back to a plain shared mapping — the
        // driver pins BOs itself around sync/submit anyway.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                prot,
                libc::MAP_SHARED | libc::MAP_LOCKED,
                fd,
                offset as libc::off_t,
            )
        };
        let ptr = if ptr != libc::MAP_FAILED {
            ptr
        } else {
            let locked_err = io::Error::last_os_error();
            if !matches!(
                locked_err.raw_os_error(),
                Some(libc::EAGAIN) | Some(libc::ENOMEM)
            ) {
                return Err(locked_err);
            }
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    prot,
                    libc::MAP_SHARED,
                    fd,
                    offset as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            ptr
        };
        Ok(Mapping {
            ptr: ptr as *mut u8,
            len,
            parent: None,
        })
    }

    /// SAFETY: same contract as [`Self::new`]. Reserves an anonymous range
    /// of len+align-1, then MAP_FIXEDs the BO over the aligned part. The
    /// full reservation (including the fixed mapping inside it) is released
    /// on drop.
    unsafe fn new_aligned(
        fd: std::os::unix::io::RawFd,
        offset: u64,
        len: usize,
        align: usize,
    ) -> io::Result<Mapping> {
        let reserve = len + align - 1;
        let parent = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reserve,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if parent == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let parent = parent as *mut u8;
        let base = (parent as usize + align - 1) & !(align - 1);
        let ptr = unsafe {
            libc::mmap(
                base as *mut libc::c_void,
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                offset as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = io::Error::last_os_error();
            // SAFETY: we mapped the reservation ourselves.
            unsafe { libc::munmap(parent as *mut libc::c_void, reserve) };
            return Err(err);
        }
        Ok(Mapping {
            ptr: ptr as *mut u8,
            len,
            parent: Some((parent, reserve)),
        })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Access the mapping as a slice. Safe for the Mapping's lifetime as
    /// long as only this Mapping accesses the range.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: [ptr, ptr+len) is mapped and exclusively owned by self.
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Mutable variant of [`Self::as_slice`].
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: see as_slice.
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: [ptr, ptr+len) is exactly what we mmap'd; the parent
        // reservation (if any) covers the aligned mapping too, so unmapping
        // both is idempotent where they overlap.
        if let Some((parent, reserve)) = self.parent {
            unsafe { libc::munmap(parent as *mut libc::c_void, reserve) };
        } else {
            unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
        }
    }
}
