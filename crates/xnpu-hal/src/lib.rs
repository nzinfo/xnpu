//! xnpu-hal: direct-DRM userspace for AMD XDNA NPUs.
//!
//! Talks to the in-tree `amdxdna` accel driver (/dev/accel/*) with raw ioctls,
//! bypassing XRT entirely. Ported from the KMQ shim of iree-amd-aie
//! (`runtime/src/iree-amd-aie/driver/amdxdna/shim/linux/kmq/`) against the
//! kernel UAPI of amdxdna in Linux 7.0 (include/uapi/drm/amdxdna_accel.h).
//!
//! All `unsafe` of the crate is confined to the modules here; the public API
//! is safe Rust.

pub mod accel;
pub mod bo;
pub mod ert;
pub mod hwctx;
pub mod ioctl;

pub use accel::{AieMetadata, Device, TileMetadata};
pub use bo::{BoType, BufferObject, Mapping, SyncDirection};
pub use ert::{syncobj_timeline_wait, StartNpuCmd, ERT_CMD_STATE_COMPLETED};
pub use hwctx::HwContext;
