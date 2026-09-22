//! Hardware contexts: create (KMQ) / configure CU / destroy.

use std::io;

use crate::accel::Device;
use crate::ioctl::{amdxdna_ioctl, raw_ioctl, AmdxdnaCmd, DRM_IOCTL_SYNCOBJ_DESTROY};
use crate::bo::{BoType, BufferObject};

/// Driver-default QoS hint used by the KMQ shim.
const DEFAULT_OPS_PER_CYCLE: u32 = 2048;

pub struct HwContext<'a> {
    device: &'a Device,
    pub handle: u32,
    pub umq_doorbell: u32,
    pub syncobj_handle: u32,
    /// PDI BO kept alive for the context lifetime (CU config references it).
    _pdi_bo: Option<BufferObject>,
}

impl<'a> HwContext<'a> {
    /// Create a bare hardware context. In kernel-mode-queue (KMQ) the umq_bo
    /// is left as 0 so the driver allocates and drives the queue itself.
    ///
    /// `num_tiles` is a QoS-style hint; the KMQ shim passes rows*cols of the
    /// full AIE grid. No CU is configured yet — call [`Self::configure_cu`].
    pub fn create(device: &'a Device, num_tiles: u32) -> io::Result<HwContext<'a>> {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct CreateHwctxArg {
            ext: u64,
            ext_flags: u64,
            qos_p: u64,
            umq_bo: u32,
            log_buf_bo: u32,
            max_opc: u32,
            num_tiles: u32,
            mem_size: u32,
            umq_doorbell: u32,
            handle: u32,
            syncobj_handle: u32,
        }
        // The driver copies the QoS struct from qos_p unconditionally
        // (qos_p=0 => EFAULT), so hand it a zeroed struct like the KMQ shim.
        let mut qos = [0u32; 6];
        let mut arg = CreateHwctxArg {
            ext: 0,
            ext_flags: 0,
            qos_p: qos.as_mut_ptr() as u64,
            umq_bo: 0,
            log_buf_bo: 0,
            max_opc: DEFAULT_OPS_PER_CYCLE,
            num_tiles,
            mem_size: 0,
            umq_doorbell: 0,
            handle: 0,
            syncobj_handle: 0,
        };
        let cmd = amdxdna_ioctl(AmdxdnaCmd::CreateHwctx, std::mem::size_of::<CreateHwctxArg>());
        // SAFETY: exact-size repr(C) UAPI struct, passed in place.
        unsafe {
            raw_ioctl(
                device.raw_fd(),
                cmd,
                &mut arg as *mut CreateHwctxArg as *mut u8,
            )?;
        }
        Ok(HwContext {
            device,
            handle: arg.handle,
            umq_doorbell: arg.umq_doorbell,
            syncobj_handle: arg.syncobj_handle,
            _pdi_bo: None,
        })
    }

    /// Load a PDI (the xclbin's PDI section) as the context's CU and configure
    /// it. The kernel's aie2_config_cu only accepts a DEV BO carved out of the
    /// client heap ("Invalid BO type" otherwise) and hands the firmware its
    /// heap offset; the bytes reach it through the heap mapping plus a cache
    /// flush. Then DRM_AMDXDNA_HWCTX_CONFIG_CU carries the packed cu_configs.
    pub fn configure_cu(&mut self, pdi: &[u8], cu_func: u8) -> io::Result<()> {
        let pdi_bo = BufferObject::new(self.device, BoType::Dev, pdi.len())?;
        self.device.write_dev_bo(&pdi_bo, pdi)?;
        // The kernel clflushes the heap pages behind the DEV BO; size is the
        // page-aligned BO size (pdi.len() is rounded up at CREATE_BO).
        pdi_bo.sync(crate::bo::SyncDirection::ToDevice, 0, pdi_bo.size() as u64)?;

        // amdxdna_hwctx_param_config_cu = { u16 num_cus; u16 rsvd[3] } followed
        // by cu_configs[] = { u32 cu_bo; u8 cu_func; u8 pad[3] } each.
        let mut param = [0u8; 16];
        param[0] = 1; // num_cus
        param[8..12].copy_from_slice(&pdi_bo.handle().to_le_bytes());
        param[12] = cu_func;

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct ConfigHwctxArg {
            handle: u32,
            param_type: u32,
            param_val: u64,
            param_val_size: u32,
            pad: u32,
        }
        let mut arg = ConfigHwctxArg {
            handle: self.handle,
            param_type: 0, // DRM_AMDXDNA_HWCTX_CONFIG_CU
            param_val: param.as_ptr() as u64,
            param_val_size: param.len() as u32,
            pad: 0,
        };
        let cmd = amdxdna_ioctl(AmdxdnaCmd::ConfigHwctx, std::mem::size_of::<ConfigHwctxArg>());
        // SAFETY: exact-size UAPI struct; param_val points at our packed
        // buffer for the duration of the call.
        unsafe {
            raw_ioctl(
                self.device.raw_fd(),
                cmd,
                &mut arg as *mut ConfigHwctxArg as *mut u8,
            )?;
        }
        self._pdi_bo = Some(pdi_bo);
        Ok(())
    }
}

impl Drop for HwContext<'_> {
    fn drop(&mut self) {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct DestroyHwctxArg {
            handle: u32,
            pad: u32,
        }
        let mut arg = DestroyHwctxArg {
            handle: self.handle,
            pad: 0,
        };
        // SAFETY: exact-size UAPI struct.
        unsafe {
            let _ = raw_ioctl(
                self.device.raw_fd(),
                amdxdna_ioctl(AmdxdnaCmd::DestroyHwctx, std::mem::size_of::<DestroyHwctxArg>()),
                &mut arg as *mut DestroyHwctxArg as *mut u8,
            );
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct SyncobjDestroyArg {
            handle: u32,
            pad: u32,
        }
        let mut arg = SyncobjDestroyArg {
            handle: self.syncobj_handle,
            pad: 0,
        };
        // SAFETY: exact-size DRM struct.
        unsafe {
            let _ = raw_ioctl(
                self.device.raw_fd(),
                DRM_IOCTL_SYNCOBJ_DESTROY,
                &mut arg as *mut SyncobjDestroyArg as *mut u8,
            );
        }
    }
}
