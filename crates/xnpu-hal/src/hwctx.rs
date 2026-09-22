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
    /// PDI BOs kept alive for the context lifetime (CU config references them
    /// by handle; one per configured CU).
    _pdi_bos: Vec<BufferObject>,
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
            _pdi_bos: Vec::new(),
        })
    }

    /// Load a PDI (the xclbin's PDI section) as the context's CU and configure
    /// it. See [`Self::configure_cus`] for the full contract.
    pub fn configure_cu(&mut self, pdi: &[u8], cu_func: u8) -> io::Result<()> {
        self.configure_cus(&[(pdi, cu_func)])
    }

    /// Attach up to 32 PDIs as the CUs of this context in a single
    /// CONFIG_HWCTX call. The driver refuses any second configuration
    /// ("Not support re-config CU", -EINVAL), so a multi-operator engine must
    /// know its full operator set up front and attach everything once. Later
    /// exec packets pick the CU via their cu_mask bit, which the firmware
    /// translates to the i-th cu_config (PDI heap address | cu_func).
    ///
    /// Each PDI needs a DEV BO carved out of the client heap ("Invalid BO
    /// type" otherwise); the kernel hands the firmware its heap offset, the
    /// bytes travel through the heap mapping plus a cache flush, and the BOs
    /// must outlive the context — kept in `self._pdi_bos`.
    pub fn configure_cus(&mut self, cus: &[(&[u8], u8)]) -> io::Result<()> {
        const MAX_NUM_CUS: usize = 32;
        if cus.is_empty() || cus.len() > MAX_NUM_CUS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("CU count must be 1..={MAX_NUM_CUS}, got {}", cus.len()),
            ));
        }
        let mut pdi_bos = Vec::with_capacity(cus.len());
        for (pdi, _func) in cus {
            let pdi_bo = BufferObject::new(self.device, BoType::Dev, pdi.len())?;
            self.device.write_dev_bo(&pdi_bo, pdi)?;
            // The kernel clflushes the heap pages behind the DEV BO; size is the
            // page-aligned BO size (pdi.len() is rounded up at CREATE_BO).
            pdi_bo.sync(crate::bo::SyncDirection::ToDevice, 0, pdi_bo.size() as u64)?;
            pdi_bos.push(pdi_bo);
        }

        // amdxdna_hwctx_param_config_cu = { u16 num_cus; u16 rsvd[3] } followed
        // by cu_configs[] = { u32 cu_bo; u8 cu_func; u8 pad[3] } each.
        let mut param = vec![0u8; 8 + 8 * cus.len()];
        param[0..2].copy_from_slice(&(cus.len() as u16).to_le_bytes());
        for (i, (_pdi, func)) in cus.iter().enumerate() {
            let off = 8 + i * 8;
            param[off..off + 4].copy_from_slice(&pdi_bos[i].handle().to_le_bytes());
            param[off + 4] = *func;
        }

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
        self._pdi_bos = pdi_bos;
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
