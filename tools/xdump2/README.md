# xdump2 — amdxdna ioctl tracer (LD_PRELOAD)

M1 `xdump` 的重建版（P20b 起用，P22/P23 FLM 逆向主力工具）。拦截 `ioctl` + `mmap`，
维护 CREATE_BO → GET_BO_INFO → mmap 的 handle→VA 映射，把每次 EXEC_CMD 的 UAPI
结构、arg-handle 列表和 CMD-BO 包字节（header + regmap）打到日志。

与 xdump 的差异（P22/P23 增量）：
- `t=<µs>` 时间戳打在每个 exec 行上（token 周期 / phase 划分全靠它）
- chain 展开：ERT_CMD_CHAIN（opcode 19）的每个 sub BO 全量解析，
  `ctrl_addr`/`ctrl_sz`（ctrl-code BO 的 xdna 地址与大小）逐 sub 报告
- `XDUMP_SUBDUMP=<n> XDUMP_SUBDUMP_DIR=<dir>`：除 wrapper BO 外，把 sub 指向的
  ctrl-code blob 本体也 dump 成 `ctrl_eN_sI_<bytes>B.bin`
- ctrl BO 从 64MB DEV_HEAP carve 出来（type=3、map_off=-1、无独立 mmap），
  host VA = heap_map_va + (ctrl_xdna − heap_xdna)，fallback 逻辑在 lib.rs 里

构建：`cargo build --release` → `target/release/libxdump2.so`

运行（FLM 全量 trace + ctrl dump）：

```bash
sudo -n rm -f /tmp/flm_xdump.log; mkdir -p /tmp/flm_subdump; chmod 777 /tmp/flm_subdump
echo "测试一下翻译：今天天气真好" | timeout 120 sudo -n env HOME=/home/nzinfo \
  XDUMP_OUT=/tmp/flm_xdump.log XDUMP_SUBDUMP=40 XDUMP_SUBDUMP_DIR=/tmp/flm_subdump \
  LD_PRELOAD=/path/to/libxdump2.so flm run hy-mt2:1.8b --quiet
```

注意：log 与 dump 目录归 root 所有，清理要 `sudo rm`。

配套分析：`../ctrl_decode.py` 解码 dump 出的 aie2 transaction ctrl-code
（指令格式已对照 IRON 编译产物 + FLM 开源 npu_cmd_*.hpp 双向校准）。
