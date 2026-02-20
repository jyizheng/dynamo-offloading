# GDS (GPU Direct Storage) Setup in Container

## Environment
- Container with overlay rootfs, running on a node with 8x NVMe drives (nvme0n1 - nvme7n1)
- GPU: NVIDIA B200
- nvidia-fs version: 2.27
- CUDA driver: 590.48.01

---

## Diagnosis Process

### Symptom
Running `./gds_test` failed at step 9 with:
```
[9] Registering file handle with cuFile (GDS capability test)...
    FAIL: cuFileHandleRegister() -> error=5030 (base+30)
*** GDS IS NOT WORKING on this filesystem/device ***
```

### Step 1: Read cufile.log
cuFile library writes detailed logs to `cufile.log` in the current working directory. The log revealed two key error lines:
```
ERROR  cufio-udev:67 udev property not found: ID_FS_USAGE nvme0n1p2
ERROR  cufio-fs:759 error getting volume attributes error for device: dev_no: 259:6
NOTICE cufio:293 cuFileHandleRegister GDS not supported or disabled by config, using cuFile posix read/write with compat mode enabled
ERROR  cufio-obj:242 unable to get volume attributes for fd 3
ERROR  cufio:311 cuFileHandleRegister error, failed to allocate file object
```
This told us: (1) GDS can't identify the filesystem via udev, (2) even compat mode fallback fails.

### Step 2: Check the filesystem where test file lives
The test was writing to `/tmp/kvbm_disk_cache/gds_test_file`. Checked:
```bash
df -Th /tmp/
# Result: overlay filesystem (container rootfs) — NOT GDS-compatible
```
GDS requires a real block filesystem (ext4/XFS) directly on NVMe. overlay/tmpfs/virtual filesystems won't work.

### Step 3: Find a GDS-compatible mount
```bash
lsblk -f
mount | grep nvme
```
Found `/mnt/nvme` mounted as ext4 on `/dev/nvme7n1p2` — a real NVMe device with a real filesystem.

### Step 4: Change test path and recompile
Changed `TEST_FILE` from `/tmp/kvbm_disk_cache/gds_test_file` to `/mnt/nvme/gds_test_file` in `gds_test.c`. Recompiled and ran — still failed with the same error 5030.

### Step 5: Re-read cufile.log after the fix
New log showed:
```
ERROR  cufio-udev:67 udev property not found: ID_FS_USAGE nvme7n1p2
```
Now targeting the right device (nvme7n1p2 instead of nvme0n1p2), but udev still can't find `ID_FS_USAGE`. This is the container's fault — no udevd running.

### Step 6: Verify udev is absent
```bash
udevadm info --query=all --name=/dev/nvme7n1p2  # command not found
ls /run/udev/data/                                # no such directory




```
Confirmed: udev database doesn't exist in this container.

### Step 7: Manually create udev database entry
GDS reads udev properties from `/run/udev/data/b<major>:<minor>`. Got the major:minor:
```bash
cat /sys/class/block/nvme7n1p2/dev   # 259:31
blkid /dev/nvme7n1p2                 # UUID, TYPE=ext4
```
Created `/run/udev/data/b259:31` with `ID_FS_USAGE=filesystem` and `ID_FS_TYPE=ext4`.

### Step 8: Success
Reran `./gds_test` — all 13 steps passed, GDS confirmed working with true GPU<->NVMe DMA transfers.

---

## Problem 1: Filesystem Not GDS-Compatible
GDS requires a real filesystem (ext4/XFS) directly on an NVMe block device. It does NOT work on:
- overlay (container rootfs)
- tmpfs
- any virtual/layered filesystem

### Fix
Use `/mnt/nvme/` which is ext4 on `/dev/nvme7n1p2` (device 259:31). Changed `TEST_FILE` in `gds_test.c` from `/tmp/kvbm_disk_cache/gds_test_file` to `/mnt/nvme/gds_test_file`.

## Problem 2: Missing udev Database in Container
GDS relies on udev to read `ID_FS_USAGE` property from the block device. Inside containers, udevd is not running, so the udev database at `/run/udev/data/` is empty. This causes:
```
ERROR cufio-udev:67 udev property not found: ID_FS_USAGE nvme7n1p2
ERROR cufio-fs:759 error getting volume attributes error for device: dev_no: 259:31
```

### Fix
Manually create the udev data file for the device:
```bash
mkdir -p /run/udev/data
cat > /run/udev/data/b259:31 << 'EOF'
E:ID_FS_USAGE=filesystem
E:ID_FS_TYPE=ext4
E:ID_FS_UUID=58122607-9721-4ea9-afab-a7899a6081cc
E:ID_PART_TABLE_TYPE=gpt
EOF
```
The file path format is `/run/udev/data/b<major>:<minor>`. Get major:minor from `cat /sys/class/block/nvme7n1p2/dev` or `stat -c '%t:%T'`.

**Note:** `/run/udev/data/` is on tmpfs — won't survive container restart. Add to entrypoint/init if needed.

## Secondary Issue: libnuma Missing
Log shows `unable to load libnuma.so.1.0.0`. Fix with `apt-get install -y libnuma1`. Not critical — GDS works without it, just disables NUMA-aware allocations.

## Diagnostic Commands
```bash
# Check filesystem type
df -Th /mnt/nvme/

# Check device info
blkid /dev/nvme7n1p2
cat /sys/class/block/nvme7n1p2/dev   # major:minor

# Check udev data
cat /run/udev/data/b259:31

# Check cufile log (written to CWD by default)
cat cufile.log

# cuFile config location
cat /usr/local/cuda/gds/cufile.json
```

## Key Files
- Test program: `/workspace/dynamo-offloading/gds_test/gds_test.c`
- cuFile config: `/usr/local/cuda/gds/cufile.json`
- cuFile log: written to current working directory as `cufile.log`
- nvidia-fs version: `/proc/driver/nvidia-fs/version`
- nvidia-fs stats: `/proc/driver/nvidia-fs/stats`
