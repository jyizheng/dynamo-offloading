# GDS (GPU Direct Storage) Setup in Container

## Environment
- Container with overlay rootfs, running on a node with 8x NVMe drives (nvme0n1 - nvme7n1)
- GPU: NVIDIA B200
- nvidia-fs version: 2.27
- CUDA driver: 590.48.01

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
