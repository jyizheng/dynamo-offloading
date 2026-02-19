# Dynamo + 3FS KVBM 部署指南

## 架构概览

```
                    ┌─────────────────────────────────┐
                    │        dynamo-qwen namespace     │
                    │                                  │
  Client ──HTTP──►  │  Frontend (cp16)                 │
                    │     │                            │
                    │     │ etcd + NATS                │
                    │     ▼                            │
                    │  Worker (cp17)                   │
                    │   ├─ TRT-LLM (Qwen3-8B)         │
                    │   │   └─ KVBM connector          │
                    │   │       G1(GPU)→G2(CPU)→G3(Disk)│
                    │   └─ FUSE sidecar                │
                    │       └─ 3FS mount (/mnt/3fs)    │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │    three-fs-ipoib namespace       │
                    │  mgmtd(cp17) meta(cp18)          │
                    │  storage(cp17,cp18,cp01)         │
                    │  IB: mlx5_0, pkey_index=1        │
                    └──────────────────────────────────┘
```

- **Frontend**: HTTP API 接收推理请求，round-robin 分发给 Worker
- **Worker**: TRT-LLM PyTorch 后端 + KVBM connector，GPU 推理 + KV cache 分层存储
- **FUSE sidecar**: 3FS FUSE 守护进程，挂载 3FS 集群到 `/mnt/3fs-root/3fs`
- **3FS 集群**: IPoIB 模式（native IB, mlx5_0），3 节点 3 链 9 副本

## 前置条件

### 集群依赖

| 组件 | Namespace | 地址 |
|------|-----------|------|
| etcd (3 节点) | baseten | 172.29.22.14:2379, 172.29.202.242:2379, 172.29.134.203:2379 |
| NATS (JetStream) | baseten | 172.29.182.124:4222 |
| 3FS mgmtd | three-fs-ipoib | RDMA://10.2.10.55:8010 |
| 3FS FDB | three-fs-ipoib | 10.2.10.55:4510 |

> **注意**: Worker 使用 `hostNetwork: true`，因此 etcd/NATS 不能使用 ClusterIP/DNS，
> 必须使用 Pod IP 地址。如果 Pod 重启 IP 变化，需要更新 `02-dynamo.yaml` 中的地址。

### 模型缓存

Worker 节点 (cp17) 必须有模型文件：

```bash
# 检查模型是否存在
ls /data/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/

# 如果不存在，运行下载 Pod（需要外网访问）
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: model-downloader
  namespace: dynamo-qwen
spec:
  restartPolicy: Never
  hostNetwork: true
  dnsPolicy: Default
  securityContext:
    runAsUser: 0
  nodeSelector:
    kubernetes.io/hostname: p02-r02-cp17
  containers:
  - name: downloader
    image: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.9.0
    command: ["python3", "-c"]
    args:
    - |
      from huggingface_hub import snapshot_download
      snapshot_download("Qwen/Qwen3-8B", local_dir=None)
    env:
    - name: HF_HOME
      value: /data/huggingface
    volumeMounts:
    - name: hf-cache
      mountPath: /data/huggingface
  volumes:
  - name: hf-cache
    hostPath:
      path: /data/huggingface
      type: DirectoryOrCreate
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
EOF
```

### 3FS 集群状态

确认 IPoIB 3FS 集群正常运行：

```bash
# 检查所有 Pod 运行状态
kubectl get pods -n three-fs-ipoib

# 从 mgmtd Pod 验证集群健康
kubectl exec -n three-fs-ipoib deploy/ipoib-mgmtd -- \
  /opt/3fs/bin/admin_cli --cfg /tmp/admin.toml "get-cluster-info"
```

## 部署步骤

### Step 1: 创建 Namespace

```bash
kubectl apply -f namespace.yaml
```

### Step 2: (可选) 部署独立 etcd + NATS

如果没有共享的 etcd/NATS，部署独立实例：

```bash
kubectl apply -f 01-infra.yaml
```

> 本集群使用 `baseten` namespace 的共享 etcd/NATS，所以 Worker 直接配置 Pod IP。

### Step 3: 获取 etcd/NATS Pod IP

```bash
# 获取 etcd endpoints
kubectl get endpoints -n baseten etcd-headless
# 输出: 172.29.22.14:2379,172.29.202.242:2379,172.29.134.203:2379

# 获取 NATS endpoint
kubectl get endpoints -n baseten nats-headless
# 输出: 172.29.182.124:4222
```

将这些 IP 更新到 `02-dynamo.yaml` 中 Worker 的环境变量：

```yaml
- name: ETCD_ENDPOINTS
  value: "172.29.22.14:2379,172.29.202.242:2379,172.29.134.203:2379"
- name: NATS_SERVER
  value: "nats://172.29.182.124:4222"
```

Frontend 可以使用 DNS（它不用 hostNetwork）：

```yaml
- name: ETCD_ENDPOINTS
  value: "etcd-headless.baseten:2379"
- name: NATS_SERVER
  value: "nats://nats-headless.baseten:4222"
```

### Step 4: 部署 Dynamo

```bash
kubectl apply -f 02-dynamo.yaml
```

这会创建以下资源：

| 资源 | 说明 |
|------|------|
| ConfigMap `trtllm-engine-config` | TRT-LLM 引擎参数 (batch size, chunked prefill 等) |
| ConfigMap `fuse-launcher-config` | 3FS FUSE 启动配置 (cluster_id, mgmtd 地址等) |
| ConfigMap `fdb-config` | FoundationDB 集群文件 |
| ConfigMap `threefs-token` | 3FS 认证 token |
| Pod `dynamo-frontend` | Frontend (cp16) |
| Service `dynamo-frontend` | Frontend ClusterIP 服务 |
| Pod `dynamo-worker` | Worker + FUSE sidecar (cp17) |

### Step 5: 验证部署

```bash
# 检查 Pod 状态
kubectl get pods -n dynamo-qwen -o wide

# 检查 FUSE sidecar 日志 (应该看到 "3FS mounted successfully")
kubectl logs -n dynamo-qwen dynamo-worker -c fuse-sidecar

# 检查 Worker 日志 (应该看到模型加载和 KVBM 初始化)
kubectl logs -n dynamo-qwen dynamo-worker -c worker -f

# 测试推理
kubectl exec -n dynamo-qwen dynamo-frontend -- \
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 关键配置说明

### TRT-LLM 引擎配置 (`trtllm-engine-config`)

```yaml
tensor_parallel_size: 1        # 单 GPU
backend: pytorch               # PyTorch 后端 (非 TensorRT engine)
max_num_tokens: 8192           # 最大 token 数
max_batch_size: 16             # 最大 batch
enable_chunked_prefill: true   # 分块 prefill
kv_cache_config:
  free_gpu_memory_fraction: 0.80  # 80% GPU 显存用于 KV cache
```

### KVBM 配置 (Worker 环境变量)

```yaml
DYN_KVBM_CPU_CACHE_GB: "20"              # G2: 20GB CPU 内存
DYN_KVBM_DISK_CACHE_GB: "50"             # G3: 50GB 磁盘缓存
DYN_KVBM_DISK_CACHE_DIR: "/mnt/3fs-root/3fs/kvbm_cache"  # 缓存目录 (当前指向 3FS)
DYN_KVBM_DISK_DISABLE_O_DIRECT: "true"   # 3FS FUSE 不支持 O_DIRECT
DYN_KVBM_DISK_ZEROFILL_FALLBACK: "true"  # 3FS 不支持 fallocate
DYN_KVBM_METRICS: "true"                 # 启用 KVBM metrics
```

> **注意**: 当前配置将 G3 直接指向 3FS FUSE 挂载点。首次启动时会进行 50GB 的磁盘缓存
> 初始化（零填充），通过 FUSE 写入约需 12 分钟（~4-5 GB/min）。

### 3FS FUSE 配置 (`fuse-launcher-config`)

```toml
cluster_id = 'ipoib'                                    # IPoIB 集群 ID
mountpoint = '/mnt/3fs-root/3fs'                         # 挂载点
token_file = '/etc/3fs-token/token'                      # 认证 token
force_use_tcp = true                                     # TCP 模式 (cp16 无 IB)
[ib_devices]
device_filter = ["mlx5_0"]                               # IB 设备
default_pkey_index = 1                                   # IPoIB pkey
[mgmtd_client]
mgmtd_server_addresses = ["RDMA://10.2.10.55:8010"]     # mgmtd 地址
```

### FUSE Sidecar 挂载传播

Worker Pod 使用 `hostPath` + `mountPropagation` 实现 FUSE sidecar 模式：

```yaml
volumes:
- name: 3fs-mount
  hostPath:
    path: /tmp/dynamo-3fs-mount     # 宿主机临时目录
    type: DirectoryOrCreate

# FUSE sidecar: Bidirectional (可以向宿主机传播挂载)
volumeMounts:
- name: 3fs-mount
  mountPath: /mnt/3fs-root
  mountPropagation: Bidirectional

# Worker: HostToContainer (接收 sidecar 的挂载)
volumeMounts:
- name: 3fs-mount
  mountPath: /mnt/3fs-root
  mountPropagation: HostToContainer
```

## 已知问题和解决方案

### 1. Worker DNS 解析失败

**症状**: Worker 启动时报 `Failed to resolve host: huggingface.co`

**原因**: `hostNetwork: true` + CoreDNS 不可达

**解决**: 使用离线模式 + 本地模型路径

```yaml
- name: HF_HUB_OFFLINE
  value: "1"
- name: TRANSFORMERS_OFFLINE
  value: "1"
# --model-path 使用本地完整路径，不用 HuggingFace repo ID
--model-path /data/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/<hash>
```

### 2. KVBM 磁盘缓存初始化慢

**症状**: Worker 启动后长时间卡在 "Allocating disk cache"

**原因**: 3FS FUSE 写入速度 ~4-5 GB/min，50GB 需要 ~12 分钟

**解决**: 正常现象，仅首次启动。可以减小 `DYN_KVBM_DISK_CACHE_GB` 加速。

### 3. 旧的 FUSE 挂载残留

**症状**: Worker 重启后 `/mnt/3fs-root/3fs` 显示 `Transport endpoint is not connected`

**解决**:

```bash
# 在宿主机上清理
ssh p02-r02-cp17
umount -l /tmp/dynamo-3fs-mount/3fs 2>/dev/null
rm -rf /tmp/dynamo-3fs-mount
```

### 4. 3FS FUSE 不支持 O_DIRECT 和 fallocate

**解决**: 设置环境变量：

```yaml
DYN_KVBM_DISK_DISABLE_O_DIRECT: "true"
DYN_KVBM_DISK_ZEROFILL_FALLBACK: "true"
```

## 常用运维命令

```bash
# 查看 Worker 实时日志
kubectl logs -n dynamo-qwen dynamo-worker -c worker -f

# 查看 FUSE 日志
kubectl logs -n dynamo-qwen dynamo-worker -c fuse-sidecar

# 重启 Worker (保留 Frontend)
kubectl delete pod -n dynamo-qwen dynamo-worker
kubectl apply -f 02-dynamo.yaml

# 清理全部
kubectl delete -f 02-dynamo.yaml
# 清理宿主机残留挂载
ssh p02-r02-cp17 "umount -l /tmp/dynamo-3fs-mount/3fs 2>/dev/null; rm -rf /tmp/dynamo-3fs-mount"

# 检查 KVBM metrics
kubectl exec -n dynamo-qwen dynamo-worker -c worker -- curl -s localhost:8081/metrics | grep kvbm

# 测试推理 (从集群外)
kubectl port-forward -n dynamo-qwen svc/dynamo-frontend 8000:8000 &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"你好"}],"max_tokens":100}'
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `namespace.yaml` | Namespace 定义 |
| `01-infra.yaml` | 独立 etcd + NATS (可选) |
| `02-dynamo.yaml` | 主部署文件: ConfigMaps + Frontend + Worker |
| `dynamo-qwen3-8b.yaml` | 早期单 Pod 测试版本 (已废弃) |
# dynamo-offloading
