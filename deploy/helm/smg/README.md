# SMG Helm Chart

Helm chart for deploying the Shepherd Model Gateway (SMG) — a high-performance inference router for LLM deployments. Supports deploying the gateway with optional worker (inference engine) pods.

## Prerequisites

- Kubernetes >= 1.26
- Helm >= 3.12
- GPU nodes with `nvidia.com/gpu` resource (for workers)

## Quick Start

### Router only (external workers)

```bash
helm install smg deploy/helm/smg \
  --set router.workerUrls[0]=http://worker-1:8000 \
  --set router.workerUrls[1]=http://worker-2:8000
```

### Router + vLLM worker

```bash
helm install smg deploy/helm/smg \
  -f deploy/helm/smg/examples/with-vllm-http.yaml \
  --set huggingface.token=hf_xxx
```

### Router + SGLang worker

```bash
helm install smg deploy/helm/smg \
  -f deploy/helm/smg/examples/with-sglang.yaml \
  --set huggingface.token=hf_xxx
```

## Images

The gateway image is on Docker Hub, engine images are on GitHub Container Registry:

| Component | Registry | Image |
|-----------|----------|-------|
| Gateway | `docker.io` | `lightseekorg/smg:1.3.3` |
| vLLM worker | `ghcr.io` | `lightseekorg/smg:1.3.3-vllm-v0.18.0` |
| TRT-LLM worker | `ghcr.io` | `lightseekorg/smg:1.3.3-trtllm-1.3.0rc8` |
| SGLang worker | `ghcr.io` | `lightseekorg/smg:1.3.3-sglang-v0.5.9` |

Worker images default to `ghcr.io`. Set `workers[].image.tag` to the engine-specific tag.

## Configuration

See [`values.yaml`](values.yaml) for the full list of configurable parameters.

### Key Sections

| Section | Description |
|---------|-------------|
| `global` | Image registry, pull secrets |
| `router` | Router deployment, routing policy, networking, observability |
| `workers` | Worker (inference engine) deployments |
| `huggingface` | HuggingFace token for gated model downloads |
| `auth` | API key authentication, rate limiting |
| `history` | Storage backend (none/memory/postgres/redis/oracle) |
| `serviceAccount` | Service account creation and annotations |
| `rbac` | RBAC for Kubernetes service discovery |

### Routing Policies

`cache_aware` (default), `round_robin`, `power_of_two`, `consistent_hashing`, `prefix_hash`, `manual`, `random`, `bucket`

### Workers

Each entry in `workers[]` creates a Deployment + Service:

```yaml
workers:
  - name: llama-8b                    # unique name
    engine: vllm                      # vllm or sglang
    model: meta-llama/Llama-3.1-8B-Instruct
    replicas: 1
    connectionMode: http              # http or grpc
    gpu:
      count: 1                        # GPUs per replica (tensor parallel)
    image:
      tag: "1.3.3-vllm-v0.18.0"      # engine image tag
```

Workers are automatically wired to the router via `--worker-urls`. The URL scheme (`http://` or `grpc://`) is set based on `connectionMode`.

#### Connection Modes

| Mode | Description | Worker Command |
|------|-------------|----------------|
| `http` | OpenAI-compatible HTTP API | vLLM: `vllm.entrypoints.openai.api_server` |
| `grpc` | SMG gRPC pipeline | vLLM: `vllm.entrypoints.grpc_server` |

SGLang uses `sglang.launch_server` for both modes (adds `--grpc-mode` for gRPC).

#### HuggingFace Token

Gated models (e.g., Llama) require a HuggingFace token:

```bash
helm install smg deploy/helm/smg \
  -f examples/with-vllm-http.yaml \
  --set huggingface.token=hf_xxx
```

The token is stored in a Kubernetes Secret and injected as `HF_TOKEN` env var into both the router and worker pods.

#### Startup Probes

Model loading takes 60-300 seconds. Workers have startup probes with generous defaults:

- HTTP mode: `httpGet /health` with `initialDelaySeconds: 60`, `failureThreshold: 30`
- gRPC mode: `tcpSocket` probe (not all engines implement gRPC health checking yet)

Override per worker:

```yaml
workers:
  - name: large-model
    startupProbe:
      initialDelaySeconds: 180
      failureThreshold: 60
```

## Examples

| File | Scenario |
|------|----------|
| [`router-only.yaml`](examples/router-only.yaml) | Minimal router with external workers |
| [`with-vllm-http.yaml`](examples/with-vllm-http.yaml) | Router + vLLM worker (HTTP) |
| [`with-vllm-grpc.yaml`](examples/with-vllm-grpc.yaml) | Router + vLLM worker (gRPC) |
| [`with-sglang.yaml`](examples/with-sglang.yaml) | Router + SGLang worker |
| [`with-multi-model.yaml`](examples/with-multi-model.yaml) | Multiple models on different engines |
| [`with-postgres.yaml`](examples/with-postgres.yaml) | PostgreSQL history backend |
| [`with-service-discovery.yaml`](examples/with-service-discovery.yaml) | K8s auto-discovery |
| [`with-ingress.yaml`](examples/with-ingress.yaml) | Ingress with TLS |
| [`with-monitoring.yaml`](examples/with-monitoring.yaml) | ServiceMonitor + Grafana dashboard |

## Testing

```bash
helm test smg
```

## Troubleshooting

### Worker not registering

Check worker pod logs:

```bash
kubectl logs -l smg.lightseek.org/worker=<worker-name> -f
```

Model loading can take several minutes. The router will register the worker once it passes the startup probe.

### Tokenizer download fails (read-only filesystem)

The router sets `HF_HOME=/tmp/hf-cache` to use the writable `/tmp` mount. If you see "Read-only file system" errors, ensure the `tmp` volume is mounted.

### Service discovery not finding pods

Verify RBAC is enabled and the selector matches your worker pod labels:

```bash
kubectl get role,rolebinding -l app.kubernetes.io/instance=smg
kubectl get pods -l <your-selector>
```

### Existing Secrets and authentication

Sensitive values (`huggingface.token`, `auth.apiKey`, control plane `key`,
PostgreSQL `url`/`username`/`password`, Redis `url`, and Oracle `dsn`/`user`/`password`)
accept either a string or a Kubernetes `secretKeyRef`. Strings are stored in the
chart-managed `<fullname>-secrets` Secret; existing Secret references avoid storing
credentials in Helm values. Secrets must exist in the release namespace.

```yaml
auth:
  apiKey:
    secretKeyRef: {name: smg-auth, key: api-key}
  controlPlaneApiKeys:
    - id: automation
      name: Automation
      role: admin
      key:
        secretKeyRef: {name: smg-auth, key: control-plane-key}
huggingface:
  token:
    secretKeyRef: {name: model-access, key: token}
history:
  backend: postgres
  postgres:
    host: postgres.database.svc
    port: 5432
    database: smg
    username:
      secretKeyRef: {name: smg-postgres, key: username}
    password:
      secretKeyRef: {name: smg-postgres, key: password}
    parameters: sslmode=require
```

PostgreSQL `url` takes precedence over composition and can reference a Secret
containing the entire connection URL. For composition, store URL-encoded username
and password values in the Secret (for example, `@` becomes `%40`). Kubernetes
expands the credential environment variables in the URL argument without a shell.
Control plane roles are `admin` or `user`; IDs and names cannot contain colons.
Secret updates require a pod restart to refresh environment variables.
