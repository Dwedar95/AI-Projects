# Kubernetes manifests

These manifests describe how the FastAPI inference service is meant to be deployed:
a Deployment with readiness and liveness probes, a ClusterIP Service, a CPU-based
HorizontalPodAutoscaler, a ConfigMap with service settings and team budgets, and a
read-only volume claim that holds the model registry.

**Validation status: syntax only.** The files were parsed with PyYAML and checked for
internal consistency (selectors, port names, volume and ConfigMap references) by
`tests/test_deploy.py`. They were NOT applied to a cluster, not run through
`kubectl apply --dry-run` or `kubeconform`, and the container image was not built,
because neither a cluster nor a Docker daemon was available in the authoring environment.

Things to adapt before use:

- `image:` is a placeholder (`ghcr.io/example/...`).
- The registry PVC uses `ReadOnlyMany`, which needs a storage class that supports it.
  An alternative is to bake the production model into the image or pull it in an init container.
- The usage log is written to an `emptyDir`, so metering data is lost when a pod restarts
  and is not shared between replicas. Per-team budgets are therefore enforced per pod.
  A shared store is required for real budget enforcement.
