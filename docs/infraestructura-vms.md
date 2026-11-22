# Inventario de VMs

Relevamiento inicial (2026-11-21). Las tres VMs usan Rocky Linux 9.8 (Blue Onyx), Docker 29.7.2 y el usuario `estudiante`.

## Recursos

| # | IP | Hostname | vCPU | RAM total | RAM disponible | Swap | Disco `/` | Disco libre | Uso |
|---|----|----------|------|-----------|----------------|------|-----------|-------------|-----|
| 1 | 10.43.97.92  | nodo-serving | 4 | 15 GiB | 12 GiB | 4 GiB | 24 GB | 20 GB | 20 % |
| 2 | 10.43.97.105 | nodo-data | 4 | 15 GiB | 13 GiB | 4 GiB | 24 GB | 21 GB | 17 % |
| 3 | 10.43.97.106 | NBAM17 | 4 | 15 GiB | 12 GiB | 4 GiB | 24 GB | 20 GB | 19 % |

**Total del clúster:** 12 vCPU, ~45 GiB de RAM (~37 GiB disponibles), ~61 GB de disco libre.

## Software instalado y firewall

| IP | firewalld | nftables | Binarios en `/usr/local/bin` |
|----|-----------|----------|------------------------------|
| 10.43.97.92  | active   | active   | rke2, rancher-system-agent, kubectl |
| 10.43.97.105 | inactive | active   | rancher-system-agent, kubectl (sin rke2) |
| 10.43.97.106 | inactive | inactive | kind, helm, kubectl |

## Límites a tener en cuenta

- **Disco (el recurso más justo):** solo ~20 GB libres por nodo. Las imágenes Docker de ML (Python + scikit-learn/MLflow) pesan entre 1 y 3 GB cada una, y Kubernetes empieza a desalojar pods cuando el disco pasa del ~85 % de uso (≈ 3,6 GB libres). Hay que limpiar imágenes con frecuencia (`docker image prune`) y no guardar datasets ni artefactos grandes en el disco local.
- **RAM:** ~12 GiB disponibles por nodo. RKE2/Rancher ya consume unos 2–2,5 GiB. Conviene que cada pod declare `requests`/`limits` y que ninguno pida más de ~8 GiB.
- **CPU:** 4 vCPU por nodo. Entrenamiento y serving deberían ir en nodos distintos para que no compitan entre sí.
- **Swap activo (4 GiB):** RKE2/kubelet lo tolera, pero tener swap encendido en nodos de Kubernetes puede generar avisos o comportamientos inesperados.

## Pendientes

- [ ] Verificar si 10.43.97.105 necesita rke2 (tiene el agente de Rancher pero no el binario `rke2`).
- [ ] Unificar la configuración de firewall entre los nodos (cada uno está distinto).
