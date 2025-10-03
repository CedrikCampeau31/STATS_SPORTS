# STATS_SPORTS
# ML Agent Loop (Rube + GitHub Actions)

Ce dépôt est structuré pour permettre à un agent IA (via Rube/MCP + ChatGPT) de:
1) Lire le code et la config,
2) Proposer de petits patches (PR),
3) Déclencher des expériences (`workflow_dispatch`),
4) Récupérer les métriques (`artifacts/results.json`),
5) Itérer jusqu’à améliorer l’objectif.

## Objectif
`maximize:F1_macro` sur l’ensemble de validation.

## Arborescence
