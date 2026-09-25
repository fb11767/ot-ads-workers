# ot-ads-workers

Workers indépendants pour adapter des visuels publicitaires Ortho Terre (draps de mise à la terre) pays par pays. Un orchestrateur externe lance plusieurs cloud agents en parallèle : chacun reçoit une tranche de file et les fichiers d'entrée, puis exécute un worker.

Le modèle, le cadrage et le prompt reprennent la chaîne actuelle : image complétée par reflet jusqu'au ratio le plus proche accepté par Replicate, édition `google/nano-banana-pro` (`resolution` 2K, `aspect_ratio=match_input_image`, `output_format=jpg`), puis recadrage aux dimensions exactes de la source. Le JPG final est écrit en qualité 92.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

Dépendances : `requests`, `Pillow`, `replicate`.

## Format de la file

La file est un fichier JSON : une liste d'objets. Champs requis :

| Champ | Rôle |
| --- | --- |
| `country` | Code pays du dossier (`it`, `de`, `be-nl`, …) |
| `src_ad` | Identifiant du visuel. Le fichier final s'appelle `<src_ad>.jpg` |
| `class` | Classe du brief (`B`, `C`, …), transmise au prompt |
| `brief` | Chemin du brief JSON |
| `copy` | Chemin du texte localisé (copie publicitaire) |
| `out` | Ancien chemin de sortie sur la machine d'origine. Il doit contenir le segment `<country>/`. Le worker n'écrit pas à cet emplacement |

Les autres champs (par exemple `priority_purchases`) sont ignorés. L'ordre de la liste est l'ordre de traitement.

Exemple :

```json
[
  {
    "priority_purchases": 13,
    "country": "it",
    "src_ad": "120249941811570637",
    "class": "C",
    "brief": "/workspace/ot_queue_0925/it/briefs/120249941811570637.json",
    "copy": "/workspace/ot_queue_0925/it/copies/120249941811570637.txt",
    "out": "/workspace/ot_queue_0925/it/adapted/120249941811570637.jpg"
  }
]
```

Les chemins viennent d'une autre machine (`/workspace/ot_queue_0925/...`). Le worker ne garde que la partie à partir du segment `<country>/` et la résout dans `--inputs`. L'exemple ci-dessus devient `it/briefs/120249941811570637.json` et `it/copies/120249941811570637.txt`.

Un brief avec `"hold": true` est enregistré avec le statut `hold` et n'appelle pas Replicate.

## Dossier d'entrées

`--inputs` est la racine qui contient un dossier par pays :

```text
ENTREES/
  it/
    briefs/120249941811570637.json
    copies/120249941811570637.txt
    sources/120249941811570637.jpg
  de/
    briefs/...
    copies/...
```

L'image source est celle du champ `source_image_path` du brief, dans cet ordre :

1. le même découpage `<country>/...` sous `--inputs`, si ce chemin existe ;
2. le chemin absolu, s'il existe sur la machine du worker ;
3. un fichier unique portant le nom de fichier de `source_image_path` quelque part sous `--inputs` ;
4. `<country>/sources/<src_ad>.jpg` (aussi `.jpeg`, `.png`, `.webp`), puis le même nom ailleurs sous `--inputs`.

Plusieurs fichiers du même nom pour un item arrêtent cet item avec une erreur.

## Variables

| Variable | Rôle |
| --- | --- |
| `REPLICATE_API_TOKEN` | Seul endroit où le jeton est lu. Obligatoire hors `--dry-run`. S'il est absent ou vide, le worker s'arrête avec un message clair et le code 1 |

Le worker n'écrit pas le jeton, ne l'affiche pas et il ne doit pas être commité. Vérification :

```bash
sh worker/check_env.sh
```

La sortie est exactement `REPLICATE_API_TOKEN set` ou `missing`.

## Lancer un worker

Depuis la racine du dépôt, pour une tranche déjà déposée avec ses entrées :

```bash
python3 worker/run_queue.py --queue /chemin/tranche.json --inputs /chemin/entrees --artifacts ./artifacts --max-per-minute 6
```

`--model` vaut `google/nano-banana-pro` par défaut. `--timeout` (défaut 900) est l'attente maximale d'une prédiction, en secondes. `--force` régénère un JPG déjà présent.

Valider la file et les chemins sans appel Replicate (le jeton n'est pas requis) :

```bash
python3 worker/run_queue.py --queue /chemin/tranche.json --inputs /chemin/entrees --artifacts ./artifacts --dry-run
```

## Sorties

Pour chaque item traité :

```text
artifacts/<country>/<src_ad>.jpg
```

Même ratio que la source (3:4 ou 4:5, ou le ratio réel du fichier) et mêmes dimensions en pixels. JPEG qualité 92.

Le rapport `artifacts/report.json` est réécrit après chaque item. Il contient le modèle, le plafond par minute, et pour chaque item : `country`, `src_ad`, `class`, `status`, `model`, `prediction_id`, `out`, `error`, `width`, `height`.

Statuts : `ok`, `skipped` (le JPG existe déjà), `dry_run`, `hold`, `error`. Le code de sortie est 1 si un item est en `error`.

## Débit et reprise

`--max-per-minute` (défaut 6) limite le nombre de prédictions créées par minute. Un HTTP 429 ou 5xx, ainsi qu'une coupure réseau, est réessayé avec une attente exponentielle (2 s, 4 s, 8 s, … plafonnée à 60 s, 6 tentatives). Si `artifacts/<country>/<src_ad>.jpg` existe et n'est pas vide, l'item est sauté.

## Consigne pour un cloud agent worker

Après un passage réel du worker (hors simple contrôle `--dry-run`) :

1. Copier chaque JPG final dans `/opt/cursor/artifacts/<country>/` en conservant le nom `<src_ad>.jpg`.
2. Lister ces chemins `/opt/cursor/artifacts/<country>/<src_ad>.jpg` dans le rapport final de l'agent.
3. Ne pas copier le jeton, ne pas l'afficher, ne pas le committer.
