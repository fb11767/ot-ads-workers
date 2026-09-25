# ot-ads-workers

Workers indépendants pour adapter des visuels publicitaires Ortho Terre (draps de mise à la terre) pays par pays. Un orchestrateur externe lance plusieurs cloud agents en parallèle : chacun reçoit une tranche de file et les fichiers d'entrée, puis exécute un worker.

Le modèle, le cadrage et le prompt reprennent la chaîne actuelle : image complétée par reflet jusqu'au ratio le plus proche accepté par Replicate, édition `google/nano-banana-pro` (`resolution` 2K, `aspect_ratio=match_input_image`, `output_format=jpg`), puis recadrage aux dimensions exactes de la source. Le JPG final est écrit en qualité 92.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

Dépendances : `requests`, `Pillow`, `replicate`, `numpy`.

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

## Checklist QA

`QA_CHECKLIST.md`, à la racine du dépôt, est la seule grille : 15 items, PASS ou FAIL. Un item raté suffit pour FAIL. Il n'y a pas de soft-pass. Le worker ne charge pas d'autre liste de critères.

Le rapport d'une image est exactement :

```json
{"country": "de", "src_ad": "1202", "verdict": "FAIL", "fails": [{"item": 4, "detail": "accent missing", "correction": "Render für exactly."}]}
```

`detail` est le champ « détail » de la checklist. `item` vaut 1 à 15, ou `"model"` quand l'appel vision échoue ou reste incertain. Une erreur ou une incertitude du modèle est un FAIL, jamais un PASS.

Le modèle vision par défaut est `openai/gpt-5` sur Replicate (`--qa-model`). Il reçoit la checklist entière, le brief, l'image adaptée et la source CA, et doit comparer chaque ligne de texte aux `expected_lines` du brief (accents, aucun mot en trop). Seul `REPLICATE_API_TOKEN` est lu.

Items décidés dans le code, avant le modèle : fichier lisible, taille égale à `target_size`, heuristique de flou (variance du laplacien, comparée à la source CA). Le modèle ne peut pas annuler ces FAIL.

### Brief QA

```json
{
  "country": "de",
  "src_ad": "1202",
  "expected_lines": ["bis zu 73 % Rabatt"],
  "expected_lines_exhaustive": true,
  "prices": ["29,90 €"],
  "percentage": 73,
  "target_size": [1080, 1350],
  "pitfalls": ["ne pas écrire rabais"]
}
```

`expected_lines_exhaustive` vaut true par défaut : ces lignes sont tout le texte autorisé. Dans une file de correction, les phrases entre guillemets « » sont des lignes obligatoires, pas forcément le texte complet, sauf si l'item fournit `expected_lines`.

### QA seul, sur un dossier

Chaque sous-dossier contient `brief.json`, une image adaptée (`adapted.jpg`, `image.jpg`, `out.jpg` ou `final.jpg`) et la source CA (`ca_source.jpg`, `ca.jpg`, `source.jpg` ou `reference.jpg`). Un dossier plat avec ces fichiers, ou un `manifest.json` (liste d'objets `adapted`, `ca_source`, `brief`), convient aussi.

```bash
python3 worker/run_queue.py --mode qa --qa-folder /chemin/dossier --artifacts /opt/cursor/artifacts
```

La même commande accepte une liste JSON à la place du dossier :

```bash
python3 worker/run_queue.py --mode qa --queue /chemin/qa_items.json --inputs /chemin/entrees --artifacts /opt/cursor/artifacts
```

### Fix et QA automatique

Le mode `fix` (moteur `gpt-image` par défaut) envoie l'image entière à `openai/gpt-image-2.5-sunburst`, sans recadrer un morceau pour le recoller. Si le QA est FAIL, les `correction` sont ajoutées au prompt suivant, jusqu'à 3 essais. L'essai conservé est le PASS, sinon celui qui a le moins d'items en échec.

```bash
python3 worker/run_queue.py --mode fix --queue /chemin/fix_queue.json --inputs /chemin/entrees --artifacts /opt/cursor/artifacts --indices 0-5 --engine gpt-image
```

Cinq ou six workers parallèles prennent des plages différentes, par exemple `--indices 0-4` et `--indices 5-9`. `0-4` est inclusif, `0:5` exclut la fin, `1,4,8` choisit des index. Chaque worker écrit son résumé à part (`batch_summary.json`, ou `batch_summary_0-4.json`) pour ne pas s'écraser. `--image-quality low` réduit le coût des essais. `--no-qa` retrouve l'ancien contrôle, sans cette boucle.

### Sorties QA

```text
/opt/cursor/artifacts/<country>/<src_ad>.jpg
/opt/cursor/artifacts/<country>/<src_ad>/report.json
/opt/cursor/artifacts/batch_summary.json
```

Le résumé donne les comptes PASS et FAIL, les chemins, et une estimation de coût (jetons vision et images générées, d'après les prix publics Replicate).

## Consigne pour un cloud agent worker

Après un passage réel du worker (hors simple contrôle `--dry-run`) :

1. Copier chaque JPG final dans `/opt/cursor/artifacts/<country>/` en conservant le nom `<src_ad>.jpg`.
2. Lister ces chemins `/opt/cursor/artifacts/<country>/<src_ad>.jpg` dans le rapport final de l'agent.
3. Ne pas copier le jeton, ne pas l'afficher, ne pas le committer.
