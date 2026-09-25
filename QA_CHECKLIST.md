# Checklist QA Ortho Terre (images portées, multi-marchés)

Verdict par image : PASS ou FAIL. Un seul item raté = FAIL (pas de soft-pass). Chaque FAIL liste l'item précis + la correction attendue. Ne pas inventer de critères hors checklist/brief.

## Comparaison obligatoire avec la source CA
1. Mise en page identique à la source CA (mêmes blocs, même ordre, même produit, mêmes personnes, même persona homme/femme/marque).
2. Taille finale = taille de la source CA (ou `target_size` du brief). Aucun texte, produit ou bandeau coupé.
3. Seuls changent : langue des textes, décor local, prix locaux.

## Texte
4. Langue native du marché, orthographe et accents exacts (ä ö ü ß å æ ø é è à ç, grec correct). Aucune lettre inventée, aucun mot déformé, aucun mélange de langues.
5. Aucun texte parasite ou résidu de la source (anglais/français québécois restant, mot tronqué, « (Das Beste) » ajouté, etc.).
6. Traduction fidèle du sens de la source (ex. « du meilleur au moins bon » bien rendu, naturel pour un natif).
7. « réduction » ou équivalent local (sconto, Rabatt, rabat, descuento, desconto, έκπτωση, korting, rabatt) ; jamais « rabais ».
8. Aucune mention de livraison (« Livraison express GRATUITE » doit être retiré) ni du nom du pays.
9. Pas de claims médicaux interdits : TENS, NMES, FDA, « guérit », promesses santé chiffrées non présentes dans la source.

## Prix et pourcentages
10. Prix = vrais prix du marché (prix_pays.json), bonne devise, format local. Jamais convertis ni inventés. Prix barrés = ceux du fichier.
11. « jusqu'à X % » = % max de la page produit du pays (be-nl/dk/it/gr 74, de 73, lu/es/pt 69, CH-DE 68, no/sv 50). Sans « jusqu'à » = 50 %.

## Visuel
12. Décor reconnaissable du marché (aucun élément québécois/canadien restant : Tremblant, lacs QC, drapeaux, panneaux).
13. Produit Ortho Terre intact (drap/tapis gris, câble, fiche) ; jamais un produit concurrent, jamais une boîte inventée.
14. Personnes, objets et véhicules intacts (pas de déformation, voiture cassée, mains ratées).
15. Net : pas de flou, pas d'artefacts, pas de bords étirés.

## Sortie
report.json : {country, src_ad, verdict, fails:[{item, détail, correction}]}.
