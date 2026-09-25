#!/bin/sh
# Affiche uniquement l'état du jeton, jamais sa valeur.
if [ -n "$(printf '%s' "${REPLICATE_API_TOKEN-}" | tr -d '[:space:]')" ]; then
  printf '%s\n' "REPLICATE_API_TOKEN set"
else
  printf '%s\n' "missing"
  exit 1
fi
