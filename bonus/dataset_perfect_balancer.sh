#!/bin/bash

# Vérifie si un fichier a été fourni
if [ $# -ne 1 ]; then
    echo "USAGE: $0 <file.txt>"
    exit 1
fi

FILE=$1

# Vérifie si le fichier existe
if [ ! -f "$FILE" ]; then
    echo "Erreur : le fichier $FILE n'existe pas."
    exit 1
fi

# Compte les occurrences exactes des mots
NOTHING_COUNT=$(grep -ow "Nothing" "$FILE" | wc -l)
CHECKMATE_COUNT=$(grep -ow "Checkmate" "$FILE" | wc -l)
CHECK_COUNT=$(grep -ow "Check" "$FILE" | wc -l)
STALEMATE_COUNT=$(grep -ow "Stalemate" "$FILE" | wc -l)

# Affiche les occurrences actuelles
echo "Occurrences actuelles :"
echo "Nothing    : $NOTHING_COUNT"
echo "Checkmate  : $CHECKMATE_COUNT"
echo "Check      : $CHECK_COUNT"
echo "Stalemate  : $STALEMATE_COUNT"

# Trouve la catégorie avec le moins d'occurrences, en ignorant les 0
MIN_COUNT=$(echo -e "$NOTHING_COUNT\n$CHECKMATE_COUNT\n$CHECK_COUNT\n$STALEMATE_COUNT" | grep -v '^0$' | sort -n | head -n 1)

# Équilibre les données
TEMP_FILE=$(mktemp)
BALANCED_FILE="${FILE%.txt}_equ.txt"

echo "Équilibrage des données..."
echo "Nombre de lignes par catégorie : $MIN_COUNT"

# Limite chaque catégorie au nombre minimum d'occurrences
for CATEGORY in "Nothing" "Checkmate" "Check" "Stalemate"; do
    grep -w "$CATEGORY" "$FILE" | head -n "$MIN_COUNT" >> "$TEMP_FILE"
done

# Mélange les lignes pour éviter les biais d'ordre
shuf "$TEMP_FILE" > "$BALANCED_FILE"

# Supprime le fichier temporaire
rm "$TEMP_FILE"

# Compte les nouvelles occurrences exactes
NEW_COUNTS=()
for CATEGORY in "Nothing" "Checkmate" "Check" "Stalemate"; do
    NEW_COUNTS+=( $(grep -ow "$CATEGORY" "$BALANCED_FILE" | wc -l) )
done

# Affiche le résumé final
echo "Nouveau fichier équilibré : $BALANCED_FILE"
echo "Nouvelles occurrences :"
echo "Nothing    : ${NEW_COUNTS[0]}"
echo "Checkmate  : ${NEW_COUNTS[1]}"
echo "Check      : ${NEW_COUNTS[2]}"
echo "Stalemate  : ${NEW_COUNTS[3]}"