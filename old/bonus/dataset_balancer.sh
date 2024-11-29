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

# Identifie la catégorie la plus fréquente et la deuxième plus fréquente
declare -A COUNTS
COUNTS=( ["Nothing"]=$NOTHING_COUNT ["Checkmate"]=$CHECKMATE_COUNT ["Check"]=$CHECK_COUNT ["Stalemate"]=$STALEMATE_COUNT )

# Trie les catégories par fréquence
SORTED_COUNTS=($(for key in "${!COUNTS[@]}"; do echo "${COUNTS[$key]} $key"; done | sort -nr | awk '{print $2}'))

MOST_FREQUENT=${SORTED_COUNTS[0]}
SECOND_MOST_FREQUENT_COUNT=${COUNTS[${SORTED_COUNTS[1]}]}

# Équilibre les données
TEMP_FILE=$(mktemp)
BALANCED_FILE="${FILE%.txt}_balanced.txt"

echo "Équilibrage des données..."
echo "Catégorie dominante : $MOST_FREQUENT ($COUNTS)"

# Réduit la catégorie dominante pour être environ égale à la deuxième plus fréquente
if [ "${COUNTS[$MOST_FREQUENT]}" -gt "$SECOND_MOST_FREQUENT_COUNT" ]; then
    grep -w "$MOST_FREQUENT" "$FILE" | head -n "$SECOND_MOST_FREQUENT_COUNT" >> "$TEMP_FILE"
else
    grep -w "$MOST_FREQUENT" "$FILE" >> "$TEMP_FILE"
fi

# Ajoute les autres catégories sans les modifier
for CATEGORY in "Nothing" "Checkmate" "Check" "Stalemate"; do
    if [ "$CATEGORY" != "$MOST_FREQUENT" ] && [ "${COUNTS[$CATEGORY]}" -gt 0 ]; then
        grep -w "$CATEGORY" "$FILE" >> "$TEMP_FILE"
    fi
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
