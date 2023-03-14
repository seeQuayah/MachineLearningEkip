#!/bin/bash

# variables
LINK="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv"
ANSWER="n"


echo -e """This script will download the dataset into $PWD/dataset/
Are you sure you want to continue? [y/n] (default: n)
"""

read ANSWER
if [ "$ANSWER" = "y" ]; then
    echo "Downloading dataset..."
    wget -O dataset/dataset.csv $LINK
    echo "Done."
else
    echo "Exiting..."
    exit 0
fi
