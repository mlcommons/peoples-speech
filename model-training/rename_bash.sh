#!/bin/bash
for dir in *; do
    if [ -d "$dir" ]; then
    a=0
    for files in ./$dir/*tar; do
    new=$(printf "$dir""_%06d.tar" "$a")
    echo $new
    mv -- "$files" "$dir"/"$new"
    let a=a+1
    done
    fi
done
