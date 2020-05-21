#!/usr/bin/env bash

touch err.txt

END=30

for ((i=1;i<=END;i++)) do
	sbatch error_bar.sh $i
done
