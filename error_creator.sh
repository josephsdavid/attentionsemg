#!/usr/bin/env bash


END=6

for ((i=1;i<=END;i++)) do
	sbatch error_bar.sh $i
done
