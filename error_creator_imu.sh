#!/usr/bin/env bash

END=6

for ((i=1;i<=END;i++)) do
	sbatch errorbar_imu.sh $i
done
