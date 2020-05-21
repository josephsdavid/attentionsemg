#!/usr/bin/env bash

END=30

for ((i=1;i<=END;i++)) do
	sbatch errorbar_imu.sh $i
done
