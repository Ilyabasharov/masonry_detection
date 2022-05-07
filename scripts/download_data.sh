#!/bin/bash

#download weights for nets

OUTPUT_PATH=../data
mkdir -p ${OUTPUT_PATH}

#for detection
wget http://mplab.sztaki.hu/~bcsaba/test/PPKE-SZTAKI-MasonryBenchmark.zip -O ${OUTPUT_PATH}/PPKE-SZTAKI-MasonryBenchmark.zip
unzip ${OUTPUT_PATH}/PPKE-SZTAKI-MasonryBenchmark.zip -d ${OUTPUT_PATH}
rm ${OUTPUT_PATH}/PPKE-SZTAKI-MasonryBenchmark.zip
