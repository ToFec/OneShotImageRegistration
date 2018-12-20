#!/bin/bash

plastimatch synth-vf --fixed img0.nii.gz --output defTmp.nii.gz --xf-trans 5 0 0
plastimatch synth-vf --fixed img0.nii.gz --gauss-center 40 30 30 --gauss-mag 3 3 3 --gauss-std 5 5 5 --output tmpGauss.nii.gz --xf-gauss
plastimatch multiply --output def0.nii.gz tmpGauss.nii.gz defTmp.nii.gz
plastimatch convert --fixed img0.nii.gz --input img0.nii.gz --xf def0.nii.gz --output img1.nii.gz


plastimatch synth-vf --fixed img0.nii.gz --output defTmp.nii.gz --xf-trans 0 5 0
plastimatch synth-vf --fixed img0.nii.gz --gauss-center 45 30 30 --gauss-mag 3 3 3 --gauss-std 5 5 5 --output tmpGauss.nii.gz --xf-gauss
plastimatch multiply --output def1.nii.gz tmpGauss.nii.gz defTmp.nii.gz
plastimatch convert --fixed img0.nii.gz --input img1.nii.gz --xf def1.nii.gz --output img2.nii.gz

plastimatch synth-vf --fixed img0.nii.gz --output defTmp.nii.gz --xf-trans -5 0 0
plastimatch synth-vf --fixed img0.nii.gz --gauss-center 45 35 30 --gauss-mag 3 3 3 --gauss-std 5 5 5 --output tmpGauss.nii.gz --xf-gauss
plastimatch multiply --output def2.nii.gz tmpGauss.nii.gz defTmp.nii.gz
plastimatch convert --fixed img0.nii.gz --input img2.nii.gz --xf def2.nii.gz --output img3.nii.gz

plastimatch synth-vf --fixed img0.nii.gz --output defTmp.nii.gz --xf-trans 0 -5 0
plastimatch synth-vf --fixed img0.nii.gz --gauss-center 40 35 30 --gauss-mag 3 3 3 --gauss-std 5 5 5 --output tmpGauss.nii.gz --xf-gauss
plastimatch multiply --output def3.nii.gz tmpGauss.nii.gz defTmp.nii.gz
plastimatch convert --fixed img0.nii.gz --input img3.nii.gz --xf def3.nii.gz --output img4.nii.gz

rm tmpGauss.nii.gz defTmp.nii.gz
