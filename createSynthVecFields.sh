#!/bin/bash

plastimatch synth --pattern sphere --background 0.1 --foreground 1 --sphere-center '32 32 32' --sphere-radius 15 --dim '64 56 60' --spacing 1 1 1 --origin '0 0 0' --output img0.nii.gz

plastimatch synth-vf --fixed img0.nii.gz --output defTmp.nrrd --xf-trans '5 0 0'
plastimatch synth-vf --fixed img0.nii.gz --gauss-center '40 30 30' --gauss-mag '3 3 3' --gauss-std '5 5 5' --output tmpGauss.nrrd --xf-gauss
unu 2op "*" defTmp.nrrd tmpGauss.nrrd -o def0.nrrd
plastimatch convert --fixed img0.nii.gz --input img0.nii.gz --xf def0.nrrd --output-img img1.nii.gz


plastimatch synth-vf --fixed img0.nii.gz --output defTmp.nrrd --xf-trans '5 0 0'
plastimatch synth-vf --fixed img0.nii.gz --gauss-center '35 30 30' --gauss-mag '3 3 3' --gauss-std '5 5 5' --output tmpGauss.nrrd --xf-gauss
unu 2op "*" defTmp.nrrd tmpGauss.nrrd -o def1.nrrd
plastimatch convert --fixed img0.nii.gz --input img1.nii.gz --xf def1.nrrd --output-img img2.nii.gz

plastimatch synth-vf --fixed img0.nii.gz --output defTmp.nrrd --xf-trans '5 0 0'
plastimatch synth-vf --fixed img0.nii.gz --gauss-center '30 30 30' --gauss-mag '3 3 3' --gauss-std '5 5 5' --output tmpGauss.nrrd --xf-gauss
unu 2op "*" defTmp.nrrd tmpGauss.nrrd -o def2.nrrd
plastimatch convert --fixed img0.nii.gz --input img1.nii.gz --xf def2.nrrd --output-img img3.nii.gz

#plastimatch synth-vf --fixed img0.nii.gz --output defTmp.nrrd --xf-trans '-5 0 0'
#plastimatch synth-vf --fixed img0.nii.gz --gauss-center '30 30 30' --gauss-mag '3 3 3' --gauss-std '5 5 5' --output tmpGauss.nrrd --xf-gauss
#unu 2op "*" defTmp.nrrd tmpGauss.nrrd -o def3.nrrd
#plastimatch convert --fixed img0.nii.gz --input img2.nii.gz --xf def3.nrrd --output-img img4.nii.gz

#plastimatch synth-vf --fixed img0.nii.gz --output defTmp.nrrd --xf-trans '-5 0 0'
#plastimatch synth-vf --fixed img0.nii.gz --gauss-center '35 30 30' --gauss-mag '3 3 3' --gauss-std '5 5 5' --output tmpGauss.nrrd --xf-gauss
#unu 2op "*" defTmp.nrrd tmpGauss.nrrd -o def4.nrrd
#plastimatch convert --fixed img0.nii.gz --input img3.nii.gz --xf def4.nrrd --output-img img5.nii.gz

cp img2.nii.gz img4.nii.gz
cp img1.nii.gz img5.nii.gz

rm tmpGauss.nrrd defTmp.nrrd
