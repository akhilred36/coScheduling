rank_choices=(1 2 3 4)
lattice_choices=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)
mesh_choices=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150)

for mesh in ${mesh_choices[@]}; do
    for lattice in ${lattice_choices[@]}; do
        name=in_${mesh}_${lattice}.lj
        echo "# 3d Lennard-Jones melt" > ${name}
        echo "" >> ${name}
        echo "units       lj" >> ${name}
        echo "atom_style  atomic">> ${name}
        echo "" >> ${name}
        echo "newton off" >> ${name}
        echo "lattice     fcc ${lattice}" >> ${name}
        echo "region      box block 0 ${mesh} 0 ${mesh} 0 ${mesh}" >> ${name}
        echo "create_box  1 box" >> ${name}
        echo "create_atoms    1 box" >> ${name}
        echo "mass        1 2.0" >> ${name}
        echo "" >> ${name}
        echo "velocity    all create 1.4 87287 loop geom" >> ${name}
        echo "" >> ${name}
        echo "pair_style  lj/cut 2.5" >> ${name}
        echo "pair_coeff  1 1 1.0 1.0 2.5" >> ${name}
        echo "" >> ${name}
        echo "neighbor    0.3 bin" >> ${name}
        echo "neigh_modify delay 0 every 20 check no" >> ${name}
        echo "fix     1 all nve" >> ${name}
        echo "thermo 10" >> ${name}
        echo "" >> ${name}
        echo "run     100" >> ${name}
    done
done
