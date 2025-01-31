rank_choices=(1 2 4 8 10 15 20 30 40 50 60 70 80 90 100 130 150 180 200 230 250 280 300 350 400 450 500)
n_choices=(10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200 205 210)
L_choices=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95 2.0 2.05 2.1)

for idx in ${!n_choices[@]}; do
    for rank in ${rank_choices[@]}; do
        name=fiesta_${rank}_${n_choices[$idx]}.lua
        cat inputParts/inputPart1.lua > ${name}
        echo "Lx = ${L_choices[$idx]}" >> ${name}
        echo "Ly = ${L_choices[$idx]}" >> ${name}
        echo "Lz = ${L_choices[$idx]}" >> ${name}
        echo "ndim = 3" >> ${name}
        echo "ni = ${n_choices[$idx]}" >> ${name}
        echo "nj = ${n_choices[$idx]}" >> ${name}
        echo "nk = ${n_choices[$idx]}" >> ${name}
        echo "procsx = 1" >> ${name}
        echo "procsy = 1" >> ${name}
        echo "procsz = ${rank}" >> ${name}
        cat inputParts/inputPart2.lua >> ${name}
    done
done
