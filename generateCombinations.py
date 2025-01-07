#apps = ["beatnik", "fiesta", "lammps", "lulesh", "minife"]
apps = ["fiesta", "lammps", "lulesh", "minife"]

combinations = []

for app1 in apps:
    for app2 in apps:
        if (app1 != app2):
            comb = set([app1,app2])
            if (comb not in combinations):
                combinations.append(comb)

print(combinations)
print(len(combinations))
