load trajectory.pdb
hide all
load_traj trajectory.xtc
intra_fit all
show cartoon, all
show sticks, resi 149 or resi 181
dist resi 149 and name CA, resi 181 and name HE2
select dihedral, (resi 180 and name CA) or (resi 180 and name N) or (resi 181 and name N) or (resi 181 and name CG)
show sticks, resi 180-181
color red, dihedral